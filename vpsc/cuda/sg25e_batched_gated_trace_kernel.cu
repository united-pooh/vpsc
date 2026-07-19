#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


namespace {

void check_float_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__device__ __forceinline__ float surrogate_derivative(float value, float scale) {
  const float denominator = 1.0f + scale * fabsf(value);
  return scale / (denominator * denominator);
}

__device__ __forceinline__ int valid_query_count(
    const int64_t* query_indices, int query_capacity) {
  int count = 0;
  while (count < query_capacity && query_indices[count] >= 0) {
    ++count;
  }
  return count;
}

__global__ void batched_gated_trace_forward_kernel(
    const float* __restrict__ drives,
    const int64_t* __restrict__ query_indices,
    const float* __restrict__ decays,
    const float* __restrict__ initial_e,
    const float* __restrict__ initial_i,
    float* __restrict__ raw,
    float* __restrict__ final_e,
    float* __restrict__ final_i,
    float* __restrict__ previous,
    float* __restrict__ writes,
    int batch,
    int time_steps,
    int state_dim,
    int query_capacity,
    float spike_threshold) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * state_dim;
  if (linear >= total) {
    return;
  }
  const int batch_index = linear / state_dim;
  const int state_index = linear - batch_index * state_dim;
  const int64_t* row_queries =
      query_indices + batch_index * query_capacity;
  const int row_query_count = valid_query_count(row_queries, query_capacity);
  const float decay_e = decays[state_index];
  const float decay_i = decays[state_dim + state_index];
  float trace_e = initial_e[linear];
  float trace_i = initial_i[linear];
  int query_cursor = 0;

  for (int time_index = 0; time_index < time_steps; ++time_index) {
    const int drive_base =
        (batch_index * time_steps + time_index) * (4 * state_dim);
    const float content_e =
        drives[drive_base + state_index] >= 0.0f ? 1.0f : 0.0f;
    const float content_i =
        drives[drive_base + state_dim + state_index] >= 0.0f ? 1.0f : 0.0f;
    const float gate_e =
        drives[drive_base + 2 * state_dim + state_index] >= 0.0f ? 1.0f : 0.0f;
    const float gate_i =
        drives[drive_base + 3 * state_dim + state_index] >= 0.0f ? 1.0f : 0.0f;
    const float write_e = content_e * gate_e;
    const float write_i = content_i * gate_i;
    const int pair_base =
        (batch_index * time_steps + time_index) * (2 * state_dim);
    previous[pair_base + state_index] = trace_e;
    previous[pair_base + state_dim + state_index] = trace_i;
    writes[pair_base + state_index] = write_e;
    writes[pair_base + state_dim + state_index] = write_i;
    trace_e = decay_e * trace_e + (1.0f - decay_e) * write_e;
    trace_i = decay_i * trace_i + (1.0f - decay_i) * write_i;

    if (query_cursor < row_query_count &&
        row_queries[query_cursor] == time_index) {
      const int raw_base =
          (batch_index * query_capacity + query_cursor) * (4 * state_dim);
      raw[raw_base + state_index] =
          trace_e >= spike_threshold ? 1.0f : 0.0f;
      raw[raw_base + state_dim + state_index] =
          trace_i >= spike_threshold ? -1.0f : 0.0f;
      raw[raw_base + 2 * state_dim + state_index] = trace_e;
      raw[raw_base + 3 * state_dim + state_index] = -trace_i;
      ++query_cursor;
    }
  }
  final_e[linear] = trace_e;
  final_i[linear] = trace_i;
}

__global__ void batched_gated_trace_backward_kernel(
    const float* __restrict__ grad_raw,
    const float* __restrict__ grad_final_e,
    const float* __restrict__ grad_final_i,
    const float* __restrict__ drives,
    const int64_t* __restrict__ query_indices,
    const float* __restrict__ decays,
    const float* __restrict__ previous,
    const float* __restrict__ writes,
    const float* __restrict__ raw,
    float* __restrict__ grad_drives,
    float* __restrict__ grad_decays,
    float* __restrict__ grad_initial_e,
    float* __restrict__ grad_initial_i,
    int batch,
    int time_steps,
    int state_dim,
    int query_capacity,
    float spike_threshold,
    float surrogate_scale) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * state_dim;
  if (linear >= total) {
    return;
  }
  const int batch_index = linear / state_dim;
  const int state_index = linear - batch_index * state_dim;
  const int64_t* row_queries =
      query_indices + batch_index * query_capacity;
  const int row_query_count = valid_query_count(row_queries, query_capacity);
  const float decay_e = decays[state_index];
  const float decay_i = decays[state_dim + state_index];
  float adjoint_e = 0.0f;
  float adjoint_i = 0.0f;
  float decay_gradient_e = 0.0f;
  float decay_gradient_i = 0.0f;
  int query_cursor = row_query_count - 1;

  for (int time_index = time_steps - 1; time_index >= 0; --time_index) {
    float direct_e = 0.0f;
    float direct_i = 0.0f;
    if (query_cursor >= 0 && row_queries[query_cursor] == time_index) {
      const int raw_base =
          (batch_index * query_capacity + query_cursor) * (4 * state_dim);
      const float trace_e = raw[raw_base + 2 * state_dim + state_index];
      const float trace_i = -raw[raw_base + 3 * state_dim + state_index];
      const float spike_signal_e = grad_raw[raw_base + state_index];
      const float spike_signal_i =
          -grad_raw[raw_base + state_dim + state_index];
      const float trace_signal_e =
          grad_raw[raw_base + 2 * state_dim + state_index];
      const float trace_signal_i =
          -grad_raw[raw_base + 3 * state_dim + state_index];
      direct_e = trace_signal_e + spike_signal_e * surrogate_derivative(
          trace_e - spike_threshold, surrogate_scale);
      direct_i = trace_signal_i + spike_signal_i * surrogate_derivative(
          trace_i - spike_threshold, surrogate_scale);
      --query_cursor;
    }
    if (time_index == time_steps - 1) {
      direct_e += grad_final_e[linear];
      direct_i += grad_final_i[linear];
    }
    adjoint_e = direct_e + decay_e * adjoint_e;
    adjoint_i = direct_i + decay_i * adjoint_i;

    const int pair_base =
        (batch_index * time_steps + time_index) * (2 * state_dim);
    const float previous_e = previous[pair_base + state_index];
    const float previous_i = previous[pair_base + state_dim + state_index];
    const float write_e = writes[pair_base + state_index];
    const float write_i = writes[pair_base + state_dim + state_index];
    decay_gradient_e += adjoint_e * (previous_e - write_e);
    decay_gradient_i += adjoint_i * (previous_i - write_i);

    const int drive_base =
        (batch_index * time_steps + time_index) * (4 * state_dim);
    const float drive_content_e = drives[drive_base + state_index];
    const float drive_content_i = drives[drive_base + state_dim + state_index];
    const float drive_gate_e = drives[drive_base + 2 * state_dim + state_index];
    const float drive_gate_i = drives[drive_base + 3 * state_dim + state_index];
    const float content_e = drive_content_e >= 0.0f ? 1.0f : 0.0f;
    const float content_i = drive_content_i >= 0.0f ? 1.0f : 0.0f;
    const float gate_e = drive_gate_e >= 0.0f ? 1.0f : 0.0f;
    const float gate_i = drive_gate_i >= 0.0f ? 1.0f : 0.0f;
    const float scale_e = (1.0f - decay_e) * adjoint_e;
    const float scale_i = (1.0f - decay_i) * adjoint_i;
    grad_drives[drive_base + state_index] = scale_e * gate_e *
        surrogate_derivative(drive_content_e, surrogate_scale);
    grad_drives[drive_base + state_dim + state_index] = scale_i * gate_i *
        surrogate_derivative(drive_content_i, surrogate_scale);
    grad_drives[drive_base + 2 * state_dim + state_index] = scale_e * content_e *
        surrogate_derivative(drive_gate_e, surrogate_scale);
    grad_drives[drive_base + 3 * state_dim + state_index] = scale_i * content_i *
        surrogate_derivative(drive_gate_i, surrogate_scale);
  }

  grad_initial_e[linear] = decay_e * adjoint_e;
  grad_initial_i[linear] = decay_i * adjoint_i;
  atomicAdd(grad_decays + state_index, decay_gradient_e);
  atomicAdd(grad_decays + state_dim + state_index, decay_gradient_i);
}

}  // namespace


std::vector<torch::Tensor> sg25e_batched_gated_trace_forward_cuda(
    torch::Tensor drives,
    torch::Tensor query_indices,
    torch::Tensor decays,
    torch::Tensor initial_e,
    torch::Tensor initial_i,
    double spike_threshold,
    double surrogate_scale) {
  check_float_cuda_contiguous(drives, "drives");
  check_float_cuda_contiguous(decays, "decays");
  check_float_cuda_contiguous(initial_e, "initial_e");
  check_float_cuda_contiguous(initial_i, "initial_i");
  TORCH_CHECK(query_indices.is_cuda(), "query_indices must be CUDA");
  TORCH_CHECK(
      query_indices.scalar_type() == at::kLong,
      "query_indices must be int64");
  TORCH_CHECK(query_indices.is_contiguous(), "query_indices must be contiguous");
  TORCH_CHECK(drives.dim() == 3, "drives must be [batch,time,4*state]");
  TORCH_CHECK(
      decays.dim() == 2 && decays.size(0) == 2,
      "decays must be [2,state]");
  TORCH_CHECK(
      query_indices.dim() == 2 && query_indices.size(1) > 0,
      "query_indices must be non-empty [batch,query]");
  const int64_t batch = drives.size(0);
  const int64_t time_steps = drives.size(1);
  const int64_t state_dim = decays.size(1);
  const int64_t query_capacity = query_indices.size(1);
  TORCH_CHECK(query_indices.size(0) == batch, "query batch mismatch");
  TORCH_CHECK(drives.size(2) == 4 * state_dim, "drive width mismatch");
  TORCH_CHECK(
      initial_e.dim() == 2 && initial_e.size(0) == batch &&
          initial_e.size(1) == state_dim,
      "initial_e shape mismatch");
  TORCH_CHECK(
      initial_i.dim() == 2 && initial_i.size(0) == batch &&
          initial_i.size(1) == state_dim,
      "initial_i shape mismatch");
  c10::cuda::CUDAGuard device_guard(drives.device());
  auto raw = torch::zeros(
      {batch, query_capacity, 4 * state_dim}, drives.options());
  auto final_e = torch::empty({batch, state_dim}, drives.options());
  auto final_i = torch::empty({batch, state_dim}, drives.options());
  auto previous = torch::empty(
      {batch, time_steps, 2 * state_dim}, drives.options());
  auto writes = torch::empty_like(previous);
  const int threads = 128;
  const int blocks = static_cast<int>(
      (batch * state_dim + threads - 1) / threads);
  batched_gated_trace_forward_kernel<<<
      blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      drives.data_ptr<float>(),
      query_indices.data_ptr<int64_t>(),
      decays.data_ptr<float>(),
      initial_e.data_ptr<float>(),
      initial_i.data_ptr<float>(),
      raw.data_ptr<float>(),
      final_e.data_ptr<float>(),
      final_i.data_ptr<float>(),
      previous.data_ptr<float>(),
      writes.data_ptr<float>(),
      static_cast<int>(batch),
      static_cast<int>(time_steps),
      static_cast<int>(state_dim),
      static_cast<int>(query_capacity),
      static_cast<float>(spike_threshold));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {raw, final_e, final_i, previous, writes};
}


std::vector<torch::Tensor> sg25e_batched_gated_trace_backward_cuda(
    torch::Tensor grad_raw,
    torch::Tensor grad_final_e,
    torch::Tensor grad_final_i,
    torch::Tensor drives,
    torch::Tensor query_indices,
    torch::Tensor decays,
    torch::Tensor previous,
    torch::Tensor writes,
    torch::Tensor raw,
    double spike_threshold,
    double surrogate_scale) {
  check_float_cuda_contiguous(grad_raw, "grad_raw");
  check_float_cuda_contiguous(grad_final_e, "grad_final_e");
  check_float_cuda_contiguous(grad_final_i, "grad_final_i");
  check_float_cuda_contiguous(drives, "drives");
  check_float_cuda_contiguous(decays, "decays");
  check_float_cuda_contiguous(previous, "previous");
  check_float_cuda_contiguous(writes, "writes");
  check_float_cuda_contiguous(raw, "raw");
  TORCH_CHECK(query_indices.is_cuda(), "query_indices must be CUDA");
  TORCH_CHECK(
      query_indices.scalar_type() == at::kLong && query_indices.is_contiguous(),
      "query_indices must be contiguous int64 CUDA");
  const int64_t batch = drives.size(0);
  const int64_t time_steps = drives.size(1);
  const int64_t state_dim = decays.size(1);
  const int64_t query_capacity = query_indices.size(1);
  c10::cuda::CUDAGuard device_guard(drives.device());
  auto grad_drives = torch::zeros_like(drives);
  auto grad_decays = torch::zeros_like(decays);
  auto grad_initial_e = torch::empty({batch, state_dim}, drives.options());
  auto grad_initial_i = torch::empty({batch, state_dim}, drives.options());
  const int threads = 128;
  const int blocks = static_cast<int>(
      (batch * state_dim + threads - 1) / threads);
  batched_gated_trace_backward_kernel<<<
      blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      grad_raw.data_ptr<float>(),
      grad_final_e.data_ptr<float>(),
      grad_final_i.data_ptr<float>(),
      drives.data_ptr<float>(),
      query_indices.data_ptr<int64_t>(),
      decays.data_ptr<float>(),
      previous.data_ptr<float>(),
      writes.data_ptr<float>(),
      raw.data_ptr<float>(),
      grad_drives.data_ptr<float>(),
      grad_decays.data_ptr<float>(),
      grad_initial_e.data_ptr<float>(),
      grad_initial_i.data_ptr<float>(),
      static_cast<int>(batch),
      static_cast<int>(time_steps),
      static_cast<int>(state_dim),
      static_cast<int>(query_capacity),
      static_cast<float>(spike_threshold),
      static_cast<float>(surrogate_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_drives, grad_decays, grad_initial_e, grad_initial_i};
}
