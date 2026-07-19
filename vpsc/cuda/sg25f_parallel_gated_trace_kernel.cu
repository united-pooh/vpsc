#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


namespace {

constexpr int kTimeThreads = 128;

void check_float_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__device__ __forceinline__ float surrogate_derivative(float value, float scale) {
  const float denominator = 1.0f + scale * fabsf(value);
  return scale / (denominator * denominator);
}

__device__ __forceinline__ int query_slot(
    const int64_t* queries, int query_count, int time_index) {
  int lower = 0;
  int upper = query_count - 1;
  while (lower <= upper) {
    const int middle = (lower + upper) >> 1;
    const int value = static_cast<int>(queries[middle]);
    if (value == time_index) {
      return middle;
    }
    if (value < time_index) {
      lower = middle + 1;
    } else {
      upper = middle - 1;
    }
  }
  return -1;
}

__global__ void parallel_gated_trace_forward_kernel(
    const float* __restrict__ packed_drives,
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
  const int time_index = threadIdx.x;
  const int block_linear = blockIdx.x;
  const int batch_index = block_linear / state_dim;
  const int state_index = block_linear - batch_index * state_dim;
  if (batch_index >= batch) {
    return;
  }
  __shared__ float prefix_a_e[kTimeThreads];
  __shared__ float prefix_b_e[kTimeThreads];
  __shared__ float prefix_a_i[kTimeThreads];
  __shared__ float prefix_b_i[kTimeThreads];
  __shared__ int row_query_count;

  const int64_t* row_queries =
      query_indices + batch_index * query_capacity;
  if (time_index == 0) {
    int count = 0;
    while (count < query_capacity && row_queries[count] >= 0) {
      ++count;
    }
    row_query_count = count;
  }
  float write_e = 0.0f;
  float write_i = 0.0f;
  const float decay_e = decays[state_index];
  const float decay_i = decays[state_dim + state_index];
  if (time_index < time_steps) {
    const int drive_base =
        ((batch_index * 4) * state_dim + state_index) * time_steps + time_index;
    const int component_stride = state_dim * time_steps;
    const float content_e =
        packed_drives[drive_base] >= 0.0f ? 1.0f : 0.0f;
    const float content_i =
        packed_drives[drive_base + component_stride] >= 0.0f ? 1.0f : 0.0f;
    const float gate_e =
        packed_drives[drive_base + 2 * component_stride] >= 0.0f ? 1.0f : 0.0f;
    const float gate_i =
        packed_drives[drive_base + 3 * component_stride] >= 0.0f ? 1.0f : 0.0f;
    write_e = content_e * gate_e;
    write_i = content_i * gate_i;
    prefix_a_e[time_index] = decay_e;
    prefix_b_e[time_index] = (1.0f - decay_e) * write_e;
    prefix_a_i[time_index] = decay_i;
    prefix_b_i[time_index] = (1.0f - decay_i) * write_i;
  } else {
    prefix_a_e[time_index] = 1.0f;
    prefix_b_e[time_index] = 0.0f;
    prefix_a_i[time_index] = 1.0f;
    prefix_b_i[time_index] = 0.0f;
  }
  __syncthreads();

  for (int offset = 1; offset < kTimeThreads; offset <<= 1) {
    float next_a_e = prefix_a_e[time_index];
    float next_b_e = prefix_b_e[time_index];
    float next_a_i = prefix_a_i[time_index];
    float next_b_i = prefix_b_i[time_index];
    if (time_index >= offset) {
      const float left_a_e = prefix_a_e[time_index - offset];
      const float left_b_e = prefix_b_e[time_index - offset];
      const float left_a_i = prefix_a_i[time_index - offset];
      const float left_b_i = prefix_b_i[time_index - offset];
      next_b_e = next_a_e * left_b_e + next_b_e;
      next_a_e = next_a_e * left_a_e;
      next_b_i = next_a_i * left_b_i + next_b_i;
      next_a_i = next_a_i * left_a_i;
    }
    __syncthreads();
    prefix_a_e[time_index] = next_a_e;
    prefix_b_e[time_index] = next_b_e;
    prefix_a_i[time_index] = next_a_i;
    prefix_b_i[time_index] = next_b_i;
    __syncthreads();
  }

  if (time_index < time_steps) {
    const int initial_index = batch_index * state_dim + state_index;
    const float initial_trace_e = initial_e[initial_index];
    const float initial_trace_i = initial_i[initial_index];
    const float trace_e =
        prefix_a_e[time_index] * initial_trace_e + prefix_b_e[time_index];
    const float trace_i =
        prefix_a_i[time_index] * initial_trace_i + prefix_b_i[time_index];
    const float previous_e = time_index == 0
        ? initial_trace_e
        : prefix_a_e[time_index - 1] * initial_trace_e +
              prefix_b_e[time_index - 1];
    const float previous_i = time_index == 0
        ? initial_trace_i
        : prefix_a_i[time_index - 1] * initial_trace_i +
              prefix_b_i[time_index - 1];
    const int pair_base =
        ((batch_index * 2) * state_dim + state_index) * time_steps + time_index;
    const int pair_stride = state_dim * time_steps;
    previous[pair_base] = previous_e;
    previous[pair_base + pair_stride] = previous_i;
    writes[pair_base] = write_e;
    writes[pair_base + pair_stride] = write_i;
    const int slot = query_slot(row_queries, row_query_count, time_index);
    if (slot >= 0) {
      const int raw_base =
          (batch_index * query_capacity + slot) * (4 * state_dim);
      raw[raw_base + state_index] =
          trace_e >= spike_threshold ? 1.0f : 0.0f;
      raw[raw_base + state_dim + state_index] =
          trace_i >= spike_threshold ? -1.0f : 0.0f;
      raw[raw_base + 2 * state_dim + state_index] = trace_e;
      raw[raw_base + 3 * state_dim + state_index] = -trace_i;
    }
    if (time_index == time_steps - 1) {
      final_e[initial_index] = trace_e;
      final_i[initial_index] = trace_i;
    }
  }
}

__global__ void parallel_gated_trace_backward_kernel(
    const float* __restrict__ grad_raw,
    const float* __restrict__ grad_final_e,
    const float* __restrict__ grad_final_i,
    const float* __restrict__ packed_drives,
    const int64_t* __restrict__ query_indices,
    const float* __restrict__ decays,
    const float* __restrict__ previous,
    const float* __restrict__ writes,
    const float* __restrict__ raw,
    float* __restrict__ grad_packed_drives,
    float* __restrict__ grad_decays,
    float* __restrict__ grad_initial_e,
    float* __restrict__ grad_initial_i,
    int batch,
    int time_steps,
    int state_dim,
    int query_capacity,
    float spike_threshold,
    float surrogate_scale) {
  const int time_index = threadIdx.x;
  const int block_linear = blockIdx.x;
  const int batch_index = block_linear / state_dim;
  const int state_index = block_linear - batch_index * state_dim;
  if (batch_index >= batch) {
    return;
  }
  __shared__ float reverse_a_e[kTimeThreads];
  __shared__ float reverse_b_e[kTimeThreads];
  __shared__ float reverse_a_i[kTimeThreads];
  __shared__ float reverse_b_i[kTimeThreads];
  __shared__ int row_query_count;
  const int64_t* row_queries =
      query_indices + batch_index * query_capacity;
  if (time_index == 0) {
    int count = 0;
    while (count < query_capacity && row_queries[count] >= 0) {
      ++count;
    }
    row_query_count = count;
  }
  __syncthreads();

  const float decay_e = decays[state_index];
  const float decay_i = decays[state_dim + state_index];
  float direct_e = 0.0f;
  float direct_i = 0.0f;
  if (time_index < time_steps) {
    const int slot = query_slot(row_queries, row_query_count, time_index);
    if (slot >= 0) {
      const int raw_base =
          (batch_index * query_capacity + slot) * (4 * state_dim);
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
    }
    if (time_index == time_steps - 1) {
      const int initial_index = batch_index * state_dim + state_index;
      direct_e += grad_final_e[initial_index];
      direct_i += grad_final_i[initial_index];
    }
    const int reverse_index = time_steps - 1 - time_index;
    reverse_a_e[reverse_index] = decay_e;
    reverse_b_e[reverse_index] = direct_e;
    reverse_a_i[reverse_index] = decay_i;
    reverse_b_i[reverse_index] = direct_i;
  }
  if (time_index >= time_steps) {
    reverse_a_e[time_index] = 1.0f;
    reverse_b_e[time_index] = 0.0f;
    reverse_a_i[time_index] = 1.0f;
    reverse_b_i[time_index] = 0.0f;
  }
  __syncthreads();

  for (int offset = 1; offset < kTimeThreads; offset <<= 1) {
    float next_a_e = reverse_a_e[time_index];
    float next_b_e = reverse_b_e[time_index];
    float next_a_i = reverse_a_i[time_index];
    float next_b_i = reverse_b_i[time_index];
    if (time_index >= offset) {
      const float left_a_e = reverse_a_e[time_index - offset];
      const float left_b_e = reverse_b_e[time_index - offset];
      const float left_a_i = reverse_a_i[time_index - offset];
      const float left_b_i = reverse_b_i[time_index - offset];
      next_b_e = next_a_e * left_b_e + next_b_e;
      next_a_e = next_a_e * left_a_e;
      next_b_i = next_a_i * left_b_i + next_b_i;
      next_a_i = next_a_i * left_a_i;
    }
    __syncthreads();
    reverse_a_e[time_index] = next_a_e;
    reverse_b_e[time_index] = next_b_e;
    reverse_a_i[time_index] = next_a_i;
    reverse_b_i[time_index] = next_b_i;
    __syncthreads();
  }

  float decay_contribution_e = 0.0f;
  float decay_contribution_i = 0.0f;
  float adjoint_e = 0.0f;
  float adjoint_i = 0.0f;
  if (time_index < time_steps) {
    const int reverse_index = time_steps - 1 - time_index;
    adjoint_e = reverse_b_e[reverse_index];
    adjoint_i = reverse_b_i[reverse_index];
    const int pair_base =
        ((batch_index * 2) * state_dim + state_index) * time_steps + time_index;
    const int pair_stride = state_dim * time_steps;
    const float previous_e = previous[pair_base];
    const float previous_i = previous[pair_base + pair_stride];
    const float write_e = writes[pair_base];
    const float write_i = writes[pair_base + pair_stride];
    decay_contribution_e = adjoint_e * (previous_e - write_e);
    decay_contribution_i = adjoint_i * (previous_i - write_i);

    const int drive_base =
        ((batch_index * 4) * state_dim + state_index) * time_steps + time_index;
    const int component_stride = state_dim * time_steps;
    const float drive_content_e = packed_drives[drive_base];
    const float drive_content_i = packed_drives[drive_base + component_stride];
    const float drive_gate_e = packed_drives[drive_base + 2 * component_stride];
    const float drive_gate_i = packed_drives[drive_base + 3 * component_stride];
    const float content_e = drive_content_e >= 0.0f ? 1.0f : 0.0f;
    const float content_i = drive_content_i >= 0.0f ? 1.0f : 0.0f;
    const float gate_e = drive_gate_e >= 0.0f ? 1.0f : 0.0f;
    const float gate_i = drive_gate_i >= 0.0f ? 1.0f : 0.0f;
    const float scale_e = (1.0f - decay_e) * adjoint_e;
    const float scale_i = (1.0f - decay_i) * adjoint_i;
    grad_packed_drives[drive_base] = scale_e * gate_e *
        surrogate_derivative(drive_content_e, surrogate_scale);
    grad_packed_drives[drive_base + component_stride] = scale_i * gate_i *
        surrogate_derivative(drive_content_i, surrogate_scale);
    grad_packed_drives[drive_base + 2 * component_stride] = scale_e * content_e *
        surrogate_derivative(drive_gate_e, surrogate_scale);
    grad_packed_drives[drive_base + 3 * component_stride] = scale_i * content_i *
        surrogate_derivative(drive_gate_i, surrogate_scale);
    if (time_index == 0) {
      const int initial_index = batch_index * state_dim + state_index;
      grad_initial_e[initial_index] = decay_e * adjoint_e;
      grad_initial_i[initial_index] = decay_i * adjoint_i;
    }
  }

  reverse_b_e[time_index] = decay_contribution_e;
  reverse_b_i[time_index] = decay_contribution_i;
  __syncthreads();
  for (int offset = kTimeThreads / 2; offset > 0; offset >>= 1) {
    if (time_index < offset) {
      reverse_b_e[time_index] += reverse_b_e[time_index + offset];
      reverse_b_i[time_index] += reverse_b_i[time_index + offset];
    }
    __syncthreads();
  }
  if (time_index == 0) {
    atomicAdd(grad_decays + state_index, reverse_b_e[0]);
    atomicAdd(grad_decays + state_dim + state_index, reverse_b_i[0]);
  }
}

}  // namespace


std::vector<torch::Tensor> sg25f_parallel_gated_trace_forward_cuda(
    torch::Tensor packed_drives,
    torch::Tensor query_indices,
    torch::Tensor decays,
    torch::Tensor initial_e,
    torch::Tensor initial_i,
    double spike_threshold,
    double surrogate_scale) {
  check_float_cuda_contiguous(packed_drives, "packed_drives");
  check_float_cuda_contiguous(decays, "decays");
  check_float_cuda_contiguous(initial_e, "initial_e");
  check_float_cuda_contiguous(initial_i, "initial_i");
  TORCH_CHECK(
      query_indices.is_cuda() && query_indices.scalar_type() == at::kLong &&
          query_indices.is_contiguous(),
      "query_indices must be contiguous int64 CUDA");
  TORCH_CHECK(
      packed_drives.dim() == 4 && packed_drives.size(1) == 4,
      "packed_drives must be [batch,4,state,time]");
  TORCH_CHECK(
      query_indices.dim() == 2 && query_indices.size(1) > 0,
      "query_indices must be non-empty [batch,query]");
  TORCH_CHECK(
      decays.dim() == 2 && decays.size(0) == 2,
      "decays must be [2,state]");
  const int64_t batch = packed_drives.size(0);
  const int64_t state_dim = packed_drives.size(2);
  const int64_t time_steps = packed_drives.size(3);
  const int64_t query_capacity = query_indices.size(1);
  TORCH_CHECK(time_steps > 0 && time_steps <= kTimeThreads, "time must be 1..128");
  TORCH_CHECK(query_indices.size(0) == batch, "query batch mismatch");
  TORCH_CHECK(decays.size(1) == state_dim, "decay state mismatch");
  TORCH_CHECK(
      initial_e.dim() == 2 && initial_e.size(0) == batch &&
          initial_e.size(1) == state_dim,
      "initial_e shape mismatch");
  TORCH_CHECK(
      initial_i.dim() == 2 && initial_i.size(0) == batch &&
          initial_i.size(1) == state_dim,
      "initial_i shape mismatch");
  c10::cuda::CUDAGuard device_guard(packed_drives.device());
  auto raw = torch::zeros(
      {batch, query_capacity, 4 * state_dim}, packed_drives.options());
  auto final_e = torch::empty({batch, state_dim}, packed_drives.options());
  auto final_i = torch::empty({batch, state_dim}, packed_drives.options());
  auto previous = torch::empty(
      {batch, 2, state_dim, time_steps}, packed_drives.options());
  auto writes = torch::empty_like(previous);
  parallel_gated_trace_forward_kernel<<<
      static_cast<int>(batch * state_dim),
      kTimeThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      packed_drives.data_ptr<float>(),
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


std::vector<torch::Tensor> sg25f_parallel_gated_trace_backward_cuda(
    torch::Tensor grad_raw,
    torch::Tensor grad_final_e,
    torch::Tensor grad_final_i,
    torch::Tensor packed_drives,
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
  check_float_cuda_contiguous(packed_drives, "packed_drives");
  check_float_cuda_contiguous(decays, "decays");
  check_float_cuda_contiguous(previous, "previous");
  check_float_cuda_contiguous(writes, "writes");
  check_float_cuda_contiguous(raw, "raw");
  const int64_t batch = packed_drives.size(0);
  const int64_t state_dim = packed_drives.size(2);
  const int64_t time_steps = packed_drives.size(3);
  const int64_t query_capacity = query_indices.size(1);
  c10::cuda::CUDAGuard device_guard(packed_drives.device());
  auto grad_packed_drives = torch::zeros_like(packed_drives);
  auto grad_decays = torch::zeros_like(decays);
  auto grad_initial_e = torch::empty({batch, state_dim}, packed_drives.options());
  auto grad_initial_i = torch::empty({batch, state_dim}, packed_drives.options());
  parallel_gated_trace_backward_kernel<<<
      static_cast<int>(batch * state_dim),
      kTimeThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      grad_raw.data_ptr<float>(),
      grad_final_e.data_ptr<float>(),
      grad_final_i.data_ptr<float>(),
      packed_drives.data_ptr<float>(),
      query_indices.data_ptr<int64_t>(),
      decays.data_ptr<float>(),
      previous.data_ptr<float>(),
      writes.data_ptr<float>(),
      raw.data_ptr<float>(),
      grad_packed_drives.data_ptr<float>(),
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
  return {grad_packed_drives, grad_decays, grad_initial_e, grad_initial_i};
}
