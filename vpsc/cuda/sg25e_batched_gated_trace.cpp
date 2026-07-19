#include <torch/extension.h>

#include <vector>


std::vector<torch::Tensor> sg25e_batched_gated_trace_forward_cuda(
    torch::Tensor drives,
    torch::Tensor query_indices,
    torch::Tensor decays,
    torch::Tensor initial_e,
    torch::Tensor initial_i,
    double spike_threshold,
    double surrogate_scale);

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
    double surrogate_scale);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "forward",
      &sg25e_batched_gated_trace_forward_cuda,
      "SG25E batched-query gated-trace forward (CUDA)");
  module.def(
      "backward",
      &sg25e_batched_gated_trace_backward_cuda,
      "SG25E batched-query gated-trace reverse adjoint (CUDA)");
}
