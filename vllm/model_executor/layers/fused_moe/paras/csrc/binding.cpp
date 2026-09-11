// SPDX-License-Identifier: Apache-2.0
// Adapted from SGLang PARAS commit 8177b7260952e3d19cfe360196c31ebc96b3733a.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

// Forward declarations from .cu (v2 kernels only)
void launch_peer_access_fused_transfer_w13_v2(
    int64_t local_buffer_ptr, int64_t* peer_buffer_ptrs, int64_t src_ep_offset,
    int64_t dst_tp_offset, int tp_rank, int tp_size, int E_local,
    int64_t I_prime_H, int num_gates, int elem_size, cudaStream_t stream);

void launch_peer_access_fused_transfer_w2_v2(
    int64_t local_buffer_ptr, int64_t* peer_buffer_ptrs, int64_t src_ep_offset,
    int64_t dst_tp_offset, int tp_rank, int tp_size, int E_local, int H,
    int I_full_bytes, int I_prime_bytes, cudaStream_t stream);

// Forward declarations for TP→EP reverse kernels
void launch_peer_access_fused_transfer_w13_ep(
    int64_t local_buffer_ptr, int64_t* peer_buffer_ptrs, int64_t src_tp_offset,
    int64_t dst_ep_offset, int tp_rank, int tp_size, int E_local,
    int64_t I_prime_H, int num_gates, int elem_size, cudaStream_t stream);

void launch_peer_access_fused_transfer_w2_ep(
    int64_t local_buffer_ptr, int64_t* peer_buffer_ptrs, int64_t src_tp_offset,
    int64_t dst_ep_offset, int tp_rank, int tp_size, int E_local, int H,
    int I_full_bytes, int I_prime_bytes, cudaStream_t stream);

// Python-facing wrappers: accept torch tensors
void launch_peer_access_fused_transfer_w13_v2_py(
    int64_t local_buffer_ptr, torch::Tensor peer_buffer_ptrs,
    int64_t src_ep_offset, int64_t dst_tp_offset, int tp_rank, int tp_size,
    int E_local, int64_t I_prime_H, int num_gates, int elem_size,
    int64_t stream_ptr) {
  TORCH_CHECK(peer_buffer_ptrs.is_cuda(), "peer_buffer_ptrs must be on GPU");
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  launch_peer_access_fused_transfer_w13_v2(
      local_buffer_ptr, peer_buffer_ptrs.data_ptr<int64_t>(), src_ep_offset,
      dst_tp_offset, tp_rank, tp_size, E_local, I_prime_H, num_gates, elem_size,
      stream);
}

void launch_peer_access_fused_transfer_w2_v2_py(
    int64_t local_buffer_ptr, torch::Tensor peer_buffer_ptrs,
    int64_t src_ep_offset, int64_t dst_tp_offset, int tp_rank, int tp_size,
    int E_local, int H, int I_full_bytes, int I_prime_bytes,
    int64_t stream_ptr) {
  TORCH_CHECK(peer_buffer_ptrs.is_cuda(), "peer_buffer_ptrs must be on GPU");
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  launch_peer_access_fused_transfer_w2_v2(
      local_buffer_ptr, peer_buffer_ptrs.data_ptr<int64_t>(), src_ep_offset,
      dst_tp_offset, tp_rank, tp_size, E_local, H, I_full_bytes, I_prime_bytes,
      stream);
}

void launch_peer_access_fused_transfer_w13_ep_py(
    int64_t local_buffer_ptr, torch::Tensor peer_buffer_ptrs,
    int64_t src_tp_offset, int64_t dst_ep_offset, int tp_rank, int tp_size,
    int E_local, int64_t I_prime_H, int num_gates, int elem_size,
    int64_t stream_ptr) {
  TORCH_CHECK(peer_buffer_ptrs.is_cuda(), "peer_buffer_ptrs must be on GPU");
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  launch_peer_access_fused_transfer_w13_ep(
      local_buffer_ptr, peer_buffer_ptrs.data_ptr<int64_t>(), src_tp_offset,
      dst_ep_offset, tp_rank, tp_size, E_local, I_prime_H, num_gates, elem_size,
      stream);
}

void launch_peer_access_fused_transfer_w2_ep_py(
    int64_t local_buffer_ptr, torch::Tensor peer_buffer_ptrs,
    int64_t src_tp_offset, int64_t dst_ep_offset, int tp_rank, int tp_size,
    int E_local, int H, int I_full_bytes, int I_prime_bytes,
    int64_t stream_ptr) {
  TORCH_CHECK(peer_buffer_ptrs.is_cuda(), "peer_buffer_ptrs must be on GPU");
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  launch_peer_access_fused_transfer_w2_ep(
      local_buffer_ptr, peer_buffer_ptrs.data_ptr<int64_t>(), src_tp_offset,
      dst_ep_offset, tp_rank, tp_size, E_local, H, I_full_bytes, I_prime_bytes,
      stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w13_v2", &launch_peer_access_fused_transfer_w13_v2_py);
  m.def("w2_v2", &launch_peer_access_fused_transfer_w2_v2_py);
  m.def("w13_ep", &launch_peer_access_fused_transfer_w13_ep_py);
  m.def("w2_ep", &launch_peer_access_fused_transfer_w2_ep_py);
}
