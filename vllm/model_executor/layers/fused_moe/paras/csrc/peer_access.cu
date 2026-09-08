// SPDX-License-Identifier: Apache-2.0
// Adapted from SGLang PARAS commit 8177b7260952e3d19cfe360196c31ebc96b3733a.
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// Maximum number of GPUs in TP group (NVSwitch supports up to 8)
#define MAX_PEERS 8

// =============================================================================
// V2 kernels: NVLink-optimized with warp-level peer assignment
// Following NVLink store guidelines:
//   - Grid: num_sms × tp_size blocks (divisible by tp_size for balanced load)
//   - Block: 256 threads = 8 warps (occupancy target: 8 blocks/SM)
//   - Peer assigned at warp level: peer = global_warp_id % tp_size
//   - 8 unrolled int4 (16B) stores per thread (#pragma unroll)
//   - Per warp: 32 lanes × 16B = 512B contiguous per store, 4KB per 8 stores
// =============================================================================

// V2 w13 kernel: warp-level peer assignment with NVLink-optimal access pattern.
// EP layout: (E_local, num_gates, tp_size, I'H)
// TP layout: (tp_size * E_local, num_gates, I'H) — contiguous per (e, k) chunk.
__global__ void peer_access_fused_transfer_w13_v2(
    const char* __restrict__ local_buffer,
    char* const* __restrict__ peer_buffers, int64_t src_ep_offset,
    int64_t dst_tp_offset, int tp_rank, int tp_size, int E_local,
    int64_t I_prime_H, int num_gates, int elem_size) {
  const int warps_in_block = blockDim.x >> 5;
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int global_warp = blockIdx.x * warps_in_block + warp_in_block;
  const int total_warps = gridDim.x * warps_in_block;

  // Warp-level peer assignment (NVLink guideline)
  const int peer = global_warp % tp_size;
  const int warp_index = global_warp / tp_size;
  const int warps_per_peer = total_warps / tp_size;

  // Self-write bypass: avoid IPC pointer for local rank (no UVA overhead)
  char* const dst_buf =
      (peer == tp_rank) ? const_cast<char*>(local_buffer) : peer_buffers[peer];

  // Each chunk = one (expert, gate) pair: I'H * elem_size contiguous bytes
  const int64_t chunk_bytes = I_prime_H * elem_size;
  const int chunks_per_peer = E_local * num_gates;
  const int64_t int4_per_chunk = chunk_bytes >> 4;  // / 16
  const int64_t total_int4 = (int64_t)chunks_per_peer * int4_per_chunk;

  // Destination base: tp_rank's expert slot (contiguous across all chunks)
  const int64_t dst_expert_base =
      dst_tp_offset + (int64_t)tp_rank * chunks_per_peer * chunk_bytes;

  // 8 unrolled × 32 lanes = 256 int4 per warp per iteration = 4KB
  int64_t pos = (int64_t)warp_index * 256 + lane;
  const int64_t stride = (int64_t)warps_per_peer * 256;

  // Use uint32 for decomposition math — avoids expensive int64 software
  // division total_int4 fits in uint32 (e.g. 32 chunks × 98304 = 3,145,728 <
  // 2^32)
  const unsigned int int4_per_chunk_u = (unsigned int)int4_per_chunk;

  while (pos < total_int4) {
#pragma unroll 8
    for (int u = 0; u < 8; u++) {
      const int64_t idx = pos + (int64_t)u * 32;
      if (idx < total_int4) {
        // Fast uint32 decomposition (hardware divider, ~20 cycles vs ~100 for
        // int64)
        const unsigned int idx_u = (unsigned int)idx;
        const unsigned int chunk_id = idx_u / int4_per_chunk_u;
        const unsigned int in_chunk = idx_u - chunk_id * int4_per_chunk_u;

        // num_gates is typically 2 — compiler optimizes to shift+mask
        const unsigned int e = chunk_id / (unsigned int)num_gates;
        const unsigned int k = chunk_id - e * (unsigned int)num_gates;

        // Source: EP[e, k, peer, :] — strided layout
        const int64_t src_off =
            src_ep_offset +
            (int64_t)(e * num_gates * tp_size + k * tp_size + peer) *
                chunk_bytes +
            (int64_t)in_chunk * 16;

        // Dest: TP[tp_rank, e, k, :] — contiguous
        const int64_t dst_off = dst_expert_base +
                                (int64_t)(e * num_gates + k) * chunk_bytes +
                                (int64_t)in_chunk * 16;

        *reinterpret_cast<int4*>(dst_buf + dst_off) =
            __ldg(reinterpret_cast<const int4*>(local_buffer + src_off));
      }
    }
    pos += stride;
  }
}

void launch_peer_access_fused_transfer_w13_v2(
    int64_t local_buffer_ptr, int64_t* peer_buffer_ptrs, int64_t src_ep_offset,
    int64_t dst_tp_offset, int tp_rank, int tp_size, int E_local,
    int64_t I_prime_H, int num_gates, int elem_size, cudaStream_t stream) {
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  int num_sms;
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));

  // num_sms × tp_size blocks (one warp-group per SM per peer), 256 threads (8
  // warps)
  const int blocks = num_sms * tp_size;
  const int threads = 256;

  peer_access_fused_transfer_w13_v2<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const char*>(local_buffer_ptr),
      reinterpret_cast<char* const*>(peer_buffer_ptrs), src_ep_offset,
      dst_tp_offset, tp_rank, tp_size, E_local, I_prime_H, num_gates,
      elem_size);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// V2 w2 kernel: warp-level peer assignment for strided row copies.
// EP layout: (E_local, H, I_full) with TP split on last dim.
// For peer r, copies columns [r*I_prime, (r+1)*I_prime) — contiguous within
// each row.
__global__ void peer_access_fused_transfer_w2_v2(
    const char* __restrict__ local_buffer,
    char* const* __restrict__ peer_buffers, int64_t src_ep_offset,
    int64_t dst_tp_offset, int tp_rank, int tp_size, int E_local, int H,
    int I_full_bytes, int I_prime_bytes) {
  const int warps_in_block = blockDim.x >> 5;
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int global_warp = blockIdx.x * warps_in_block + warp_in_block;
  const int total_warps = gridDim.x * warps_in_block;

  const int peer = global_warp % tp_size;
  const int warp_index = global_warp / tp_size;
  const int warps_per_peer = total_warps / tp_size;

  // Self-write bypass: avoid IPC pointer for local rank (no UVA overhead)
  char* const dst_buf =
      (peer == tp_rank) ? const_cast<char*>(local_buffer) : peer_buffers[peer];

  const int n_int4_per_row_dst = I_prime_bytes >> 4;  // I_prime_bytes / 16
  const int peer_shard_byte_off = peer * I_prime_bytes;
  const int64_t total_int4 = (int64_t)E_local * H * n_int4_per_row_dst;

  // 8 unrolled × 32 lanes = 256 int4 per warp per iteration = 4KB
  int64_t pos = (int64_t)warp_index * 256 + lane;
  const int64_t stride = (int64_t)warps_per_peer * 256;

  // Use uint32 for decomposition math — avoids expensive int64 software
  // division total_int4 fits in uint32 (e.g. 16*2048*48 = 1,572,864 < 2^32)
  const unsigned int n_int4_per_row_dst_u = (unsigned int)n_int4_per_row_dst;
  const unsigned int H_u = (unsigned int)H;

  while (pos < total_int4) {
#pragma unroll 8
    for (int u = 0; u < 8; u++) {
      const int64_t idx = pos + (int64_t)u * 32;
      if (idx < total_int4) {
        // Fast uint32 decomposition (hardware divider)
        const unsigned int idx_u = (unsigned int)idx;
        const unsigned int eh_idx = idx_u / n_int4_per_row_dst_u;
        const unsigned int col = idx_u - eh_idx * n_int4_per_row_dst_u;
        const unsigned int e = eh_idx / H_u;
        const unsigned int h = eh_idx - e * H_u;

        // Source: EP[e, h, peer*I_prime + col] — strided rows
        const int64_t src_off = src_ep_offset + (int64_t)e * H * I_full_bytes +
                                (int64_t)h * I_full_bytes +
                                peer_shard_byte_off + col * 16;

        // Dest: TP[(tp_rank*E_local+e), h, col]
        const int64_t dst_off =
            dst_tp_offset +
            (int64_t)(tp_rank * E_local + e) * H * I_prime_bytes +
            (int64_t)h * I_prime_bytes + col * 16;

        *reinterpret_cast<int4*>(dst_buf + dst_off) =
            __ldg(reinterpret_cast<const int4*>(local_buffer + src_off));
      }
    }
    pos += stride;
  }
}

void launch_peer_access_fused_transfer_w2_v2(
    int64_t local_buffer_ptr, int64_t* peer_buffer_ptrs, int64_t src_ep_offset,
    int64_t dst_tp_offset, int tp_rank, int tp_size, int E_local, int H,
    int I_full_bytes, int I_prime_bytes, cudaStream_t stream) {
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  int num_sms;
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));

  const int blocks = num_sms * tp_size;
  const int threads = 256;

  peer_access_fused_transfer_w2_v2<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const char*>(local_buffer_ptr),
      reinterpret_cast<char* const*>(peer_buffer_ptrs), src_ep_offset,
      dst_tp_offset, tp_rank, tp_size, E_local, H, I_full_bytes, I_prime_bytes);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// TP→EP reverse kernels: write TP slot[i] data back to peer EP slot[i+1].
// Used for: (a) correctness verification, (b) future dp_size>1 support.
// Layer ordering: REVERSE (N-1→0) with per-layer barrier.
// Same NVLink optimizations as v2 kernels: warp-level peer assignment,
// int4 vectorized stores, 8-store unrolling, self-write bypass, __ldg(),
// uint32.
// =============================================================================

// TP→EP w13 kernel: reverse of EP→TP w13_v2.
// Source: TP slot[i], layout (E_total, num_gates, I'H) — contiguous per
// (expert, gate). Dest:   peer's EP slot[i+1], layout (E_local, num_gates,
// tp_size, I'H). For peer r, reads experts [r*E_local, (r+1)*E_local) from
// local TP slot, writes to peer r's EP slot at the tp_rank shard position.
__global__ void peer_access_fused_transfer_w13_ep(
    const char* __restrict__ local_buffer,
    char* const* __restrict__ peer_buffers, int64_t src_tp_offset,
    int64_t dst_ep_offset, int tp_rank, int tp_size, int E_local,
    int64_t I_prime_H, int num_gates, int elem_size) {
  const int warps_in_block = blockDim.x >> 5;
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int global_warp = blockIdx.x * warps_in_block + warp_in_block;
  const int total_warps = gridDim.x * warps_in_block;

  // Warp-level peer assignment (NVLink guideline)
  const int peer = global_warp % tp_size;
  const int warp_index = global_warp / tp_size;
  const int warps_per_peer = total_warps / tp_size;

  // Self-write bypass: avoid IPC pointer for local rank (no UVA overhead)
  char* const dst_buf =
      (peer == tp_rank) ? const_cast<char*>(local_buffer) : peer_buffers[peer];

  // Each chunk = one (expert, gate) pair: I'H * elem_size contiguous bytes
  const int64_t chunk_bytes = I_prime_H * elem_size;
  const int chunks_per_peer = E_local * num_gates;
  const int64_t int4_per_chunk = chunk_bytes >> 4;  // / 16
  const int64_t total_int4 = (int64_t)chunks_per_peer * int4_per_chunk;

  // Source base: peer's expert group in TP slot
  // TP layout: peer r owns experts [r*E_local, (r+1)*E_local), each with
  // num_gates chunks
  const int64_t src_expert_base =
      src_tp_offset + (int64_t)peer * chunks_per_peer * chunk_bytes;

  // 8 unrolled × 32 lanes = 256 int4 per warp per iteration = 4KB
  int64_t pos = (int64_t)warp_index * 256 + lane;
  const int64_t stride = (int64_t)warps_per_peer * 256;

  // Use uint32 for decomposition math — avoids expensive int64 software
  // division
  const unsigned int int4_per_chunk_u = (unsigned int)int4_per_chunk;

  while (pos < total_int4) {
#pragma unroll 8
    for (int u = 0; u < 8; u++) {
      const int64_t idx = pos + (int64_t)u * 32;
      if (idx < total_int4) {
        // Fast uint32 decomposition (hardware divider, ~20 cycles vs ~100 for
        // int64)
        const unsigned int idx_u = (unsigned int)idx;
        const unsigned int chunk_id = idx_u / int4_per_chunk_u;
        const unsigned int in_chunk = idx_u - chunk_id * int4_per_chunk_u;

        // num_gates is typically 2 — compiler optimizes to shift+mask
        const unsigned int e = chunk_id / (unsigned int)num_gates;
        const unsigned int k = chunk_id - e * (unsigned int)num_gates;

        // Source: TP[(peer*E_local + e), k, :] — contiguous in TP layout
        const int64_t src_off = src_expert_base +
                                (int64_t)(e * num_gates + k) * chunk_bytes +
                                (int64_t)in_chunk * 16;

        // Dest: peer's EP[e, k, tp_rank, :] — strided layout
        const int64_t dst_off =
            dst_ep_offset +
            (int64_t)(e * num_gates * tp_size + k * tp_size + tp_rank) *
                chunk_bytes +
            (int64_t)in_chunk * 16;

        *reinterpret_cast<int4*>(dst_buf + dst_off) =
            __ldg(reinterpret_cast<const int4*>(local_buffer + src_off));
      }
    }
    pos += stride;
  }
}

void launch_peer_access_fused_transfer_w13_ep(
    int64_t local_buffer_ptr, int64_t* peer_buffer_ptrs, int64_t src_tp_offset,
    int64_t dst_ep_offset, int tp_rank, int tp_size, int E_local,
    int64_t I_prime_H, int num_gates, int elem_size, cudaStream_t stream) {
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  int num_sms;
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));

  // num_sms × tp_size blocks (one warp-group per SM per peer), 256 threads (8
  // warps)
  const int blocks = num_sms * tp_size;
  const int threads = 256;

  peer_access_fused_transfer_w13_ep<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const char*>(local_buffer_ptr),
      reinterpret_cast<char* const*>(peer_buffer_ptrs), src_tp_offset,
      dst_ep_offset, tp_rank, tp_size, E_local, I_prime_H, num_gates,
      elem_size);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// TP→EP w2 kernel: reverse of EP→TP w2_v2.
// Source: TP slot[i], layout (E_total, H, I') — contiguous rows of width I'.
// Dest:   peer's EP slot[i+1], layout (E_local, H, I_full) — TP split on last
// dim. For peer r, reads experts [r*E_local, (r+1)*E_local) from local TP slot,
// writes tp_rank's column shard to peer r's EP slot.
__global__ void peer_access_fused_transfer_w2_ep(
    const char* __restrict__ local_buffer,
    char* const* __restrict__ peer_buffers, int64_t src_tp_offset,
    int64_t dst_ep_offset, int tp_rank, int tp_size, int E_local, int H,
    int I_full_bytes, int I_prime_bytes) {
  const int warps_in_block = blockDim.x >> 5;
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int global_warp = blockIdx.x * warps_in_block + warp_in_block;
  const int total_warps = gridDim.x * warps_in_block;

  const int peer = global_warp % tp_size;
  const int warp_index = global_warp / tp_size;
  const int warps_per_peer = total_warps / tp_size;

  // Self-write bypass: avoid IPC pointer for local rank (no UVA overhead)
  char* const dst_buf =
      (peer == tp_rank) ? const_cast<char*>(local_buffer) : peer_buffers[peer];

  const int n_int4_per_row = I_prime_bytes >> 4;  // I_prime_bytes / 16
  const int tp_rank_shard_byte_off = tp_rank * I_prime_bytes;
  const int64_t total_int4 = (int64_t)E_local * H * n_int4_per_row;

  // 8 unrolled × 32 lanes = 256 int4 per warp per iteration = 4KB
  int64_t pos = (int64_t)warp_index * 256 + lane;
  const int64_t stride = (int64_t)warps_per_peer * 256;

  // Use uint32 for decomposition math — avoids expensive int64 software
  // division
  const unsigned int n_int4_per_row_u = (unsigned int)n_int4_per_row;
  const unsigned int H_u = (unsigned int)H;

  while (pos < total_int4) {
#pragma unroll 8
    for (int u = 0; u < 8; u++) {
      const int64_t idx = pos + (int64_t)u * 32;
      if (idx < total_int4) {
        // Fast uint32 decomposition (hardware divider)
        const unsigned int idx_u = (unsigned int)idx;
        const unsigned int eh_idx = idx_u / n_int4_per_row_u;
        const unsigned int col = idx_u - eh_idx * n_int4_per_row_u;
        const unsigned int e = eh_idx / H_u;
        const unsigned int h = eh_idx - e * H_u;

        // Source: TP[(peer*E_local+e), h, col] — rows of width I'
        const int64_t src_off =
            src_tp_offset + (int64_t)(peer * E_local + e) * H * I_prime_bytes +
            (int64_t)h * I_prime_bytes + col * 16;

        // Dest: peer's EP[e, h, tp_rank*I' + col] — rows of width I_full
        const int64_t dst_off = dst_ep_offset + (int64_t)e * H * I_full_bytes +
                                (int64_t)h * I_full_bytes +
                                tp_rank_shard_byte_off + col * 16;

        *reinterpret_cast<int4*>(dst_buf + dst_off) =
            __ldg(reinterpret_cast<const int4*>(local_buffer + src_off));
      }
    }
    pos += stride;
  }
}

void launch_peer_access_fused_transfer_w2_ep(
    int64_t local_buffer_ptr, int64_t* peer_buffer_ptrs, int64_t src_tp_offset,
    int64_t dst_ep_offset, int tp_rank, int tp_size, int E_local, int H,
    int I_full_bytes, int I_prime_bytes, cudaStream_t stream) {
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  int num_sms;
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));

  const int blocks = num_sms * tp_size;
  const int threads = 256;

  peer_access_fused_transfer_w2_ep<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const char*>(local_buffer_ptr),
      reinterpret_cast<char* const*>(peer_buffer_ptrs), src_tp_offset,
      dst_ep_offset, tp_rank, tp_size, E_local, H, I_full_bytes, I_prime_bytes);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
