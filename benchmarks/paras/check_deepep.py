# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed BF16 DeepEP dispatch/combine round trip with CUDA graph replay."""

import os

import deep_ep
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
    cpu_group = dist.new_group(backend="gloo")
    torch.manual_seed(rank)
    size = dist.get_world_size()
    hidden, experts, capacity = 2048, 128, 64
    buffer = deep_ep.Buffer(
        cpu_group,
        0,
        deep_ep.Buffer.get_low_latency_rdma_size_hint(capacity, hidden, size, experts),
        low_latency_mode=True,
        num_qps_per_rank=experts // size,
        allow_nvlink_for_low_latency_mode=True,
        explicitly_destroy=True,
    )
    x = torch.randn(7 if rank == 0 else 3, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(experts, (x.shape[0], experts), device="cuda").argsort(dim=1)
    ids = ids[:, :8].contiguous().to(torch.int64)
    weights = torch.full(ids.shape, 0.125, device="cuda", dtype=torch.float32)

    def run():
        received, _, handle, _, hook = buffer.low_latency_dispatch(
            x, ids, capacity, experts, use_fp8=False, return_recv_hook=True
        )
        hook()
        result, _, hook = buffer.low_latency_combine(
            received, ids, weights, handle, return_recv_hook=True
        )
        hook()
        return result

    for _ in range(3):
        result = run()
    torch.accelerator.synchronize()
    torch.testing.assert_close(result, x, atol=0.01, rtol=0.01)
    dist.barrier(group=cpu_group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    for _ in range(5):
        x.mul_(0.5)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(result, x, atol=0.001, rtol=0.01)
    print(f"rank={rank} BF16 round trip and five CUDA graph replays passed", flush=True)
    dist.barrier(group=cpu_group)
    del graph
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
