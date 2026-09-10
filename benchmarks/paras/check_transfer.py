# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real managed-view transport correctness and full-model timing via torchrun."""

import argparse
import json
import math
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.model_executor.layers.fused_moe.paras.storage import ExpertArena, ExpertLayout
from vllm.model_executor.layers.fused_moe.paras.transfer import WeightTransfer


def pattern(shape, layer, weight, expert_start, intermediate_start, mode, device):
    # Use integer arithmetic, then reinterpret all BF16 bit patterns. This catches
    # element permutations (including gate/up and row/column ordering) exactly.
    e, h, i = (shape[0], shape[2], shape[1] // 2) if weight == "w13" else shape
    expert = torch.arange(e, device=device, dtype=torch.int32) + expert_start
    hidden = torch.arange(h, device=device, dtype=torch.int32)
    inter = torch.arange(i, device=device, dtype=torch.int32) + intermediate_start
    if weight == "w13":
        gates = torch.stack((inter, inter + 2003)).reshape(-1)
        value = (
            expert[:, None, None] * 53
            + gates[None, :, None] * 71
            + hidden[None, None, :] * 17
        )
    else:
        value = (
            expert[:, None, None] * 53
            + hidden[None, :, None] * 17
            + inter[None, None, :] * 71
        )
    return (value + layer * 97).to(torch.int16).view(torch.bfloat16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["nccl", "peer_access"], required=True)
    p.add_argument("--layers", type=int)
    p.add_argument("--model", help="Local model config; test all routed layers")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo")
    gpu_group = dist.new_group(backend="nccl")
    size = dist.get_world_size()
    if args.model:
        from transformers import AutoConfig

        if args.layers is not None:
            p.error("--model tests every routed layer; do not combine with --layers")
        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        layout = ExpertLayout.from_model(config, ep_size=size, expert_tp_size=size)
    else:
        layout = ExpertLayout(args.layers or 48, 128, 2048, 768, size, size)
    arena = ExpertArena(layout, args.method)
    arena.materialize(f"cuda:{rank}")
    transfer = WeightTransfer(arena, args.method, dist.group.WORLD, gpu_group)
    for layer in layout.layer_indices:
        for weight in ("w13", "w2"):
            target = arena.view(f"ep.{layer}.{weight}")
            target.copy_(
                pattern(
                    target.shape,
                    layer,
                    weight,
                    rank * (layout.experts // size),
                    0,
                    "ep",
                    target.device,
                )
            )
    torch.accelerator.synchronize()
    pointers = {k: v.data_ptr() for k, v in arena.views.items()}
    times = {"tp": [], "ep": []}
    for iteration in range(args.rounds):
        for mode in ("tp", "ep"):
            dist.barrier()
            times[mode].append(transfer.move(mode))
            if iteration in (0, args.rounds - 1):
                for layer in layout.layer_indices:
                    for weight in ("w13", "w2"):
                        target = arena.view(f"{mode}.{layer}.{weight}")
                        expected = pattern(
                            target.shape,
                            layer,
                            weight,
                            rank * (layout.experts // size) if mode == "ep" else 0,
                            rank * (layout.intermediate // size) if mode == "tp" else 0,
                            mode,
                            target.device,
                        )
                        if not torch.equal(
                            target.view(torch.int16), expected.view(torch.int16)
                        ):
                            raise AssertionError(
                                f"rank={rank} {mode}.{layer}.{weight} mismatch"
                            )
                print(
                    f"rank={rank} exact {mode} weights verified round={iteration}",
                    flush=True,
                )
    assert pointers == {k: v.data_ptr() for k, v in arena.views.items()}
    results = {
        "rank": rank,
        "world_size": size,
        "method": args.method,
        "layers": layout.layers,
        "expert_layer_indices": layout.layer_indices,
        "model": args.model,
        "arena_bytes": arena.nbytes,
        "exact_roundtrip": "passed",
        "times_ms": times,
        "summary": {
            mode: {
                "median_ms": statistics.median(t[1:]),
                "p95_ms": sorted(t[1:])[math.ceil(0.95 * (len(t) - 1)) - 1],
                "logical_GB_per_second": layout.layers
                * arena.slab_bytes
                / statistics.median(t[1:])
                / 1e6,
            }
            for mode, t in times.items()
        },
    }
    all_results = [None] * size
    dist.all_gather_object(all_results, results)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2))
        print(json.dumps(all_results, indent=2), flush=True)
    transfer.close()
    dist.destroy_process_group(gpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
