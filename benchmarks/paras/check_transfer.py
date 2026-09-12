# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real managed-view transport correctness and full-model timing via torchrun."""

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from vllm.model_executor.layers.fused_moe.paras.runtime import initialize_tp_scales
from vllm.model_executor.layers.fused_moe.paras.storage import ExpertArena, ExpertLayout
from vllm.model_executor.layers.fused_moe.paras.transfer import WeightTransfer


def pattern(shape, layer, weight, expert_start, intermediate_start, dtype, device):
    # Reinterpret integer patterns to catch permutations, including FP8 NaN bits.
    gated = weight.startswith("w13")
    e, h, i = (shape[0], shape[2], shape[1] // 2) if gated else shape
    expert = torch.arange(e, device=device, dtype=torch.int32) + expert_start
    hidden = torch.arange(h, device=device, dtype=torch.int32)
    inter = torch.arange(i, device=device, dtype=torch.int32) + intermediate_start
    if gated:
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
    bits = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[dtype.itemsize]
    return (value + layer * 97).to(bits).view(dtype)


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
    resident_scales = {}
    if layout.weight_block_size:
        for layer in layout.layer_indices:
            ep, tp = SimpleNamespace(), SimpleNamespace()
            for weight, shape in zip(("w13", "w2"), layout.shapes("ep")):
                e, n, k = shape
                source = pattern(
                    (e, n // 128, k // 128),
                    layer,
                    weight,
                    rank * (layout.experts // size),
                    0,
                    torch.float32,
                    arena.buffer.device,
                )
                _, tp_n, tp_k = layout.tensors("tp")[weight][0]
                target = torch.empty(
                    (layout.experts, tp_n // 128, tp_k // 128),
                    device=source.device,
                    dtype=torch.float32,
                )
                setattr(ep, f"{weight}_weight_scale_inv", source)
                setattr(tp, f"{weight}_weight_scale_inv", target)
            initialize_tp_scales(ep, tp, gpu_group)
            for mode, scales in (("ep", ep), ("tp", tp)):
                for weight in ("w13", "w2"):
                    tensor = getattr(scales, f"{weight}_weight_scale_inv")
                    inter = tensor.shape[1] // 2 if weight == "w13" else tensor.shape[2]
                    expected = pattern(
                        tensor.shape,
                        layer,
                        weight,
                        rank * (layout.experts // size) if mode == "ep" else 0,
                        rank * inter if mode == "tp" else 0,
                        tensor.dtype,
                        tensor.device,
                    )
                    assert torch.equal(
                        tensor.view(torch.uint8), expected.view(torch.uint8)
                    )
                    assert not arena.is_managed(tensor)
                    resident_scales[f"{mode}.{layer}.{weight}"] = (tensor, expected)
    for layer in layout.layer_indices:
        for weight in layout.parameter_names:
            target = arena.view(f"ep.{layer}.{weight}")
            target.copy_(
                pattern(
                    target.shape,
                    layer,
                    weight,
                    rank * (layout.experts // size),
                    0,
                    target.dtype,
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
                    for weight in layout.parameter_names:
                        target = arena.view(f"{mode}.{layer}.{weight}")
                        inter = (
                            target.shape[1] // 2
                            if weight.startswith("w13")
                            else target.shape[2]
                        )
                        expected = pattern(
                            target.shape,
                            layer,
                            weight,
                            rank * (layout.experts // size) if mode == "ep" else 0,
                            rank * inter if mode == "tp" else 0,
                            target.dtype,
                            target.device,
                        )
                        if not torch.equal(
                            target.view(torch.uint8), expected.view(torch.uint8)
                        ):
                            raise AssertionError(
                                f"rank={rank} {mode}.{layer}.{weight} mismatch"
                            )
                print(
                    f"rank={rank} exact {mode} weights verified round={iteration}",
                    flush=True,
                )
    assert pointers == {k: v.data_ptr() for k, v in arena.views.items()}
    for name, (tensor, expected) in resident_scales.items():
        assert torch.equal(tensor.view(torch.uint8), expected.view(torch.uint8)), name
    results = {
        "rank": rank,
        "world_size": size,
        "method": args.method,
        "layers": layout.layers,
        "expert_layer_indices": layout.layer_indices,
        "model": args.model,
        "arena_bytes": arena.nbytes,
        "exact_roundtrip": "passed",
        "resident_scale_tensors": len(resident_scales),
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
