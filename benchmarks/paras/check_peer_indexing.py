# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate eight-way peer-kernel indexing in separate arenas on ONE GPU.

This isolates layout/indexing from the fabric. It does not validate CUDA IPC,
remote visibility, NCCL fences, or eight-GPU transport performance.
"""

import argparse
import json
from pathlib import Path

import torch
from check_transfer import pattern

from vllm.model_executor.layers.fused_moe.paras.storage import ExpertArena, ExpertLayout
from vllm.model_executor.layers.fused_moe.paras.transfer import (
    WeightTransfer,
    load_peer_kernel,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layout = ExpertLayout(48, 128, 2048, 768, 8, 8)
    arenas = [ExpertArena(layout, "peer_access") for _ in range(8)]
    for arena in arenas:
        arena.materialize("cuda")
    pointers = {id(a): {k: v.data_ptr() for k, v in a.views.items()} for a in arenas}
    peer_pointers = torch.tensor(
        [a.view("ep.0.w13").data_ptr() for a in arenas],
        device="cuda",
        dtype=torch.int64,
    )
    kernel = load_peer_kernel()
    stream = torch.cuda.current_stream()
    transfers = []
    for rank, arena in enumerate(arenas):
        # Exercise production launch arguments against eight local destinations.
        # No distributed transport is initialized in this diagnostic.
        transfer = WeightTransfer.__new__(WeightTransfer)
        assert arena.buffer is not None
        transfer.arena, transfer.buffer = arena, arena.buffer
        transfer.rank, transfer.kernel = rank, kernel
        transfer.peer_pointers, transfer.stream = peer_pointers, stream
        transfers.append(transfer)
        for layer in range(layout.layers):
            for weight in ("w13", "w2"):
                value = arena.view(f"ep.{layer}.{weight}")
                value.copy_(
                    pattern(
                        value.shape, layer, weight, rank * 16, 0, "ep", value.device
                    )
                )
    for iteration in range(2):
        for mode in ("tp", "ep"):
            for layer in arenas[0].transfer_order(mode):
                for transfer in transfers:
                    transfer._peer_layer(layer, mode)
            torch.accelerator.synchronize()
            for rank, arena in enumerate(arenas):
                for layer in range(layout.layers):
                    for weight in ("w13", "w2"):
                        value = arena.view(f"{mode}.{layer}.{weight}")
                        expected = pattern(
                            value.shape,
                            layer,
                            weight,
                            rank * 16 if mode == "ep" else 0,
                            rank * 96 if mode == "tp" else 0,
                            mode,
                            value.device,
                        )
                        assert torch.equal(
                            value.view(torch.int16), expected.view(torch.int16)
                        ), (rank, layer, weight, mode)
            print(
                f"Local eight-way {mode} indexing exact, round {iteration}", flush=True
            )
    assert pointers == {
        id(a): {k: v.data_ptr() for k, v in a.views.items()} for a in arenas
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "passed": True,
                "physical_gpus": 1,
                "logical_ranks": 8,
                "layers": 48,
                "roundtrips": 2,
                "arena_bytes_per_rank": arenas[0].nbytes,
                "validation": "kernel indexing only; no IPC or distributed fences",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
