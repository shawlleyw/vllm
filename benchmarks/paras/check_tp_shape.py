# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check Qwen3 TP8's 96-channel Triton experts and decode graph replay."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    w13 = torch.randn(128, 192, 2048, device=device, dtype=dtype) / 2048**0.5
    w2 = torch.randn(128, 2048, 96, device=device, dtype=dtype) / 96**0.5
    quant = FusedMoEQuantConfig.make(None)
    evidence = []
    for batch in (1, 2, 4, 8, 16, 32, 64, 512):
        x = torch.randn(batch, 2048, device=device, dtype=dtype)
        scores = torch.randn(batch, 128, device=device).softmax(-1)
        weights, indices = scores.topk(8, dim=-1)
        weights = (weights / weights.sum(-1, keepdim=True)).contiguous()
        indices = indices.to(torch.int32)

        def run(x=x, weights=weights, indices=indices):
            return fused_experts(x, w13, w2, weights, indices, quant_config=quant)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                actual = run()
        torch.cuda.current_stream().wait_stream(stream)
        # Match Triton's BF16 gate/up, activation, down, and weighted output
        # rounding boundaries while evaluating the expert arithmetic in PyTorch.
        parts = torch.empty(batch, 8, 2048, device=device, dtype=dtype)
        for expert in range(128):
            rows, slots = torch.where(indices == expert)
            if rows.numel() == 0:
                continue
            gate, up = F.linear(x[rows], w13[expert]).chunk(2, dim=-1)
            activated = (F.silu(gate.float()) * up.float()).to(dtype)
            down = F.linear(activated, w2[expert])
            parts[rows, slots] = (down.float() * weights[rows, slots, None]).to(dtype)
        expected = parts.sum(1)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run()
        for _ in range(3):
            graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(captured, actual, atol=0, rtol=0)
        evidence.append(
            {
                "batch": batch,
                "max_abs_error": (actual - expected).abs().max().item(),
                "graph_replays": 3,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "passed": True,
                "w13_shape": list(w13.shape),
                "w2_shape": list(w2.shape),
                "checks": evidence,
            },
            indent=2,
        )
    )
    print(args.output.read_text())


if __name__ == "__main__":
    main()
