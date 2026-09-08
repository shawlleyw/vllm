# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-vocabulary comparisons under matching token and expert-mode histories."""

import argparse
import json
from pathlib import Path

import torch


def read(directory):
    directory = Path(directory)
    generation = json.loads((directory / "generation.json").read_text())
    return (
        torch.cat(torch.load(directory / "rank0.pt", weights_only=True)).float(),
        generation.get("history_tokens", generation["tokens"]),
        json.loads((directory / "rank0-modes.json").read_text()),
    )


def metrics(value, reference):
    delta = (value - reference).abs()
    kl = (
        reference.softmax(-1) * (reference.log_softmax(-1) - value.log_softmax(-1))
    ).sum(-1)
    return {
        "max_absolute_logit_error": delta.max().item(),
        "mean_absolute_logit_error": delta.mean().item(),
        "max_KL": max(0.0, kl.max().item()),
        "mean_KL": max(0.0, kl.mean().item()),
        "greedy_token_agreement": (value.argmax(-1) == reference.argmax(-1))
        .float()
        .mean()
        .item(),
    }


def main(args):
    references = {mode: read(getattr(args, mode)) for mode in ("ep", "tp")}
    assert references["ep"][1] == references["tp"][1]
    cross = metrics(references["ep"][0], references["tp"][0])
    reverse_cross = metrics(references["tp"][0], references["ep"][0])
    oracle = read(args.switch_reference) if args.switch_reference else None
    results = []
    for directory in args.candidates:
        value, tokens, modes = read(directory)
        assert value.shape == (32, 151936) and len(modes) == len(tokens) == 32
        assert torch.isfinite(value).all()
        for index, mode in enumerate(modes):
            assert tokens[:index] == references[mode][1][:index], (
                f"Token history differs at {index}"
            )
        reference = torch.stack([references[m][0][i] for i, m in enumerate(modes)])
        stats = metrics(value, reference)
        result = {
            "candidate": str(directory),
            "steps": 32,
            "vocabulary_size": 151936,
            "modes": modes,
            "static_mode_comparison": stats,
        }
        if len(set(modes)) == 1:
            torch.testing.assert_close(value, reference, atol=0.25, rtol=0.01)
            result["bitwise_equal_to_static"] = torch.equal(value, reference)
        else:
            # Retained KV contains activations computed under earlier expert
            # arithmetic. A pure-TP history is therefore not an exact oracle.
            # First bound its deviation by the measured static EP/TP envelope,
            # then require exact agreement with a transfer-free mixed-mode run.
            for key in (
                "max_absolute_logit_error",
                "mean_absolute_logit_error",
                "max_KL",
                "mean_KL",
            ):
                assert stats[key] <= max(cross[key], reverse_cross[key]) + 1e-6
            assert oracle is not None, (
                "Mixed-mode validation requires a disjoint oracle"
            )
            assert tokens == oracle[1] and modes == oracle[2], "Oracle histories differ"
            torch.testing.assert_close(value, oracle[0], atol=0, rtol=0)
            result["bitwise_equal_to_disjoint_oracle"] = True
        assert stats["greedy_token_agreement"] == 1
        result["passed"] = True
        results.append(result)
    output = {
        "static_EP_TP_arithmetic_difference": cross,
        "static_TP_EP_arithmetic_difference": reverse_cross,
        "switch_reference": args.switch_reference,
        "comparisons": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ep", required=True)
    p.add_argument("--tp", required=True)
    p.add_argument("--switch-reference")
    p.add_argument("--output", required=True)
    p.add_argument("candidates", nargs="+")
    main(p.parse_args())
