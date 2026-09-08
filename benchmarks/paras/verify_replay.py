# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Require CUDA runtime graph launches in each worker's profiler trace."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import ijson
import regex as re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    traces = sorted((args.run / "profiles").glob("*.pt.trace.json*"))
    assert traces, "No profiler traces found"
    evidence = []
    for trace in traces:
        opener = gzip.open if trace.suffix == ".gz" else open
        counts = Counter()
        with opener(trace, "rb") as source:
            for event in ijson.items(source, "traceEvents.item"):
                if event.get("cat") in ("cuda_runtime", "cuda_driver"):
                    name = event.get("name", "")
                    if "GraphLaunch" in name or "GraphInstantiate" in name:
                        counts[name] += 1
        if counts:
            evidence.append({"trace": trace.name, "cuda_graph_calls": dict(counts)})
    expected = 4 if (args.run / "live.json").exists() else 2
    assert len(evidence) == expected, (
        f"Expected {expected} worker traces, got {evidence}"
    )
    observed_ranks = {
        int(match.group(1))
        for rank in evidence
        if (match := re.match(r"dp(\d+)_", rank["trace"]))
    }
    assert observed_ranks == {0, 1}, evidence
    for rank in evidence:
        calls = rank["cuda_graph_calls"]
        assert sum(v for k, v in calls.items() if "GraphLaunch" in k) > 0, rank
        assert not any("GraphInstantiate" in k for k in calls), rank
    (args.run / "replay.json").write_text(json.dumps(evidence, indent=2))
    result_path = args.run / ("live.json" if expected == 4 else "results.json")
    results = json.loads(result_path.read_text())
    results["graph_replay"] = "passed: both workers launch existing CUDA graphs"
    result_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
