# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse reference Triton tuning choices for numerical comparisons across servers."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from vllm.triton_utils import triton

_hits: set[str] = set()


class FrozenAutotuneCacheManager(triton.runtime.cache.FileCacheManager):
    def get_file(self, filename):
        if filename.endswith(".autotune.json"):
            root = Path(os.environ["PARAS_FROZEN_AUTOTUNE"])
            path = root / self.key / filename
            if not path.is_file():
                raise RuntimeError(f"Missing reference autotune signature: {path}")
            _hits.add(f"{self.key}/{filename}")
            return str(path)
        return super().get_file(filename)


def status():
    return {"directory": os.environ.get("PARAS_FROZEN_AUTOTUNE"), "hits": sorted(_hits)}


def export(source: Path, output: Path, rank: int):
    if output.exists():
        raise FileExistsError(f"Use a fresh reference directory: {output}")
    files = {}
    for path in sorted(source.rglob("*.autotune.json")):
        if f"/triton/{rank}/" not in str(path) and f"/rank_0_{rank}/" not in str(path):
            continue
        key = f"{path.parent.name}/{path.name}"
        if key in files:
            assert path.read_bytes() == files[key].read_bytes(), (
                f"Conflicting tuning results for {key}; narrow --source"
            )
        files[key] = path
    if not files:
        raise ValueError(f"No rank {rank} autotuning results under {source}")
    output.mkdir(parents=True)
    manifest = {}
    for key, path in files.items():
        destination = output / key
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, destination)
        manifest[key] = {
            "source": str(path.resolve()),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(files)} tuning signatures from rank {rank} to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    export(args.source, args.output, args.rank)
