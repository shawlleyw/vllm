# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check selected GPUs are idle and can create CUDA contexts before launch."""

import csv
import os
import subprocess

import torch


def main():
    selected = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(selected) not in (2, 4, 8) or len(set(selected)) != len(selected):
        raise SystemExit("PARAS_GPUS must contain 2, 4, or 8 distinct GPU indices")
    rows = list(
        csv.reader(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).splitlines()
        )
    )
    available = {
        index.strip(): (int(memory), int(utilization))
        for index, memory, utilization in rows
    }
    idle_memory_limit = int(os.environ.get("PARAS_IDLE_MEMORY_MIB", "128"))
    for index in selected:
        if index not in available:
            raise SystemExit(f"Unknown GPU index {index}")
        memory, utilization = available[index]
        if memory > idle_memory_limit or utilization > 0:
            raise SystemExit(f"GPU {index} is busy: {memory} MiB, {utilization}%")
    # A GPU can report zero utilization yet reject contexts after a memory error.
    # This process exits before the server starts, releasing every test context.
    for local, index in enumerate(selected):
        try:
            value = torch.ones(1, device=f"cuda:{local}")
            assert value.item() == 1
        except Exception as error:
            raise SystemExit(
                f"GPU {index} failed CUDA initialization: {error}"
            ) from error
    print(f"CUDA context checks passed on GPUs {','.join(selected)}", flush=True)


if __name__ == "__main__":
    main()
