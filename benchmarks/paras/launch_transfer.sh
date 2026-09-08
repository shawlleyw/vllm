#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
method=${1:?Usage: launch_transfer.sh peer_access|nccl output.json}
output=${2:?output.json required}
case "$method" in peer_access|nccl) ;; *) exit 2 ;; esac
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[[ -d .venv/conda-meta ]]
export CUDA_VISIBLE_DEVICES=6,7
export CUDA_HOME=/usr/local/cuda-13.0
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=4
export OMP_NUM_THREADS=1
export PATH="$PWD/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib"
unset PYTHONPATH NVSHMEM_DIR
.venv/bin/python - <<'PY'
import csv
import subprocess
rows = csv.reader(subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
    '--format=csv,noheader,nounits'], text=True).splitlines())
for index, memory, utilization in rows:
    if int(index) in (6, 7) and (int(memory) > 128 or int(utilization) > 0):
        raise SystemExit(f'GPU {index} is busy: {memory} MiB, {utilization}%')
PY
exec .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  benchmarks/paras/check_transfer.py --method "$method" --layers 48 --rounds 12 \
  --output "$output"
