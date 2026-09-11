#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
method=${1:?Usage: launch_transfer.sh peer_access|nccl output.json}
output=${2:?output.json required}
case "$method" in peer_access|nccl) ;; *) exit 2 ;; esac
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[[ -d .venv/conda-meta ]]
export CUDA_VISIBLE_DEVICES="${PARAS_GPUS:?Set PARAS_GPUS explicitly, e.g. 4,5,6,7 for four ranks}"
IFS=, read -ra paras_devices <<< "$CUDA_VISIBLE_DEVICES"
paras_size=${#paras_devices[@]}
export CUDA_HOME=/usr/local/cuda-13.0
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=4
export OMP_NUM_THREADS=1
export PATH="$PWD/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib"
unset PYTHONPATH NVSHMEM_DIR
.venv/bin/python benchmarks/paras/check_gpus.py
layout_args=(--layers 48)
if [[ -n "${PARAS_MODEL:-}" ]]; then
  layout_args=(--model "$PARAS_MODEL")
fi
exec .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node="$paras_size" \
  benchmarks/paras/check_transfer.py --method "$method" "${layout_args[@]}" --rounds 12 \
  --output "$output"
