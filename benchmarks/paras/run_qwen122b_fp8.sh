#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PARAS_GPUS=${PARAS_GPUS:-0,1,2,3,4,5,6,7}
export PARAS_MODEL=${PARAS_MODEL:-$PWD/.venv/models/Qwen3.5-122B-A10B-FP8}
export PARAS_LOAD_FORMAT=${PARAS_LOAD_FORMAT:-dummy}
export PARAS_LANGUAGE_MODEL_ONLY=1
export PARAS_EP_MOE_BACKEND=${PARAS_EP_MOE_BACKEND:-${PARAS_MOE_BACKEND:-triton}}
export PARAS_TP_MOE_BACKEND=${PARAS_TP_MOE_BACKEND:-${PARAS_MOE_BACKEND:-triton}}
if [[ "$PARAS_EP_MOE_BACKEND" == deep_gemm || "$PARAS_TP_MOE_BACKEND" == deep_gemm ]]; then
  export VLLM_USE_DEEP_GEMM_E8M0=${VLLM_USE_DEEP_GEMM_E8M0:-0}
fi
export PARAS_MAMBA_CACHE_MODE=align
source benchmarks/paras/qwen30b_env.sh
mode=${1:-switch}
out=${2:-$PWD/.venv/var/paras/qwen122b-fp8-$mode}
case "$mode" in
  switch) exec benchmarks/paras/launch_paras.sh "${PARAS_TRANSPORT:-peer_access}" "$out" ;;
  ep|tp) exec benchmarks/paras/launch_static.sh "$mode" "$out" ;;
  *) echo 'Usage: run_qwen122b_fp8.sh [switch|ep|tp] [output-directory]' >&2; exit 2 ;;
esac
