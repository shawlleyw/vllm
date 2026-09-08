#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
mode=${1:?Usage: launch_static.sh ep|tp [output-directory]}
case "$mode" in
  ep) experts=(--enable-expert-parallel --all2all-backend deepep_low_latency --moe-backend batched_triton) ;;
  tp) experts=(--no-enable-expert-parallel --all2all-backend allgather_reducescatter --moe-backend triton) ;;
  *) echo "Mode must be ep or tp" >&2; exit 2 ;;
esac
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
out=${2:-/data/shaoyuw/paras/vllm-milestone/runs/$mode}
mkdir -p "$out"
out=$(realpath "$out")
export CUDA_VISIBLE_DEVICES=6,7
export VLLM_USE_V2_MODEL_RUNNER=0
export CUDA_HOME=/usr/local/cuda-13.0
export OMP_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=4
export NVSHMEM_REMOTE_TRANSPORT=none
export NVSHMEM_IB_ENABLE_IBGDA=0
export NVSHMEM_QP_DEPTH=2048
[[ -d .venv/conda-meta ]]
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD/benchmarks/paras"
export NVSHMEM_DIR="$PWD/.venv/lib/python3.12/site-packages/nvidia/nvshmem"
export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib"
export VLLM_NCCL_SO_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
export VLLM_CACHE_ROOT="${PARAS_CACHE_ROOT:-$out/cache}"
export VLLM_SERVER_DEV_MODE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv > "$out/gpus-before.csv"
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
args=(/data/shaoyuw/models/Qwen3-30B-A3B
  --host 127.0.0.1 --port 8765 --served-model-name paras-qwen
  --dtype bfloat16 --tensor-parallel-size 1 --data-parallel-size 2
  --data-parallel-backend mp --distributed-executor-backend mp
  --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 512
  --gpu-memory-utilization 0.85 --enable-chunked-prefill --enable-prefix-caching
  --no-async-scheduling --no-enable-dbo --no-enable-eplb --no-enable-elastic-ep
  --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":2}'
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'
  --worker-extension-cls "${PARAS_WORKER_EXTENSION:-static_probe.StaticProbe}"
  --profiler-config.profiler torch
  --profiler-config.torch_profiler_dir "$out/profiles"
  --profiler-config.torch_profiler_with_stack false
  --profiler-config.ignore_frontend true
  --cudagraph-metrics --seed 0)
if [[ -n "${PARAS_TRANSPORT:-}" ]]; then
  [[ "$mode" == ep ]]
  args+=(--paras-config "{\"expert_tp_size\":2,\"weight_transfer_method\":\"$PARAS_TRANSPORT\"}" --api-server-count 1)
fi
printf '%q ' .venv/bin/python -m vllm.entrypoints.cli.main serve "${args[@]}" "${experts[@]}" > "$out/command.txt"
printf '\n' >> "$out/command.txt"
.venv/bin/python -m vllm.entrypoints.cli.main serve "${args[@]}" "${experts[@]}" 2>&1 | tee "$out/server.log"
