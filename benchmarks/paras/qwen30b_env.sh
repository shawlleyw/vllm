#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Source from any directory before launching the local Qwen3 setup.
paras_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$paras_root/.venv/bin/activate"
export PATH="$paras_root/.tools:$PATH"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export TORCH_CUDA_ARCH_LIST=9.0
export PARAS_GPUS=${PARAS_GPUS:-0,1,2,3}
export PARAS_IDLE_MEMORY_MIB=${PARAS_IDLE_MEMORY_MIB:-2048}
export PARAS_MODEL=${PARAS_MODEL:-$paras_root/.venv/models/Qwen3-30B-A3B}
export PARAS_LOAD_FORMAT=${PARAS_LOAD_FORMAT:-dummy}
export HF_HOME="$paras_root/.venv/var/cache/huggingface"
export HF_HUB_OFFLINE=1
export PRE_COMMIT_HOME="$paras_root/.venv/var/cache/pre-commit"
export UV_CACHE_DIR="$paras_root/.cache/uv"
export NVSHMEM_DIR="$paras_root/.venv/lib/python3.12/site-packages/nvidia/nvshmem"
export EP_NCCL_ROOT_DIR="$paras_root/.venv/lib/python3.12/site-packages/nvidia/nccl"
export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:$EP_NCCL_ROOT_DIR/lib"
export VLLM_NCCL_SO_PATH="$EP_NCCL_ROOT_DIR/lib/libnccl.so.2"
export NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_IB_ENABLE_IBGDA=0 NVSHMEM_QP_DEPTH=2048
export CUDA_VISIBLE_DEVICES="$PARAS_GPUS"
unset paras_root
