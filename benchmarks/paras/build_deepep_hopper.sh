#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source_repo=${1:-/opt/tiger/DeepEP-v1}
repo="$PWD/.venv/src/DeepEP-v1-hopper"
[[ -x .venv/bin/python && -f "$source_repo/deep_ep/buffer.py" ]]
mkdir -p "$repo"
tar -C "$source_repo" --exclude=.git --exclude=.venv --exclude='build*' --exclude='dist*' \
  --exclude='*.egg-info' --exclude='__pycache__' --exclude='*.so' -cf - . | tar -C "$repo" -xf -
sha256sum "$source_repo/setup.py" "$source_repo/deep_ep/buffer.py" \
  "$source_repo"/csrc/kernels/*.cu > "$repo/SOURCE_SHA256"
export PATH="$PWD/.tools:$PWD/.venv/bin:$PATH"
export UV_CACHE_DIR="$PWD/.cache/uv"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=${MAX_JOBS:-8}
export CPATH="$CUDA_HOME/include/cccl${CPATH:+:$CPATH}"
export NVSHMEM_DIR="$PWD/.venv/lib/python3.12/site-packages/nvidia/nvshmem"
export EP_NCCL_ROOT_DIR="$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl"
export LIBRARY_PATH="$NVSHMEM_DIR/lib:$EP_NCCL_ROOT_DIR/lib:$CUDA_HOME/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:$EP_NCCL_ROOT_DIR/lib"
unset PYTHONPATH DISABLE_SM90_FEATURES
.venv/bin/python - "$repo" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
path = root / "setup.py"
path.write_text(path.read_text().replace(
    "else:\n        disable_nvshmem = False",
    "else:\n        disable_nvshmem = False\n"
    "        nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)",
))
path = root / "deep_ep/buffer.py"
path.write_text(path.read_text().replace(
    "os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'",
    "os.environ.setdefault('NVSHMEM_IB_ENABLE_IBGDA', '1')",
))
path = root / "csrc/kernels/internode_ll.cu"
path.write_text(path.read_text().replace(
    "            EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts);",
    """            for (int peer = lane_id; peer < num_ranks; peer += 32) {
                if (nvshmemi_get_p2p_ptr(reinterpret_cast<uint64_t>(rdma_recv_count), rank, peer) == 0)
                    EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts);
            }""",
))
PY
uv pip install --python .venv/bin/python --no-build-isolation --reinstall-package deep-ep "$repo"
.venv/bin/python -c 'import torch, deep_ep; print(torch.__version__, deep_ep.__file__); assert hasattr(deep_ep.Buffer, "low_latency_dispatch")'
