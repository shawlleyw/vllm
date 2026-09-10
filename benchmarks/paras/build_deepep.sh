#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
repo=${1:-/data/shaoyuw/paras/vllm-milestone/DeepEP-sm80}
expected=8e57c764c7d3fdb0999fe3e34a03371c8f53ae1b
[[ -x .venv/bin/python ]]
export PATH="$PWD/.tools:$PWD/.venv/bin:$PATH"
export UV_CACHE_DIR="$PWD/.cache/uv"
[[ $(git -C "$repo" rev-parse HEAD) == "$expected" ]]
export CUDA_HOME=/usr/local/cuda-13.0
export NVSHMEM_DIR="$PWD/.venv/lib/python3.12/site-packages/nvidia/nvshmem"
export LD_LIBRARY_PATH="$NVSHMEM_DIR/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia/nccl/lib"
export CPATH="$CUDA_HOME/include/cccl"
export LIBRARY_PATH="$CUDA_HOME/lib64/stubs"
export DISABLE_SM90_FEATURES=1 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=8
unset PYTHONPATH
.venv/bin/python - "$repo" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
p = root / 'csrc/kernels/configs.cuh'
s = p.read_text().replace(
    '#ifndef DISABLE_SM90_FEATURES\n#include <cuda_fp8.h>',
    '#if !defined(DISABLE_SM90_FEATURES) || __has_include(<cuda_fp8.h>)\n#include <cuda_fp8.h>')
p.write_text(s)
p = root / 'setup.py'
s = p.read_text()
if '        nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)\n        # internode' not in s:
    s = s.replace('        # internode.cu uses TMA',
                  '        nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)\n        # internode.cu uses TMA')
p.write_text(s)
p = root / 'deep_ep/buffer.py'
s = p.read_text().replace("os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'",
                          "os.environ.setdefault('NVSHMEM_IB_ENABLE_IBGDA', '1')")
p.write_text(s)
p = root / 'csrc/kernels/internode_ll.cu'
s = p.read_text().replace(
    '\n            EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts);',
    '''
            for (int peer = lane_id; peer < num_ranks; peer += 32) {
                if (nvshmemi_get_p2p_ptr(reinterpret_cast<uint64_t>(rdma_recv_count), rank, peer) == 0)
                    EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts);
            }''')
p.write_text(s)
PY
uv pip install --python .venv/bin/python --no-build-isolation --reinstall-package deep-ep "$repo"
.venv/bin/python - <<'PY'
import deep_ep, deep_ep_cpp, torch
print('torch', torch.__version__, 'DeepEP', deep_ep_cpp.__file__)
assert hasattr(deep_ep.Buffer, 'low_latency_dispatch')
assert hasattr(deep_ep.Buffer, 'low_latency_combine')
PY
