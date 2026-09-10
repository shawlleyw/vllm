#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
expected=2cf0a6915ce544dc493a0990f2ea38d81601128a
if ! git merge-base --is-ancestor "$expected" HEAD; then
  echo "Expected a checkout based on vLLM v0.28.0 at $expected" >&2
  exit 1
fi
export PATH="$PWD/.tools:$PWD/.venv/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export UV_CACHE_DIR="$PWD/.cache/uv"
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
export PRE_COMMIT_HOME="$PWD/.venv/var/cache/pre-commit"
unset LD_LIBRARY_PATH PYTHONPATH NVSHMEM_DIR
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT=$expected
export VLLM_PRECOMPILED_WHEEL_VARIANT=cu130
cached_wheel="$PWD/.tools/wheels/vllm-0.28.0-cp38-abi3-manylinux_2_28_x86_64.whl"
if [[ -f "$cached_wheel" && -z "${VLLM_PRECOMPILED_WHEEL_LOCATION:-}" ]]; then
  export VLLM_PRECOMPILED_WHEEL_LOCATION="$cached_wheel"
fi
if [[ -d "$PWD/.tools/wheels" && -z "${UV_FIND_LINKS:-}" ]]; then
  export UV_FIND_LINKS="$PWD/.tools/wheels"
fi
uv pip install --python .venv/bin/python -e . --torch-backend=cu130
uv pip install --python .venv/bin/python -r requirements/lint.txt wheel ninja setuptools pytest pytest-asyncio nvidia-nvshmem-cu13
.venv/bin/pre-commit install
uv pip check --python .venv/bin/python
