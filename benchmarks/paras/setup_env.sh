#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
expected=2cf0a6915ce544dc493a0990f2ea38d81601128a
if ! git merge-base --is-ancestor "$expected" HEAD; then
  echo "Expected a checkout based on vLLM v0.28.0 at $expected" >&2
  exit 1
fi
if [[ ! -d .venv/conda-meta ]]; then
  if [[ -e .venv ]]; then
    echo '.venv exists but is not a conda environment' >&2
    exit 1
  fi
  conda create --prefix "$PWD/.venv" --override-channels -c conda-forge python=3.12 uv -y
fi
export PATH="$PWD/.venv/bin:$PATH"
export PRE_COMMIT_HOME="$PWD/.venv/var/cache/pre-commit"
export UV_CACHE_DIR="$PWD/.venv/var/cache/uv"
conda env config vars set --prefix "$PWD/.venv" PRE_COMMIT_HOME="$PRE_COMMIT_HOME" UV_CACHE_DIR="$UV_CACHE_DIR"
unset LD_LIBRARY_PATH PYTHONPATH NVSHMEM_DIR
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT=$expected
.venv/bin/uv pip install --python .venv/bin/python -e . --torch-backend=cu130
.venv/bin/uv pip install --python .venv/bin/python -r requirements/lint.txt wheel pytest pytest-asyncio
.venv/bin/pre-commit install
.venv/bin/uv pip check --python .venv/bin/python
