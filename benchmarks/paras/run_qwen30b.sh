#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source benchmarks/paras/qwen30b_env.sh
mode=${1:-switch}
out=${2:-$PWD/.venv/var/paras/qwen30b-$mode}
case "$mode" in
  switch) exec benchmarks/paras/launch_paras.sh "${PARAS_TRANSPORT:-peer_access}" "$out" ;;
  ep|tp) exec benchmarks/paras/launch_static.sh "$mode" "$out" ;;
  *) echo 'Usage: run_qwen30b.sh [switch|ep|tp] [output-directory]' >&2; exit 2 ;;
esac
