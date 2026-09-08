#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export PARAS_TRANSPORT=${1:?Usage: launch_paras.sh peer_access|nccl output-directory}
case "$PARAS_TRANSPORT" in peer_access|nccl) ;; *) exit 2 ;; esac
exec "$(dirname "${BASH_SOURCE[0]}")/launch_static.sh" ep "${2:?output-directory required}"
