# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V1 runner hooks; storage and transport have no dependency on runner internals."""

from vllm.model_executor.layers.fused_moe.paras.runtime import get_runtime


def initialize(model):
    runtime = get_runtime()
    runtime.initialize(model)
    return runtime


def profile_both(runner, operation):
    runtime = runner.paras
    results = []
    for mode in ("ep", "tp"):
        runtime.initialize_mode(mode)
        results.append(operation())
    runtime.initialize_mode("ep")
    return sum(x or 0 for x in results)


def capture_both(runner):
    runtime = runner.paras
    # The largest prefill and decode workspaces of both modes must have been
    # warmed before the first persistent graph fixes their addresses.
    for mode in ("ep", "tp"):
        runtime.initialize_mode(mode)
        runner._dummy_run(runner.max_num_tokens, skip_eplb=True)
    total = 0
    for mode in ("ep", "tp"):
        runtime.initialize_mode(mode)
        total += runner._capture_model_single()
    runtime.initialize_mode("ep")
    for mode, count in runtime.status()["graphs"].items():
        if count == 0:
            raise RuntimeError(f"PARAS did not capture {mode} decode graphs")
    from vllm.model_executor.layers.fused_moe.paras.state import StationaryState

    runtime.stationary_state = StationaryState(runner.model, runtime.arena)
    runtime.assert_addresses()
    runtime.initialized = True
    return total
