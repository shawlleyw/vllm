# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for fixed weights/cache bindings across routed state replacement.

These catch accidental registration or rebinding. GPU serving tests must also
compare logits and recurrent-state evolution while replaying both graph sets.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.paras.runtime import MoEStates
from vllm.model_executor.layers.fused_moe.paras.state import StationaryState
from vllm.model_executor.layers.fused_moe.paras.storage import ExpertArena, ExpertLayout
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


def routed(arena, mode):
    module = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(module)
    module.moe_config = SimpleNamespace(mode=mode)
    for name in ("w13", "w2"):
        module.register_parameter(
            name + "_weight",
            torch.nn.Parameter(arena.view(f"{mode}.2.{name}"), requires_grad=False),
        )
    return module


def model_and_arena():
    arena = ExpertArena(ExpertLayout(1, 8, 32, 64, 4, 4, (2,)), "peer_access")
    arena.materialize("cpu")
    model = torch.nn.Module()
    model.dense = torch.nn.Linear(32, 32)
    model.attention = torch.nn.Module()
    model.attention.kv_cache = (torch.zeros(2, 32, 4), torch.zeros(2, 32, 32))
    model.moe = torch.nn.Module()
    model.moe.shared_experts = torch.nn.Linear(32, 32)
    model.moe.routed_experts = routed(arena, "ep")
    return model, arena


def test_switch_preserves_dense_shared_and_recurrent_storage():
    model, arena = model_and_arena()
    state = StationaryState(model, arena)
    modes = MoEStates(model.moe, model.moe.routed_experts, routed(arena, "tp"), 2)
    x = torch.randn(3, 32)
    expected = model.moe.shared_experts(model.dense(x)).detach()
    for mode in ("tp", "ep", "tp", "ep"):
        # Simulate destructive routed-weight writes while fixed state is live.
        arena.buffer.fill_(7)
        modes.activate(mode)
        model.attention.kv_cache[1].add_(1)
        state.check()
        torch.testing.assert_close(model.moe.shared_experts(model.dense(x)), expected)
        assert model.moe.shared_experts._moe_config is getattr(modes, mode).moe_config
    assert torch.all(model.attention.kv_cache[1] == 4)


@pytest.mark.parametrize("target", ["dense", "shared", "recurrent"])
def test_stationary_state_rejects_rebinding(target):
    model, arena = model_and_arena()
    state = StationaryState(model, arena)
    if target == "recurrent":
        conv, recurrent = model.attention.kv_cache
        model.attention.kv_cache = (conv, recurrent.clone())
    else:
        module = model.dense if target == "dense" else model.moe.shared_experts
        module.weight = torch.nn.Parameter(module.weight.detach().clone())
    with pytest.raises(RuntimeError, match="bindings changed"):
        state.check()


def test_stationary_state_rejects_strided_arena_alias():
    model, arena = model_and_arena()
    model.attention.kv_cache = arena.view("ep.2.w13").transpose(1, 2)
    with pytest.raises(RuntimeError, match="overlaps expert arena"):
        StationaryState(model, arena)
