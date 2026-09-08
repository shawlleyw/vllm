# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The arena must keep unread source layers intact while reusing slabs.

CPU views are sufficient to catch placement, alignment, and overwrite errors;
actual CUDA transport correctness lives in the distributed transfer test.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.paras.storage import ExpertArena, ExpertLayout


@pytest.mark.parametrize("method", ["nccl", "peer_access"])
def test_arena_preserves_unread_sources_and_bounds(method):
    layout = ExpertLayout(4, 8, 32, 16)
    arena = ExpertArena(layout, method)
    assert arena.nbytes == (5 + (method == "nccl")) * arena.slab_bytes
    arena.materialize("cpu", arena.nbytes)
    pointers = {k: v.data_ptr() for k, v in arena.views.items()}
    for target in ("tp", "ep"):
        source = "ep" if target == "tp" else "tp"
        for layer in range(layout.layers):
            for weight in ("w13", "w2"):
                arena.view(f"{source}.{layer}.{weight}").fill_(layer + 1)
        for layer in arena.transfer_order(target):
            for weight in ("w13", "w2"):
                src = arena.view(f"{source}.{layer}.{weight}")
                assert torch.all(src == layer + 1)
                arena.view(f"{target}.{layer}.{weight}").fill_(-1)
    assert pointers == {k: v.data_ptr() for k, v in arena.views.items()}
    for name, tensor in arena.views.items():
        assert arena.entries[name].offset % 256 == 0
        assert arena.is_managed(tensor)
        assert arena.view(name, (-1,)).data_ptr() == tensor.data_ptr()
    with pytest.raises(ValueError):
        arena.reserve("late", (1,), torch.bfloat16)
    with pytest.raises(RuntimeError):
        arena.view("ep.0.w13", (1,))


def test_actual_qwen_shape_budget_and_replica_guard():
    arena = ExpertArena(ExpertLayout(48, 128, 2048, 768), "peer_access")
    assert arena.slab_bytes == 576 * 2**20
    assert arena.nbytes == 49 * 576 * 2**20
    with pytest.raises(ValueError, match="budget"):
        arena.materialize("cpu", arena.nbytes - 1)
    with pytest.raises(ValueError, match="replicas"):
        ExpertLayout(48, 128, 2048, 768, ep_size=4)
