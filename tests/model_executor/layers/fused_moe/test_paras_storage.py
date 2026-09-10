# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The arena must keep unread source layers intact while reusing slabs.

CPU views are sufficient to catch placement, alignment, and overwrite errors;
actual CUDA transport correctness lives in the distributed transfer test.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.paras.storage import ExpertArena, ExpertLayout


@pytest.mark.parametrize("indices", [(0, 1, 2, 3), (2, 3, 4, 5), (1, 3, 5, 7)])
@pytest.mark.parametrize("size", [2, 4, 8])
@pytest.mark.parametrize("method", ["nccl", "peer_access"])
def test_arena_preserves_unread_sources_and_bounds(method, size, indices):
    layout = ExpertLayout(4, 8, 32, 64, size, size, indices)
    arena = ExpertArena(layout, method)
    assert arena.nbytes == (5 + (method == "nccl")) * arena.slab_bytes
    arena.materialize("cpu", arena.nbytes)
    pointers = {k: v.data_ptr() for k, v in arena.views.items()}
    for target in ("tp", "ep"):
        source = "ep" if target == "tp" else "tp"
        for layer in indices:
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
        arena.view(f"ep.{indices[0]}.w13", (1,))


@pytest.mark.parametrize("size", [2, 4, 8])
def test_actual_qwen_shape_budget_and_replica_guard(size):
    arena = ExpertArena(ExpertLayout(48, 128, 2048, 768, size, size), "peer_access")
    assert arena.slab_bytes == (1152 // size) * 2**20
    assert arena.nbytes == 49 * (1152 // size) * 2**20
    with pytest.raises(ValueError, match="budget"):
        arena.materialize("cpu", arena.nbytes - 1)
    with pytest.raises(ValueError, match="replicas"):
        ExpertLayout(48, 128, 2048, 768, ep_size=4, expert_tp_size=2)


@pytest.mark.parametrize(
    "model_type,extra,indices",
    [
        ("qwen3_moe", {"mlp_only_layers": [0, 1]}, (2, 3, 4, 5)),
        ("qwen2_moe", {"decoder_sparse_step": 2}, (1, 3, 5)),
        ("qwen3_next", {"mlp_only_layers": [0, 3]}, (1, 2, 4, 5)),
        ("glm4_moe", {"first_k_dense_replace": 2}, (2, 3, 4, 5)),
        ("glm4_moe_lite", {"first_k_dense_replace": 1}, (1, 2, 3, 4, 5)),
        ("deepseek_v3", {"first_k_dense_replace": 1, "moe_layer_freq": 2}, (2, 4)),
        ("deepseek_v2", {"first_k_dense_replace": 2}, (2, 3, 4, 5)),
        ("qwen3_5_moe_text", {}, (0, 1, 2, 3, 4, 5)),
    ],
)
def test_only_routed_layers_reserve_slabs(model_type, extra, indices):
    from types import SimpleNamespace

    config = SimpleNamespace(
        model_type=model_type,
        num_hidden_layers=6,
        num_experts=8,
        n_routed_experts=8,
        hidden_size=32,
        moe_intermediate_size=64,
        **extra,
    )
    layout = ExpertLayout.from_model(config, ep_size=4, expert_tp_size=4)
    assert layout.layer_indices == indices
    arena = ExpertArena(layout, "peer_access")
    assert arena.nbytes == (len(indices) + 1) * arena.slab_bytes
    arena.materialize("cpu")
    assert list(arena.transfer_order("ep")) == list(indices)
    assert list(arena.transfer_order("tp")) == list(reversed(indices))
    for slot, layer_id in enumerate(indices):
        for mode in ("ep", "tp"):
            entry = arena.entries[f"{mode}.{layer_id}.w13"]
            assert entry.offset == (slot + (mode == "tp")) * arena.slab_bytes
    for dense in set(range(6)) - set(indices):
        for mode in ("ep", "tp"):
            with pytest.raises(KeyError):
                arena.view(f"{mode}.{dense}.w13")


def test_multimodal_wrapper_uses_text_expert_layout():
    from types import SimpleNamespace

    text = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        num_hidden_layers=40,
        num_experts=256,
        hidden_size=2048,
        moe_intermediate_size=512,
    )
    config = SimpleNamespace(model_type="qwen3_5_moe", text_config=text)
    layout = ExpertLayout.from_model(config, ep_size=4, expert_tp_size=4)
    assert layout.layers == 40
    assert layout.shapes("tp") == ((256, 256, 2048), (256, 2048, 128))


def test_managed_registration_uses_original_layer_ids(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.paras import runtime

    arena = ExpertArena(ExpertLayout(2, 8, 32, 64, 4, 4, (2, 5)), "peer_access")
    arena.materialize("cpu")
    monkeypatch.setattr(runtime, "get_runtime", lambda: SimpleNamespace(arena=arena))
    monkeypatch.setattr(
        runtime.UnquantizedFusedMoEMethod,
        "__init__",
        lambda self, moe: setattr(self, "moe", moe),
    )
    for layer_id in (2, 5):
        for mode in ("ep", "tp"):
            method = runtime.ManagedMoEMethod(
                SimpleNamespace(has_bias=False),
                mode,
                f"language_model.model.layers.{layer_id}.mlp.experts",
            )
            layer = torch.nn.Module()
            shapes = arena.layout.shapes(mode)
            method.create_weights(
                layer, shapes[0][0], 32, shapes[0][1] // 2, torch.bfloat16
            )
            for name in ("w13", "w2"):
                param = getattr(layer, name + "_weight")
                view = arena.view(f"{mode}.{layer_id}.{name}")
                param.data.fill_(layer_id)
                assert param.data_ptr() == view.data_ptr()
                assert torch.all(view == layer_id)
