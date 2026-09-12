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
@pytest.mark.parametrize("fp8", [False, True])
def test_arena_preserves_unread_sources_and_bounds(method, size, indices, fp8):
    layout = ExpertLayout(
        4,
        8,
        512 if fp8 else 32,
        128 * size if fp8 else 64,
        size,
        size,
        indices,
        (128, 128) if fp8 else None,
    )
    arena = ExpertArena(layout, method)
    assert arena.nbytes == (5 + (method == "nccl")) * arena.slab_bytes
    arena.materialize("cpu", arena.nbytes)
    pointers = {k: v.data_ptr() for k, v in arena.views.items()}
    for target in ("tp", "ep"):
        source = "ep" if target == "tp" else "tp"
        for layer in indices:
            for weight in layout.parameter_names:
                arena.view(f"{source}.{layer}.{weight}").view(torch.uint8).fill_(
                    layer + 1
                )
        for layer in arena.transfer_order(target):
            for weight in layout.parameter_names:
                src = arena.view(f"{source}.{layer}.{weight}").view(torch.uint8)
                assert torch.all(src == layer + 1)
                arena.view(f"{target}.{layer}.{weight}").view(torch.uint8).fill_(255)
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


def test_qwen122b_fp8_arena_reserves_only_weights():
    layout = ExpertLayout(48, 256, 3072, 1024, 8, 8, weight_block_size=(128, 128))
    tensors = layout.tensors("tp")
    assert tensors["w13"] == ((256, 256, 3072), torch.float8_e4m3fn)
    assert set(tensors) == {"w13", "w2"}
    arena = ExpertArena(layout, "peer_access")
    assert arena.slab_bytes == 288 * 2**20
    assert arena.nbytes == 49 * arena.slab_bytes
    assert all(entry.dtype == torch.float8_e4m3fn for entry in arena.entries.values())


def test_fp8_scale_rows_do_not_constrain_weight_transfer_alignment():
    layout = ExpertLayout(1, 8, 128, 1024, 8, 8, weight_block_size=(128, 128))
    assert layout.shapes("tp") == ((8, 256, 128), (8, 128, 128))


@pytest.mark.parametrize("intermediate,block", [(768, (128, 128)), (1024, (64, 128))])
def test_fp8_rejects_tp_shards_that_split_quantization_blocks(intermediate, block):
    with pytest.raises(ValueError, match="128x128"):
        ExpertLayout(1, 256, 3072, intermediate, 8, 8, weight_block_size=block)


def test_fp8_layout_reads_quantization_from_multimodal_wrapper():
    from types import SimpleNamespace

    text = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        num_hidden_layers=48,
        num_experts=256,
        hidden_size=3072,
        moe_intermediate_size=1024,
    )
    config = SimpleNamespace(
        text_config=text,
        quantization_config={
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    )
    layout = ExpertLayout.from_model(config, ep_size=8, expert_tp_size=8)
    assert layout.weight_dtype == torch.float8_e4m3fn
    assert layout.weight_block_size == (128, 128)
    config.quantization_config["activation_scheme"] = "static"
    with pytest.raises(ValueError, match="dynamic block FP8"):
        ExpertLayout.from_model(config, ep_size=8, expert_tp_size=8)


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


@pytest.mark.parametrize("mode", ["ep", "tp"])
def test_fp8_registration_keeps_scales_outside_arena_with_loader_metadata(
    monkeypatch, mode
):
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.paras import runtime
    from vllm.model_executor.layers.quantization import fp8

    layout = ExpertLayout(1, 8, 512, 1024, 8, 8, weight_block_size=(128, 128))
    arena = ExpertArena(layout, "peer_access")
    arena.materialize("cpu")
    monkeypatch.setattr(runtime, "get_runtime", lambda: SimpleNamespace(arena=arena))
    monkeypatch.setattr(fp8, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(fp8, "select_fp8_moe_backend", lambda **kw: (None, None))
    layer = torch.nn.Module()
    layer.moe_config = SimpleNamespace(has_bias=False, w13_num_shards=2)
    config = fp8.Fp8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
    )
    method = runtime.ManagedFp8MoEMethod(
        config, layer, mode, "model.layers.0.mlp.experts"
    )
    w13, _ = layout.shapes(mode)
    loader = object()
    method.create_weights(
        layer, w13[0], 512, w13[1] // 2, torch.bfloat16, weight_loader=loader
    )
    for name, param_name in layout.parameter_names.items():
        param = getattr(layer, param_name)
        assert param.data_ptr() == arena.view(f"{mode}.0.{name}").data_ptr()
        assert param.dtype == layout.tensors(mode)[name][1]
        assert param.weight_loader is loader
    scales = [layer.w13_weight_scale_inv, layer.w2_weight_scale_inv]
    assert [tuple(p.shape) for p in scales] == (
        [(1, 16, 4), (1, 4, 8)] if mode == "ep" else [(8, 2, 4), (8, 4, 1)]
    )
    arena.buffer.fill_(255)
    for param in scales:
        assert param.device.type == "cpu"
        assert param.dtype == torch.float32 and torch.all(param == 1)
        assert not arena.is_managed(param)
        assert param.weight_loader is loader
        assert param.quant_method == "block"
    assert layer.w13_input_scale is None and layer.w2_input_scale is None


@pytest.mark.parametrize(
    "ep_backend,tp_backend", [("deep_gemm", "triton"), ("triton", "deep_gemm")]
)
@pytest.mark.parametrize("hopper,e8m0", [(True, False), (True, True), (False, False)])
def test_deepgemm_switch_rejects_layouts_that_requantize_weights(
    monkeypatch, ep_backend, tp_backend, hopper, e8m0
):
    from types import SimpleNamespace as NS

    from vllm.config.compilation import CUDAGraphMode
    from vllm.config.paras import ParasConfig
    from vllm.platforms import current_platform

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM_E8M0", str(int(e8m0)))
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: hopper
    )
    model = NS(
        dtype=torch.bfloat16,
        quantization="fp8",
        enforce_eager=False,
        hf_config=NS(
            model_type="qwen3_5_moe_text",
            num_hidden_layers=48,
            num_experts=256,
            hidden_size=3072,
            moe_intermediate_size=1024,
            quantization_config={
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
            },
        ),
    )
    parallel = NS(
        _api_process_count=1,
        data_parallel_external_lb=False,
        data_parallel_hybrid_lb=False,
        tensor_parallel_size=1,
        data_parallel_size=8,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_size_local=8,
        enable_expert_parallel=True,
        all2all_backend="deepep_low_latency",
        enable_dbo=False,
        enable_eplb=False,
        enable_elastic_ep=False,
    )
    config = NS(
        model_config=model,
        parallel_config=parallel,
        kernel_config=NS(moe_backend=ep_backend),
        speculative_config=None,
        lora_config=None,
        compilation_config=NS(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
    )
    paras = ParasConfig(expert_tp_size=8, expert_tp_backend=tp_backend)
    if hopper and not e8m0:
        paras.validate(config)
    else:
        with pytest.raises(ValueError, match="Hopper and FP32 scales"):
            paras.validate(config)
