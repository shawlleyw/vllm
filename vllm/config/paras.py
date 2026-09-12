# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from vllm.config.utils import config, get_hash_factors, hash_factors


@config
class ParasConfig:
    """Explicit expert EP/TP switching with stationary attention and KV storage."""

    expert_tp_size: Literal[2, 4, 8] | None = None
    """Explicit expert TP ranks (2, 4, or 8); must equal the attention DP size."""
    weight_transfer_method: Literal["peer_access", "nccl"] = "peer_access"
    """Transport used to reshard the managed routed expert weights."""
    expert_tp_backend: Literal["triton", "deep_gemm"] = "triton"
    """Expert TP kernel backend. EP uses the configured moe_backend."""

    def compute_hash(self) -> str:
        return hash_factors(get_hash_factors(self, set()))

    def validate(self, vllm_config) -> None:
        import torch

        import vllm.envs as envs
        from vllm.config.compilation import CUDAGraphMode

        if self.expert_tp_size is None:
            raise ValueError("PARAS requires an explicit expert_tp_size (2, 4, or 8)")

        p = vllm_config.parallel_config
        m = vllm_config.model_config
        requirements = {
            "one API process and internal DP load balancing": (
                p._api_process_count == 1
                and not p.data_parallel_external_lb
                and not p.data_parallel_hybrid_lb
            ),
            "attention TP1 with matching DP/expert TP on one node": (
                p.tensor_parallel_size == 1
                and p.data_parallel_size == self.expert_tp_size
                and p.pipeline_parallel_size == 1
                and p.prefill_context_parallel_size == 1
                and p.decode_context_parallel_size == 1
                and p.data_parallel_size_local == self.expert_tp_size
            ),
            "EP startup with DeepEP low latency": (
                p.enable_expert_parallel and p.all2all_backend == "deepep_low_latency"
            ),
            "supported expert kernels": (
                vllm_config.kernel_config.moe_backend in ("triton", "deep_gemm")
                if m is not None and m.quantization == "fp8"
                else vllm_config.kernel_config.moe_backend == "batched_triton"
                and self.expert_tp_backend == "triton"
            ),
            "BF16 activations with BF16 or block FP8 routed experts": (
                m is not None
                and m.dtype == torch.bfloat16
                and m.quantization in (None, "fp8")
            ),
            "V1 model runner": not envs.VLLM_USE_V2_MODEL_RUNNER,
            "no DBO/EPLB/elastic EP": not (
                p.enable_dbo or p.enable_eplb or p.enable_elastic_ep
            ),
            "no speculative decoding or LoRA": (
                vllm_config.speculative_config is None
                and vllm_config.lora_config is None
            ),
            "full decode graphs": (
                not m.enforce_eager
                and vllm_config.compilation_config.cudagraph_mode
                == CUDAGraphMode.FULL_DECODE_ONLY
            ),
        }
        failures = [name for name, valid in requirements.items() if not valid]
        if failures:
            raise ValueError("PARAS requires " + ", ".join(failures))

        if "deep_gemm" in (
            vllm_config.kernel_config.moe_backend,
            self.expert_tp_backend,
        ):
            from vllm.platforms import current_platform

            if (
                not current_platform.is_device_capability_family(90)
                or envs.VLLM_USE_DEEP_GEMM_E8M0
            ):
                raise ValueError(
                    "PARAS DeepGEMM requires Hopper and FP32 scales "
                    "(VLLM_USE_DEEP_GEMM_E8M0=0) to preserve checkpoint weights"
                )

        from vllm.model_executor.layers.fused_moe.paras.storage import ExpertLayout

        ExpertLayout.from_model(
            m.hf_config,
            ep_size=p.data_parallel_size,
            expert_tp_size=self.expert_tp_size,
        )
