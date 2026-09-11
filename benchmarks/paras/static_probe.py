# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in worker diagnostics for the PARAS static baseline gate."""

import dataclasses
import os
import sys

import torch

from vllm.compilation.counter import compilation_counter
from vllm.distributed import get_dp_group, get_tp_group

if os.environ.get("PARAS_REFERENCE_NUMERICS") == "1":
    torch.backends.cuda.preferred_blas_library(backend="cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (False, False)
    torch.backends.cuda.matmul.fp32_precision = "ieee"


class StaticProbe:
    def paras_static_snapshot(self):
        runner = self.model_runner
        layers = []
        for name, module in runner.model.named_modules():
            if hasattr(module, "w13_weight") and hasattr(module, "w2_weight"):
                parallel = module.moe_config.moe_parallel_config
                layers.append(
                    {
                        "name": name,
                        "parallel": dataclasses.asdict(parallel),
                        "w13_shape": list(module.w13_weight.shape),
                        "w2_shape": list(module.w2_weight.shape),
                        "w13_address": module.w13_weight.data_ptr(),
                        "w2_address": module.w2_weight.data_ptr(),
                        "method": type(module.quant_method).__name__,
                    }
                )
        from vllm.model_executor.layers.fused_moe.paras.state import StationaryState
        from vllm.model_executor.layers.fused_moe.paras.storage import ExpertLayout

        layout = ExpertLayout.from_model(
            self.vllm_config.model_config.hf_text_config,
            ep_size=get_dp_group().world_size,
            expert_tp_size=get_dp_group().world_size,
        )
        attention_backends = {}
        for (
            name,
            module,
        ) in self.vllm_config.compilation_config.static_forward_context.items():
            if hasattr(module, "get_attn_backend"):
                attention_backends[name] = module.get_attn_backend().get_name()
        snapshot = {
            "expert_layout": dataclasses.asdict(layout),
            "attention_backends": attention_backends,
            "stationary_tensors": StationaryState(runner.model).signatures,
            "stationary_transfer_checks": getattr(self, "_paras_state_checks", []),
            "runner": type(runner).__module__,
            "async_scheduling": runner.use_async_scheduling,
            "max_concurrent_batches": self.vllm_config.max_concurrent_batches,
            "attention_tp_ranks": get_tp_group().ranks,
            "attention_dp_ranks": get_dp_group().ranks,
            "dp_rank": self.parallel_config.data_parallel_rank,
            "graph_mode": str(self.vllm_config.compilation_config.cudagraph_mode),
            "torch_deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
            "counters": dataclasses.asdict(compilation_counter),
            "layers": layers,
            "kv_addresses": [t.data_ptr() for t in runner.kv_caches],
            "memory_allocated": torch.accelerator.memory_allocated(),
            "memory_reserved": torch.accelerator.memory_reserved(),
        }
        if "paras_tuning" in sys.modules:
            snapshot["frozen_autotune"] = sys.modules["paras_tuning"].status()
        if hasattr(runner, "paras"):
            snapshot["paras"] = runner.paras.status()
            snapshot["all_weight_addresses"] = {
                name: tensor.data_ptr()
                for name, tensor in runner.paras.arena.views.items()
            }
        # DP utility calls return only the first engine's result to HTTP.
        # Gather diagnostics on the CPU group so that result includes both ranks.
        snapshots = [None] * get_dp_group().world_size
        torch.distributed.all_gather_object(
            snapshots, snapshot, group=get_dp_group().cpu_group
        )
        return snapshots

    def paras_tensor_digests(self):
        """Hash model parameters for explicit, untimed corruption diagnostics."""
        import hashlib

        hashes = {}
        for name, tensor in self.model_runner.model.named_parameters():
            raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            hashes[name] = hashlib.sha256(memoryview(raw.numpy())).hexdigest()
        result = [None] * get_dp_group().world_size
        torch.distributed.all_gather_object(
            result, hashes, group=get_dp_group().cpu_group
        )
        return result

    def paras_logits_start(self, forced_tokens=None):
        runner = self.model_runner
        self._paras_logits = []
        self._paras_logit_modes = []
        self._paras_compute_logits = runner.model.compute_logits

        def record(hidden_states, *args, **kwargs):
            logits = self._paras_compute_logits(hidden_states, *args, **kwargs)
            if logits is not None:
                self._paras_logits.append(logits.detach().cpu())
                mode = (
                    runner.paras.mode
                    if hasattr(runner, "paras")
                    else ("ep" if self.parallel_config.enable_expert_parallel else "tp")
                )
                self._paras_logit_modes.append(mode)
                if forced_tokens is not None:
                    # Save the unmodified scores, then force the sampled token.
                    # This retains one live request and its decode/KV history.
                    step = len(self._paras_logits) - 1
                    assert logits.shape[0] == 1 and step < len(forced_tokens)
                    forced = torch.full_like(logits, float("-inf"))
                    forced[:, forced_tokens[step]] = 0
                    return forced
            return logits

        runner.model.compute_logits = record
        return True

    def paras_logits_stop(self, directory):
        from pathlib import Path

        self.model_runner.model.compute_logits = self._paras_compute_logits
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        destination = path / f"rank{get_dp_group().rank_in_group}.pt"
        torch.save(self._paras_logits, destination)
        import json

        (path / f"rank{get_dp_group().rank_in_group}-modes.json").write_text(
            json.dumps(self._paras_logit_modes)
        )
        self._paras_logits = []
        return str(destination)

    def paras_watch_stationary_state(self, switches=2):
        """Fingerprint all fixed weights/cache bytes around selected live transfers.

        This diagnostic adds bandwidth overhead and is disabled for timing samples.
        It reads complete backing storages, including hybrid conv/recurrent views,
        without allocating a second cache or copying the model to the CPU.
        """
        runtime = self.model_runner.paras
        self._paras_watch_remaining = switches
        if hasattr(self, "_paras_original_move"):
            return True
        self._paras_original_move = runtime.transfer.move
        self._paras_state_checks = []

        def fingerprint():
            tensors = list(self.model_runner.model.parameters())
            tensors.extend(self.model_runner.kv_caches)
            storages = {}
            arena_address = runtime.arena.buffer.untyped_storage().data_ptr()
            for tensor in tensors:
                storage = tensor.untyped_storage()
                if tensor.device.type != "cuda" or not storage.nbytes():
                    continue
                if storage.data_ptr() != arena_address:
                    storages[storage.data_ptr()] = storage
            sums = []
            total_bytes = 0
            for storage in storages.values():
                raw = torch.empty(0, dtype=torch.uint8, device=self.device).set_(
                    storage, 0, (storage.nbytes(),), (1,)
                )
                size = raw.numel()
                # uint8 -> int64 reductions may materialize the conversion.
                # Bound that temporary to 256 MiB even for large cache storages.
                for start in range(0, size, 32 * 1024**2):
                    chunk = raw[start : start + 32 * 1024**2]
                    aligned = chunk.numel() // 8 * 8
                    sums.append(chunk[:aligned].view(torch.int64).sum())
                    sums.append(chunk.sum(dtype=torch.int64))
                total_bytes += size
            return torch.stack(sums).cpu(), total_bytes, len(storages)

        def checked_move(target):
            if self._paras_watch_remaining <= 0:
                return self._paras_original_move(target)
            torch.accelerator.synchronize()
            before, total_bytes, count = fingerprint()
            elapsed = self._paras_original_move(target)
            torch.accelerator.synchronize()
            after, _, _ = fingerprint()
            if not torch.equal(before, after):
                raise RuntimeError(
                    "Transfer modified stationary weights or attention state"
                )
            self._paras_state_checks.append(
                dict(target=target, passed=True, bytes=total_bytes, storages=count)
            )
            self._paras_watch_remaining -= 1
            return elapsed

        runtime.transfer.move = checked_move
        return True
