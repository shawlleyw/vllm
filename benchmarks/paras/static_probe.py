# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only worker diagnostics for the PARAS static baseline gate."""

import dataclasses

import torch

from vllm.compilation.counter import compilation_counter
from vllm.distributed import get_dp_group, get_tp_group


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
        snapshot = {
            "runner": type(runner).__module__,
            "attention_tp_ranks": get_tp_group().ranks,
            "attention_dp_ranks": get_dp_group().ranks,
            "dp_rank": self.parallel_config.data_parallel_rank,
            "graph_mode": str(self.vllm_config.compilation_config.cudagraph_mode),
            "counters": dataclasses.asdict(compilation_counter),
            "layers": layers,
            "kv_addresses": [t.data_ptr() for t in runner.kv_caches],
            "memory_allocated": torch.accelerator.memory_allocated(),
            "memory_reserved": torch.accelerator.memory_reserved(),
        }
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

    def paras_logits_start(self):
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
