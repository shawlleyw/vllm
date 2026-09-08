# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable MoE wrapper states. Only routed experts and expert communication switch."""

import copy
import dataclasses
import time
import weakref
from typing import Any

import regex as re
import torch
import torch.distributed as dist

from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.utils import set_weight_attrs

from .storage import ExpertArena, ExpertLayout
from .transfer import WeightTransfer

_runtime = None


def get_runtime():
    global _runtime
    if _runtime is None:
        _runtime = ParasRuntime(get_current_vllm_config())
    return _runtime


class ManagedMoEMethod(UnquantizedFusedMoEMethod):
    def __init__(self, moe, mode, layer_name):
        super().__init__(moe)
        self.mode = mode
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", layer_name)
        if match is None:
            raise ValueError(f"Unknown expert layer name: {layer_name}")
        self.layer_index = int(match.group(1))

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **attrs,
    ):
        if params_dtype != torch.bfloat16 or self.moe.has_bias:
            raise ValueError("PARAS currently requires unbiased BF16 routed experts")
        arena = get_runtime().arena
        shapes = (
            (num_experts, 2 * intermediate_size_per_partition, hidden_size),
            (num_experts, hidden_size, intermediate_size_per_partition),
        )
        for name, shape in zip(("w13", "w2"), shapes):
            tensor = arena.view(f"{self.mode}.{self.layer_index}.{name}")
            if tuple(tensor.shape) != shape:
                raise ValueError(f"Managed {name} shape differs from backend: {shape}")
            param = torch.nn.Parameter(tensor, requires_grad=False)
            layer.register_parameter(name + "_weight", param)
            set_weight_attrs(param, attrs)

    def process_weights_after_loading(self, layer):
        before = (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr())
        super().process_weights_after_loading(layer)
        after = (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr())
        if before != after:
            raise RuntimeError("MoE backend replaced managed expert storage")


class ParasRoutedExperts(RoutedExperts):
    def __init__(self, *args, paras_mode="ep", **kwargs):
        self.paras_mode = paras_mode
        super().__init__(*args, **kwargs)

    def _get_quant_method(self, prefix, quant_config, moe_config):
        if quant_config is not None:
            raise ValueError("PARAS quantization is not implemented")
        return ManagedMoEMethod(moe_config, self.paras_mode, prefix)


@dataclasses.dataclass
class MoEStates:
    runner: torch.nn.Module
    ep: RoutedExperts
    tp: RoutedExperts

    def activate(self, mode):
        experts = getattr(self, mode)
        self.runner.routed_experts = experts
        self.runner.moe_config = experts.moe_config
        if self.runner.shared_experts is not None:
            self.runner.shared_experts._moe_config = experts.moe_config


class ParasRuntime:
    def __init__(self, config):
        self.config = config
        config.paras_config.validate(config)
        layout = ExpertLayout.from_model(config.model_config.hf_config)
        self.arena = ExpertArena(layout, config.paras_config.weight_transfer_method)
        self.arena.materialize(
            torch.device("cuda", torch.accelerator.current_device_index())
        )
        self.mode = "ep"
        self.epoch = 0
        self.last_timings = {}
        self.failed = False
        self.initialized = False
        self.states = []
        self.graph_states: weakref.WeakKeyDictionary[Any, dict[str, dict]] = (
            weakref.WeakKeyDictionary()
        )
        self.graph_pools = {
            mode: torch.cuda.graph_pool_handle() for mode in ("ep", "tp")
        }
        self.prepared = None
        self.last_requests = []

    def initialize(self, model):
        from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

        dp = get_dp_group()
        # Expert topology is independent of attention TP. The two-rank milestone
        # has one expert TP replica spanning both attention DP ranks.
        self.expert_ranks = tuple(dp.ranks)
        self.cpu_group = dp.cpu_group
        self.transfer_group = dist.new_group(
            ranks=list(self.expert_ranks), backend="nccl"
        )
        communicator = get_ep_group().device_communicator
        assert communicator is not None
        self.communicator = communicator
        self.managers = {
            "ep": self.communicator.all2all_manager,
            "tp": AgRsAll2AllManager(get_ep_group().cpu_group),
        }
        parallel = copy.copy(self.config.parallel_config)
        parallel.enable_expert_parallel = False
        parallel.all2all_backend = "allgather_reducescatter"
        tp_parallel = FusedMoEParallelConfig.make(
            tp_size_=1,
            pcp_size_=1,
            dp_size_=2,
            sp_size_=1,
            vllm_parallel_config=parallel,
        )
        for runner in list(model.modules()):
            if not isinstance(runner, MoERunner):
                continue
            ep = runner.routed_experts
            if not isinstance(ep, ParasRoutedExperts):
                raise ValueError("Every routed expert layer must use managed storage")
            cfg = dataclasses.replace(
                ep.moe_config,
                moe_parallel_config=tp_parallel,
                num_local_experts=ep.global_num_experts,
                moe_backend="triton",
                intermediate_size_per_partition_unpadded=None,
            )
            mapping = ExpertMapManager(
                max_num_batched_tokens=cfg.max_num_tokens,
                top_k=cfg.experts_per_token,
                global_num_experts=cfg.num_experts,
                num_redundant_experts=0,
                num_expert_group=ep.num_expert_group,
                moe_parallel_config=tp_parallel,
                placement_strategy="linear",
                enable_eplb=False,
                num_fused_shared_experts=0,
                rocm_aiter_enabled=False,
            )
            kwargs = {
                name: getattr(ep, name)
                for name in (
                    "ckpt_gate_proj_name",
                    "ckpt_down_proj_name",
                    "ckpt_up_proj_name",
                    "is_fused_checkpoint_transposed",
                    "renormalize",
                    "use_grouped_topk",
                    "num_expert_group",
                    "topk_group",
                    "custom_routing_function",
                    "scoring_func",
                    "routed_scaling_factor",
                    "swiglu_limit",
                    "swiglu_alpha",
                    "swiglu_beta",
                    "e_score_correction_bias",
                    "apply_router_weight_on_input",
                )
            }
            tp = ParasRoutedExperts(
                ep.layer_name,
                ep.params_dtype,
                cfg,
                None,
                expert_map_manager=mapping,
                paras_mode="tp",
                **kwargs,
            )
            tp.quant_method.process_weights_after_loading(tp)
            self.states.append(MoEStates(runner, ep, tp))
        if len(self.states) != self.arena.layout.layers:
            raise ValueError("Model layers differ from reserved expert layout")
        self.transfer = WeightTransfer(
            self.arena,
            self.config.paras_config.weight_transfer_method,
            self.cpu_group,
            self.transfer_group,
        )
        self.assert_addresses()

    def assert_addresses(self):
        for index, state in enumerate(self.states):
            for mode in ("ep", "tp"):
                experts = getattr(state, mode)
                for name in ("w13", "w2"):
                    tensor = getattr(experts, name + "_weight")
                    if (
                        tensor.data_ptr()
                        != self.arena.view(f"{mode}.{index}.{name}").data_ptr()
                    ):
                        raise RuntimeError("Managed weight address changed")

    def select_graphs(self, target):
        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        for wrapper in list(CUDAGraphWrapper._all_instances):
            if wrapper.vllm_config.paras_config is None:
                continue
            if wrapper not in self.graph_states:
                self.graph_states[wrapper] = {"ep": {}, "tp": {}}
                self.graph_states[wrapper][self.mode] = (
                    wrapper.concrete_cudagraph_entries
                )
            wrapper.concrete_cudagraph_entries = self.graph_states[wrapper][target]
            wrapper.graph_pool = self.graph_pools[target]

    def activate(self, target):
        for state in self.states:
            state.activate(target)
        self.communicator.all2all_manager = self.managers[target]
        self.select_graphs(target)
        self.mode = target

    def initialize_mode(self, target):
        if target != self.mode:
            self.transfer.move(target)
        self.activate(target)

    def prepare(self, target, epoch, requests=None):
        error = None
        if target not in ("ep", "tp") or epoch != self.epoch + (target != self.mode):
            error = "Invalid target or transition epoch"
        if self.failed or not self.initialized:
            error = "PARAS is unavailable"
        try:
            self.assert_addresses()
            torch.accelerator.synchronize()
        except Exception as exc:
            error = str(exc)
        records: list[Any] = [None] * len(self.expert_ranks)
        dist.all_gather_object(
            records, (self.mode, self.epoch, target, epoch, error), group=self.cpu_group
        )
        if any(r != records[0] or r[-1] is not None for r in records):
            raise RuntimeError(f"PARAS readiness failed: {records}")
        self.last_requests = [None] * len(self.expert_ranks)
        dist.all_gather_object(self.last_requests, requests, group=self.cpu_group)
        self.prepared = (target, epoch)
        return self.status()

    def commit(self, target, epoch):
        if self.prepared != (target, epoch):
            raise RuntimeError("PARAS transfer was not prepared")
        if target == self.mode:
            self.prepared = None
            return self.status()
        started = time.perf_counter()
        # Once any layer moves, source weights may be overwritten. Leave all
        # engines paused and reject further transitions after ANY failure.
        self.failed = True
        transfer_ms = self.transfer.move(target)
        self.activate(target)
        self.assert_addresses()
        dist.barrier(group=self.cpu_group)
        self.epoch = epoch
        self.last_timings = {
            "transfer_ms": transfer_ms,
            "worker_switch_ms": (time.perf_counter() - started) * 1000,
        }
        self.failed = False
        self.prepared = None
        return self.status()

    def shutdown(self):
        global _runtime
        # The worker retires execution before releasing model resources. Drop
        # both graph sets before closing this process's imported IPC mappings.
        for modes in self.graph_states.values():
            for entries in modes.values():
                entries.clear()
        self.graph_states.clear()
        if hasattr(self, "transfer") and self.transfer.ipc is not None:
            self.transfer.stream.synchronize()
            self.transfer.ipc.close()
        self.states.clear()
        _runtime = None

    def status(self):
        assert self.arena.buffer is not None
        return {
            "mode": self.mode,
            "epoch": self.epoch,
            "transport": self.config.paras_config.weight_transfer_method,
            "failed": self.failed,
            "initialized": self.initialized,
            "rank": get_dp_group().rank_in_group,
            "expert_ranks": self.expert_ranks,
            "arena_bytes": self.arena.nbytes,
            "arena_address": self.arena.buffer.data_ptr(),
            "last_timings": self.last_timings,
            "requests_at_pause": self.last_requests,
            "graph_pools": self.graph_pools,
            "replays": {
                mode: sum(w.paras_replays[mode] for w in self.graph_states)
                for mode in ("ep", "tp")
            },
            "graphs": {
                mode: sum(len(s[mode]) for s in self.graph_states.values())
                for mode in ("ep", "tp")
            },
        }
