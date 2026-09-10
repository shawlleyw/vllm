# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep dense/shared weights and attention state outside the expert arena."""

import torch

from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from .storage import ExpertArena


class StationaryState:
    """Record bindings after warmup, including hybrid recurrent cache views.

    Values may evolve during generation. Switching must preserve their storage,
    shape and dtype; the routed submodules alone may change between modes.
    """

    def __init__(self, model: torch.nn.Module, arena: ExpertArena | None = None):
        self.model = model
        self.arena = arena
        self.signatures = self.snapshot()

    def snapshot(self) -> dict:
        result = {}
        routed_prefixes: list[str] = []

        def record(name, value):
            if isinstance(value, torch.Tensor):
                if (
                    self.arena is not None
                    and self.arena.buffer is not None
                    and value.numel()
                    and value.device == self.arena.buffer.device
                    and value.untyped_storage().data_ptr()
                    == self.arena.buffer.untyped_storage().data_ptr()
                ):
                    raise RuntimeError(
                        f"Stationary tensor overlaps expert arena: {name}"
                    )
                result[name] = (
                    value.data_ptr(),
                    str(value.device),
                    str(value.dtype),
                    tuple(value.shape),
                    tuple(value.stride()),
                )
            elif isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    record(f"{name}.{index}", item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    record(f"{name}.{key}", item)

        for name, module in self.model.named_modules():
            if any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in routed_prefixes
            ):
                continue
            if isinstance(module, RoutedExperts):
                routed_prefixes.append(name)
                continue
            for key, param in module.named_parameters(recurse=False):
                record(f"{name}.parameter.{key}", param)
            for key, buffer in module.named_buffers(recurse=False):
                record(f"{name}.buffer.{key}", buffer)
            # Cache bindings are not necessarily registered torch buffers. Hybrid
            # layers expose a tuple of convolution and recurrent state views.
            for key in ("kv_cache", "topk_indices_buffer"):
                if hasattr(module, key):
                    record(f"{name}.{key}", getattr(module, key))
        return result

    def check(self) -> None:
        current = self.snapshot()
        if current != self.signatures:
            changed = sorted(
                name
                for name in current.keys() | self.signatures.keys()
                if current.get(name) != self.signatures.get(name)
            )
            raise RuntimeError(f"Stationary tensor bindings changed: {changed[:8]}")
