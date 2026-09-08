# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aligned reservations and stable typed views, adapted from SGLang PARAS.

Attention parameters and KV caches never belong to this arena. Configurations
with multiple TP replicas require a different expert arena budget.
"""

from dataclasses import dataclass
from math import prod

import torch


def align(size: int, alignment: int = 256) -> int:
    return (size + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class ExpertLayout:
    layers: int
    experts: int
    hidden: int
    intermediate: int
    ep_size: int
    expert_tp_size: int

    def __post_init__(self):
        if min(self.layers, self.experts, self.hidden, self.intermediate) <= 0:
            raise ValueError("Expert dimensions must be positive")
        if self.ep_size not in (2, 4, 8) or self.expert_tp_size != self.ep_size:
            raise ValueError(
                "EP and expert TP must match at 2, 4, or 8; replicas need a new plan"
            )
        if self.experts % self.ep_size or self.intermediate % self.expert_tp_size:
            raise ValueError("Experts and intermediate width must divide their groups")
        if (self.intermediate // self.expert_tp_size * 2) % 16:
            raise ValueError("Peer transfer rows must be multiples of 16 bytes")

    @classmethod
    def from_model(cls, hf_config, *, ep_size: int, expert_tp_size: int):
        if hf_config.model_type != "qwen3_moe":
            raise ValueError("No PARAS expert layout registered for this model")
        return cls(
            hf_config.num_hidden_layers,
            hf_config.num_experts,
            hf_config.hidden_size,
            hf_config.moe_intermediate_size,
            ep_size=ep_size,
            expert_tp_size=expert_tp_size,
        )

    def shapes(self, mode: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if mode not in ("ep", "tp"):
            raise ValueError(mode)
        e = self.experts // self.ep_size if mode == "ep" else self.experts
        i = (
            self.intermediate
            if mode == "ep"
            else self.intermediate // self.expert_tp_size
        )
        return (e, 2 * i, self.hidden), (e, self.hidden, i)


@dataclass(frozen=True)
class Reservation:
    offset: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * self.dtype.itemsize


class ExpertArena:
    def __init__(self, layout: ExpertLayout, transport: str):
        if transport not in ("nccl", "peer_access"):
            raise ValueError(transport)
        self.layout = layout
        self.entries: dict[str, Reservation] = {}
        self.views: dict[str, torch.Tensor] = {}
        self.buffer: torch.Tensor | None = None
        self.nbytes = 0
        w13, w2 = layout.shapes("ep")
        self.w2_offset = align(prod(w13) * 2)
        self.slab_bytes = self.w2_offset + align(prod(w2) * 2)
        for mode in ("ep", "tp"):
            for layer in range(layout.layers):
                base = (layer + (mode == "tp")) * self.slab_bytes
                for name, shape, offset in zip(
                    ("w13", "w2"), layout.shapes(mode), (base, base + self.w2_offset)
                ):
                    self.reserve(
                        f"{mode}.{layer}.{name}", shape, torch.bfloat16, offset
                    )
        if transport == "nccl":
            self.reserve("scratch.w13", w13, torch.bfloat16)
            self.reserve("scratch.w2", w2, torch.bfloat16)

    def reserve(self, name, shape, dtype, offset=None) -> None:
        if self.buffer is not None or name in self.entries:
            raise ValueError("Reservations must be unique and precede materialization")
        entry = Reservation(
            align(self.nbytes) if offset is None else offset, tuple(shape), dtype
        )
        if entry.offset < 0 or entry.offset % 256 or min(entry.shape) <= 0:
            raise ValueError("Invalid reservation or alignment")
        self.entries[name] = entry
        self.nbytes = max(self.nbytes, align(entry.offset + entry.nbytes))

    def materialize(self, device, limit_bytes: int | None = None) -> None:
        if self.buffer is not None:
            raise RuntimeError("Arena already materialized")
        if limit_bytes is not None and self.nbytes > limit_bytes:
            raise ValueError("Expert arena exceeds its budget")
        self.buffer = torch.empty(self.nbytes, dtype=torch.uint8, device=device)
        for name, entry in self.entries.items():
            self.views[name] = (
                self.buffer.narrow(0, entry.offset, entry.nbytes)
                .view(entry.dtype)
                .view(entry.shape)
            )

    def view(self, name: str, shape=None) -> torch.Tensor:
        tensor = self.views[name]
        return tensor if shape is None else tensor.view(shape)

    def is_managed(self, tensor: torch.Tensor) -> bool:
        if self.buffer is None or tensor.device != self.buffer.device:
            return False
        start = tensor.data_ptr() - self.buffer.data_ptr()
        return tensor.is_contiguous() and 0 <= start <= self.nbytes - tensor.nbytes

    def transfer_order(self, target: str):
        if target == "tp":
            return range(self.layout.layers - 1, -1, -1)
        if target == "ep":
            return range(self.layout.layers)
        raise ValueError(target)
