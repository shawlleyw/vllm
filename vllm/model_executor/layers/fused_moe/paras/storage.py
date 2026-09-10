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
    # Original transformer layer IDs with routed experts; dense layers are omitted.
    layer_indices: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.layer_indices is None:
            object.__setattr__(self, "layer_indices", tuple(range(self.layers)))
        indices = self.layer_indices
        assert indices is not None
        if (
            len(indices) != self.layers
            or tuple(sorted(set(indices))) != indices
            or any(i < 0 for i in indices)
        ):
            raise ValueError("Expert layer indices must be unique and increasing")
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
        # Multimodal wrappers describe the routed experts in their text config.
        config = getattr(hf_config, "text_config", None) or hf_config
        model_type = config.model_type
        indices = tuple(range(config.num_hidden_layers))
        if model_type in ("qwen2_moe", "qwen3_moe", "qwen3_next"):
            dense = set(getattr(config, "mlp_only_layers", []))
            step = getattr(config, "decoder_sparse_step", 1)
            if step < 1:
                raise ValueError("decoder_sparse_step must be positive")
            indices = tuple(
                i for i in indices if i not in dense and (i + 1) % step == 0
            )
            experts = config.num_experts
        elif model_type == "qwen3_5_moe_text":
            # Attention types alternate, but every decoder layer has routed experts.
            experts = config.num_experts
        elif model_type in ("glm4_moe", "glm4_moe_lite", "deepseek_v2", "deepseek_v3"):
            first = config.first_k_dense_replace
            step = (
                1 if model_type == "glm4_moe" else getattr(config, "moe_layer_freq", 1)
            )
            if first < 0 or first >= config.num_hidden_layers or step < 1:
                raise ValueError("Invalid dense prefix or MoE layer frequency")
            indices = tuple(i for i in indices if i >= first and i % step == 0)
            experts = config.n_routed_experts
        else:
            raise ValueError(f"No PARAS expert layout registered for {model_type}")
        return cls(
            len(indices),
            experts,
            config.hidden_size,
            config.moe_intermediate_size,
            ep_size=ep_size,
            expert_tp_size=expert_tp_size,
            layer_indices=indices,
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
        assert layout.layer_indices is not None
        for mode in ("ep", "tp"):
            for slot, layer_id in enumerate(layout.layer_indices):
                base = (slot + (mode == "tp")) * self.slab_bytes
                for name, shape, offset in zip(
                    ("w13", "w2"), layout.shapes(mode), (base, base + self.w2_offset)
                ):
                    self.reserve(
                        f"{mode}.{layer_id}.{name}", shape, torch.bfloat16, offset
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
        assert self.layout.layer_indices is not None
        if target == "tp":
            return reversed(self.layout.layer_indices)
        if target == "ep":
            return iter(self.layout.layer_indices)
        raise ValueError(target)
