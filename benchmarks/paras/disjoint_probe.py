# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical oracle ONLY: keep independent weight copies, bypass live transfers.

Imported as a diagnostic worker extension before model construction. This costs
twice the steady expert storage per rank and is never selected by the normal launcher.
"""

from dataclasses import replace

from static_probe import StaticProbe

from vllm.model_executor.layers.fused_moe.paras.storage import ExpertArena
from vllm.model_executor.layers.fused_moe.paras.transfer import WeightTransfer

_plan = ExpertArena.__init__
_move = WeightTransfer.move


def disjoint_plan(self, layout, transport):
    if transport != "peer_access":
        raise ValueError(
            "The disjoint oracle uses peer_access for initial materialization"
        )
    _plan(self, layout, transport)
    for layer in range(layout.layers):
        for name in ("w13", "w2"):
            key = f"tp.{layer}.{name}"
            self.entries[key] = replace(
                self.entries[key],
                offset=self.entries[key].offset + (layout.layers - 1) * self.slab_bytes,
            )
    self.nbytes = 2 * layout.layers * self.slab_bytes
    self.reference_tp_materialized = False


def fixed_weights(self, target):
    if not self.arena.reference_tp_materialized:
        if target != "tp":
            raise RuntimeError("Oracle must materialize TP before switching back")
        elapsed = _move(self, target)
        self.arena.reference_tp_materialized = True
        return elapsed
    return 0.0


ExpertArena.__init__ = disjoint_plan
WeightTransfer.move = fixed_weights


class DisjointProbe(StaticProbe):
    pass
