# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in scheduler evidence for async overlap and request-preserving pauses."""

import json
import os
from pathlib import Path

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState


class AsyncProbeScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.scheduler_config.async_scheduling
        self.trace_dir = Path(os.environ["PARAS_SCHEDULER_TRACE_DIR"])
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.scheduled_steps = 0
        self.steps_with_pending_outputs = 0
        self.pauses = []
        self.resumes = []

    def _update_after_schedule(self, output):
        if output.total_num_scheduled_tokens:
            self.scheduled_steps += 1
            if any(
                self.requests[req_id].num_output_placeholders > 0
                for req_id in output.num_scheduled_tokens
            ):
                self.steps_with_pending_outputs += 1
        super()._update_after_schedule(output)

    def set_pause_state(self, state):
        if state == PauseState.PAUSED_ALL:
            self.pauses.append(self.snapshot())
        elif state == PauseState.UNPAUSED and self.pause_state == PauseState.PAUSED_ALL:
            snapshot = self.snapshot()
            assert snapshot["pending_outputs"] == 0, snapshot
            self.resumes.append(snapshot)
        super().set_pause_state(state)
        self.write_trace()

    def snapshot(self):
        return {
            "running": [r.request_id for r in self.running],
            "waiting": [r.request_id for r in self.waiting],
            "pending_outputs": sum(
                r.num_output_placeholders for r in self.requests.values()
            ),
            "scheduled_steps": self.scheduled_steps,
            "steps_with_pending_outputs": self.steps_with_pending_outputs,
        }

    def write_trace(self):
        record = {
            "rank": self.parallel_config.data_parallel_rank,
            "async_scheduling": True,
            **self.snapshot(),
            "pauses": self.pauses,
            "resumes": self.resumes,
        }
        path = self.trace_dir / f"rank-{record['rank']}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(path)
