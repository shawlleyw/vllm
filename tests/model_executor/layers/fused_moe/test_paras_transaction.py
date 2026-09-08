# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Control transactions must preserve scheduling or fail closed.

A fake engine catches ordering, cancellation, and error recovery without GPUs;
real serving and cross-rank transfers are exercised by benchmarks/paras.
"""

import asyncio
from types import SimpleNamespace

import pytest

from vllm.v1.engine.async_llm import AsyncLLM


class Client:
    paras_switch = AsyncLLM.paras_switch
    _paras_switch = AsyncLLM._paras_switch
    _paras_status = AsyncLLM._paras_status
    paras_status = AsyncLLM.paras_status

    def __init__(self, failure=None):
        self.vllm_config = SimpleNamespace(paras_config=True)
        self._paras_lock = asyncio.Lock()
        self._paras_failed = False
        self._paras_tasks = set()
        self._paras_last_timings = {}
        self.calls = []
        self.failure = failure
        self.mode = "ep"
        self.epoch = 0
        self.paused = False
        self.transferring = asyncio.Event()
        self.proceed = asyncio.Event()
        self.proceed.set()

    async def pause_generation(self, *, mode, clear_cache):
        assert mode == "keep" and clear_cache is False
        self.calls.append("pause")
        self.paused = True

    async def resume_generation(self):
        assert not self._paras_failed
        self.calls.append("resume")
        self.paused = False

    async def collective_rpc(self, method, args=(), timeout=None):
        if method == "paras_status":
            return [
                {
                    "mode": self.mode,
                    "epoch": self.epoch,
                    "failed": False,
                    "last_timings": {"transfer_ms": 1},
                }
            ]
        self.calls.append(method)
        assert self.paused
        if method == self.failure:
            raise RuntimeError("injected failure")
        if method == "paras_commit":
            self.transferring.set()
            await self.proceed.wait()
            self.mode, self.epoch = args


@pytest.mark.asyncio
async def test_prepare_failure_resumes_preserved_requests():
    client = Client("paras_prepare")
    with pytest.raises(RuntimeError, match="injected"):
        await client.paras_switch("tp")
    assert not client.paused and client.mode == "ep" and client.epoch == 0
    assert client.calls == ["pause", "paras_prepare", "resume"]


@pytest.mark.asyncio
async def test_destructive_failure_leaves_execution_stopped():
    client = Client("paras_commit")
    with pytest.raises(RuntimeError, match="remains stopped"):
        await client.paras_switch("tp")
    assert client.paused and client._paras_failed
    with pytest.raises(RuntimeError, match="restart"):
        await client.paras_switch("ep")
    assert "resume" not in client.calls


@pytest.mark.asyncio
async def test_disconnected_control_request_completes_and_serializes():
    client = Client()
    client.proceed.clear()
    first = asyncio.create_task(client.paras_switch("tp"))
    await client.transferring.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.create_task(client.paras_switch("ep"))
    client.proceed.set()
    result = await second
    assert result["epoch"] == 2 and client.mode == "ep" and not client.paused
    calls = list(client.calls)
    result = await client.paras_switch("ep")
    assert result["noop"] and result["epoch"] == 2 and client.calls == calls
