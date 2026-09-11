# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live PARAS acceptance: queued/prefill/decode/idle/cancellation and replay."""

import argparse
import asyncio
import contextlib
import json
import math
import statistics
import time
from pathlib import Path

import aiohttp
from check_static import ranks


async def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as client:

        async def raw_post(path, body=None, rank=None):
            async with client.post(
                args.url + path,
                json=body,
                headers={} if rank is None else {"X-data-parallel-rank": str(rank)},
            ) as r:
                text = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"{path}: HTTP {r.status}: {text}")
                return json.loads(text) if text else None

        async def post(path, body=None, rank=None):
            # Worker collectives/profiling must not interleave with async forwards.
            if path not in ("/collective_rpc", "/start_profile", "/stop_profile"):
                return await raw_post(path, body, rank)
            await raw_post("/pause?mode=keep&clear_cache=false", {})
            try:
                return await raw_post(path, body, rank)
            finally:
                await raw_post("/resume", {})

        async def status():
            async with client.get(args.url + "/paras/status") as r:
                r.raise_for_status()
                return await r.json()

        async def snapshot():
            data = await post("/collective_rpc", {"method": "paras_static_snapshot"})
            result = sorted(ranks(data), key=lambda r: r["dp_rank"])
            assert [r["dp_rank"] for r in result] == list(range(args.world_size))
            return result

        async def generate(rank, prompt, tokens, seen=None, label=""):
            body = {
                "model": "paras-qwen",
                "prompt": prompt,
                "max_tokens": tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
                "logprobs": 5,
                "return_tokens_as_token_ids": True,
            }
            started = time.perf_counter()
            text, ids, logprobs = "", set(), []
            usage = None
            finished = 0
            async with client.post(
                args.url + "/v1/completions",
                json=body,
                headers={"X-data-parallel-rank": str(rank)},
            ) as r:
                r.raise_for_status()
                async for line in r.content:
                    if (
                        not line.startswith(b"data: ")
                        or line.strip() == b"data: [DONE]"
                    ):
                        continue
                    event = json.loads(line[6:])
                    ids.add(event["id"])
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event["choices"]:
                        text += choice["text"]
                        if choice.get("logprobs"):
                            logprobs.extend(choice["logprobs"]["tokens"])
                        finished += choice.get("finish_reason") == "length"
                        if seen is not None and len(logprobs) >= 4:
                            seen.set()
            assert usage and usage["completion_tokens"] == tokens, (label, usage)
            assert finished == 1 and len(ids) == 1, (label, ids, finished)
            assert len(logprobs) == tokens, (label, len(logprobs))
            return {
                "label": label,
                "rank": rank,
                "id": ids.pop(),
                "text": text,
                "tokens": logprobs,
                "usage": usage,
                "seconds": time.perf_counter() - started,
            }

        for _ in range(300):
            try:
                initial = await status()
                break
            except (aiohttp.ClientError, OSError):
                await asyncio.sleep(2)
        else:
            raise TimeoutError("PARAS server did not initialize")
        assert initial["mode"] == "ep" and initial["epoch"] == 0
        before = await snapshot()
        (out / "before.json").write_text(json.dumps(before, indent=2))
        transitions, requests = [], []
        epoch = 0

        async def switch(target):
            nonlocal epoch
            previous = await status()
            result = await post("/paras/switch", {"target": target})
            epoch += previous["mode"] != target
            assert result["mode"] == target and result["epoch"] == epoch, result
            assert result["noop"] == (previous["mode"] == target), result
            transitions.append(result)
            return result

        await switch("ep")
        # Each destination mode must make progress while all other ranks are idle.
        for target in ("tp", "ep"):
            await switch(target)
            await switch(target)
            requests.append(
                await generate(
                    0, "The capital of France is", 24, label=f"idle-{target}"
                )
            )

        # More than 64 rank-0 requests guarantees queueing. A streaming token
        # event establishes that switching occurs during decode, not beforehand.
        first_tokens = asyncio.Event()
        tasks = [
            asyncio.create_task(
                generate(0, f"Explain integer {i}.", 96, first_tokens, f"queued-{i}")
            )
            for i in range(70)
        ]
        tasks += [
            asyncio.create_task(
                generate(rank, f"Explain number {i}.", 144, label=f"uneven-{rank}-{i}")
            )
            for rank in range(1, args.world_size)
            for i in range(1 + rank % 3)
        ]
        await asyncio.wait_for(first_tokens.wait(), 120)
        await post(
            "/collective_rpc", {"method": "paras_watch_stationary_state", "args": [4]}
        )
        active_counts = []
        for target in ("tp", "ep", "tp", "ep"):
            active_counts.append(sum(not t.done() for t in tasks))
            await switch(target)
        assert active_counts[0] > 0, active_counts
        requests.extend(await asyncio.gather(*tasks))

        # Large distinct prompts, above the scheduling budget, keep prefill
        # chunks active. Confirm a request is still waiting for its first tokens.
        await post(
            "/collective_rpc", {"method": "paras_watch_stationary_state", "args": [2]}
        )
        prefill_events = []
        tasks = []
        for rank in range(args.world_size):
            for index in range(args.prefill_requests_per_rank):
                seen = asyncio.Event()
                prefill_events.append(seen)
                tasks.append(
                    asyncio.create_task(
                        generate(
                            rank,
                            f"Document {rank}-{index}: "
                            + "A different short sentence. " * 1000,
                            96,
                            seen,
                            f"prefill-{rank}-{index}",
                        )
                    )
                )
        await asyncio.sleep(0.03)
        prefill_before_first_token = any(not seen.is_set() for seen in prefill_events)
        prefill_switches = [await switch("tp"), await switch("ep")]
        requests.extend(await asyncio.gather(*tasks))
        assert prefill_before_first_token
        prefill_active = [
            sum(
                0 < request["computed_tokens"] < request["prompt_tokens"]
                for rank in transition["requests_at_pause"]
                for request in rank
            )
            for transition in prefill_switches
        ]
        assert any(prefill_active), "No switch overlapped partial prefill"

        # Cancelling a generation request while peers continue must not poison
        # future scheduling or consume a request twice after a switch.
        cancel_seen = asyncio.Event()
        cancelled = asyncio.create_task(
            generate(0, "Write a very long story.", 2048, cancel_seen, "cancelled")
        )
        survivor_seen = asyncio.Event()
        survivor = asyncio.create_task(
            generate(1, "Count the positive integers.", 256, survivor_seen, "survivor")
        )
        await asyncio.wait_for(cancel_seen.wait(), 120)
        cancelled.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cancelled
        await switch("tp")
        requests.append(await survivor)
        await switch("ep")
        settled = await snapshot()
        # Repeated idle transitions detect continuing allocation growth and
        # produce transport/switch timing samples under identical conditions.
        idle_start = len(transitions)
        for _ in range(12):
            await switch("tp")
            await switch("ep")
        idle_samples = transitions[idle_start:]
        after = await snapshot()
        (out / "after.json").write_text(json.dumps(after, indent=2))
        (out / "settled.json").write_text(json.dumps(settled, indent=2))
        for first, warm, last in zip(before, settled, after):
            assert first["attention_tp_ranks"] == last["attention_tp_ranks"]
            assert first["attention_dp_ranks"] == last["attention_dp_ranks"]
            assert first["kv_addresses"] == last["kv_addresses"], "KV moved"
            assert first["stationary_tensors"] == last["stationary_tensors"]
            assert first["all_weight_addresses"] == last["all_weight_addresses"]
            assert len(last["stationary_transfer_checks"]) == 6
            assert all(check["passed"] for check in last["stationary_transfer_checks"])
            assert first["counters"] == last["counters"], (
                "Compilation/capture after init"
            )
            assert warm["memory_allocated"] == last["memory_allocated"], "Memory growth"
            assert warm["memory_reserved"] == last["memory_reserved"], "Reserved growth"
            assert last["paras"]["graphs"] == {"ep": 7, "tp": 7}
            assert all(
                last["paras"]["replays"][m] > first["paras"]["replays"][m]
                for m in ("ep", "tp")
            ), "Missing both-mode replay"
            assert (
                last["paras"]["graph_pools"]["ep"] != last["paras"]["graph_pools"]["tp"]
            )
        assert len({r["id"] for r in requests}) == len(requests)
        # Actual CUDA runtime traces supplement the mode-specific replay counters.
        for target in ("ep", "tp"):
            await switch(target)
            await post("/start_profile", {"profile_prefix": target})
            await generate(0, "The capital of Italy is", 32, label=f"profile-{target}")
            await post("/stop_profile")
        await switch("ep")
        samples = idle_samples
        summary = {}
        for target in ("ep", "tp"):
            values = [
                r["last_timings"]
                for r in samples
                if r["mode"] == target and not r["noop"]
            ]
            summary[target] = {
                key: {
                    "median": statistics.median(v[key] for v in values),
                    "p95": sorted(v[key] for v in values)[
                        math.ceil(0.95 * len(values)) - 1
                    ],
                }
                for key in values[0]
            }
        async_evidence = []
        if (out / "scheduler").is_dir():
            for rank in range(args.world_size):
                record = json.loads(
                    (out / "scheduler" / f"rank-{rank}.json").read_text()
                )
                assert record["async_scheduling"]
                assert record["steps_with_pending_outputs"] > 0
                assert len(record["pauses"]) == len(record["resumes"])
                retained = 0
                for paused, resumed in zip(record["pauses"], record["resumes"]):
                    assert resumed["pending_outputs"] == 0
                    assert paused["scheduled_steps"] == resumed["scheduled_steps"]
                    retained += len(set(paused["running"]) & set(resumed["running"]))
                assert retained > 0
                async_evidence.append(
                    {
                        "rank": rank,
                        "steps_with_pending_outputs": record[
                            "steps_with_pending_outputs"
                        ],
                        "retained_running_requests": retained,
                        "pauses_with_pending_outputs": sum(
                            p["pending_outputs"] > 0 for p in record["pauses"]
                        ),
                    }
                )
            assert any(r["pauses_with_pending_outputs"] for r in async_evidence)
        result = {
            "passed": True,
            "async_scheduling": before[0]["async_scheduling"],
            "async_evidence": async_evidence,
            "transport": initial["transport"],
            "world_size": args.world_size,
            "completed_requests": len(requests),
            "cancelled_requests": 1,
            "active_requests_at_switch": active_counts,
            "prefill_before_first_token": prefill_before_first_token,
            "prefill_active_at_switch": prefill_active,
            "transitions": transitions,
            "requests": requests,
            "timings_ms": summary,
        }
        (out / "live.json").write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("transitions", "requests")
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefill-requests-per-rank", type=int, default=1)
    args = parser.parse_args()
    if args.prefill_requests_per_rank < 1:
        parser.error("prefill-requests-per-rank must be positive")
    asyncio.run(main(args))
