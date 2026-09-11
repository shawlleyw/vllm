# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise a static DP server and record the evidence needed by the gate."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import numpy as np


def ranks(value):
    if isinstance(value, dict):
        if "dp_rank" in value:
            yield value
        else:
            for item in value.values():
                yield from ranks(item)
    elif isinstance(value, list):
        for item in value:
            yield from ranks(item)


async def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as client:

        async def raw_post(path, body=None, headers=None):
            async with client.post(
                args.url + path, json=body, headers=headers
            ) as response:
                response.raise_for_status()
                raw = await response.text()
                return json.loads(raw) if raw else None

        async def post(path, body=None, headers=None):
            # Worker collectives/profiling must not interleave with async forwards.
            if path not in ("/collective_rpc", "/start_profile", "/stop_profile"):
                return await raw_post(path, body, headers)
            await raw_post("/pause?mode=keep&clear_cache=false", {})
            try:
                return await raw_post(path, body, headers)
            finally:
                await raw_post("/resume", {})

        async def snapshot():
            result = await post(
                "/collective_rpc", {"method": "paras_static_snapshot", "timeout": 60}
            )
            result = sorted(ranks(result), key=lambda x: x["dp_rank"])
            assert [r["dp_rank"] for r in result] == list(range(args.world_size)), (
                result
            )
            return result

        async def generate(rank, prompt, tokens=64):
            start = time.perf_counter()
            result = await post(
                "/v1/completions",
                {
                    "model": "paras-qwen",
                    "prompt": prompt,
                    "max_tokens": tokens,
                    "temperature": 0,
                    "ignore_eos": True,
                    "logprobs": 5,
                    "return_tokens_as_token_ids": True,
                },
                {"X-data-parallel-rank": str(rank)},
            )
            elapsed = time.perf_counter() - start
            assert result["usage"]["completion_tokens"] == tokens, result
            assert result["choices"][0]["finish_reason"] == "length", result
            return {"rank": rank, "seconds": elapsed, "response": result}

        deadline = time.monotonic() + 600
        while True:
            try:
                async with client.get(args.url + "/health") as response:
                    if response.status == 200:
                        break
            except aiohttp.ClientError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Server did not become healthy in 600 seconds")
            await asyncio.sleep(2)

        before = await snapshot()
        (out / "before.json").write_text(json.dumps(before, indent=2))
        for rank in before:
            assert rank["runner"] == "vllm.v1.worker.gpu_model_runner", rank
            assert rank["graph_mode"] == "FULL_DECODE_ONLY", rank
            assert len(rank["attention_tp_ranks"]) == 1, rank
            assert len(rank["attention_dp_ranks"]) == args.world_size, rank
            assert rank["counters"]["num_cudagraph_captured"] > 0, rank
            layout = rank["expert_layout"]
            assert len(rank["layers"]) == layout["layers"], rank
            e, h, i = layout["experts"], layout["hidden"], layout["intermediate"]
            if args.mode == "ep":
                e //= args.world_size
            else:
                i //= args.world_size
            shapes = ([e, 2 * i, h], [e, h, i])
            for layer in rank["layers"]:
                assert (layer["w13_shape"], layer["w2_shape"]) == shapes, layer
                parallel = layer["parallel"]
                assert parallel["ep_size"] == (
                    args.world_size if args.mode == "ep" else 1
                )
                assert parallel["tp_size"] == (
                    1 if args.mode == "ep" else args.world_size
                )

        prompt = "The capital of France is"
        # Each rank must also make progress while all others have no requests.
        idle = [await generate(rank, prompt, 24) for rank in range(args.world_size)]
        # BF16 ReduceScatter can use different summation orders for each
        # destination rank. Record continuation differences; numerical tests
        # compare the same rank and explicitly matching input histories.
        idle_texts = [r["response"]["choices"][0]["text"] for r in idle]
        started = time.perf_counter()
        uneven = await asyncio.gather(
            *(generate(0, f"Explain the integer {i}.") for i in range(8)),
            *(
                generate(rank, f"Explain the integer {i}.")
                for rank in range(1, args.world_size)
                for i in range(1 + rank % 3)
            ),
        )
        wall = time.perf_counter() - started
        chunked = await asyncio.gather(
            *(
                generate(rank, "A short sentence. " * 400, 32)
                for rank in range(args.world_size)
            )
        )
        after = await snapshot()
        (out / "after.json").write_text(json.dumps(after, indent=2))
        for first, last in zip(before, after):
            assert first["layers"] == last["layers"], "Weight layout changed"
            assert first["counters"] == last["counters"], "Compiled or captured again"
            assert first["kv_addresses"] == last["kv_addresses"], "KV moved"
            assert first["stationary_tensors"] == last["stationary_tensors"]

        profiled = None
        if not args.skip_profile:
            await post("/start_profile")
            try:
                profiled = await generate(0, "The capital of Italy is", 32)
            finally:
                await post("/stop_profile")
        durations = [r["seconds"] for r in uneven]
        result = {
            "mode": args.mode,
            "world_size": args.world_size,
            "idle_rank_continuations_identical": len(set(idle_texts)) == 1,
            "serving_checks": "passed",
            "graph_replay": "not profiled in this run"
            if args.skip_profile
            else "requires verification of profiles",
            "requests": idle + uneven + chunked,
            "profiled_request": profiled,
            "uneven_batch_output_tokens_per_second": sum(
                r["response"]["usage"]["completion_tokens"] for r in uneven
            )
            / wall,
            "uneven_request_latency_median_seconds": float(np.median(durations)),
            "uneven_request_latency_p95_seconds": float(np.percentile(durations, 95)),
        }
        (out / "results.json").write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("requests", "profiled_request")
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--mode", required=True, choices=("ep", "tp"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-profile", action="store_true")
    asyncio.run(main(parser.parse_args()))
