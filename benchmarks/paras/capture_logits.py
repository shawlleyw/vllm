# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture full logits outside decode graphs for matching-history comparisons."""

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp


async def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as client:

        async def post(path, body):
            async with client.post(
                args.url + path, json=body, headers={"X-data-parallel-rank": "0"}
            ) as r:
                r.raise_for_status()
                raw = await r.text()
                return json.loads(raw) if raw else None

        if not args.already_started:
            await post("/collective_rpc", {"method": "paras_logits_start"})
        if args.history:
            history = json.loads(Path(args.history).read_text())["tokens"]
            assert len(history) == 32
            prompt_ids = (
                await post(
                    "/tokenize",
                    {
                        "model": "paras-qwen",
                        "prompt": "The capital of France is",
                        "add_special_tokens": False,
                    },
                )
            )["tokens"]
            # Isolate numerical measurements from earlier diagnostic requests.
            await post("/reset_prefix_cache", {})
            parts, tokens = [], []
            for step in range(32):
                result = await post(
                    "/v1/completions",
                    {
                        "model": "paras-qwen",
                        "prompt": prompt_ids
                        + [int(t.removeprefix("token_id:")) for t in history[:step]],
                        "max_tokens": 1,
                        "temperature": 0,
                        "ignore_eos": True,
                        "logprobs": 5,
                        "return_tokens_as_token_ids": True,
                    },
                )
                choice = result["choices"][0]
                parts.append(choice["text"])
                tokens.extend(choice["logprobs"]["tokens"])
            await post(
                "/collective_rpc", {"method": "paras_logits_stop", "args": [str(out)]}
            )
            (out / "generation.json").write_text(
                json.dumps(
                    {
                        "text": "".join(parts),
                        "tokens": tokens,
                        "history_tokens": history,
                        "forced_history": True,
                        "transition": None,
                    },
                    indent=2,
                )
            )
            print(f"Captured 32 forced-history steps into {out}", flush=True)
            return
        count = 0
        switched = False
        transition = None
        parts, tokens = [], []
        body = {
            "model": "paras-qwen",
            "prompt": "The capital of France is",
            "max_tokens": 32,
            "temperature": 0,
            "ignore_eos": True,
            "logprobs": 5,
            "return_tokens_as_token_ids": True,
            "stream": True,
        }
        async with client.post(
            args.url + "/v1/completions",
            json=body,
            headers={"X-data-parallel-rank": "0"},
        ) as r:
            r.raise_for_status()
            async for line in r.content:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                event = json.loads(line[6:])
                for choice in event["choices"]:
                    parts.append(choice["text"])
                    if choice.get("logprobs"):
                        tokens.extend(choice["logprobs"]["tokens"])
                        count = len(tokens)
                    if args.switch and count >= 4 and not switched:
                        transition = await post("/paras/switch", {"target": "tp"})
                        switched = True
        await post(
            "/collective_rpc", {"method": "paras_logits_stop", "args": [str(out)]}
        )
        (out / "generation.json").write_text(
            json.dumps(
                {"text": "".join(parts), "tokens": tokens, "transition": transition},
                indent=2,
            )
        )
        assert count == 32
        print(f"Captured {count} steps into {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8765")
    p.add_argument("--output", required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--history", help="generation.json providing forced input histories"
    )
    mode.add_argument("--switch", action="store_true")
    p.add_argument("--already-started", action="store_true")
    asyncio.run(main(p.parse_args()))
