# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequential static and switching acceptance on explicitly selected idle GPUs.

Downloads and CPU checks must finish before invoking this GPU-only stage. The
launcher rechecks GPU availability before every server. Only child processes
started by this script are terminated on failure or completion.
"""

import argparse
import contextlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from check_static import ranks

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
BENCH = ROOT / "benchmarks/paras"
MODELS = ("GLM-4.7-Flash", "GLM-4.5-Air", "Qwen3.6-35B-A3B")


def request(url, path, body=None, *, quiesce=True):
    if quiesce and path in ("/collective_rpc", "/start_profile", "/stop_profile"):
        request(url, "/pause?mode=keep&clear_cache=false", {})
        try:
            return request(url, path, body, quiesce=False)
        finally:
            request(url, "/resume", {})
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def stop(process):
    # The launcher may exit before its workers. Always retire its process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=30)
    time.sleep(2)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def main(args):
    if args.resume and args.logits_only:
        raise ValueError("--resume reuses completed serving stages only")
    gpus = args.gpus.split(",")
    if len(gpus) != args.world_size or len(set(gpus)) != len(gpus) or "1" in gpus:
        raise ValueError("Select four distinct GPUs, excluding broken GPU 1")
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        PARAS_GPUS=args.gpus,
        PARAS_PORT=str(args.port),
        PARAS_LANGUAGE_MODEL_ONLY="1",
    )
    env.pop("PARAS_TRANSPORT", None)
    execution_env = {
        key: value
        for key, value in env.items()
        if key.startswith(("NCCL_", "CUBLAS_", "CUBLASLT_"))
        or key
        in (
            "VLLM_BATCH_INVARIANT",
            "PARAS_CACHE_ROOT",
            "TRITON_CACHE_MANAGER",
            "PARAS_FROZEN_AUTOTUNE",
            "PARAS_REFERENCE_NUMERICS",
            "PARAS_ASYNC_SCHEDULING",
            "PARAS_SCHEDULER_TRACE",
        )
    }
    url = f"http://127.0.0.1:{args.port}"
    try:
        request(url, "/health")
    except (urllib.error.URLError, TimeoutError):
        pass
    else:
        raise RuntimeError(f"A server already owns {url}")

    def run(script, *arguments, output):
        command = [str(PYTHON), str(BENCH / script), *map(str, arguments)]
        print("Running:", " ".join(command), flush=True)
        with output.open("w") as log:
            subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    for model in args.models:
        path = args.model_root / model
        if not (path / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(f"Missing model checkpoint index: {path}")
        # Verify every indexed shard exists before occupying any GPU.
        index = json.loads((path / "model.safetensors.index.json").read_text())
        for shard in set(index["weight_map"].values()):
            if not (path / shard).is_file():
                raise FileNotFoundError(path / shard)
        synthetic_path = path / "synthetic.json"
        synthetic = (
            json.loads(synthetic_path.read_text()) if synthetic_path.is_file() else None
        )
        env["PARAS_MODEL"] = str(path)
        env["PARAS_MEMORY_UTILIZATION"] = os.environ.get(
            "PARAS_MEMORY_UTILIZATION",
            "0.95" if model == "GLM-4.5-Air" and synthetic is None else "0.85",
        )
        if model == "Qwen3.6-35B-A3B":
            env["PARAS_MAMBA_CACHE_MODE"] = "align"
        else:
            env.pop("PARAS_MAMBA_CACHE_MODE", None)
        out = args.output / model
        out.mkdir(exist_ok=True)
        for method in () if args.logits_only else ("peer_access", "nccl"):
            with (out / f"transfer-{method}.log").open("w") as log:
                subprocess.run(
                    [
                        str(BENCH / "launch_transfer.sh"),
                        method,
                        str(out / f"transfer-{method}.json"),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        references = {mode: out / f"static-{mode}" / "logits" for mode in ("ep", "tp")}
        candidates = [
            out / method / f"logits-{target}"
            for method in ("peer_access", "nccl")
            for target in ("ep", "tp")
        ]
        for mode in ("static-ep", "static-tp", "peer_access", "nccl"):
            stage = out / mode
            stage.mkdir(exist_ok=True)
            switching = mode in ("peer_access", "nccl")
            if args.resume:
                result_path = stage / ("live.json" if switching else "results.json")
                logits_dirs = (
                    [stage / f"logits-{target}" for target in ("ep", "tp")]
                    if switching
                    else [stage / "logits"]
                )
                if (
                    result_path.is_file()
                    and (stage / "replay.json").is_file()
                    and all((p / "generation.json").is_file() for p in logits_dirs)
                ):
                    result = json.loads(result_path.read_text())
                    assert result["world_size"] == args.world_size
                    assert (
                        result["passed"]
                        if switching
                        else result["serving_checks"] == "passed"
                    )
                    launch = json.loads((stage / "launch.json").read_text())
                    assert launch["gpus"] == gpus and launch["model"] == str(path)
                    snapshots = json.loads((stage / "before.json").read_text())
                    assert all(
                        r.get("async_scheduling")
                        == (env.get("PARAS_ASYNC_SCHEDULING", "1") == "1")
                        for r in snapshots
                    )
                    assert all(
                        json.loads((p / "generation.json").read_text()).get("execution")
                        == "one_prefill_then_decode"
                        for p in logits_dirs
                    )
                    print(f"Reusing passed serving stage: {stage}", flush=True)
                    continue
            launcher = "launch_paras.sh" if switching else "launch_static.sh"
            argument = mode if switching else mode.removeprefix("static-")
            command = [str(BENCH / launcher), argument, str(stage)]
            (stage / "launch.json").write_text(
                json.dumps(
                    dict(
                        command=command,
                        model=str(path),
                        gpus=gpus,
                        execution_env=execution_env,
                    ),
                    indent=2,
                )
            )
            print(f"Starting {model}: {mode}", flush=True)
            with (stage / "launcher.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    deadline = time.monotonic() + 2400
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f"Server exited: {stage}/server.log")
                        try:
                            request(url, "/health")
                            break
                        except (urllib.error.URLError, TimeoutError):
                            if time.monotonic() > deadline:
                                raise TimeoutError(f"Server startup: {stage}") from None
                            time.sleep(2)
                    if args.logits_only:
                        before = sorted(
                            ranks(
                                request(
                                    url,
                                    "/collective_rpc",
                                    {"method": "paras_static_snapshot"},
                                )
                            ),
                            key=lambda r: r["dp_rank"],
                        )
                        (stage / "before.json").write_text(json.dumps(before, indent=2))
                    if switching:
                        if not args.logits_only:
                            run(
                                "check_live.py",
                                "--prefill-requests-per-rank",
                                8 if synthetic else 1,
                                "--world-size",
                                args.world_size,
                                "--url",
                                url,
                                "--output",
                                stage,
                                output=stage / "check.log",
                            )
                        for target in ("ep", "tp"):
                            request(url, "/paras/switch", {"target": target})
                            destination = stage / f"logits-{target}"
                            if args.logits_only:
                                request(
                                    url, "/start_profile", {"profile_prefix": target}
                                )
                            run(
                                "capture_logits.py",
                                "--url",
                                url,
                                "--output",
                                destination,
                                "--history",
                                references["ep"] / "generation.json",
                                output=stage / f"capture-{target}.log",
                            )
                            if args.logits_only:
                                request(url, "/stop_profile", {})
                    else:
                        target = argument
                        if not args.logits_only:
                            run(
                                "check_static.py",
                                "--world-size",
                                args.world_size,
                                "--mode",
                                target,
                                "--url",
                                url,
                                "--output",
                                stage,
                                output=stage / "check.log",
                            )
                        history = (
                            []
                            if target == "ep"
                            else ["--history", references["ep"] / "generation.json"]
                        )
                        if args.logits_only:
                            request(url, "/start_profile", {"profile_prefix": target})
                        run(
                            "capture_logits.py",
                            "--url",
                            url,
                            "--output",
                            references[target],
                            *history,
                            output=stage / "capture.log",
                        )
                        if args.logits_only:
                            request(url, "/stop_profile", {})
                    if args.logits_only:
                        after = sorted(
                            ranks(
                                request(
                                    url,
                                    "/collective_rpc",
                                    {"method": "paras_static_snapshot"},
                                )
                            ),
                            key=lambda r: r["dp_rank"],
                        )
                        (stage / "after.json").write_text(json.dumps(after, indent=2))
                        assert len(before) == len(after) == args.world_size
                        for first, last in zip(before, after):
                            assert first["counters"] == last["counters"]
                            assert first["kv_addresses"] == last["kv_addresses"]
                            assert (
                                first["stationary_tensors"]
                                == last["stationary_tensors"]
                            )
                            if switching:
                                assert (
                                    first["all_weight_addresses"]
                                    == last["all_weight_addresses"]
                                )
                                assert last["paras"]["graphs"] == {"ep": 7, "tp": 7}
                                assert all(
                                    last["paras"]["replays"][m]
                                    > first["paras"]["replays"][m]
                                    for m in ("ep", "tp")
                                )
                    run(
                        "verify_replay.py",
                        "--world-size",
                        args.world_size,
                        stage,
                        output=stage / "replay.log",
                    )
                finally:
                    stop(process)
            # CUDA process teardown can briefly outlive the frontend.
            time.sleep(5)
        run(
            "compare_logits.py",
            "--ep",
            references["ep"],
            "--tp",
            references["tp"],
            "--output",
            out / "correctness.json",
            *candidates,
            output=out / "compare.log",
        )
        (out / "complete.json").write_text(
            json.dumps(
                dict(
                    model=model,
                    synthetic=synthetic,
                    gpus=gpus,
                    serving_checks=not args.logits_only,
                    batch_invariant=env.get("VLLM_BATCH_INVARIANT", "0"),
                    execution_env=execution_env,
                    static_modes=["ep", "tp"],
                    transports=["peer_access", "nccl"],
                    correctness=str(out / "correctness.json"),
                ),
                indent=2,
            )
        )
        print(f"{model}: passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--world-size", type=int, choices=(4,), required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--model-root", type=Path, default=Path("/data/shaoyuw/models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse passed serving stages for the same model and GPUs",
    )
    parser.add_argument(
        "--logits-only",
        action="store_true",
        help="Run numerical captures and graph checks without serving/transfer suites",
    )
    main(parser.parse_args())
