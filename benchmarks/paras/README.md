# PARAS EP ↔ expert TP on vLLM V1

This experimental implementation supports matching EP/TP sizes 2, 4, and 8
on one node. It keeps attention TP1/DP, requests,
prefix caching, and KV storage stationary while explicitly switching routed
experts. The original validated target is Qwen3-30B-A3B BF16 on two A100 GPUs, based on
vLLM `2cf0a6915ce544dc493a0990f2ea38d81601128a` (v0.28.0).

## Environment

```bash
benchmarks/paras/setup_env.sh
benchmarks/paras/build_deepep.sh /data/shaoyuw/paras/vllm-milestone/DeepEP-sm80
conda activate /home/shaoyuw/vllm/.venv
```

`setup_env.sh` creates a **conda** prefix at `.venv`, with Python 3.12 and uv,
installs editable vLLM and native extensions for the exact upstream commit, and
installs pre-commit hooks. Python dependencies, NCCL, NVSHMEM, and the PARAS
extension cache are inside this prefix. The system CUDA 13.0 toolkit supplies
`nvcc`. Launchers replace inherited library paths with the conda paths.

The DeepEP source is an isolated clone of the user-approved A100 port at
`8e57c764c7d3fdb0999fe3e34a03371c8f53ae1b`. The vLLM Dockerfile's original pin
`d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb` rejects SM80 builds. The build script
applies four idempotent CUDA 13/local NVLink compatibility fixes: native CUDA FP8
header declarations, NVSHMEM host SONAME resolution, configurable IBGDA enablement,
and an RDMA queue assertion restricted to RDMA peers. The milestone uses BF16.

The launcher sets `NVSHMEM_REMOTE_TRANSPORT=none`, `NVSHMEM_IB_ENABLE_IBGDA=0`, and
`NVSHMEM_QP_DEPTH=2048`. The last setting accommodates the 512-token scheduling
budget. There is no RDMA or multi-node transport in this milestone.

## Static baseline gate

Run servers sequentially and stop each foreground server with Ctrl-C:

```bash
PARAS_GPUS=6,7 benchmarks/paras/launch_static.sh ep /data/shaoyuw/paras/vllm-milestone/runs/ep-new
PARAS_GPUS=6,7 benchmarks/paras/launch_static.sh tp /data/shaoyuw/paras/vllm-milestone/runs/tp-new
```

For each server, run the matching checks from another shell:

```bash
.venv/bin/python benchmarks/paras/check_static.py --world-size 2 --mode ep \
  --output /data/shaoyuw/paras/vllm-milestone/runs/ep-new
.venv/bin/python benchmarks/paras/verify_replay.py --world-size 2 \
  /data/shaoyuw/paras/vllm-milestone/runs/ep-new
```

Both static configurations passed before runtime code was introduced. Evidence
is preserved in `runs/ep-conda-final` and `runs/tp-conda-final` under the milestone
artifact directory. EP uses DeepEP low latency and batched Triton experts; TP uses
AllGather/ReduceScatter and Triton experts across the attention DP ranks.

All launchers check GPU availability immediately before launch. They use the
original V1 model runner, FlashAttention 2, BF16, 8192-token context, 64 sequences
per rank, 512-token scheduling budget, chunked prefill, prefix caching, and full
decode graphs at sizes 1, 2, 4, 8, 16, 32, 64. Async scheduling, DBO, EPLB, elastic
EP, and speculative decoding are disabled. GPU memory utilization is 0.85.

## Switching server

```bash
PARAS_GPUS=6,7 benchmarks/paras/launch_paras.sh peer_access /data/shaoyuw/paras/vllm-milestone/runs/peer-new
# Use nccl instead of peer_access for the transfer baseline.
```

This adds:

```text
--paras-config '{"expert_tp_size":2,"weight_transfer_method":"peer_access"}'
--api-server-count 1
```

PARAS requires one API process to serialize control transactions. Its engine
client broadcasts utility operations to **all** DP engines, including idle
ranks. The model starts and finishes initialization in EP mode.

```bash
curl -s http://127.0.0.1:8765/paras/status
curl -s http://127.0.0.1:8765/paras/switch \
  -H 'Content-Type: application/json' -d '{"target":"tp"}'
curl -s http://127.0.0.1:8765/paras/switch \
  -H 'Content-Type: application/json' -d '{"target":"ep"}'
```

Status reports mode, epoch, transport, last phase timings, graph counts/pools,
replay counts, arena size, and the request positions recorded at the last pause.
An active-mode switch is a no-op. The client pauses with `mode="keep"` and
`clear_cache=False`, validates readiness and a common epoch, transfers weights,
activates destination execution/graph state, commits, and resumes. A failed
preparation resumes the old mode. A destructive transfer failure leaves execution
stopped and requires engine restart. Control-client disconnection does not cancel
the transaction. PARAS uses a fixed one-step DP consensus cadence on all ranks,
reducing the upstream 32-step pause delay without changing attention topology.

## Storage and graphs

The shared MoE factory selects managed routed expert parameters; checkpoint
loading writes directly into EP views. Backend postprocessing must preserve
addresses. Each layer has separate EP/TP experts, maps, prepare/finalize state,
and kernels behind the same registered MoE wrapper and model router.

For EP2/TP2 the expert arena has 49 slabs of 576 MiB for 48 layers
(27.5625 GiB per rank). EP8/TP8 uses 49 slabs of 144 MiB (6.890625 GiB).
EP layer `i` occupies slab `i`; TP layer `i` occupies slab `i+1`. EP→TP transfers
run backward through layers; TP→EP runs forward. CUDA IPC uses the four weight
kernels adapted from SGLang PARAS `8177b7260952e3d19cfe360196c31ebc96b3733a`, an
explicit CUDA stream, and per-layer cross-rank completion fences. NCCL permutes
into one preallocated pair of staging views, adding one slab, then performs
all-to-all. IPC mappings retain allocation-base offsets and are closed at teardown.
Selected transports must initialize successfully; there is no silent fallback.

Attention and KV allocations are outside the arena. Unequal EP/TP sizes are
rejected: multiple TP replicas need a separately budgeted expert layout.
The transfer group and expert topology are represented independently of attention
TP. Runner hooks and model-layout metadata are isolated from the storage core.

Both modes are profiled/warmed before persistent graph capture. Each has a separate
pool and batch-descriptor graph dictionary. Switching selects existing dictionaries
and weight views; it does not reload, compile, recapture, or clear caches. The
existing graph-capture guard remains enabled after initialization.

## Verification and measurements

```bash
.venv/bin/python -m pytest --confcutdir=tests/model_executor/layers/fused_moe \
  tests/model_executor/layers/fused_moe -q
.venv/bin/python benchmarks/paras/check_live.py --world-size 2 \
  --output /data/shaoyuw/paras/vllm-milestone/runs/peer-new
.venv/bin/python benchmarks/paras/verify_replay.py --world-size 2 \
  /data/shaoyuw/paras/vllm-milestone/runs/peer-new
```

The focused CPU tests do not need the repository's broad root test fixtures.
They cover safe overwrite order, alignment, bounded allocation, typed aliases,
transaction serialization, cancellation, and pre/post-transfer failures.

The live suite covers more requests than the per-rank limit, unequal batches,
idle ranks, decode, long chunked prefill, cancellation, repeated round trips,
mode-specific graph replay, stable weight/KV addresses, unchanged capture and
compilation counters, and no continuing allocated/reserved memory growth. Torch
profiler traces independently verify CUDA runtime graph launches in both modes.

With the server stopped and GPUs idle, run full-shape transport checks/benchmarks:

```bash
PARAS_GPUS=6,7 benchmarks/paras/launch_transfer.sh peer_access \
  /data/shaoyuw/paras/vllm-milestone/transfers/peer-new.json
```

Repeat with `nccl` as the first argument. Synthetic patterns test exact BF16 storage bits,
including expert ownership and gate/up/intermediate permutations, through real
managed views. Timing reports logical per-rank weight bytes divided by transfer
time; this is not a measurement of physical NVLink traffic.

`capture_logits.py` uses the opt-in diagnostic worker extension to save complete
151936-vocabulary logits outside graphs. Compare static and PARAS modes with
matching token histories; `--switch` changes to TP during the same generation.
`check_static.py --skip-profile` provides the same steady-mode workload without
adding profiler traces to a live-suite run. These short workload measurements
are smoke benchmarks, not saturation or production capacity estimates.

Launchers bind localhost and enable development RPC/profiling only for these
experiments. Runner V2, automatic switching, quantization, other model layouts,
multiple API processes, multiple TP replicas, and multi-node execution remain outside this
milestone.

For a switched request, KV retains the arithmetic of earlier EP steps. Therefore
pure-TP logits need not match it bit for bit even with the same token history.
The numerical checker bounds this difference by the observed static EP/TP
arithmetic difference and also requires **exact** agreement with a separate
mixed-mode reference that never transfers weights during generation:

```bash
PARAS_GPUS=6,7 PARAS_WORKER_EXTENSION=disjoint_probe.DisjointProbe \
  benchmarks/paras/launch_paras.sh peer_access /data/shaoyuw/paras/vllm-milestone/runs/oracle-new
.venv/bin/python benchmarks/paras/capture_logits.py --switch \
  --output /data/shaoyuw/paras/vllm-milestone/numerics/oracle-new
```

The diagnostic extension reserves disjoint EP and TP weights
(54 GiB per rank at size 2; 13.5 GiB at size 8),
materializes TP once during initialization, and thereafter switches only the
execution state. It is benchmark-only. Compare captures with:

```bash
.venv/bin/python benchmarks/paras/compare_logits.py \
  --ep /data/shaoyuw/paras/vllm-milestone/numerics/static-ep \
  --tp /data/shaoyuw/paras/vllm-milestone/numerics/static-tp \
  --switch-reference /data/shaoyuw/paras/vllm-milestone/numerics/disjoint-switch \
  --output /data/shaoyuw/paras/vllm-milestone/numerics/comparison.json \
  /data/shaoyuw/paras/vllm-milestone/numerics/final-peer-ep \
  /data/shaoyuw/paras/vllm-milestone/numerics/final-peer-tp \
  /data/shaoyuw/paras/vllm-milestone/numerics/final-peer-switch
```

The checker requires matching token histories and, for the oracle comparison,
matching recorded expert-mode histories. Repeat for the NCCL captures. Reported
p95 values use the nearest-rank percentile. Pre-commit hook environments and uv
caches are configured beneath the conda prefix as well.

## Eight GPUs

Set `PARAS_GPUS=0,1,2,3,4,5,6,7` for EP8 ↔ TP8 with attention TP1/DP8.
There is no implicit GPU selection or PARAS parallel size. Set `PARAS_GPUS=6,7`
explicitly for two-rank reproductions. Launchers infer both
DP size and expert TP size from this list and use one API process for comparable
static and switching runs. Preflight checks require idle devices and successful
CUDA context creation, since an idle GPU can still need driver recovery.

Qwen3-30B-A3B's per-layer EP8 weights are `[16,1536,2048]` and
`[16,2048,768]`; TP8 weights are `[128,192,2048]` and `[128,2048,96]`.
The 96-channel Triton shape works on A100, including CUDA graph replay; a larger
model is not required for shape compatibility. Check it independently with:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/paras/check_tp_shape.py \
  --output /data/shaoyuw/paras/vllm-eight-gpu/tp8-shape.json
```

`check_peer_indexing.py` additionally exercises the production eight-way transfer
kernels against eight independent full-model arenas on one GPU (55.125 GiB).
It verifies exact indexing and overwrite order, **not** IPC or distributed fences:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST=8.0 \
  .venv/bin/python benchmarks/paras/check_peer_indexing.py \
  --output /data/shaoyuw/paras/vllm-eight-gpu/peer8-local-indexing.json
```

After stopping any server, run each transport sequentially:

```bash
PARAS_GPUS=0,1,2,3,4,5,6,7 benchmarks/paras/launch_transfer.sh peer_access \
  /data/shaoyuw/paras/vllm-eight-gpu/transfer8-peer.json
PARAS_GPUS=0,1,2,3,4,5,6,7 benchmarks/paras/launch_transfer.sh nccl \
  /data/shaoyuw/paras/vllm-eight-gpu/transfer8-nccl.json
```

Run the static EP8 and TP8 gates sequentially before switching acceptance:

```bash
PARAS_GPUS=0,1,2,3,4,5,6,7 benchmarks/paras/launch_static.sh ep \
  /data/shaoyuw/paras/vllm-eight-gpu/static8-ep
# In another shell; repeat with mode tp and a static8-tp directory.
.venv/bin/python benchmarks/paras/check_static.py --world-size 8 --mode ep \
  --output /data/shaoyuw/paras/vllm-eight-gpu/static8-ep
.venv/bin/python benchmarks/paras/verify_replay.py --world-size 8 \
  /data/shaoyuw/paras/vllm-eight-gpu/static8-ep
```

Then run both switching transports sequentially:

```bash
PARAS_GPUS=0,1,2,3,4,5,6,7 benchmarks/paras/launch_paras.sh peer_access \
  /data/shaoyuw/paras/vllm-eight-gpu/live8-peer
# In another shell; repeat with transport nccl and a live8-nccl directory.
.venv/bin/python benchmarks/paras/check_live.py --world-size 8 \
  --output /data/shaoyuw/paras/vllm-eight-gpu/live8-peer
.venv/bin/python benchmarks/paras/verify_replay.py --world-size 8 \
  /data/shaoyuw/paras/vllm-eight-gpu/live8-peer
```

When BF16 reductions produce different greedy continuations, use
`capture_logits.py --history /path/to/reference/generation.json` for each steady
mode. This generates one token for each explicitly forced prefix and records
input histories separately from sampled outputs. Use the same history and
protocol for both static and PARAS captures. The numerical diagnostic resets
prefix caching before measurement; the switch operation itself never does so.
Mixed-mode live captures still use `--switch` without `--history` and require a
disjoint oracle with matching recorded histories.

The suites check all DP ranks, including uneven loads and idle ranks, with eight
worker traces per captured mode. Logit capture and comparison commands above are
unchanged. Four-rank checks use four GPU indices and `--world-size 4`.

On September 8, 2026, full eight-GPU validation was blocked before model loading:
GPU 1 rejected CUDA context creation and reported recovery action `Reset`,
uncorrectable DRAM errors, and failed row remapping. A reset requires an
administrator password unavailable in the session. Eight-rank storage tests and
the standalone TP8 kernel and local eight-way transfer indexing checks pass.
On healthy GPUs 4–7, both transports also pass static/live four-rank serving,
graph replay, and matching-history steady-mode logits. This does **not** establish
eight-rank serving correctness. Health diagnostics and available-device test results are
under `/data/shaoyuw/paras/vllm-eight-gpu`.

## Explicit four-GPU run

No launcher or acceptance checker falls back to two ranks. `--paras-config`
requires an explicit `expert_tp_size`; the launchers derive that value from the
required `PARAS_GPUS` list. For the four-GPU run, use:

```bash
PARAS_GPUS=4,5,6,7 benchmarks/paras/launch_paras.sh peer_access \
  /data/shaoyuw/paras/vllm-four-gpu-explicit
# In another shell:
.venv/bin/python benchmarks/paras/check_live.py --world-size 4 \
  --output /data/shaoyuw/paras/vllm-four-gpu-explicit
.venv/bin/python benchmarks/paras/verify_replay.py --world-size 4 \
  /data/shaoyuw/paras/vllm-four-gpu-explicit
```

This explicitly selects attention TP1/DP4, EP4 at startup, and expert TP4 after
switching. It passes `--data-parallel-size 4` and
`--paras-config '{"expert_tp_size":4,"weight_transfer_method":"peer_access"}'`.
