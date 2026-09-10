# PARAS EP ↔ expert TP on vLLM V1

This experimental implementation supports matching EP/TP sizes 2, 4, and 8
on one node. It keeps attention TP1/DP, requests,
prefix caching, and KV storage stationary while explicitly switching routed
experts. The original validated target is Qwen3-30B-A3B BF16 on two A100 GPUs, based on
vLLM `2cf0a6915ce544dc493a0990f2ea38d81601128a` (v0.28.0).

## Environment

### FP8 EP8 ↔ TP8: Qwen3.5-122B-A10B

PARAS supports serialized E4M3 FP8 experts with dynamic activation quantization
and 128×128 weight blocks, as used by
[Qwen3.5-122B-A10B-FP8](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8).
Activations outside the FP8 kernels remain BF16. By default, EP uses DeepEP low latency and
batched Triton; expert TP uses Triton with AllGather/ReduceScatter. Attention,
shared experts, and recurrent/KV state keep their existing placement.
The FP8 launcher uses `--moe-backend triton`; vLLM selects its batched variant
for DeepEP automatically.

The managed arena includes FP8 weights and their FP32 block scales. Both are
resharded together without requantization or a persistent BF16 expert copy.
At TP8, down-projection scale rows contain one FP32 value; the peer transfer
kernel supports these four-byte rows. NCCL transfers use byte views to preserve
the exact FP8 representations. Unsupported quantization formats, ignored routed
experts, and TP partitions that split a quantization block are rejected.

The local environment from the H800 setup below can run all 48 layers on GPUs
0–7. The expert arena uses approximately 13.785 GiB per GPU. Model configuration,
tokenizer, and processor files are under `.venv/models/Qwen3.5-122B-A10B-FP8`;
the launcher defaults to dummy weights and text-only inference:

```bash
benchmarks/paras/run_qwen122b_fp8.sh
```

Use `run_qwen122b_fp8.sh ep` or `run_qwen122b_fp8.sh tp` for static baselines.
Set one option to use DeepGEMM for both EP and expert TP:

```bash
PARAS_MOE_BACKEND=deep_gemm benchmarks/paras/run_qwen122b_fp8.sh
```

`PARAS_MOE_BACKEND=triton` selects Triton for both (the default).
`PARAS_EP_MOE_BACKEND` and `PARAS_TP_MOE_BACKEND` optionally override each mode.
DeepGEMM switching currently requires Hopper and FP32 block scales. The launcher
sets `VLLM_USE_DEEP_GEMM_E8M0=0` for DeepGEMM so initialization preserves the
checkpoint's weights and scales. PARAS's TP state selects DeepGEMM directly, including
narrow shards that vLLM's usual selector would route to Triton. The CLI equivalent
uses `--moe-backend deep_gemm` and
`--paras-config '{"expert_tp_size":8,"expert_tp_backend":"deep_gemm"}'`.
Static TP baselines retain vLLM's usual shape-based backend selection.

For real weights, supply a local copy of the checkpoint and set
`PARAS_LOAD_FORMAT=auto`. The switch API is unchanged:

```bash
curl -fsS http://127.0.0.1:8765/paras/status
curl -fsS http://127.0.0.1:8765/paras/switch \
  -H 'Content-Type: application/json' -d '{"target":"tp"}'
curl -fsS http://127.0.0.1:8765/paras/switch \
  -H 'Content-Type: application/json' -d '{"target":"ep"}'
```

Run the existing transport checker with `--model` to include the checkpoint's
scale tensors in the bit-exact round trip. Run the live checker with
`--world-size 8`. Dummy runs validate execution and transfer correctness;
evaluation with the trained checkpoint is still required to assess model quality.

Validated on eight H800s with all 48 layers and dummy weights:

- Triton/Triton and DeepGEMM/DeepGEMM each passed 95 completed requests, one
  cancellation, and 36 live mode changes, including partial prefill and queued
  decode. KV/recurrent state and weight/scale storage remained stable.
- Both configurations preserved all 32×248,320 captured logits bit for bit after
  an EP→TP→EP round trip with matching token histories.
- Profiler traces confirmed graph replay on every rank and actual DeepGEMM
  grouped expert kernels in both EP and TP.
- FP8 weight/scale and BF16 regression transfers passed bit-exact checks over
  both peer access and NCCL. The CPU regression suite passed 71 tests; Ruff,
  clang-format, and ShellCheck passed.

Results are summarized in `.venv/var/paras/fp8-validation.json`, with detailed
artifacts in `qwen122b-fp8-switch` and `qwen122b-fp8-deepgemm` under that directory.

### Local H800 setup under `/opt/tiger/vllm`

The local Python 3.12 environment is a uv venv at `.venv`. It uses the
upstream v0.28.0 native extensions, PyTorch CUDA 13.0 wheels, and the system
CUDA toolkit at `/usr/local/cuda`. Recreate dependencies with
`benchmarks/paras/setup_env.sh` (requires `uv` in `PATH` or `.tools/uv`).
Build DeepEP with:

```bash
benchmarks/paras/build_deepep_hopper.sh /opt/tiger/DeepEP-v1
```

This builds DeepEP v1.2.1 in an isolated copy at `.venv/src/DeepEP-v1-hopper`,
targeting SM90. It applies the existing local compatibility fixes to NVSHMEM
library naming, IBGDA configuration, and the RDMA queue assertion. The original
source is unchanged. The newer `/opt/tiger/DeepEP` checkout requires NCCL APIs
absent from PyTorch's pinned NCCL version.

Qwen3-30B-A3B configuration and tokenizer files are at
`.venv/models/Qwen3-30B-A3B`. The following command starts the full 48-layer
model with dummy weights on GPUs 0–3, initially in EP4 mode, with runtime
switching to expert TP4. Attention stays at TP1/DP4 in both modes.

```bash
benchmarks/paras/run_qwen30b.sh
```

From another shell:

```bash
curl -fsS http://127.0.0.1:8765/paras/status
curl -fsS http://127.0.0.1:8765/paras/switch \
  -H 'Content-Type: application/json' -d '{"target":"tp"}'
curl -fsS http://127.0.0.1:8765/paras/switch \
  -H 'Content-Type: application/json' -d '{"target":"ep"}'
```

Use `run_qwen30b.sh ep` or `run_qwen30b.sh tp` for static baselines. Override
`PARAS_GPUS` to select another four GPUs. The H800 environment allows up to
2048 MiB of idle device memory to accommodate the host's `hold.py` reservation,
while rejecting nonzero utilization. Outputs default to
`.venv/var/paras/qwen30b-MODE`.
Pause the existing `hold.py` GPU burn-in process before launching; it periodically
uses all GPUs. Resume it after stopping the server if it is still needed.
For real weights, provide a downloaded BF16 checkpoint:

```bash
PARAS_MODEL=/path/to/Qwen3-30B-A3B PARAS_LOAD_FORMAT=auto \
  benchmarks/paras/run_qwen30b.sh
```

Dummy output is only useful for execution and switching checks. To run the
existing four-rank DeepEP and live acceptance checks:

```bash
source benchmarks/paras/qwen30b_env.sh
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/paras/check_deepep.py
# With the switching server running:
.venv/bin/python benchmarks/paras/check_live.py --world-size 4 \
  --output .venv/var/paras/qwen30b-switch
.venv/bin/python benchmarks/paras/verify_replay.py --world-size 4 \
  .venv/var/paras/qwen30b-switch
```

Validated on 2026-09-11 with Python 3.12.14, PyTorch 2.13.0+cu130,
DeepEP 1.2.1, NCCL 2.29.7, and NVSHMEM 3.4.5. The four-rank DeepEP round trip
and five graph replays passed. Live acceptance passed with async scheduling:
83 completed requests, one cancellation, and 36 mode changes, including
switches during queued decode and partial prefill. KV and weight addresses
remained stable, and profiler traces confirmed graph replay in both modes on
all four ranks. These checks used dummy weights and do not assess model quality.

Results and logs are under `.venv/var/paras/qwen30b-switch`; the dependency
snapshot is `.venv/requirements-paras.txt`. Ruff and ShellCheck passed for the
modified Python and shell scripts.

### Original A100 setup

```bash
benchmarks/paras/setup_env.sh
benchmarks/paras/build_deepep.sh /data/shaoyuw/paras/vllm-milestone/DeepEP-sm80
source .venv/bin/activate
```

`setup_env.sh` creates a uv venv at `.venv`, with Python 3.12,
installs editable vLLM and native extensions for the exact upstream commit, and
installs pre-commit hooks. Python dependencies, NCCL, NVSHMEM, and the PARAS
extension cache are inside this prefix. The system CUDA 13.0 toolkit supplies
`nvcc`. Launchers replace inherited library paths with the venv paths.

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
full-vocabulary logits outside graphs. Compare static and PARAS modes with
matching token histories; `--switch` changes to TP during the same generation.
`check_static.py --skip-profile` provides the same steady-mode workload without
adding profiler traces to a live-suite run. These short workload measurements
are smoke benchmarks, not saturation or production capacity estimates.

Launchers bind localhost and enable development RPC/profiling only for these
experiments. Runner V2, automatic switching, other quantization formats,
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
mode. The worker records unmodified logits, then forces the sampled token to
keep one live request on the reference history. Every capture uses one prefill
followed by 31 decode steps, preserving the same attention path and KV evolution.
Use the same history and protocol for both static and PARAS captures. The
numerical diagnostic resets prefix caching before measurement; the switch
operation itself never does so.
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

## Dense prefixes, shared experts, and hybrid attention

PARAS now reserves slabs for routed MoE layers only. `ExpertLayout.layer_indices`
records the original transformer layer IDs, omitting dense prefixes and gaps
between MoE layers. Runtime states, reservations (for example, `ep.3.w13`), and
transfer order use those IDs. Only the allocator uses compact slab positions. The shared factory retains each model's routing and
separate shared-expert module. Attention TP remains 1, so dense MLPs and shared
experts stay replicated and outside the arena. Fused shared-expert slots are
rejected; context/pipeline parallelism remains unsupported. Quantized expert
switching is limited to the block FP8 layout described above.

Layout metadata covers Qwen2/3 MoE, Qwen3-Next, Qwen3.5/3.6 MoE text configurations,
GLM4 MoE/Lite, and DeepSeek V2/V3. A layout entry establishes weight geometry,
not GPU acceptance for every checkpoint. Qwen3.6-35B-A3B uses the existing
`Qwen3_5MoeForConditionalGeneration` implementation and its nested text config.

`StationaryState` records non-routed parameter/buffer bindings after both graph
sets are captured. It also records attention cache attributes, including tuples
of convolution and recurrent state used by linear attention. Each switch checks
storage addresses, shapes, strides, and dtypes. Values are allowed to evolve
while generating; live model comparisons separately validate correctness.

The launcher accepts `PARAS_MODEL`, `PARAS_ATTENTION_CONFIG`, `PARAS_PORT`, and
`PARAS_LANGUAGE_MODEL_ONLY`. Attention defaults to automatic backend selection
with FlashAttention version 2 for applicable backends. On A100, MLA can select
Triton MLA instead of forcing the ordinary FlashAttention backend. Hybrid models
use `mamba_cache_mode=align` with prefix caching. Model-specific kernels must still
support full single-token decode graphs. DSA on A100 is not supported by the
available sparse attention backends and is not enabled by these layout changes.

After downloading the models and completing CPU checks, select four idle GPUs
(excluding broken GPU 1 on the development host) and run:

```bash
.venv/bin/python benchmarks/paras/run_model_matrix.py \
  --gpus 0,2,3,4 --world-size 4 \
  --output /data/shaoyuw/paras/vllm-model-extension/runs
```

The GPU list is an example, not an availability claim. Launchers check immediately
before use and fail if a GPU is occupied. The matrix tests GLM-4.7-Flash,
GLM-4.5-Air, and Qwen3.6-35B-A3B sequentially. Use `--models MODEL` to run one.
It covers full-layout exact synthetic transfers with both transports, static
EP/TP serving, live queued/decode/prefill/cancellation switching, graph replay,
fixed state bindings, and steady-mode full-vocabulary logits under matched token
histories. Qwen runs with `--language-model-only`; image/video serving is outside
this attention/MoE acceptance matrix. Results are written per model and stage.

The fixed compilation counters track vLLM compilation and CUDA graph capture.
First-use Triton JIT events on non-full-graph paths are reported separately in
the server logs, including for static baselines. Graph preservation is verified
for the configured full-decode capture sizes.

```bash
.venv/bin/python -m pytest --confcutdir=tests/model_executor/layers/fused_moe \
  tests/model_executor/layers/fused_moe/test_paras_storage.py \
  tests/model_executor/layers/fused_moe/test_paras_state.py \
  tests/model_executor/layers/fused_moe/test_paras_transaction.py -q
```

The matrix reserves 95% of GPU memory for GLM-4.5-Air and 85% for the smaller
models; `PARAS_MEMORY_UTILIZATION` overrides this choice. Model weights remain
BF16, including the dense prefix and separate shared experts.

For an isolated numerical comparison, `--logits-only` runs the four server
configurations with matching histories, profiles real decode graph replay, and
checks that capture counters and weight/cache bindings stay fixed. It omits the
live-serving and synthetic-transfer suites and labels its completion record
accordingly. `VLLM_BATCH_INVARIANT=1` is incompatible with the required
batched Triton EP backend in this checkout and fails during initialization.
The explicit `paras_tensor_digests` diagnostic RPC hashes all model parameters
on every rank for untimed corruption checks; its CPU copies are not part of the
switch implementation or timing measurements.

To exercise model architectures with smaller memory and compile costs, create
canonical synthetic checkpoints from their original tensor shapes:

```bash
.venv/bin/python benchmarks/paras/make_dummy_models.py \
  --output /data/shaoyuw/paras/vllm-model-extension/dummy-models
.venv/bin/python benchmarks/paras/run_model_matrix.py \
  --gpus 4,5,6,7 --world-size 4 \
  --model-root /data/shaoyuw/paras/vllm-model-extension/dummy-models \
  --output /data/shaoyuw/paras/vllm-model-extension/runs-dummy
```

Defaults are five GLM layers (one dense plus four routed/shared MoE layers),
and eight Qwen layers (six linear-attention plus two full-attention layers).
Hidden dimensions, expert counts and intermediate sizes remain unchanged.
Each checkpoint tensor is seeded by its logical name, and all EP/TP runs load
that same checkpoint through the ordinary loader. Vision and prediction-head
weights are omitted. Norms have unit effective scale: Qwen offset norms use
zero weights, while its gated linear-attention norms use ones.
These runs test architecture and switching correctness,
not pretrained model quality or full-depth serving capacity.

Synthetic runs use 85% memory reservation and eight distinct long prompts per
rank so that switching overlaps chunked prefill even with fewer layers. The
probe confirms partially computed prompts in the scheduler pause snapshots.
`--resume` can reuse passed serving stages with matching model/GPU metadata;
use it only when the changes being tested do not affect those prior results.

For numerical comparisons of autotuned attention, reuse the same reference
kernel choices across servers. Independently tuned linear-attention prefill
kernels can produce different recurrent states even with matching weights and
token histories. This is separate from preserving cache contents during a
switch. Export the rank-0 tuning results from a completed static EP baseline:

```bash
.venv/bin/python benchmarks/paras/paras_tuning.py \
  --source /path/to/static-ep/cache --output /path/to/reference-tuning
TRITON_CACHE_MANAGER=paras_tuning:FrozenAutotuneCacheManager \
PARAS_FROZEN_AUTOTUNE=/path/to/reference-tuning \
.venv/bin/python benchmarks/paras/run_model_matrix.py \
  --gpus 4,5,6,7 --world-size 4 --models Qwen3.6-35B-A3B \
  --model-root /path/to/dummy-models --output /path/to/comparison-runs
```

The cache manager reuses only matching autotuning signatures, leaves compiled
binary caching intact, and fails if a required signature is missing. Snapshots
record the signatures used on every rank. Launch/completion metadata records
the cache and numerical environment settings. This does not enable batch
invariance or alter the comparison tolerances.

`PARAS_REFERENCE_NUMERICS=1` with `CUBLAS_WORKSPACE_CONFIG=:4096:8` adds
explicit BF16 numerical-reference settings to both launchers: cuBLASLt with
full-precision accumulation and split-K disabled, deterministic Inductor
selection, no benchmark-driven fusion/combo kernels, and preserved intermediate
precision casts. Use the same settings for static and switching servers.
Global PyTorch deterministic algorithms are not enabled: on this version they
make indexed writes into Qwen's strided recurrent cache allocate a copy of the
entire state view. Full decode graphs remain enabled with the reference settings.

## Async scheduling and switch synchronization

Launchers enable async scheduling by default. Set `PARAS_ASYNC_SCHEDULING=0`
for a synchronous baseline. DBO remains disabled. V1 async scheduling permits
two concurrent batches, overlapping CPU scheduling with model execution.

Switching pauses new scheduling with `mode="keep", clear_cache=False`.
Already-dispatched batches retire their outputs; running and waiting requests
stay in the scheduler. DP pause consensus requires an empty async batch queue
on every rank. Ranks continue participating in expert collectives until that
common boundary, including dummy execution on idle ranks. Worker readiness,
per-layer transfer fences, and the commit/resume barriers then serialize the
weight transition before scheduling restarts. No request-completion drain is
required.

Set `PARAS_SCHEDULER_TRACE=1` when running the model matrix to record optional
scheduler diagnostics. Live checks verify scheduling with prior outputs still
pending, no new scheduled tokens between pause and resume, settled output
placeholders at resume, and retained running requests. These traces measure
pending-output overlap; use a profiler for CPU/GPU timing or speedup claims.
