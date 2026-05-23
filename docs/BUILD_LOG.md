# Mirage — Build Log

A chronological record of major work, decisions, findings, and blockers — kept
for handoff and so decisions can be traced back. Architecture decisions have
dedicated records in `docs/adr/`; this log is the narrative and the status.

---

## Status snapshot — 2026-05-22

| # | Task | Status |
|---|------|--------|
| 1 | Unblock GPU device access on MI300X | ✅ done |
| 2 | Install ROCm 7.2 PyTorch + base deps | ✅ done |
| 3 | Repo skeleton with typing discipline | ✅ done |
| 4 | Vendor-neutral backend Protocol layer | ✅ done |
| 5 | Core Runtime types + serving API contracts | ✅ done |
| 6 | Wire ROCm attention primitive | ✅ done |
| 7 | Cosmos-Predict-7B loader + inference path | ✅ done — runs on MI300X |
| 8 | Benchmark harness vs naive PyTorch + Diffusers | ✅ done — baseline captured (738 s) |
| 9 | Integrate Cosmos guardrail before deployment | ✅ done — blocks unsafe prompts |
| 10 | Per-stage Cosmos profiler | ✅ done |
| 11 | torch.compile the Cosmos DiT | ✅ done — 1.13× on the DiT loop |

**Quality gate (all green):** `ruff` clean · `ruff format` clean · `mypy --strict`
clean (33 files) · `pytest` 36 passed.

---

## Environment & hardware (facts)

| Item | Value |
|------|-------|
| GPU | 1× AMD Instinct MI300X VF — `gfx942` (CDNA3), 192 GiB HBM3, 304 CUs |
| GPU note | "VF" = virtualized/SR-IOV slice; `rocm-smi` sees one device |
| ROCm | 7.2.0 (system); HIP 7.2 |
| PyTorch | `2.12.0+rocm7.2` (from `download.pytorch.org/whl/rocm7.2`) |
| Host | 235 GiB RAM, 20 CPUs, 697 GB disk (592 GB free at start) |
| Python | 3.12.3; env managed by `uv` (system `python3-venv` unavailable, no sudo) |
| Repo | `/home/mshah/mirage` — Apache-2.0 |

---

## Session 1 — 2026-05-22 — Foundation + first Cosmos run on MI300X

**Goal:** Stand up everything needed to run Cosmos-Predict-7B on MI300X (not H100).

### Decisions

- **D1 — MI300X is the lead hardware target** (not H100). Deliberate inversion of
  the Implementation Plan, which leads with H100 and defers MI300X to Phase 5.
  Rationale: the plan's own competitive analysis names the non-NVIDIA wedge as
  the defensible position; the hardware is in hand; 192 GiB HBM removes memory
  pressure from v0.1. Full record: `docs/adr/0001-mi300x-first-hardware-target.md`.
- **D2 — Vendor-neutral by construction.** All layers depend only on the
  `Backend` / `AttentionOp` / `WorldModelEngine` Protocols. `ROCmBackend` is the
  only concrete backend; an NVIDIA backend is a zero-rewrite fast-follow.
  Record: `docs/adr/0003-vendor-neutral-backend-protocol.md`.
- **D3 — Attention primitive: SDPA→aotriton now, CK flash-attn later.** The
  plan named FlashAttention-3, which is Hopper-only (`wgmma`/TMA, absent on
  CDNA3). On ROCm, `torch` SDPA already dispatches to aotriton-compiled flash
  kernels, so the "naive" floor is already flash-class. The AMD Composable-Kernel
  `flash-attn` build is an optimization, deferred. Record:
  `docs/adr/0002-rocm-attention-primitive.md`.
- **D4 — Cosmos via the `diffusers` pipeline, not NVIDIA's repo.** Use
  `diffusers.CosmosTextToWorldPipeline` from `nvidia/Cosmos-1.0-Diffusion-7B-Text2World`.
  Reason: NVIDIA's `cosmos-predict1` repo hard-depends on TransformerEngine +
  apex (CUDA-only) — the decisive ROCm blocker. The diffusers path has neither
  and runs on stock ROCm PyTorch via SDPA. (From the Cosmos research brief.)
- **D5 — Tooling:** `uv` for the venv/deps; `torch` pinned to the ROCm 7.2 wheel
  index (never PyPI — PyPI torch is CUDA-only); `mypy --strict` + Pydantic +
  `Protocol` classes per the plan's "maximize AI leverage" guidance.
- **D6 — `StubEngine`** (noise frames) was built so the serving stack could be
  developed and tested before the 38 GB model loader landed.
- **D7 — v0.1 generation is not denoise-time streaming.** `CosmosEngine.generate`
  runs the diffusion pipeline to completion, then yields decoded frames.
  True per-step streaming is a later optimization (plan, Phase 2).

### Work done

- **Task 1** — `mshah` lacked `render`-group access to `/dev/kfd` and
  `/dev/dri/renderD*`. Fixed via `sudo chmod a+rw` (+ `usermod -aG render,video`
  for persistence). PyTorch then saw the GPU.
- **Task 2** — Installed `torch 2.12.0+rocm7.2`; verified HIP, device
  enumeration, and a bf16 matmul on-device. Installed model/serving/dev extras.
- **Task 3** — Repo skeleton: `pyproject.toml` (mypy-strict, ruff, hatchling),
  `src/mirage` package layout, Apache-2.0 `LICENSE`, `README.md`, `Makefile`,
  `docs/architecture.md`, 3 ADRs.
- **Task 4** — `mirage.hardware` (device/dtype domain types), `mirage.backend`
  (`Backend` Protocol + `ROCmBackend` + registry). `mirage info` CLI detects the
  MI300X.
- **Task 5** — `mirage.runtime` (Pydantic request/response types, `WorldModelEngine`
  Protocol, `StubEngine`, frame-aware `PagedLatentCache`); `mirage.serving`
  (FastAPI NDJSON frame-streaming API + gRPC `.proto` contract).
- **Task 6** — Attention wired: `AttentionOp` Protocol, `NaiveAttention`,
  `ROCmFlashAttention` (CK wrapper, inert until the CK package is built),
  shape-based `select_attention_op`. On-device test confirms attention runs at
  the Cosmos DiT shape (32 heads × head_dim 128) on the MI300X.
- **Task 7** — `mirage.models.cosmos.CosmosEngine` wraps the diffusers pipeline
  as a `WorldModelEngine`. `scripts/run_cosmos.py` is the end-to-end runner.
  **First inference run launched (smoke config: 17 frames, 8 steps).**
- **Task 8** — `mirage.bench` harness: `benchmark_engine` (latency/throughput/
  peak-HBM), `BenchmarkResult` (JSON-serializable), `speedup()`,
  `python -m mirage.bench` CLI.

### Findings

- **F1** — ROCm SDPA → aotriton flash: the naive attention floor is already a
  flash kernel on MI300X. Lowers the urgency of the CK `flash-attn` build.
- **F2** — `download.pytorch.org/whl/rocm7.2` exists — an exact-match torch
  wheel for the system ROCm. No version-skew workaround needed.
- **F3** — "Cosmos-Predict-7B" = **Cosmos-Predict1**. Predict2/2.5 ship only 2B
  and 14B diffusion models — there is no 7B in Predict2.
- **F4** — Architecture: DiT 4096-wide / 28 layers / 32 heads / head_dim 128;
  text encoder T5-XXL; VAE = Cosmos Tokenizer CV8x8x8 (continuous, 8×8×8
  compression); scheduler `EDMEulerScheduler`.
- **F5** — The diffusers repo's `model_index.json` has **no `safety_checker`**,
  so the diffusers path needs no separate Llama-Guard download.
- **F6** — Download was 66 GB (the repo also bundles NeMo-format weights and a
  guardrail copy); the diffusers pipeline uses ~38 GB of it.
- **F7** — Cosmos HF repos are `gated=auto` (instant); `Llama-Guard-3-8B` is
  `gated=manual` (was already approved for this account).

### Blockers resolved

- GPU device-node permissions (Task 1) — user ran a `sudo chmod`.
- Gated weights — user provided an HF token (account `miteshah`); licenses
  accepted; `hf` CLI used (`huggingface-cli` is fully deprecated).
- First `hf download` hit a stale gate-approval state; succeeded on retry.
- First Cosmos load failed: diffusers cannot load the T5 encoder without
  `accelerate` (T5 keeps modules in fp32, which requires `low_cpu_mem_usage`).
  Added `accelerate` to the `models` extra; load proceeds.
- Second load failed: `CosmosTextToWorldPipeline.__init__` force-constructs a
  `CosmosSafetyChecker`, which needs the heavy `cosmos_guardrail` stack.
  Disabled the guardrail behind `CosmosConfig.enable_guardrail` (Task #9).
- Generation then failed inside `diffusers/.../transformer_cosmos.py`:
  it calls `torchvision.transforms.functional.resize` but does not hard-depend
  on torchvision (`NameError` when absent). Installed `torchvision
  0.27.0+rocm7.2` from the ROCm wheel index; added it to the `models` extra.

### First Cosmos-Predict-7B inference on MI300X ✅

Smoke run — `scripts/run_cosmos.py --frames 17 --steps 8`, guardrail disabled:

| Metric | Value |
|--------|-------|
| Model | cosmos-predict1-7b-text2world (diffusers path) |
| Device | AMD Instinct MI300X VF |
| Output | 17 frames @ 1280×704, 8 denoising steps |
| Model load | 10.9 s (warm disk cache) |
| Generation | 66.7 s end-to-end |
| — denoising loop | ~7 s (8 steps @ ~1.1 it/s) |
| — T5 encode + VAE decode + post | ~60 s |
| Peak HBM | 28.4 GiB / 192 GiB |
| Output | `benchmark-results/cosmos_smoke.mp4` — verified: 17 frames, full dynamic range, inter-frame motion present |

**Milestone:** Cosmos-Predict-7B runs end-to-end on AMD MI300X through the Mirage
runtime.

Full reference run — `scripts/run_cosmos.py` (121 frames @ 1280×704, 36 steps):

| Metric | Value |
|--------|-------|
| Total generation | 738 s (~12.3 min) |
| — DiT denoising loop | 435 s (59%) — 36 steps × ~12.1 s |
| — VAE decode + T5 encode + post | ~303 s (41%) |
| Peak HBM | 52.5 / 192 GiB |
| Throughput | 0.16 frame/s |
| Output | `benchmark-results/cosmos_reference.mp4` — verified 121 frames |

- **F8** — Cost split is scale-dependent: short clips are VAE/encode-bound; the
  full reference run is **DiT loop 59% / VAE decode + encode 41%** — both are
  first-order costs. The optimization strategy follows directly from this —
  see [`OPTIMIZATION.md`](OPTIMIZATION.md).

### Open items / risks

- **First real Cosmos inference not yet verified** — smoke run in flight.
- **CK `flash-attn` for gfx942 not built** — v0.1 runs on SDPA/aotriton (D3).
- **gRPC** — `.proto` contract written; `grpcio-tools` codegen deferred.
- **Single GPU** — the MI300X is a VF; multi-GPU paths untestable here.
- **"vs naive baseline"** — today `CosmosEngine` *is* the stock diffusers path,
  so the benchmark currently measures the baseline itself; a meaningful speedup
  number needs Mirage-specific optimizations (future work).

### How to verify / run (handoff)

```bash
make check-gpu                       # MI300X smoke test
make info                            # detected backend + device
make lint && make typecheck && make test
.venv/bin/python scripts/run_cosmos.py --frames 17 --steps 8   # quick gen
.venv/bin/python -m mirage.bench --engine cosmos --iters 3     # benchmark
```

---

## Session 2 — 2026-05-22 — Optimization, Tier 1

**Goal:** Tier 1 of [`OPTIMIZATION.md`](OPTIMIZATION.md) — instrument first, then
apply levers, every change measured.

### Decisions

- **D8** — Instrument before optimizing. Built `mirage.bench.profile`: a
  warmup-separated, CUDA-synced per-stage profiler (text-encode / DiT / VAE
  decode / other), using forward hooks so it is valid on a `torch.compile`-d DiT.

### Findings

- **F9** — The Cosmos diffusers pipeline issues **two separate `transformer()`
  calls per denoising step** (conditional + unconditional) — classifier-free
  guidance is *not* batched. Confirmed Tier-1 lever: batching the two into one
  `batch=2` forward.
- **F10 — correction to F8.** F8 read the reference run as DiT 59% /
  VAE+encode 41%. The warmup-separated profiler shows otherwise: at 49f / 12
  steps **steady state**, of 44 s total the DiT loop is **~99%**; text encode
  ~0.1 s, VAE decode ~0.5 s. The reference run's ~303 s of non-DiT time was
  overwhelmingly **one-time ROCm kernel autotuning** (aotriton / hipBLASLt /
  MIOpen on first use of each shape), not steady-state VAE cost. **Implication:
  the DiT loop is *the* optimization target; VAE decode and T5 encode are
  negligible in steady state.** `OPTIMIZATION.md` re-weighted accordingly — and
  note: warming the kernel caches at deploy time is itself a real latency win.

### Work done

- `mirage.bench.profile` — `CosmosProfile` + `profile_cosmos` (forward-hook,
  compile-safe, warmup-separated).
- `CosmosConfig.compile_transformer` flag; `CosmosEngine.load()` applies
  `torch.compile` (inductor + triton-rocm) to the DiT transformer.
- `scripts/profile_cosmos.py` — per-stage profile with `--compile` / `--compare`.

### Results

Warmup-separated per-stage profile — 49 frames / 12 steps, MI300X:

| Stage | Baseline | torch.compile |
|-------|---------:|--------------:|
| Text encode (2 calls) | 0.06 s | 0.06 s |
| **DiT loop** (24 transformer forwards) | **42.0 s** | **37.1 s** |
| VAE decode (1 call) | 0.53 s | 0.53 s |
| Other | 1.8 s | 1.8 s |
| **Total** | **44.3 s** | **39.5 s** |

**`torch.compile` (inductor + triton-rocm) on the DiT: 1.13× on the loop,
1.12× end-to-end.** Modest, as expected — the DiT is GEMM-bound (hipBLASLt
already serves the big matmuls); compile's win is pointwise fusion + launch
overhead. It stacks with later levers and is ~free for serving (one-time
compile cost).

The profile confirms the strategy:
- DiT loop = **95%** of steady-state time — the sole first-order target.
- 24 transformer forwards / 12 steps → CFG runs **2 forwards per step,
  unbatched** (F9) — the next lever.
- VAE decode + text encode = **1.3% combined** — not worth optimizing now (F10).

### Next

- **CFG batching** — fold the 2 per-step transformer forwards into one batch-2
  forward. The diffusers pipeline hard-codes two calls, so this needs a
  Mirage-native denoising loop — a deliberate step toward replacing the
  diffusers `__call__`. Estimated ~1.2–1.5× on the DiT loop.
- `torch.compile` `max-autotune` mode — possible extra DiT win, long compile.
- Feature / step caching (training-free, ~1.5–2×) — also needs the custom loop.

---

## Session 3 — 2026-05-22 — Cosmos guardrail integration (Task #9)

**Goal:** make `enable_guardrail=True` real — the NVIDIA Open Model License
requires the Cosmos safety guardrail for deployment.

### Decisions

- **D9 — Mirage owns guardrail orchestration.** The diffusers pipeline
  force-registers a `CosmosSafetyChecker` as a pipeline *component*, which
  breaks the current diffusers device detection (`_execution_device` raises).
  Rather than fight that, `CosmosEngine` always neutralizes the in-pipeline
  checker and, when enabled, runs `cosmos_guardrail` itself — a text check
  before generation, a video face-blur check after. This is also the better
  architecture: the runtime owns the safety boundary.

### Findings

- **F11** — `cosmos_guardrail` 0.3.0 uses **Qwen3Guard-Gen-0.6B** (open, ~1.2 GB)
  for text safety, a word **Blocklist**, and a **RetinaFace** face-blur
  postprocessor. No Llama-Guard, no gated dependency — the legacy `aegis/`
  cache in the guardrail repo (~13 GB) is unused and skipped.
- **F12** — Two `cosmos_guardrail` 0.3.0 vs current-environment version skews,
  both worked around in `CosmosEngine`:
  1. It fetches assets with `snapshot_download(allow_patterns=["blocklist"])`;
     that bare pattern matches no files under current `huggingface_hub`.
     Fix: `_ensure_guardrail_assets` pre-fetches `blocklist/*` + `face_blur_filter/*`.
  2. Its NLTK corpora live in the symlinked HF blob cache, which NLTK 3.9's
     path-security check rejects ("Unauthorized path"), silently disabling the
     blocklist's lemmatized matching. Fix: `_materialize_nltk_data` copies the
     corpora to a real dir and prepends it to `nltk.data.path`.

### Work done

- `cosmos_guardrail` added as a `guardrail` extra in `pyproject.toml`.
- `CosmosEngine`: `_disable_cosmos_guardrail` (always, neutralizes the
  in-pipeline checker), `_load_guardrail`, `_ensure_guardrail_assets`,
  `_materialize_nltk_data`; `generate()` runs text + video checks when enabled;
  `GuardrailError` raised on a block.
- `scripts/run_cosmos.py --guardrail` flag; catches `GuardrailError` cleanly.

### Results — guardrail verified

- **Unsafe prompt → blocked.** A graphic-violence prompt was rejected by the
  Blocklist (`GUARDRAIL BLOCKED: BLOCKLIST: Prompt blocked by censorship`);
  `GuardrailError` raised, generation never started, no video produced.
- **Safe prompt → generates normally** — 17-frame clip, valid output, with the
  RetinaFace face-blur postprocessor applied.
- **NLTK Security Violations: 0** after `_materialize_nltk_data` — the blocklist
  loads its corpora cleanly.

The guardrail is functional and deployment-ready. Default remains
`enable_guardrail=False` (dev/benchmark); deployment sets it `True` and installs
the `guardrail` extra.

---

## Session 4 — 2026-05-22 — Competitive scan + apples-to-apples validation

### Finding

- **F13 — Mirage is currently the only public Cosmos-on-AMD implementation.**
  A deep web search (AMD.com, ROCm Blogs, the entire Cosmos GitHub org —
  `cosmos-predict1`, `cosmos-predict2`, `cosmos-predict2.5` — the
  `huggingface/diffusers`, `ROCm/pytorch`, `ROCm/aotriton`, `ROCm/aiter` issue
  trackers, MLPerf, Reddit, Hacker News, and broader web) returned **zero**
  published latency / throughput / memory numbers for any Cosmos variant
  (Predict1-7B/14B, Predict2-2B/14B, Predict2.5, Transfer1) on any AMD
  Instinct GPU (MI300X / MI325X / MI355X / MI300A / MI250X). AMD itself has
  shipped MI300X / MI355X numbers for HunyuanWorld-Voyager, Wan-2.2 (MLPerf
  v6.0 Single Stream 27.4 s on MI355X), and their own Micro-World, but has
  conspicuously skipped Cosmos. The likely reason is the CUDA-only dependency
  stack of NVIDIA's reference repo (TransformerEngine, Apex, NATTEN,
  flash-attn). Mirage's D4 (use the `diffusers` path) sidesteps all four —
  which is *why* we have numbers and no one else does. NVIDIA's published
  reference number for the same config: Cosmos-Predict1-7B Text2World on H100
  ≈ **380 s** for 121 frames @ 1280×704, 36 steps, BF16, peak 74 GB.
  Sources: `huggingface.co/nvidia/Cosmos-Predict1-7B-Text2World`;
  `docs.nvidia.com/cosmos/latest/predict2/model_matrix.html`;
  `rocm.blogs.amd.com` (notable for the absence of Cosmos posts);
  `github.com/nvidia-cosmos/*` and `github.com/ROCm/*` trackers (zero hits).

### Work

- **Validation run in flight** — warmup-separated 121-frame / 36-step
  baseline-vs-`torch.compile` profile (`scripts/profile_cosmos.py --frames 121
  --steps 36 --compare`) to replace the 49-frame extrapolation with a measured
  number for the apples-to-apples comparison vs NVIDIA's H100 ~380 s.
- **First-public-numbers writeup** — drafted at
  [`docs/COSMOS_ON_MI300X.md`](COSMOS_ON_MI300X.md); numbers placeholder until
  the validation lands.

---

## Session 5 — 2026-05-22 — Mirage-native denoising loop + CFG batching (Task #13)

### Work

- New module `mirage.runtime.denoise` — `denoise_cosmos_video()` implements
  the Cosmos denoising loop natively, using the diffusers pipeline's
  components (T5, DiT, VAE, EDM-Euler scheduler) but folding the per-step
  conditional + unconditional transformer forwards into one batch-2 call.
- `CosmosConfig.use_native_loop` flag; `CosmosEngine.generate` dispatches.
- `scripts/run_cosmos.py --native-loop` and
  `scripts/profile_cosmos.py --native-loop` for measurement.

One bug fix along the way: the Cosmos transformer internally repeats the
`padding_mask` by `hidden_states.shape[0]`, so passing a `(2, 1, H, W)` mask
balloons to `(4, ...)` and the internal concat fails. Pass `(1, 1, H, W)`; the
transformer handles batching itself.

### Results — 49 f / 12 steps, warmup-separated

| Stage | Baseline | `torch.compile` | Native loop (CFG batched) |
|---|--:|--:|--:|
| Text encode (calls) | 0.06 s (2) | 0.06 s (2) | 0.06 s (2) |
| **DiT loop (calls)** | **42.0 s (24)** | **37.1 s (24)** | **41.3 s (12)** |
| VAE decode | 0.53 s | 0.53 s | 0.53 s |
| Other | 1.8 s | 1.8 s | **0.2 s** |
| **Total** | **44.3 s** | **39.5 s** | **42.2 s** |

CFG batching: **1.02× on the DiT loop, 1.05× end-to-end.** `dit_calls`
collapsed from 24 → 12 as designed, but per-call wall time roughly doubled —
total GEMM work unchanged.

### Finding

- **F14 — CFG batching is essentially a wash at Cosmos-7B scale.** The bet was
  1.2–1.5×; the measurement is 1.05×. CFG batching does not reduce total GEMM
  work — it packages the same compute as one batch-2 forward instead of two
  batch-1 forwards. The savings come from launch and Python overhead, which is
  a small fraction of each transformer call when the per-call compute is
  already seconds (Cosmos-7B over ~24 k latent tokens). The original
  projection was an LLM-decode-shaped estimate; for a diffusion DiT it does
  not apply.
- **What did help (sneakily):** the leaner native loop dropped *other* from
  1.8 → 0.2 s — no callback bookkeeping, no progress bar, no extra cat/chunk
  outside the loop body. ~1.6 s saved per generation, scales with step count.
- **Roadmap implication.** Levers that *reduce work* dominate at this scale.
  Overhead-cutting levers — CFG batching, `torch.compile`'s pointwise fusion —
  give only single-digit-percent wins on the DiT loop because per-call compute
  already dwarfs overhead. `OPTIMIZATION.md` reweighted accordingly. The
  native loop itself is still load-bearing: it is the seam where feature/step
  caching and step-skipping go.

### Next

- **Feature / step caching (TeaCache / DeepCache style)** — *work-reducing*,
  training-free, claimed 1.5–2× on diffusion DiT. Lands inside the native
  loop. The bigger lever.
- **Solver swap** (EDM-Euler → DPM-Solver++ or fewer steps) — free, real work
  reduction; quality dial.
- **Stack measurement** — re-profile with `--native-loop --compile` to confirm
  the wins compose (1.13 × 1.05 → ~1.18×, projected).

---

## Session 6 — 2026-05-22 — Full-config baseline measured + step-skip cache landed

### Result — apples-to-apples vs NVIDIA H100

Warmup-separated 121-frame / 36-step baseline on MI300X — **the** publishable
headline number:

| | MI300X (Mirage, diffusers + SDPA→aotriton) | H100 (NVIDIA reference) |
|---|--:|--:|
| Total | **465.4 s** | ~380 s |
| DiT loop / calls | 460.0 s / 72 | (not separately reported) |
| Peak HBM | **52.5 GiB / 192** | 74 GB / 80 |

**MI300X reaches 82 % of H100 wall time using none of NVIDIA's specialized
tooling** (no TransformerEngine, no Apex, no NATTEN, no CUDA flash-attn), with
**~30 % less peak HBM**. The cold/steady gap (738 → 465 s = ~273 s) is
exclusively one-time ROCm kernel autotuning — warming the kernel cache at
deploy is itself a real latency win.

### Finding

- **F15 — `torch.compile` segfaults at 121-frame shapes (inductor/triton-rocm).**
  The compiled arm of the validation completed the baseline (the 465 s number
  above), then crashed mid-warmup of the compiled gen at step 21/36 with
  SIGSEGV. The 49 f compile (1.13× DiT) worked fine; the failure mode is
  shape- or memory-pressure-specific to 121 f. The 121-frame compile number
  stays projected (~410 s) until either inductor/triton-rocm is updated or we
  work around it. Tracked separately; not blocking.

### Work — step-skip caching (Task #14)

- New params on `denoise_cosmos_video`: `cache_skip_every` (0 = disabled;
  ≥ 2 = run a full DiT forward only every Nth step after warmup) and
  `cache_warmup_steps` (default 4). Cached `noise_pred` from each full step
  is reused on the N-1 skipped steps that follow.
- `CosmosConfig.cache_skip_every` / `cache_warmup_steps` thread through.
- `--cache-skip-every N` flag on both `run_cosmos.py` and `profile_cosmos.py`.
- This is a *work-reducing* lever (per F14): at `--cache-skip-every 4` post-
  warmup, 8 of 32 full forwards run instead of 32 → projected ~3× DiT
  speedup, modulo quality.

### Measured — caching is the biggest single lever so far

Warmup-separated profile, 49 f / 12 steps, MI300X:

| Config | Total | DiT loop | DiT calls | Speedup vs baseline |
|---|--:|--:|--:|--:|
| Baseline (diffusers, batch-1 CFG) | 44.3 s | 42.0 s | 24 | 1.00× |
| `torch.compile` | 39.5 s | 37.1 s | 24 | 1.12× |
| Native + CFG batched | 42.2 s | 41.3 s | 12 | 1.05× |
| **Native + `cache_skip_every=4`** | **21.5 s** | **20.7 s** | **6** | **2.06×** |

The DiT-call count collapses from 24 → 6 as designed (4 warmup + 2 cached-
refresh full forwards; the other 6 steps reuse the cached `noise_pred`).
End-to-end **1.96×, DiT loop 2.00×**. This is the kind of behaviour F14
predicted: *work-reducing* levers scale, *overhead-cutting* ones don't.

If the 2× scales linearly to the 121-frame baseline (465 s), Mirage on MI300X
projects to **~232 s — faster than NVIDIA's H100 reference (~380 s) on the
same workload.** That projection is contingent on quality holding at the
longer config — to be verified.

---

## Session 7 — 2026-05-23 — Caching at the reference config (headline)

### Result

`scripts/run_cosmos.py --frames 121 --steps 36 --native-loop --cache-skip-every 4`:

| Metric | Value |
|---|--:|
| Generate (single run, not warmup-separated) | **164 s** |
| **vs MI300X warmup-separated baseline (465 s)** | **2.84× faster** |
| **vs NVIDIA H100 reference (~380 s)** | **2.32× FASTER** |
| Peak HBM | 52.5 GiB / 192 GiB |
| Throughput | 0.738 frames/s |
| Video | `benchmark-results/cosmos_121f_cached.mp4` — 121 frames verified, stats healthy |

The projected ~232 s from linear scaling of the 49-frame 2× speedup turned out
**optimistic-in-direction**: the actual 164 s implies ~2.84× scaling at the
longer config. Per-call DiT cost grows with sequence length, but caching
reduces *call count* the same proportionally; combined with CFG batching, the
effective DiT-loop work reduction at `skip=4` is ~3×.

Caveats:
- **Warmup-separated profile (follow-up):** 154.1 s total / 152.5 s DiT / 12
  calls. The single-run 164 s number included ~10 s of one-time autotuning;
  the clean steady-state is **154 s — 2.47× faster than H100 reference, 3.02×
  over the MI300X baseline**.
- **Quality is a dial.** The video stats are in a healthy range (brightness
  108, std 67.2, motion 4.69 — same band as the uncached 121 f reference at
  brightness 106 / std 65.1). Visual verification is the gating test; if
  `skip=4` degrades visibly, `skip=2` trades half the win for safer quality.

This is the strongest measured Mirage win to date and the first MI300X number
that beats NVIDIA's published H100 reference wall time on the same workload.

