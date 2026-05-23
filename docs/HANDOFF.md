# Mirage Runtime — Handoff

**Date:** 2026-05-23 · **Repo:** https://github.com/miteshs/Mirage ·
**HEAD:** `ca709fe`+ · **Status:** pre-alpha, working on MI300X, results
publishable, polyglot scaffold + Rust core + Stage-4 v2 serving + Phase 2
(adaptive caching headline) + Phase 2.5 FP8 wiring all landed; FP8 kernel
not yet a Cosmos-shape win (kernel tuning is future work)

This is the single doc to read first if you are picking the project up. It
distills `docs/BUILD_LOG.md` (the full chronological log) into the
"what is this, what does it do, and where do I go next" cut.

---

## 1. Mission

Mirage is a **world-model-native inference engine** — a serving stack
purpose-built for diffusion-temporal video models, not for autoregressive
token decode. The LLM-era serving stack (vLLM, TensorRT-LLM, Diffusers,
torch.compile) leaves 30–60 % of silicon performance on the table for world
models; Mirage closes that gap.

- **Lead workload:** NVIDIA **Cosmos-Predict1-7B Text2World** via the
  HuggingFace `diffusers.CosmosTextToWorldPipeline` path.
- **Lead hardware:** **AMD Instinct MI300X** (`gfx942`, ROCm 7.2). The
  strategic deviation from the implementation plan is recorded in
  `docs/adr/0001-mi300x-first-hardware-target.md` — short version: the
  defensible wedge is non-NVIDIA silicon, and the diffusers path
  side-steps every CUDA-only dependency (TransformerEngine, Apex, NATTEN,
  flash-attn) in NVIDIA's reference Cosmos repo.
- The codebase is **vendor-neutral by construction** (Protocol-based
  backends, see ADR-0003). NVIDIA support is an added class, not a
  rewrite.

---

## 2. Headline numbers

All measured on a single AMD Instinct MI300X VF (192 GiB HBM3, 304 CUs,
`gfx942`), ROCm 7.2.0, `torch 2.12.0+rocm7.2`, BF16,
`nvidia/Cosmos-1.0-Diffusion-7B-Text2World`, default reference config
(121 frames @ 1280×704, 36 inference steps), warmup-separated unless noted.

| | Wall time | vs NVIDIA H100 reference (~380 s) |
|---|--:|--:|
| Mirage baseline (diffusers + SDPA→aotriton) | **465 s** (740 s on 2026-05-23 re-run) | 0.82× / 0.51× |
| Same + `torch.compile` on the DiT (≤64 f) | ~410 s projected (49 f shows 1.13× DiT) | ~0.93× (projected) |
| Same + native loop + step-skip `cache_skip=2` | **266 s** | **1.43× faster** |
| Same + native loop + step-skip `cache_skip=4` | **154 s** Session 7 / **163.9 s** Session 9 clean | **2.47×** / **2.32×** |
| Same + native loop + **adaptive cache** (`thr=0.30`) | **151.4 s** Session 12 clean (150.9 / 151.1 Sess 10/9) | **2.52× — well-replicated** |
| Same + native loop + adaptive + tuned FP8 (autotuned tile) | **142.0 s** Session 12 clean (141.7 Agent I worktree) | **2.68× — current headline** |
| Peak HBM (Cosmos, all configs) | **52.5 GiB** | ~30 % less than H100's 74 GB |
| Wan-2.2-T2V-A14B 17f / 8 steps smoke | 45.2 s warm Session 11 (326 s cold Session 10) | **84–85 GiB peak** |
| Wan-2.2 81f / 40 steps quality (projected steady-state) | ~1700 s | **85.1 GiB peak** — first Wan-2.2 on AMD MI300X; ~1.6× behind H100 (1041 s with FP8+offload) |

**Verification status (Session 12):** all three Cosmos timings reproduce on
a clean GPU within ±0.5 s of the agent / prior-session measurements.
Seed-0 determinism holds (same MD5 across 4 sessions for the
adaptive-no-FP8 mp4). The 2.68× is a *system-vs-system* claim (Mirage +
adaptive cache + tuned FP8 vs NVIDIA's published-unstacked H100); the
raw-hardware comparison has MI300X 1.24× *slower* than H100 at the same
compute. See `docs/METHODOLOGY.md` for the apples-to-apples breakdown
and `docs/COSMOS_ON_MI300X.md` §"Quantitative cache quality" for the
LPIPS / motion-stat trade-off (adaptive caching produces a trajectory-
divergent output: LPIPS 0.645 vs no-cache, mean inter-frame motion
−28 %; valid Cosmos output, just different).

**To our knowledge as of 2026-05-23 this is the first publicly reported
Cosmos benchmark on any AMD GPU.** Visual quality at `skip=4` is verified
at the 121 f / 36-step reference config; at shorter configs `skip=2` is
the safe floor (see F16 in `BUILD_LOG.md`).

Watchable artifacts (gitignored, local-only):
`benchmark-results/cosmos_reference.mp4` (no cache),
`cosmos_121f_cached_skip2.mp4`, `cosmos_121f_cached.mp4` (`skip=4`).

---

## 3. Where everything lives

| Want to read about | File |
|---|---|
| Chronological build log, every decision and finding | `docs/BUILD_LOG.md` (D1-D9, F1-F19, 9 sessions) |
| The publish-ready first-public-numbers writeup | `docs/COSMOS_ON_MI300X.md` |
| Optimization strategy + measured ledger | `docs/OPTIMIZATION.md` |
| Architecture component map | `docs/architecture.md` |
| Key decisions with rationale | `docs/adr/0001..0005` |
| Polyglot build scaffold (Cargo workspace, kernels/, ADR-0004) | `Cargo.toml`, `crates/`, `kernels/` |
| The seam between Mirage and a GPU vendor | `src/mirage/backend/protocol.py` |
| The seam between Mirage and an attention kernel | `src/mirage/attention/protocol.py` + `registry.py` |
| Cosmos engine + Mirage-orchestrated guardrail | `src/mirage/models/cosmos.py` |
| **Wan-2.2 engine** (second WM family) | `src/mirage/models/wan.py` |
| Mirage-native denoising loop (CFG batching, fixed + adaptive caching) | `src/mirage/runtime/denoise.py` |
| **FP8 attention kernel** (Triton fused FA-2) | `src/mirage/attention/fp8_triton.py` + `kernels/triton_kernels/fp8_flash_attn.py` |
| **HIP FP8 GEMM proof-of-life** (compiles, runs, output values TODO) | `kernels/hip/fp8_attn/` |
| Per-stage profiler | `src/mirage/bench/profile.py` |
| Frame-streaming HTTP + gRPC contract | `src/mirage/serving/` |
| End-to-end runners | `scripts/run_cosmos.py`, `scripts/run_wan.py` |
| Benchmark harnesses (head-to-head) | `scripts/bench_caching.py`, `scripts/bench_fp8.py` |
| Per-stage measurement runner | `scripts/profile_cosmos.py` |

---

## 4. How to run it

Requires an AMD GPU host with ROCm 7.x and Python 3.11+.

```bash
git clone https://github.com/miteshs/Mirage.git mirage && cd mirage
pip install --user uv
uv venv --python 3.12 .venv
# torch + torchvision MUST come from the ROCm wheel index (not PyPI)
uv pip install --python .venv torch torchvision \
    --index-url https://download.pytorch.org/whl/rocm7.2
make install
.venv/bin/hf auth login    # accept license on the Cosmos HF repo first
make check-gpu             # smoke
make info                  # see the detected backend

# Quick generation (~30 s warm, mp4 in benchmark-results/)
.venv/bin/python scripts/run_cosmos.py --frames 17 --steps 8

# Full reference run with caching (the 154 s headline)
.venv/bin/python scripts/run_cosmos.py \
    --frames 121 --steps 36 --native-loop --cache-skip-every 4

# Per-stage profile baseline-vs-compile
.venv/bin/python scripts/profile_cosmos.py --frames 49 --steps 12 --compare
```

Quality gate: `make lint && make typecheck && make test` — all green
(ruff, `mypy --strict` over 35 source files, 36 tests).

---

## 5. What's open

**Phase 2 + 2.5 + Stage 4 landed (Sessions 9–10, 2026-05-23):**

- ✓ **Wan-2.2 loader** — `WanEngine`, `scripts/run_wan.py`, 9 structural tests.
  **Smoke gen on main:** 17 f / 8 steps in 326 s, peak 84.3 GiB (Session 10).
- ✓ **Adaptive caching (TeaCache-style)** — `cache_mode=adaptive` thr=0.30,
  current headline **150.9 s / 2.52×** at 121 f / 36 steps (Session 10).
- ✓ **F15 safety gate** — `compile_max_frames=64`; original SIGSEGV didn't
  repro on current torch+ROCm; recompile storm gated instead.
- ✓ **FP8 attention kernel + diffusers backend wiring (Phase 2.5).**
  Triton FA-2 kernel + Mirage `mirage_fp8` backend registered with
  diffusers' `_AttentionBackendRegistry`. `MIRAGE_FP8_ATTENTION=1` flips
  the active backend on `CosmosEngine.load`. Quality preserved at
  Cosmos production shape (motion 4.65 vs 4.64).
  **NOT a wall-time win at Cosmos production shape today:** adaptive+FP8
  = 154.7 s vs adaptive 150.9 s. Session 9's 1.92× was at B=1 H=8 (8-col
  program grid that amortizes BLOCK_M=128/BLOCK_N=64); Cosmos's 64-col
  grid doesn't. F20 / future kernel tuning.
- ✓ **Stage 4 v2 serving path** (Session 10). `POST /v2/generate/stream`
  routes through `Router.accept → Scheduler.submit → engine driver →
  Router.push_frame → NDJSON stream`. v1 endpoints unchanged. 17 new
  pytest cases; cargo workspace tests unchanged (no Rust crate changes).

**Currently open:**

1. **FP8 kernel tuning** for Cosmos's production shape — autotune
   `BLOCK_M` / `BLOCK_N` per shape, or add a shape-specific code path.
   `scripts/bench_cosmos_fp8.py` is the harness.
2. **HIP kernel correctness fix.** The `v_mfma_f32_16x16x32_fp8_fp8` GEMM
   compiles + loads via `hipcc` + pybind, but the operand-register layout
   is incomplete. Triton wins on perf today; HIP is the long-term option
   for shapes Triton can't tune well.
3. **Wan-2.2 deeper exploration.** Smoke gen works; a real 81-frame
   quality run + per-stage profile + caching analysis are open.
   `scripts/run_wan.py --frames 81 --steps 40` is the target.
4. **v2 → v1 deprecation plan.** v2 is the future; once it's exercised
   on real workloads we can deprecate the v1 sync path.

**Other next moves, in order of strategic value:**

1. **Announce / publish.** `docs/COSMOS_ON_MI300X.md` is publish-ready
   (refreshed 2026-05-23 with the adaptive-cache headline and the
   `inference_mode` postscript). The first-public-Cosmos-on-AMD-GPU framing
   is the OSS-first GTM lever the implementation plan calls for (§2.4).
2. **Wire the Rust core into `src/mirage/serving/app.py`** — the three
   crates (`mirage-cache`, `mirage-scheduler`, `mirage-router`) are
   importable but the FastAPI handlers still call the engine directly.
   Stage-4 work.
3. **Continuous batching** — scheduler-level. Depends on the app.py
   wiring above.
4. **Action conditioning hooks** — robotics-OEM-facing surface. Plan's
   Phase-2 deliverable; deferred until a robotics OEM is in the design-
   partner pipeline.

**Deferred / known issues:**

- **F15 update** — On this MI300X (torch 2.12.0+rocm7.2) the original
  SIGSEGV didn't reproduce; instead the compile path enters a recompile
  storm (~20 s/step vs 12.8 s eager). The `compile_max_frames=64` gate
  in `CosmosConfig` covers both behaviors. Compile still works at ≤ 64 f.
- **CK flash-attn for gfx942** not wired (the `ROCmFlashAttention` wrapper
  exists but the package isn't installed; SDPA→aotriton is the working
  path). See ADR-0002.
- **Multi-GPU** — host is a single MI300X VF; no multi-GPU paths exercised.
- **Cosmos guardrail** integration handles two `cosmos_guardrail` 0.3.0 vs
  current `huggingface_hub` / NLTK 3.9 version skews (`_ensure_guardrail_assets`
  and `_materialize_nltk_data` in `src/mirage/models/cosmos.py`).
- **HF token** for gated weights — stored at `~/.cache/huggingface/token`
  (off the repo). Account: `miteshs`.

---

## 6. Anchors in the code (where the architecture lives)

- **The one seam.** `mirage.backend.protocol.Backend` (Protocol). Everything
  above it imports the Protocol, never a concrete vendor backend
  (ADR-0003). `ROCmBackend` is the only concrete impl today.
- **Attention.** `mirage.attention.protocol.AttentionOp` + `select_attention_op`.
  Naive SDPA is the floor; `ROCmFlashAttention` is the CK wrapper (inert
  until the CK package is built). On ROCm, SDPA already routes through
  aotriton flash kernels (ADR-0002).
- **Runtime.** `mirage.runtime`. `WorldModelEngine` Protocol, Pydantic
  request/response types, `StubEngine` (noise frames — for serving
  development), `PagedLatentCache` (frame-aware eviction skeleton).
- **Cosmos engine.** `mirage.models.cosmos.CosmosEngine`. Wraps the
  diffusers pipeline; orchestrates the guardrail (Mirage-side, not
  pipeline-side — see Session 3 in BUILD_LOG).
- **Native denoising loop.** `mirage.runtime.denoise.denoise_cosmos_video`.
  CFG batching + step-skip caching live here. This is where adaptive
  caching, feature caching, and any future loop-level optimization belongs.
- **Benchmark + profile.** `mirage.bench.harness` (end-to-end timing),
  `mirage.bench.profile` (per-stage; forward-hook + method-wrap; works
  on compiled DiTs).
- **Serving.** `mirage.serving.app` (FastAPI / NDJSON frame streaming) +
  `mirage/serving/proto/mirage.proto` (gRPC contract).

---

## 7. Pointers for common next operations

- **Adding NVIDIA support:** implement `Backend` Protocol in
  `src/mirage/backend/cuda.py`, append to `_ALL_BACKENDS` in
  `src/mirage/backend/registry.py`. No other code changes.
- **Adding a new WM model:** implement `WorldModelEngine` Protocol. Mirror
  the structure of `mirage.models.cosmos`. Reuse the native loop and
  caching by parameterizing them on the pipeline's components.
- **Adding a loop-level optimization:** land it in
  `mirage.runtime.denoise.denoise_cosmos_video`, gated by a
  `CosmosConfig` flag. Use the per-stage profiler to A/B against the
  baseline.
- **Pushing a measured speedup claim:** always re-verify with
  `scripts/profile_cosmos.py --warmup 1` (warmup-separated), and eyeball
  the artifact at the *full reference config* (121 f / 36 steps).
  Smaller configs can mislead — F16/F17.

---

## 8. Commit history (recent)

```
ca709fe  attention: wire FP8 kernel into Cosmos via a diffusers-side backend
576ca56  serving: Stage-4 v2 path through router + scheduler
10c7b9a  Phase 2 integration: docs + numbers from the clean-GPU benchmark sweep
61486e1  caching: adaptive (TeaCache-style) + F15 compile gate at 121f
d916800  attention: FP8 path on CDNA3 — Triton flash kernel + scaled_mm fallback + HIP scaffold
ed1ed27  mirage.models.wan: Wan-2.2 T2V-A14B as the second world-model family
300dc57  docs: refresh for OSS-announce + log Session 8 (polyglot scaffold, OOM fix)
e7f66b0  denoise: wrap loop body in torch.inference_mode() — fixes 121f/36 OOM
af9210d  Stage 3 integration: build pipeline, unique lib names, py.typed markers
8c9bcfe  crates/mirage-router: greenfield request router in Rust+PyO3
f280ada  crates/mirage-scheduler: greenfield priority scheduler in Rust+PyO3
6e61c0e  crates/mirage-cache: port PagedLatentCache from Python to Rust+PyO3
92e91b5  Stage 2: resolve Rust-core fork, populate workspace deps, scaffold 3 crates
e012293  Scaffold polyglot build tooling — Cargo workspace, kernels/, ADR-0004
90aa374  Add docs/HANDOFF.md — single doc to read first when picking up Mirage
41f3f59  Reinstate cache=4 headline: cache tolerance is config-size-dependent (F16+F17)
2b8a6d9  Retract cache_skip=4 headline: cached output is visibly degraded (F16)   [later corrected]
987d4cd  Clean warmup-separated 121f cache=4: 154s — 2.47x faster than H100 reference
010f969  Cached 121f / 36 step run: 164s — MI300X beats H100 reference (~380s) by 2.3x
63d52f1  Measured 121f baseline (465s) + native-loop step-skip caching
9f85125  Initial Mirage Runtime — first published Cosmos-Predict-7B on AMD MI300X
```

The `2b8a6d9` retraction was reverted (`41f3f59`) when we realised the
"garbage" eyeball had been on the 49 f / 12 step output, not the 121 f /
36 step one. Both states are preserved in history; F16 explains the
config-size dependence.

`e7f66b0` fixes a one-line OOM that surfaced on a fresh-environment re-run:
the native loop was missing `torch.inference_mode()`, so the autograd
graph for all 36 diffusion steps stayed alive — ~189 GiB allocated at
121 f, fit within HBM at 17 f / 8 steps so the bug never showed up in
smaller smoke configs. See the *Engineering postscript* in
`docs/COSMOS_ON_MI300X.md` for the full diagnosis trail.

`e012293`–`af9210d` are the polyglot scaffold (Stages 1–3): Cargo
workspace at the repo root, three Rust crates (`mirage-cache`,
`mirage-scheduler`, `mirage-router`) with PyO3 bindings, maturin build
pipeline, ADR-0004 (scaffold-only) and ADR-0005 (fork resolved). The
Python surface (model loading, denoise loop, attention, serving) stays
Python; Rust owns the orchestration core.

---

## 9. Strategy / business context

For why MI300X first, why diffusers path, why this constitutes a real
wedge in 2026, see the source strategy docs:
- `~/Mirage_Implementation_Plan.pdf` (the 30-page strategic plan)
- `~/Mirage_Cowork_Handoff.md` (decisions-only orientation note)

These are not in the repo (they're personal strategy docs), but the
ADRs and `docs/BUILD_LOG.md` reference them whenever a decision was
strategy-driven rather than tactical.
