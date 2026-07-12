# Repercep Runtime — Handoff

**Date:** 2026-05-25 · **Repo:** https://github.com/miteshs/Repercep ·
**HEAD:** `main`, in sync with `origin/main` (4 commits past Session-16
close) · **Status:** pre-alpha, working on MI300X **and** H100 **and**
CPU, results publishable + **independently verified on a clean GPU**
+ **H100 cache quality now measured** (Session 17).  Polyglot
scaffold + Rust core + Stage-4 v2 serving + Phase 2 (adaptive caching)
+ Phase 2.5 FP8 wiring + autotuned FP8 kernel + **NVIDIA H100 port
(Session 14)** + **F29 FP8 autotune grid + Intel CPU AMX substrate
(Session 15)** + **FA-3 source build + bridge wiring + V-JEPA 2
(Session 16)** + **Wan-2.2 H100 first numbers + F40 confirmed +
H100 cache quality measured + POSITIONING.md (Session 17)** all landed.

**Headlines (post-Session-17):**

- **Cosmos H100 = 99.6 ± 3.9 s / 3.81× NVIDIA's published reference
  (~380 s)** at 121 f / 36 steps; cache quality measured (LPIPS 0.61,
  trajectory-divergent per F23, brightness/std preserved).  3.81 ×
  decomposes as cache 2.75 × × FA-3 1.39 × — see
  `docs/POSITIONING.md` for the honest source-of-speedup.
- **Wan-2.2 H100 = 1552.8 s / 72.6 GiB** at 81 f / 40 steps /
  1280 × 720 (both 14 B MoE experts BF16-resident, no offload — the
  Wan team's 1041 s requires CPU offload + FP8 weight convert which
  we do not yet replicate).  Smoke 17 f / 8 step is
  **37.66 ± 0.31 s** across 5 prompts × 5 seeds (1.59 % spread).
- **Cosmos MI300X = 142 s / 2.68× NVIDIA reference** unchanged from
  Session 13.
- **Wan-2.2 MI300X = ~1700 s projected steady-state** at the same
  shape; H100 is ~9 % faster, silicon delta only.
- **CPU AMX substrate** per ADR-0007 (cannot be exercised on this VM
  — hypervisor masks `amx_bf16`; see Session 16 close for the
  bare-metal numbers).

This is the single doc to read first if you are picking the project up. It
distills `docs/BUILD_LOG.md` (the full chronological log) into the
"what is this, what does it do, and where do I go next" cut.

## TL;DR for a new session

**If you're picking up from Session 17 close (2026-05-25):** read
`docs/SESSION_17_CLOSE.md` first — focused "what happened today,
what to do next" cut that supersedes the rankings below.  Then skim
`docs/POSITIONING.md` (strategic framing, what's defensible to
claim).  Headlines from Session 17:

- Wan-2.2-T2V-A14B end-to-end on H100 for the first time: smoke
  **37.66 ± 0.31 s / 66.4 GiB** (5 prompts × 5 seeds, 1.59 %
  spread), quality reference **1552.8 s / 72.6 GiB** at the
  canonical 81 f / 40 / 1280 × 720 shape.  Session-14 F30 (FUSE
  quota under HF parallel downloader) sidestepped with
  `HF_HUB_ENABLE_HF_TRANSFER=0 --max-workers 4`; 118 GB landed in
  ~8 min.
- **`WanConfig.vae_tiling`** added — required to fit A14B in 80 GiB
  H100 (without it, OOMs in `AutoencoderKLWan.forward` per-frame
  `torch.cat`).  Output bit-stable; F38 in BUILD_LOG.
- **F40 confirmed empirically** via counter-based trace
  (`scripts/trace_wan_attention.py`).  `REPERCEP_FP8_ATTENTION=fa`
  bridge engages 0 times on Wan; `WanTransformer3DModel` bypasses
  `_AttentionBackendRegistry` and calls `torch.F.scaled_dot_
  product_attention` directly.  Three fix paths ranked in BUILD_LOG
  F40; path 1 (custom `WanAttnProcessor` via
  `set_attn_processor`) is the right starting point — the
  highest-leverage Wan move per POSITIONING.md.
- **Cosmos H100 cache quality measured** — closes the Session-15
  deferred item.  LPIPS 0.6067 vs no-cache, motion −34 %,
  brightness/std preserved.  Matches MI300X regime (F23: LPIPS
  0.645, motion −28 %).  The 3.81 × headline now rests on a
  measured H100 quality result, not an assumed one.
- **`docs/POSITIONING.md` created** — strategic-framing doc: what's
  defensible to claim, where the 3.81 × actually comes from (cache
  2.75 × × FA-3 1.39 ×), what is and isn't a moat, options matrix.

**If you're picking up from Session 16 close (2026-05-25 early
morning):** read `docs/SESSION_16_CLOSE.md` first.  Headline:
**Cosmos H100 = 99.6 ± 3.9 s / 3.81× NVIDIA's published reference**
across 5 prompts at 121 f / 36 steps, FA-3 source build + diffusers
bridge wired (`REPERCEP_FP8_ATTENTION=fa`), V-JEPA 2 served end-to-end
on all three targets.

**If you're picking up from Session 13 close (2026-05-24):** read
`docs/SESSION_13_CLOSE.md` first — it's the focused "what happened
today, what to do next" cut and supersedes the open-work ranking
below. The rest of this section is the general orientation.

Pick up here:

1. **Read `docs/METHODOLOGY.md` next** — covers what we measured, how, and
   the honest apples-to-apples accounting. Especially §"Reproducibility
   envelope": the timing claims (142 / 151 / 470 s) are independently
   verified, and the cache-quality claim ("preserved") has been
   **explicitly retracted** in favor of "trajectory-divergent valid Cosmos
   output, ~28 % less inter-frame motion than no-cache." Don't
   over-claim.
2. **The open work, ranked.** All four items in §5 below are tractable
   in a single session; none are blocked on external dependencies:
   - (highest signal) Threshold-quality sweep
     (`--cache-adaptive-threshold` in {0.05, 0.10, 0.20, 0.30, 0.50},
     LPIPS each vs the **no-cache reference** that already exists at
     `benchmark-results/cosmos_no_cache_clean.mp4`). Lets a serving user
     pick the threshold knowingly.
   - (highest reproducibility value) Multi-prompt / multi-seed sweep
     via the ready-to-run `scripts/verify_timing.py --N 3 --prompts 5`.
     ~25 min GPU. Reports mean ± std across 5 prompts; tightens the
     headline to a confidence interval.
   - (highest rigor) FVD against a held-out reference set — the right
     "is cache quality-preserved" arbiter. Needs an eval set of ~50
     reference Cosmos generations across different prompts; doesn't
     exist yet.
   - (publish-ready) OSS announcement. `docs/COSMOS_ON_MI300X.md` +
     `docs/WAN_ON_MI300X.md` are publish-ready with honest framing; the
     remaining work is the social/distribution step (X thread, GitHub
     release notes, blog post draft).

Everything in §3–§9 is reference material. §10 (new this session) is the
12-session timeline if you need it.

---

## 1. Mission

Repercep is a **world-model-native inference engine** — a serving stack
purpose-built for diffusion-temporal video models, not for autoregressive
token decode. The LLM-era serving stack (vLLM, TensorRT-LLM, Diffusers,
torch.compile) leaves 30–60 % of silicon performance on the table for world
models; Repercep closes that gap.

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
| Repercep baseline (diffusers + SDPA→aotriton) | **465 s** (740 s on 2026-05-23 re-run) | 0.82× / 0.51× |
| Same + `torch.compile` on the DiT (≤64 f) | ~410 s projected (49 f shows 1.13× DiT) | ~0.93× (projected) |
| Same + native loop + step-skip `cache_skip=2` | **266 s** | **1.43× faster** |
| Same + native loop + step-skip `cache_skip=4` | **154 s** Session 7 / **163.9 s** Session 9 clean | **2.47×** / **2.32×** |
| Same + native loop + **adaptive cache** (`thr=0.30`) | **151.4 s** Session 12 clean (150.9 / 151.1 Sess 10/9) | **2.52× — well-replicated** |
| Same + native loop + adaptive + tuned FP8 (autotuned tile) | **142.0 s** Session 12 clean (141.7 Agent I worktree) | **2.68× — current headline** |
| Peak HBM (Cosmos, all configs) | **52.5 GiB** | ~30 % less than H100's 74 GB |
| Wan-2.2-T2V-A14B 17f / 8 steps smoke | 45.2 s warm Session 11 (326 s cold Session 10) | **84–85 GiB peak** |
| Wan-2.2 81f / 40 steps quality (projected steady-state) | ~1700 s | **85.1 GiB peak** — first Wan-2.2 on AMD MI300X; ~1.6× behind H100 (1041 s with FP8+offload) |

**Verification status (Session 12 + 13):** all three Cosmos timings
reproduce on a clean GPU within ±0.5 s of the agent / prior-session
measurements. Seed-0 determinism holds (same MD5 across 4 sessions for
the adaptive-no-FP8 mp4). The 2.68× is a *system-vs-system* claim
(Repercep + adaptive cache + tuned FP8 vs NVIDIA's published-unstacked
H100); the raw-hardware comparison has MI300X 1.24× *slower* than H100
at the same compute. See `docs/METHODOLOGY.md` for the apples-to-apples
breakdown.

**Multi-prompt variance (Session 13, 5 distinct prompts):** adaptive
caching wall = mean 154.48 s, std 5.96 s (3.86 % relative); no-cache
wall = 469.84 ± 0.32 s (0.07 %). The headline replicates within noise
across all 5 prompts.

**FVD on 5-pair set (Session 13, 8 clips/video → 40 features/side):**
**166.3** between adaptive and no-cache. Per-prompt LPIPS mean 0.616 ±
0.069. Both small-N preliminary; the FVD literature uses N≥1000.
Pattern is defensible (monotone with threshold, stable across prompts);
the absolute number should be cited as preliminary. Threshold-FVD
trace also measured: 110.7 (0.05) → 233.4 (0.50). See
`docs/COSMOS_ON_MI300X.md` §"Multi-prompt FVD" for the full curve.

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
| The publish-ready first-public-numbers writeup (AMD MI300X) | `docs/COSMOS_ON_MI300X.md` |
| The NVIDIA H100 port-ready writeup (framework; numbers pending Session 15) | `docs/COSMOS_ON_H100.md` |
| Optimization strategy + measured ledger | `docs/OPTIMIZATION.md` |
| Architecture component map | `docs/architecture.md` |
| Key decisions with rationale | `docs/adr/0001..0005` |
| Polyglot build scaffold (Cargo workspace, kernels/, ADR-0004) | `Cargo.toml`, `crates/`, `kernels/` |
| The seam between Repercep and a GPU vendor | `src/repercep/backend/protocol.py` |
| The seam between Repercep and an attention kernel | `src/repercep/attention/protocol.py` + `registry.py` |
| Cosmos engine + Repercep-orchestrated guardrail | `src/repercep/models/cosmos.py` |
| **Wan-2.2 engine** (second WM family) | `src/repercep/models/wan.py` |
| Repercep-native denoising loop (CFG batching, fixed + adaptive caching) | `src/repercep/runtime/denoise.py` |
| **FP8 attention kernel** (Triton fused FA-2) | `src/repercep/attention/fp8_triton.py` + `kernels/triton_kernels/fp8_flash_attn.py` |
| **HIP FP8 GEMM proof-of-life** (compiles, runs, output values TODO) | `kernels/hip/fp8_attn/` |
| Per-stage profiler | `src/repercep/bench/profile.py` |
| Frame-streaming HTTP + gRPC contract | `src/repercep/serving/` |
| End-to-end runners | `scripts/run_cosmos.py`, `scripts/run_wan.py` |
| Benchmark harnesses (head-to-head) | `scripts/bench_caching.py`, `scripts/bench_fp8.py` |
| Per-stage measurement runner | `scripts/profile_cosmos.py` |

---

## 4. How to run it

Requires an AMD GPU host with ROCm 7.x and Python 3.11+.

```bash
git clone https://github.com/miteshs/Repercep.git repercep && cd repercep
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
  Triton FA-2 kernel + Repercep `repercep_fp8` backend registered with
  diffusers' `_AttentionBackendRegistry`. `REPERCEP_FP8_ATTENTION=1` flips
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

0. **(highest signal — closes the methodology asymmetry) H100
   benchmark sweep.** Session 14 landed the NVIDIA backend, the FA-3
   wrapper, the FP8 Hopper Triton kernel, and the optional TE op
   (ADR-0006); `docs/COSMOS_ON_H100.md` is the framework. What's
   missing is the actual measurement on the attached H100 SXM5.
   Running the same harnesses that anchored the MI300X writeup —
   `scripts/run_cosmos.py` at 121 f / 36 steps × {no-cache,
   adaptive, adaptive+FP8, TE-FP8}, plus `verify_timing.py` for
   multi-prompt variance — yields the stack-vs-stack-on-same-silicon
   comparison `docs/METHODOLOGY.md` §3 has been honest about lacking.

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
2. **Wire the Rust core into `src/repercep/serving/app.py`** — the three
   crates (`repercep-cache`, `repercep-scheduler`, `repercep-router`) are
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
  and `_materialize_nltk_data` in `src/repercep/models/cosmos.py`).
- **HF token** for gated weights — stored at `~/.cache/huggingface/token`
  (off the repo). Account: `miteshs`.

---

## 6. Anchors in the code (where the architecture lives)

- **The one seam.** `repercep.backend.protocol.Backend` (Protocol). Everything
  above it imports the Protocol, never a concrete vendor backend
  (ADR-0003). `ROCmBackend` is the only concrete impl today.
- **Attention.** `repercep.attention.protocol.AttentionOp` + `select_attention_op`.
  Naive SDPA is the floor; `ROCmFlashAttention` is the CK wrapper (inert
  until the CK package is built). On ROCm, SDPA already routes through
  aotriton flash kernels (ADR-0002).
- **Runtime.** `repercep.runtime`. `WorldModelEngine` Protocol, Pydantic
  request/response types, `StubEngine` (noise frames — for serving
  development), `PagedLatentCache` (frame-aware eviction skeleton).
- **Cosmos engine.** `repercep.models.cosmos.CosmosEngine`. Wraps the
  diffusers pipeline; orchestrates the guardrail (Repercep-side, not
  pipeline-side — see Session 3 in BUILD_LOG).
- **Native denoising loop.** `repercep.runtime.denoise.denoise_cosmos_video`.
  CFG batching + step-skip caching live here. This is where adaptive
  caching, feature caching, and any future loop-level optimization belongs.
- **Benchmark + profile.** `repercep.bench.harness` (end-to-end timing),
  `repercep.bench.profile` (per-stage; forward-hook + method-wrap; works
  on compiled DiTs).
- **Serving.** `repercep.serving.app` (FastAPI / NDJSON frame streaming) +
  `repercep/serving/proto/repercep.proto` (gRPC contract).

---

## 7. Pointers for common next operations

- **Adding NVIDIA support:** done in Session 14 — see
  `src/repercep/backend/cuda.py`, `src/repercep/attention/hopper_flash.py`,
  `src/repercep/attention/fp8_hopper_triton.py`,
  `src/repercep/attention/transformer_engine.py`, ADR-0006, and
  `docs/COSMOS_ON_H100.md`. Confirmed the port is exactly what
  ADR-0003 promised — one Backend class + one registry entry +
  attention ops below the seam, no changes to model loading, the
  denoise loop, the engine, or the serving handlers.
- **Adding a new WM model:** implement `WorldModelEngine` Protocol. Mirror
  the structure of `repercep.models.cosmos`. Reuse the native loop and
  caching by parameterizing them on the pipeline's components.
- **Adding a loop-level optimization:** land it in
  `repercep.runtime.denoise.denoise_cosmos_video`, gated by a
  `CosmosConfig` flag. Use the per-stage profiler to A/B against the
  baseline.
- **Pushing a measured speedup claim:** always re-verify with
  `scripts/profile_cosmos.py --warmup 1` (warmup-separated), and eyeball
  the artifact at the *full reference config* (121 f / 36 steps).
  Smaller configs can mislead — F16/F17.

---

## 8. Commit history (recent)

```
231ee3e  verification (Session 12): timings reproduce ±0.5s; cache is trajectory-divergent
c673e3e  wan: 81f/40-step quality reference + WAN_ON_MI300X.md + inline profiler
83479e1  docs: clean numbers — autotuned FP8 = 141.7s e2e / 1.13x kernel
8984c84  attention: autotune the FP8 Triton kernel per shape signature
2cbf88d  verification: scripts/verify_quality.py — LPIPS + MSE + PSNR for two mp4s
a24a2dc  verification: methodology doc + multi-seed timing harness
ca709fe  attention: wire FP8 kernel into Cosmos via a diffusers-side backend
576ca56  serving: Stage-4 v2 path through router + scheduler
10c7b9a  Phase 2 integration: docs + numbers from the clean-GPU benchmark sweep
61486e1  caching: adaptive (TeaCache-style) + F15 compile gate at 121f
d916800  attention: FP8 path on CDNA3 — Triton flash kernel + scaled_mm fallback + HIP scaffold
ed1ed27  repercep.models.wan: Wan-2.2 T2V-A14B as the second world-model family
300dc57  docs: refresh for OSS-announce + log Session 8 (polyglot scaffold, OOM fix)
e7f66b0  denoise: wrap loop body in torch.inference_mode() — fixes 121f/36 OOM
af9210d  Stage 3 integration: build pipeline, unique lib names, py.typed markers
8c9bcfe  crates/repercep-router: greenfield request router in Rust+PyO3
f280ada  crates/repercep-scheduler: greenfield priority scheduler in Rust+PyO3
6e61c0e  crates/repercep-cache: port PagedLatentCache from Python to Rust+PyO3
92e91b5  Stage 2: resolve Rust-core fork, populate workspace deps, scaffold 3 crates
e012293  Scaffold polyglot build tooling — Cargo workspace, kernels/, ADR-0004
90aa374  Add docs/HANDOFF.md — single doc to read first when picking up Repercep
41f3f59  Reinstate cache=4 headline: cache tolerance is config-size-dependent (F16+F17)
2b8a6d9  Retract cache_skip=4 headline: cached output is visibly degraded (F16)   [later corrected]
987d4cd  Clean warmup-separated 121f cache=4: 154s — 2.47x faster than H100 reference
010f969  Cached 121f / 36 step run: 164s — MI300X beats H100 reference (~380s) by 2.3x
63d52f1  Measured 121f baseline (465s) + native-loop step-skip caching
9f85125  Initial Repercep Runtime — first published Cosmos-Predict-7B on AMD MI300X
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
workspace at the repo root, three Rust crates (`repercep-cache`,
`repercep-scheduler`, `repercep-router`) with PyO3 bindings, maturin build
pipeline, ADR-0004 (scaffold-only) and ADR-0005 (fork resolved). The
Python surface (model loading, denoise loop, attention, serving) stays
Python; Rust owns the orchestration core.

---

## 9. Strategy / business context

For why MI300X first, why diffusers path, why this constitutes a real
wedge in 2026, see the source strategy docs:
- `~/Repercep_Implementation_Plan.pdf` (the 30-page strategic plan)
- `~/Mirage_Cowork_Handoff.md` (decisions-only orientation note)

These are not in the repo (they're personal strategy docs), but the
ADRs and `docs/BUILD_LOG.md` reference them whenever a decision was
strategy-driven rather than tactical.

---

## 10. Session arc — what landed in each session

| Session | Theme | Headline at session end |
|---|---|---|
| 1 (2026-05-22) | Foundation + first Cosmos run | 17 f / 8 step smoke |
| 2 (2026-05-22) | Optimization Tier 1 | `torch.compile` 1.13× DiT |
| 3 (2026-05-22) | Cosmos guardrail | Safety integrated |
| 4 (2026-05-22) | Competitive scan + validation | Apples-to-apples baseline |
| 5 (2026-05-22) | Repercep-native loop + CFG batching | F9 / F14 measured |
| 6 (2026-05-22) | Full-config baseline + step-skip | 465 s baseline, 154 s w/ `skip=4` |
| 7 (2026-05-23) | 121 f / 36 step caching | **2.47× H100, headline** |
| 8 (2026-05-23) | Polyglot scaffold + Rust core + OOM fix | Stage 1–3, F18 `inference_mode` |
| 9 (2026-05-23) | Phase 2: Wan, adaptive cache, FP8 kernel | adaptive 151 s = **2.51× H100** |
| 10 (2026-05-23) | Phase 2.5 (FP8 wiring) + Stage 4 (v2 serving) | Wan smoke 326 s; FP8 wired but a wash (F20) |
| 11 (2026-05-23) | FP8 autotune (Agent I) + Wan 81f (Agent J) | FP8 142 s = **2.68× H100**; Wan ~1700 s projected |
| 12 (2026-05-24) | **Verification campaign — rigorous** | All timings reproduced ±0.5 s; cache is trajectory-divergent (F23) |
| 13 (2026-05-24) | Threshold sweep + multi-prompt variance + 5-pair FVD | Adaptive 154.48 ± 5.96 s across 5 prompts; FVD 166.3 (small-N preliminary); F24 (FVD harness) |
| 14 (2026-05-24) | **NVIDIA H100 port — architecture + kernels** | CUDABackend lands; Hopper FA-3 + FP8 Triton + TE optional; ADR-0006 closes the "Revisit if" of ADR-0001; F25 + F26 |
| 15 (2026-05-24) | **CPU AMX substrate + H100 follow-ups** | CPUBackend (Vendor.INTEL) + AMX BF16 flash kernel + ADR-0007; F29 FP8 Hopper autotune grid expansion (BLOCK_M=192, num_stages=5, num_warps=12 on 256x256); FA-3 wheel built from source on Hopper; F31 (MooseFS write-quota incident) + F32 (HF Hub offline-mode metadata writes) recorded |
| 16 (2026-05-25 early) | **FA-3 source build + bridge + V-JEPA 2** | Cosmos H100 = **99.6 ± 3.9 s / 3.81× NVIDIA pub** (5 prompts × 5 seeds); FA-3 minimal-config wheel wired via `REPERCEP_FP8_ATTENTION=fa`; TE source-build path validated; V-JEPA 2 served end-to-end on all 3 targets; F33–F37 |
| 17 (2026-05-25) | **Wan-2.2 H100 + F40 confirmed + quality measured** | Wan smoke = 37.66 ± 0.31 s; Wan 81f/40 = **1552.8 s / 72.6 GiB**; `WanConfig.vae_tiling` added; **F40 confirmed empirically** via counter-based trace (FA-3 bridge engages 0× on Wan); Cosmos H100 cache quality measured (LPIPS 0.61, trajectory-divergent per F23); `docs/POSITIONING.md` created; F38–F41 |

Findings (F1–F41) are cross-referenced in `docs/BUILD_LOG.md`. ADRs
0001–0007 cover the structural decisions (MI300X-first, attention
primitive, vendor-neutral Backend, polyglot tooling scaffold, Rust-core
fork resolved, NVIDIA H100 parallel target, Intel CPU AMX substrate).

---

## 11. Quick commands (for resuming)

```bash
# Verify the environment is good
make lint typecheck test
sg render -c "sg video -c 'make check-gpu'"

# Headline reproducer (~3 min, single GPU)
sg render -c "sg video -c '\
    REPERCEP_FP8_ATTENTION=1 .venv/bin/python scripts/run_cosmos.py \
        --frames 121 --steps 36 --native-loop \
        --cache-mode adaptive --cache-adaptive-threshold 0.30 \
        --cache-force-full-every 16'"
# Expect: generate_seconds ≈ 142 s, peak_hbm_gib = 52.5

# Quality vs no-cache reference (the load-bearing comparison)
.venv/bin/python scripts/verify_quality.py \
    benchmark-results/cosmos_no_cache_clean.mp4 \
    benchmark-results/cosmos_adaptive_fp8_tuned_clean.mp4 \
    --device cpu
# Expect: LPIPS ≈ 0.64 ("substantially different" — trajectory-divergent,
# not garbage; see Session 12 / F23 notes)

# Multi-seed campaign (if pushing further)
sg render -c "sg video -c '.venv/bin/python scripts/verify_timing.py --N 3 --prompts 5'"
# ~25 min GPU; writes benchmark-results/verify_timing_<ts>.json
```
