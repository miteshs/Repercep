# Mirage Runtime — Handoff

**Date:** 2026-05-23 · **Repo:** https://github.com/miteshs/Mirage ·
**HEAD:** `41f3f59` · **Status:** pre-alpha, working on MI300X, results
publishable

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
| Mirage baseline (diffusers + SDPA→aotriton) | **465 s** | 0.82× (slower) |
| Same + `torch.compile` on the DiT | ~410 s projected (49 f shows 1.13× DiT) | ~0.93× (projected) |
| Same + native loop + step-skip `cache_skip=2` | **266 s** | **1.43× faster** |
| Same + native loop + step-skip `cache_skip=4` | **154 s** | **2.47× faster** |
| Peak HBM (all configs) | **52.5 GiB** | ~30 % less than H100's 74 GB |

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
| Chronological build log, every decision and finding | `docs/BUILD_LOG.md` (D1-D9, F1-F17, 7 sessions) |
| The publish-ready first-public-numbers writeup | `docs/COSMOS_ON_MI300X.md` |
| Optimization strategy + measured ledger | `docs/OPTIMIZATION.md` |
| Architecture component map | `docs/architecture.md` |
| Key decisions with rationale | `docs/adr/0001..0003` |
| The seam between Mirage and a GPU vendor | `src/mirage/backend/protocol.py` |
| Cosmos engine + Mirage-orchestrated guardrail | `src/mirage/models/cosmos.py` |
| Mirage-native denoising loop (CFG batching, step caching) | `src/mirage/runtime/denoise.py` |
| Per-stage profiler | `src/mirage/bench/profile.py` |
| Frame-streaming HTTP + gRPC contract | `src/mirage/serving/` |
| End-to-end runner | `scripts/run_cosmos.py` |
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

**Next moves, in order of strategic value:**

1. **Announce / publish.** `docs/COSMOS_ON_MI300X.md` is publish-ready. The
   first-public-Cosmos-on-AMD-GPU framing is the OSS-first GTM lever the
   implementation plan calls for (§2.4).
2. **Wan-2.2 as the second model** — Phase 2 of the plan (Runtime v0.5 + 2nd
   model family). Most of the runtime infra reuses; main new work is the
   model loader. Proves the runtime is WM-native, not Cosmos-specific.
3. **Adaptive caching (TeaCache-style)** — skip on input-similarity rather
   than a fixed cadence. The right long-term shape; replaces F16/F17's
   "rule of thumb."
4. **Fix the F15 inductor segfault** at 121 f shapes so `torch.compile`
   stacks with caching.
5. **FP8 (CDNA3 native MFMA)** — Phase 2, real ~2× math throughput.

**Deferred / known issues:**

- **F15** — `torch.compile` (inductor + triton-rocm) segfaults mid-warmup at
  121 f shapes; 49 f compiles fine. Repro: `profile_cosmos.py --frames 121
  --steps 36 --compare --warmup 1`.
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

---

## 9. Strategy / business context

For why MI300X first, why diffusers path, why this constitutes a real
wedge in 2026, see the source strategy docs:
- `~/Mirage_Implementation_Plan.pdf` (the 30-page strategic plan)
- `~/Mirage_Cowork_Handoff.md` (decisions-only orientation note)

These are not in the repo (they're personal strategy docs), but the
ADRs and `docs/BUILD_LOG.md` reference them whenever a decision was
strategy-driven rather than tactical.
