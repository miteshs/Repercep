# Session 21 H100 handoff

**Date:** 2026-05-25 · **HEAD:** `74355ea` on `main`, in sync with
`origin/main` (Session 20 close, 8 commits ahead of Session 19 close).

Goal for the next pod: pick **one** of the two open strategic levers
(Wan-shaped adaptive cache, or N≥50-prompt FVD) and commit a focused
~1-2 week sprint.  Read `docs/SESSION_20_CLOSE.md` first for what just
landed and `docs/POSITIONING.md` for the strategic framing.

## State of the world at session start

**Headline numbers** (all reproducible end-to-end on a fresh H100 pod
with the current `main`):

| Workload | Config | Wall | Peak HBM | vs ref |
|---|---|--:|--:|--:|
| Cosmos H100 (no-cache) | 121f/36 native | **320.9 s** | 52.5 GiB | **1.18 × NVIDIA pub** |
| Cosmos H100 (adaptive thr=0.30) | 121f/36 native + cache | **101.8 s** | 52.5 GiB | **3.73 × NVIDIA pub** |
| Wan TI2V-5B (smoke) | 17f / 8 step | **6.0 s** | 42.6 GiB | n/a |
| Wan A14B (smoke, Session 17) | 17f / 8 step | 37.66 ± 0.31 s | 66.4 GiB | n/a |
| Wan A14B (quality, Session 17) | 81f / 40 step | 1552.8 s | 72.6 GiB | 1.49 × slower than Wan team's offload+FP8 |

Cosmos cache lever on this pod: **3.15 ×** (no-cache 320.9 → adaptive 101.8).

**Plumbing landed and verified:**

- **F40 fix-path 1** (Wan diffusers bridge) engages 960 × on Wan
  TI2V-5B 17f/8 under `REPERCEP_FP8_ATTENTION=fa` — up from 0 × in
  Session 17's F40 baseline.  `wan_processor.maybe_install_repercep_
  wan_attention(pipe)` in `WanEngine.load()` is doing exactly what
  the F40 spec needed it to.
- **Native PyO3 control plane** is live: `PagedLatentCache` ←
  `repercep_cache._native`, `Router` ← `repercep_router._native`,
  `Scheduler` ← `repercep_scheduler._native`.  Python fallbacks
  available but not active.
- **`mypy --strict src/repercep` clean across all 49 files** for the
  first time since the lazy `QuantizedLinearModule` pattern.
- **AMX CI compile-smoke + routing tests:** `REPERCEP_AMX_FORCE_BUILD=1
  make kernels-cpu` builds all three AMX kernels (BF16 / INT8 / FP16)
  on the hypervisor-AMX-masked dev VM.  4 capability-gate routing
  tests verify the registry's INTEL branch under monkeypatched
  detectors.

**Env shape on the H100 pod:**

- `.venv` symlinks to `/workspace/repercep-venv`.
- torch 2.11.0+cu128, diffusers 0.37.1, transformers 5.9.0,
  triton 3.6.0, fastapi 0.136.3.
- `flash-attn` and `flash_attn_3` are **NOT** installed.  The
  `REPERCEP_FP8_ATTENTION=fa` bridge engages the diffusers dispatcher
  and falls through `_native_fallback` → `torch.nn.functional.scaled_
  dot_product_attention` (cuDNN-FA-3 on Hopper), not the FA-3 Python
  wrapper.  This is the path that produced the 101.8 s headline.
- `transformer_engine` (TE) is **NOT** installed.  TE-FP8 is not
  in the current path.
- HF cache lives at `/workspace/.cache/huggingface`; `~/.cache/
  huggingface/{hub,xet}` are symlinks.  **Keep the symlinks** — the
  root overlay is only 50 GB and fills up if duplicates accumulate.
- HF + GitHub auth: `source ~/extra.sh` (sets `HF_TOKEN`, runs
  `gh auth login`).
- Available models on disk (no re-download needed):
  - `nvidia/Cosmos-1.0-Diffusion-7B-Text2World` (~66 GB)
  - `nvidia/Cosmos-1.0-Guardrail` (assets)
  - `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (~32 GB)
- Wan A14B (~118 GB) and `Wan-AI/Wan2.2-T2V-A14B-Diffusers` are NOT
  cached locally.

**Verified on this H100 pod (re-runnable commands at bottom):**

- `pytest -q` → 233 passed / 24 skipped.
- `ruff check src tests scripts` → clean.
- `mypy --strict src/repercep` → clean (49 files).
- `REPERCEP_AMX_FORCE_BUILD=1 make kernels-cpu` → all three kernels
  build to `_native*.so`.

## What's open after Session 20

Per `docs/POSITIONING.md` "What we recommend doing in the next 2 weeks":

1. ✅ Cosmos H100 quality measurement (Session 17).
2. ✅ F40 fix-path 1 (Session 20, F45).
3. 🟡 **Wan-shaped adaptive cache** — open, ~1-2 weeks, highest lever.
4. 🟡 **FVD with N ≥ 50 prompts** — open, ~1 week of GPU.

Pick *one* of (3) or (4) and commit.  Both are 1-2 weeks; doing both
in parallel is overcommit per POSITIONING.md.

### Lane A — Wan-shaped adaptive cache (widens the wedge)

The Cosmos adaptive cache hardcodes `CosmosTransformer3DModel` block
topology in `src/repercep/runtime/denoise.py::denoise_cosmos_video`.
A Wan equivalent needs:

- A native `denoise_wan_video` in the same module (mirror the Cosmos
  pattern but parameterize on `pipe.transformer` /
  `pipe.transformer_2` for A14B's MoE).
- A Wan-specific cache-trigger gate that respects the MoE high-noise /
  low-noise expert boundary (`pipe.config.boundary_ratio`).  The
  rel-L1 accumulation across steps that triggers the Cosmos full
  forward needs an A14B-aware partition because the two experts
  produce different feature distributions.
- A `WanConfig.cache_mode` knob that mirrors `CosmosConfig`'s.
- Tests parallel to `tests/test_denoise.py` (mirror the Cosmos cache
  unit tests but for Wan shapes).
- Integration into `WanEngine.generate` so `use_native_loop=True`
  + `cache_mode="adaptive"` short-circuits the diffusers
  `WanPipeline.__call__` and drives the loop directly, the way
  `CosmosEngine.generate` already does.

**Why this matters:** turns the Wan H100 1552 s headline into ~700-800 s
projected (POSITIONING.md option matrix).  Re-positions the story
from "Cosmos-specific cache lever" to "world-model-family-general
cache lever proven on two diffusion-video families × three silicon
targets."  That's a structural moat-level claim.

**Prereqs the pod needs:**

- Wan A14B repo on disk for the headline (~118 GB download — set
  `HF_TOKEN`, then `from huggingface_hub import snapshot_download;
  snapshot_download('Wan-AI/Wan2.2-T2V-A14B-Diffusers', ...)`).
  TI2V-5B alone won't validate the MoE-aware cache.
- Enough disk: pod has /workspace = 232 T avail, plenty.

### Lane B — FVD with N ≥ 50 prompts (bullet-proofs the existing wedge)

`docs/METHODOLOGY.md` §"Held-out reference set for FVD" already
documents the workflow.  `scripts/compute_fvd.py` is in tree;
`scripts/eval_cpu_quality.py` chains LPIPS + FVD.  What's missing:

- A held-out eval set of ~50 prompts × 2 outputs each (no-cache and
  adaptive) generated and saved.  Estimate: 50 × 320.9 s no-cache +
  50 × 101.8 s adaptive ≈ 5.9 hr GPU on this pod for the references
  alone.
- FVD computation across the 50-pair set with the I3D backbone.
  ~10 min once the references are generated.
- A writeup in `docs/COSMOS_ON_H100.md` §"Quantitative cache quality"
  expanding the current pixel-LPIPS analysis to distribution-level
  FVD.

**Why this matters:** the 3.73-3.81 × Cosmos headline currently
includes an unmeasured-at-distribution-level cache quality claim.
Per POSITIONING.md "the dependency that quietly blocks the most
others is FVD/LPIPS quality measurement on H100 — every claim that
includes the cache currently rests on an unmeasured quality
assertion, and that's the one thing every skeptical external reader
will probe first."  FVD lands the credibility piece needed before
external publication.

**Prereqs:** Cosmos repo on disk (already there).  A held-out prompt
set — generate it from a published source (e.g., the prompts used in
Cosmos's model card eval section, or a small thematic cluster).

### Recommendation if forced to choose

**Lane A (Wan-shaped cache)** if the goal is widening the wedge —
turns one more model family into a cache-positive case and aligns
with the "world-model-native engine, not Cosmos-specific" story.

**Lane B (FVD)** if the goal is making the existing wedge defensible
for external publication / blog post / OSS announcement.

Both are real and unblock different next-quarter moves.  POSITIONING.md
itself doesn't pick between them; it lays out the trade.

## Lower-priority open items

5. **Install `flash-attn` (or `flash-attn-3`) + re-run F45 trace.**
   Counter flips from `native_fallback → SDPA = 960` to `→ FA-3 = 960`.
   Then benchmark Wan TI2V-5B with the bridge taking the FA-3 path vs
   without to quantify the FA-3 lift on Wan specifically.  **Build
   `flash-attn` with `MAX_JOBS=8`** to avoid the 266-process compile
   storm we killed mid-Session-20.  ~half-day.

6. **Wan-2.2-A14B 81f/40 re-verification.**  Session 17's 1552.8 s /
   72.6 GiB number was explicitly scoped out of Session 20 (118 GB
   download).  When pod time allows, this is the third leg of the
   headline triple — currently the only one not re-verified on the
   current torch+diffusers stack.

7. **F40 fix-path coverage for VAE + text encoder.**  The 1100 direct
   SDPA calls observed in F45's trace are `AutoencoderKLWan` decoder
   + UMT5-XXL text encoder bypassing the dispatcher entirely.  Not
   load-bearing for the DiT-dominated headline.  Defer until the
   bridge is shown to actually win on the DiT path first
   (needs flash-attn for the lift).

8. **Bare-metal AMX measurement.**  Hardware-blocked.  When a
   Sapphire / Emerald / Granite Rapids host opens up, the
   `REPERCEP_AMX_FORCE_BUILD` smoke + capability-gate tests are CI-
   ready; what's missing is real silicon for headline numbers.

9. **Ada FP8 wiring decision (F42 vs F44).**  Deprioritized in
   Session 20 by explicit user choice ("forget about Ada for now").

10. **Production serving driver hardening.**  Different axis entirely
    per POSITIONING.md.

## Multi-prompt F47 follow-up (smaller win, ~50 min GPU)

If picking up Lane A or Lane B feels too heavy, the natural smaller
follow-up is a multi-prompt re-measure of F47's no-cache delta:

- 3 prompts × 3 seeds × {no-cache, adaptive} = 18 runs.
- ~3 min/run for adaptive + ~5.4 min/run for no-cache ≈ 50-60 min GPU.
- Sets up POSITIONING.md update: replaces the "446.3 s no-cache /
  138.4 s adaptive / 99.6 s adaptive+FA-3" decomposition row with
  fresh measurements on the current stack.

This converts F47 from "single-prompt this-pod pointer" into "n=9
defensible POSITIONING.md update."  `scripts/verify_timing.py` is
the harness; would need a flag added for `--cache-mode none` if it
doesn't already accept it.

## Quick reproducers

```bash
cd /home/mshah/Repercep
source ~/extra.sh                          # HF_TOKEN + gh auth login

# Sanity checks (all clean on Session 20 close)
PYTHONPATH=$(pwd)/src .venv/bin/python -m ruff check src tests scripts
PYTHONPATH=$(pwd)/src .venv/bin/python -m mypy src/repercep
PYTHONPATH=$(pwd)/src .venv/bin/python -m pytest -q

# Confirm native PyO3 control plane is live (Session 19's check)
PYTHONPATH=$(pwd)/src .venv/bin/python - <<'PY'
from repercep.runtime.latent_cache import PagedLatentCache
from repercep.runtime.router import Router
from repercep.runtime.scheduler import Scheduler
print(PagedLatentCache.__module__)  # expect: repercep_cache._native
print(Router.__module__)            # expect: repercep_router._native
print(Scheduler.__module__)         # expect: repercep_scheduler._native
PY

# Wan F40 dispatch trace — expect bridge=960, fallback→SDPA=960
REPERCEP_FP8_ATTENTION=fa PYTHONPATH=$(pwd)/src .venv/bin/python \
    scripts/trace_wan_attention.py --small --frames 17 --steps 8

# Cosmos H100 headline (cached) — expect ~100 s ± 4 s, 52.5 GiB
REPERCEP_FP8_ATTENTION=fa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$(pwd)/src .venv/bin/python -u scripts/run_cosmos.py \
        --frames 121 --steps 36 --native-loop \
        --cache-mode adaptive --cache-adaptive-threshold 0.30 \
        --cache-force-full-every 16

# Cosmos H100 no-cache baseline — expect ~321 s ± n, 52.5 GiB
REPERCEP_FP8_ATTENTION=fa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$(pwd)/src .venv/bin/python -u scripts/run_cosmos.py \
        --frames 121 --steps 36 --native-loop --cache-mode none

# Wan TI2V-5B smoke — expect ~6 s, 42.6 GiB
REPERCEP_FP8_ATTENTION=fa PYTHONPATH=$(pwd)/src .venv/bin/python \
    scripts/run_wan.py --small --frames 17 --steps 8 --profile

# AMX CI compile-smoke + routing tests (works on AMX-less VMs)
REPERCEP_AMX_FORCE_BUILD=1 make kernels-cpu
PYTHONPATH=$(pwd)/src .venv/bin/python -m pytest \
    tests/test_attention_cpu.py tests/test_amx_int8.py \
    tests/test_amx_fp16.py -q
```

## When you resume — start here

1. Read this doc.
2. Skim `docs/SESSION_20_CLOSE.md` for everything that landed.
3. Skim `docs/POSITIONING.md` "Strategic options matrix" + "What we
   recommend doing in the next 2 weeks" — pick a lane.
4. If picking up Lane A (Wan cache): start with the A14B download
   (~118 GB), then sketch `denoise_wan_video` mirroring `denoise_
   cosmos_video` in `src/repercep/runtime/denoise.py`.
5. If picking up Lane B (FVD): pick the prompt set source, then
   pre-generate the 50 no-cache + 50 adaptive references (~6 hr
   GPU), then run `scripts/compute_fvd.py` over the pairs.
6. CPU AMX work is hardware-blocked; the CI smoke + routing tests
   landed in Session 20 cover the movable-today scope.
7. Ada FP8 / multi-GPU CP / production serving driver are all
   explicitly deprioritized — don't sink time there without a
   user redirect.

Have a good day.
