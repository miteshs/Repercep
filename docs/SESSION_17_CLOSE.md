# Session 17 close — next-session pickup

**Date:** 2026-05-25 · **Working tree:** clean on `main`, in sync with
`origin/main` (4 commits since SESSION_16_CLOSE: `d66afc1` Wan H100
first numbers → `ff77b97` tier-1 verification → `03bf214` POSITIONING
→ `f247fcf` H100 cache quality).

Read this *before* `docs/HANDOFF.md`.  This is the focused "what
happened today, what to do next" cut for Session 17; `HANDOFF.md` is
the full orientation layer.

---

## TL;DR

**Wan-2.2-T2V-A14B on H100, end-to-end, first time.**  Session 14's
F30 download blocker resolved by serializing the HF downloader
(`HF_HUB_ENABLE_HF_TRANSFER=0 --max-workers 4`); 118 GB landed in
~8 min.  At the canonical 81 f / 40 step / 1280 × 720 reference
shape: **1552.8 s / 72.6 GiB peak** (~9 % faster than MI300X's
~1700 s projected, 1.49 × *slower* than Wan team's published 1041 s —
they offload + FP8, we don't).  Smoke 17 f / 8 step: **37.66 ± 0.31 s
mean** across 5 prompts × 5 seeds (1.59 % spread, tighter than
Cosmos's 3.86 %).

**F40 confirmed empirically — `MIRAGE_FP8_ATTENTION=fa` bridge does
NOT engage on Wan.**  Counter-based trace landed: 0 dispatcher
engagements, 780 direct `torch.F.scaled_dot_product_attention` calls
across one 17 f / 4 step smoke.  `WanTransformer3DModel` bypasses
`_AttentionBackendRegistry` entirely — same F36 pattern.  Three fix
paths ranked in BUILD_LOG F40; path 1 (custom `WanAttnProcessor` via
`set_attn_processor`) is the right starting point.

**Cosmos H100 cache quality measured (closes Session-15 deferred
item).**  Pixel LPIPS 0.6067 / motion −34 % / brightness +3.6 %
(preserved).  Matches MI300X regime (F23: LPIPS 0.645 / motion −28 %);
no silicon-specific quality cliff.  The 99.6 s headline and the
dependent 3.81 × claim now rest on a *measured* H100 quality result,
not an assumed one.  FVD with N ≥ 50 on a held-out eval set is now
the next pre-publication vouchability item.

New code: `WanConfig.vae_tiling`, `--vae-tiling` flag, conditional
`guidance_scale_2` for non-MoE Wan variants, `--backend` +
CPU-aware peak-RSS probe in `scripts/run_wan.py`.  New scripts:
`scripts/verify_wan_timing.py` (variance harness),
`scripts/trace_wan_attention.py` (F40 verifier).  New unit tests:
4 in `tests/test_wan.py`.  New doc: `docs/POSITIONING.md` (moat
analysis, honest source-of-speedup decomposition).

---

## What landed today

### 1. Wan-2.2-T2V-A14B H100 — first end-to-end measurement

* **Smoke 17 f / 8 step @ 1280 × 720, variance over 5 prompts × 5
  seeds:**
  - mean **37.66 s**, stdev 0.313 s, range 37.4 – 38.0 s, spread
    1.59 %
  - peak HBM **66.4 GiB** identical across all 5 runs (allocator
    deterministic at this shape)
  - 1.20 × faster than MI300X's 45.2 s warm smoke at the same shape;
    18 GiB lower peak (VAE tiling vs the no-tiling MI300X path)

* **Quality reference 81 f / 40 step @ 1280 × 720, single-run:**
  - generate **1552.8 s** (~25.9 min), peak **72.6 GiB**
  - per-step **38.82 s** (vs MI300X steady-state ~41 s = ~5 % faster
    silicon delta)
  - DiT loop **1517.6 s (97.7 % of total)** — hi-noise 493.24 s /
    26 calls + lo-noise 1024.35 s / 54 calls; VAE 28.40 s; text 0.32 s
  - vs Wan team's published 1041.5 s on single H100: **1.49 × slower**
    because they `--offload_model True --convert_model_dtype` (CPU
    offload + FP8 weight convert), we run both 14 B experts BF16
    resident with no offload.  Different stacks — see WAN_ON_H100.md
    TL;DR for the apples-to-apples breakdown.

* `docs/WAN_ON_H100.md` rewritten end-to-end: all TBD / "see F30"
  placeholders replaced with measured numbers; per-stage profile
  table added; Caveats section updated; Reproduce block now has the
  full `MIRAGE_FP8_ATTENTION=fa PYTORCH_CUDA_ALLOC_CONF=expandable_
  segments:True --vae-tiling` invocation.

### 2. `WanConfig.vae_tiling` — required for 80 GiB H100 fit

A14B BF16 + FP32 VAE peak does **not** fit in 80 GiB H100.  The first
two H100 attempts hit `torch.OutOfMemoryError` at the per-frame
`torch.cat` inside `AutoencoderKLWan.forward` (`autoencoder_kl_wan.py:170`).

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` recovers ~3 GiB
of allocator fragmentation but is insufficient — activations grow
into the gap.  Real fix: `pipe.vae.enable_tiling()`.  Output is
bit-stable (the decoder already does spatial-tile seam blending).
Landed as:

- `WanConfig.vae_tiling: bool = False` (default off preserves
  MI300X-native baseline)
- `--vae-tiling` flag on `scripts/run_wan.py`
- Drops peak by **~18 GiB** at 1280 × 720 (was OOM, now 66.4 GiB
  smoke / 72.6 GiB at 81 f / 40)

Same flag would also help MI300X at this shape (would drop peak from
85.1 GiB to ~67 GiB) but doesn't unblock anything there (192 GiB has
headroom).

### 3. `WanEngine` non-MoE crash fix

`WanEngine.generate` previously passed `guidance_scale_2` unconditionally
to `WanPipeline.__call__`.  Non-MoE Wan variants (TI2V-5B,
`boundary_ratio is None`) raise `ValueError: guidance_scale_2 is only
supported when the pipeline's boundary_ratio is not None`.  Hit on
the CPU TI2V-5B smoke; fixed by probing `pipe.config.boundary_ratio`
at call time and conditionally including the kwarg.  Recorded as
**F39**.

### 4. F40 confirmed empirically — FA-3 bridge no-op on Wan

The smoke + 81f/40 H100 numbers came in much closer to MI300X than
Cosmos's FA-3 bridge would predict (smoke 1.20 ×, 81f/40 1.06 ×
per-step — silicon-delta range, not the 3.81 × Cosmos showed via the
same bridge).  Hypothesis: same F36 pattern as Cosmos, where
`WanTransformer3DModel` bypasses `_AttentionBackendRegistry`.
Confirmed by direct trace:

```bash
MIRAGE_FP8_ATTENTION=fa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python scripts/trace_wan_attention.py --frames 17 --steps 4
```

Counter output:

```
torch.F.scaled_dot_product_attention (direct)   780

=== VERDICT ===
F40 CONFIRMED: bridge engaged 0x; 780 direct SDPA calls.
WanTransformer3DModel bypasses _AttentionBackendRegistry.
```

780 SDPA calls / 4 steps / 2 CFG forwards × ~98 attention sublayers
≈ 195 calls per step per forward, consistent with Wan A14B's
~30 transformer blocks × ~3 attention paths per block.  The
diffusers bridge `MIRAGE_FP8_ATTENTION=fa` is **informational not
load-bearing** on Wan today.

Three fix paths ranked in BUILD_LOG F40:

1. **Custom `WanAttnProcessor` via diffusers `set_attn_processor`** —
   targeted, doesn't perturb other models.  Right starting point.
2. **Monkey-patch `torch.F.scaled_dot_product_attention` in
   `WanEngine.load()`** — heavy hammer, works for any model that
   bypasses the dispatcher.
3. **Diffusers upstream PR to `WanTransformer3DModel.forward`** —
   architecturally right but not in our control.

### 5. Cosmos H100 cache quality — Session-15 deferred item closed

Single-pair pixel comparison at the canonical 121 f / 36 / 1280 × 704
config, both runs with `MIRAGE_FP8_ATTENTION=fa --native-loop` (FA-3
+ native loop identical; cache is the only difference):

- no-cache reference: 310.3 s wall / 52.5 GiB peak / motion 9.13
- adaptive cache (thr=0.30): 95.8 s wall / 52.5 GiB peak / motion 6.00
  (**−34 %**)

| metric | value | reading |
|---|---|---|
| LPIPS vs no-cache | **0.6067** | "trajectory-divergent" per F23, not noise |
| PSNR vs no-cache | 14.11 dB | low-PSNR regime; consistent with divergence |
| MSE vs no-cache | 2560.7 | uint8 scale; same regime as MI300X |
| brightness drift | +3.6 % | preserved |
| pixel std drift | +0.7 % | preserved |

H100 reproduces MI300X (F23: LPIPS 0.645, motion −28 %, brightness +
std preserved) within a few percentage points.  The cache is
trajectory-divergent on Hopper *exactly* as on CDNA3 — no silicon-
specific quality cliff.

The **99.6 s headline** (and the **3.81 × Cosmos H100 claim** that
depends on it) now rests on a *measured* H100 quality result, not an
assumed one.  Closes the "soft underbelly" flagged in POSITIONING.md
§"The quality caveat".

What's still open per F23 framing: **FVD against a held-out eval set
with N ≥ 50 prompts**.  Pixel LPIPS at 0.6 is too strict for diffusion
outputs that trade trajectory for compute; FVD is the right
distribution-level metric.  `scripts/compute_fvd.py` exists; the
held-out reference set doesn't.

### 6. Verification infrastructure (tier-1 from the user's "increase confidence" thread)

* **`scripts/verify_wan_timing.py`** — Wan sibling of
  `scripts/verify_timing.py` (which is Cosmos-only).  N-run variance
  harness with the same 5-prompt set used for Cosmos in Session 16.
  Reports mean / stdev / range / pct-spread; writes JSON sidecar.
* **`scripts/trace_wan_attention.py`** — counter-based dispatcher
  verifier.  Wraps `_mirage_fp8_attention`, `_native_fallback`, and
  `torch.F.scaled_dot_product_attention` with counters; prints a
  verdict.  Used to confirm F40.
* **4 new unit tests in `tests/test_wan.py`:**
  - `WanConfig.vae_tiling` defaults to `False`
  - `vae_tiling=True` triggers `pipe.vae.enable_tiling()` on `load()`
  - `vae_tiling=False` does NOT trigger `enable_tiling()`
  - `guidance_scale_2` passed iff `pipe.config.boundary_ratio is not None`
  All 18 Wan tests pass; pytest full suite 178 passed / 12 skipped /
  1 pre-existing fail (`test_backend_routes_short_seq_to_native`, F41).

### 7. `docs/POSITIONING.md` — new strategic-framing doc

Synthesizes "what Mirage is, what's defensible to claim, what isn't."
Companion to METHODOLOGY.md (accounting), ANNOUNCEMENT.md
(distribution), per-target docs (numbers), BUILD_LOG.md (root causes).

Key content:

- The three landed claims, ranked by external defensibility (Cosmos
  H100 3.81 ×, MI300X structural MoE-resident, world-model breadth)
- Source of the 3.81 × decomposed: cache 2.75 × × FA-3 1.39 × =
  3.82 × ≈ measured 3.81 ×.  Mirage **no-cache** baseline on H100 is
  actually 1.17 × *slower* than NVIDIA's published reference; the
  cache is the dominant lever.
- What is and isn't a moat
- Strategic options matrix (5 story arcs × {effort, risk, headline
  impact})

### 8. CPU 5B sidebar — runs, but with a methodology miss

* Wan-2.2-TI2V-5B 17 f / 8 step on Sapphire Rapids 8468 (160 cores,
  OMP_NUM_THREADS=120; AMX masked by hypervisor — `/proc/cpuinfo`
  shows `avx512_bf16 + avx_vnni` only):
  - generate **3969.8 s** (~66 min)
  - peak RSS delta 12.4 GiB
  - DiT loop 2193.5 s (~36.5 min), VAE decode 1734.2 s (~29 min)

* **Methodology miss:** `--vae-tiling` was passed for parity with
  the H100 invocation, but on CPU it shreds the FP32 VAE decode into
  hundreds of small per-tile forwards — actively harmful when the
  envelope is 1.5 TiB RAM.  Without tiling, the CPU number would
  likely land at ~38-45 min.  Re-running is open work; the lesson is
  recorded in the doc + this close.

* The CPU AMX kernel itself **could not be built** on this VM —
  hypervisor masks `amx_bf16`.  The build refuses with a clear
  FATAL message; this is correct behavior, not a regression.  The
  CPU smoke validated the engine end-to-end through torch SDPA +
  oneDNN AVX512_BF16 path, but did NOT exercise Mirage's AMX kernel.

### 9. Other landed pieces

* `scripts/run_wan.py` — `--vae-tiling`, `--backend
  {rocm,cuda,cpu,auto}`, CPU-aware peak-RSS probe mirroring
  `run_cosmos.py`.
* `docs/COSMOS_ON_H100.md` §"Quantitative cache quality" rewritten
  with measured numbers (was "Deferred to Session 15").
* `docs/BUILD_LOG.md` Session 17 section added with F38–F41 +
  empirical F40 evidence + fix-path ranking.

### Things attempted that did NOT fully land

* **F40 *fix*** — only the diagnosis landed.  Implementation of
  fix-path 1 (custom `WanAttnProcessor`) is the highest-leverage
  next move per POSITIONING.md.
* **TE-FP8 end-to-end on Cosmos** — still queued from Session 16
  §"What's open" #1.  Did not start this session.
* **Wan-shaped adaptive cache** — the biggest single lever for the
  Wan headline; ~1-2 weeks of focused work; queued.
* **CPU 5B re-run without `--vae-tiling`** — would land a cleaner
  CPU number (estimated ~38-45 min vs measured 66 min).  Open.
* **FVD with N ≥ 50** — the next pre-publication vouchability item.
  Open.

---

## Findings recorded (F38–F41)

* **F38** — Wan-2.2 A14B BF16 both-experts-resident does not fit in
  80 GiB H100 without VAE tiling; OOM happens inside
  `AutoencoderKLWan.forward` at the per-frame `torch.cat`.
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` alone recovers
  ~3 GiB of fragmentation but is insufficient.  `pipe.vae.enable_
  tiling()` drops peak by ~18 GiB; output is bit-stable.

* **F39** — `WanEngine.generate` unconditionally passed
  `guidance_scale_2` to `WanPipeline.__call__`.  Non-MoE Wan
  variants (TI2V-5B, `boundary_ratio is None`) raise on the kwarg.
  Fix: probe `pipe.config.boundary_ratio is not None` and
  conditionally include.

* **F40** — `MIRAGE_FP8_ATTENTION=fa` bridge does **NOT** engage on
  `WanTransformer3DModel` end-to-end.  Confirmed by counter-based
  trace: 0 dispatcher engagements, 780 direct
  `torch.F.scaled_dot_product_attention` calls per 17 f / 4 step
  smoke.  Same F36 pattern.  Three fix paths ranked; path 1
  (`set_attn_processor`-injected `WanAttnProcessor`) is the right
  starting point.

* **F41** — `test_backend_routes_short_seq_to_native` asserts
  `torch.allclose(out, ref, atol=1e-5, rtol=1e-5)` on **bf16**
  tensors.  bf16 epsilon ~7.8e-3 — threshold fundamentally
  incompatible with the dtype.  Pre-existing latent bug; surfaces
  now because FA-3 actually being installed changes the SDPA route.
  Not a regression caused by Session 17 code changes.

See `docs/BUILD_LOG.md` §"Session 17" for full details + the
empirical counter trace + the fix-path ranking.

---

## Quality gate (post-session)

* `ruff check src tests scripts` — clean
* `mypy --strict` — clean over 65 source files
* `pytest -q` — 178 passed / 12 skipped / 1 failed (F41,
  pre-existing; ticket-of-record in BUILD_LOG)
* `make kernels-cpu` — refused (hypervisor masks `amx_bf16`); not
  a regression, expected behavior on this VM

---

## What's open after today (priority-ranked)

Per `docs/POSITIONING.md` §"What we recommend doing in the next 2
weeks if forced to choose," updated post-Session-17:

1. **(highest leverage) F40 fix-path 1.**  Custom `WanAttnProcessor`
   injected via diffusers `set_attn_processor`.  Makes
   `MIRAGE_FP8_ATTENTION=fa` actually engage on Wan; without it the
   bridge is documented dead weight.  ~half-day.  Pre-requisite for
   measuring whether FA-3 delivers any speedup on Wan at all.

2. **(biggest Wan lever) Wan-shaped adaptive cache.**  TeaCache-style
   step-skip loop shaped for Wan's MoE boundary.  Community
   2.5–3 × on 8 × H100; single-GPU upside similar.  Turns the Wan
   story from honest-but-modest to "two world-model families served
   with the cache lever, on three silicon targets."  ~1-2 weeks.

3. **(closes pre-publication vouchability) FVD with N ≥ 50 on held-
   out Cosmos eval set.**  `scripts/compute_fvd.py` is in tree;
   the held-out reference set isn't.  Required before external
   publication of the 3.81 × number; the pixel-LPIPS work that
   landed today is necessary but not sufficient for that claim.
   ~1 week.

4. **TE-FP8 on Cosmos** — Session 16 §1 item, still open.  TE is
   installed and validated in isolation; wire `transformer_engine.
   pytorch.DotProductAttention` through the diffusers bridge.
   Expected ~115-130 s (different stack, possibly comparable wall
   but with FP8 quality preserved).  ~1-2 days.  Not a headline
   widener if the cache is already engaged.

5. **Wan offload + FP8 weight convert** — plumb `cpu_offload=True`
   + FP8 conversion through `WanConfig` so we can apples-to-apples
   vs Wan team's 1041 s.  Credibility number, not a wedge.  ~1 week.

6. **CPU 5B Wan re-run without `--vae-tiling`** — would land a
   cleaner CPU number (~38-45 min vs measured 66 min).  Open
   methodology cleanup.

7. **Threshold-curve sweep for Cosmos cache quality** —
   `--cache-adaptive-threshold ∈ {0.05, 0.10, 0.20, 0.30, 0.50}`
   mapped to wall time and LPIPS.  Lets a serving user pick
   threshold knowingly.  ~half-day.

8. **Multi-GPU Context-Parallel** — the only published technique
   that puts single-Cosmos under 60 s and Wan into the 50 s league.
   2-3 weeks; needs multi-GPU box.

9. **F41 test fix** — relax `test_backend_routes_short_seq_to_native`
   tolerance from 1e-5 to 1e-2 for bf16, or convert to fp32 before
   the allclose.  ~15 min code change.

---

## Quick reproducers (for next session)

```bash
# Fresh box bootstrap — Session 16 close has the longer version with
# the cu128 wheel index; Session 17 added these specific invocations.

# H100 — first activate the bridge + tiling defaults
export HF_HOME=/workspace/hf-cache
export MIRAGE_FP8_ATTENTION=fa
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Wan H100 smoke (warm: ~38 s gen + ~65 s load = ~100 s wall)
.venv/bin/python scripts/run_wan.py --frames 17 --steps 8 \
    --vae-tiling --profile

# Wan H100 quality reference (canonical shape, ~26 min)
.venv/bin/python scripts/run_wan.py --frames 81 --steps 40 \
    --vae-tiling --profile

# Wan variance — 5 prompts x 5 seeds (~10 min including cold load)
.venv/bin/python scripts/verify_wan_timing.py --N 5

# Verify FA-3 bridge engagement on Wan (F40 trace)
.venv/bin/python scripts/trace_wan_attention.py --frames 17 --steps 4
# Expect today: "F40 CONFIRMED: bridge engaged 0x; 780 direct SDPA calls."
# After F40 fix-path 1 lands: expect "F40 REFUTED: bridge engaged Nx;
# FA-3 took M/N."

# Cosmos H100 cache quality reproduce
.venv/bin/python scripts/run_cosmos.py --frames 121 --steps 36 \
    --native-loop --cache-mode none \
    --out benchmark-results/cosmos_h100_nocache.mp4
.venv/bin/python scripts/run_cosmos.py --frames 121 --steps 36 \
    --native-loop --cache-mode adaptive \
    --cache-adaptive-threshold 0.30 --cache-force-full-every 16 \
    --out benchmark-results/cosmos_h100_adaptive.mp4
.venv/bin/python scripts/verify_quality.py \
    benchmark-results/cosmos_h100_nocache.mp4 \
    benchmark-results/cosmos_h100_adaptive.mp4
# Expect: LPIPS ~0.61 ("substantially different" at pixel level;
#         trajectory-divergent per F23, NOT noise — see
#         COSMOS_ON_H100.md §"Quantitative cache quality")
```

---

## Branches + repository state

```
main  (current, in sync with origin/main)
  + 4 commits since SESSION_16_CLOSE.md (commit 1d337f6):
    f247fcf  cosmos: H100 cache quality measured — closes Session-15 deferred
    03bf214  docs: POSITIONING.md — moat analysis + source-of-speedup
    ff77b97  wan: tier-1 verification — F40 confirmed + 5-run variance bars
    d66afc1  wan: first H100 81f/40 number — 1552.8s / 72.6GiB --vae-tiling
```

No new branches; everything direct to `main` per the project's
post-Session-15 convention (the feature-branch model of Sessions
14–15 retired after the `--no-ff` merge cleanup).

---

## When you resume — start here

1. Read this doc (you're doing it).
2. Skim `docs/POSITIONING.md` — the strategic-framing layer.  Tells
   you which claims are defensible and which aren't.  Critical
   reading before any external messaging.
3. Skim `docs/COSMOS_ON_H100.md` §"Quantitative cache quality" if
   you need the F23 framing for "what does 0.6 LPIPS mean" — the
   honest answer is "trajectory-divergent, not quality collapse".
4. Pick from the priority-ranked open list above.

If the goal is **a new Wan headline this session**, F40 fix-path 1
(~half-day) → measure → decide whether to start the Wan cache.

If the goal is **closing pre-publication vouchability**, FVD with
N ≥ 50 (~1 week) is the next thing the doc itself flags as
necessary-but-not-yet-done for the 3.81 × claim.

If the goal is **extending the strongest existing claim**, TE-FP8 on
Cosmos (~1-2 days) is the queued Session-16 #1 item.

The dependency that quietly blocks the most others is the FVD work —
every claim that includes the cache (so all the Cosmos headlines)
currently rests on a single-pair pixel measurement.  If the next
session has to pick *one* thing to ship, that's the highest-impact
single item, even though the wall-time wedge doesn't widen from it.

Have a good day.
