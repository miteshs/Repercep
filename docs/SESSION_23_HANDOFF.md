# Session 23 handoff — full force on Lane A (Wan adaptive cache) + FVD N≥1000

**Date:** 2026-05-25 · **HEAD:** added by Session 22's docs-only commit,
on `main`, in sync with `origin/main` after push.

**The Session 23 commitment is explicit and dual-track:**

1. **Lane A — Wan-shaped adaptive cache.** Mirror Mirage's Cosmos
   cache lever onto Wan-2.2.  Validate on TI2V-5B (non-MoE) first;
   then A14B with MoE-aware boundary handling.  Widens the wedge
   from Cosmos-specific to world-model-family-general.
2. **FVD N≥1000.** Bullet-proof every published cache claim against
   the literature-standard distribution arbiter.  Single-MI300X
   budget: ~7 days continuous GPU.  Land the resumable harness
   first, kick off the batch, ride it across the week.

Read `docs/SESSION_22_CLOSE.md` first for the pod state Session 23
inherits.  Read `docs/POSITIONING.md` for the strategic framing both
lanes are pulling on.

---

## Pod state at session start

This is the **MI300X pod** (not the H100 pod Session 21 was written
from).  Session 22 brought it up clean against `docs/COSMOS_ON_MI300X.md`
and verified:

| | |
|---|---|
| GPU | AMD Instinct MI300X VF, 192 GiB HBM3, gfx942 |
| ROCm | 7.2.0 (HIP 7.2.53211) |
| Python | 3.12.3 |
| torch | 2.12.0+rocm7.2 |
| diffusers | 0.37.1 |
| transformers | 5.9.0 |
| `make check-gpu` | green |
| Cosmos cold e2e (121f/36/adaptive thr=0.30) | **420.2 s / 52.5 GiB peak** (reproduces COSMOS_ON_MI300X.md cold budget) |
| AMX CI smoke (BF16/INT8/FP16) | all three build on AMX-masked EMR VM |
| AMX routing tests | 32 passed / 1 skipped |
| HF cache: Cosmos-Predict-7B (~66 GiB) | on disk |
| HF cache: Wan-AI/Wan2.2-TI2V-5B (~32 GiB) | on disk |
| HF cache: Wan-AI/Wan2.2-T2V-A14B (~118 GiB) | **NOT** on disk — stretch |

**Pod quirk to know about:** the agent's shells in Session 22
inherited the pre-`usermod` group set; GPU commands were wrapped in
`sg render -c "sg video -c '...'"`.  A `claude` restart (or any
fresh terminal session) will pick up the render/video groups
properly.  Session 23: prefer restart over `sg` wrapping.

---

## Lane A — Wan-shaped adaptive cache

### Goal

Make Mirage's adaptive cache lever (the one that produces Cosmos's
3.10× wall reduction on MI300X) work on Wan-2.2 — first on the
non-MoE TI2V-5B variant, then on the MoE A14B variant with
boundary-aware behaviour.

POSITIONING.md projects this turns the Wan-A14B MI300X 1700 s
steady-state into ~700–900 s.  More importantly it converts
the cache claim from "Cosmos-specific lever" to
"world-model-family-general lever" — that's the structural moat
piece, not just a number.

### Concrete scope, in execution order

#### A1. Implement `denoise_wan_video` (new function in `src/mirage/runtime/denoise.py`)

Template: `denoise_cosmos_video` in the same file (lines 64-371).
The Cosmos pattern is well-factored: a public wrapper that gates on
`torch.inference_mode()` (don't omit — Cosmos OOMed at 121 f / 36
without this; same risk on Wan), then `_denoise_impl` with the
cache state machine.  Mirror the structure.

**Wan-specific deltas from the Cosmos pattern:**

| Concern | Cosmos | Wan |
|---|---|---|
| Transformer call | one transformer (`pipe.transformer`) | MoE has `pipe.transformer` (high-noise) + `pipe.transformer_2` (low-noise); swap at boundary |
| Boundary signal | n/a | `pipe.config.boundary_ratio` × `pipe.scheduler.config.num_train_timesteps`; if `boundary_ratio is None` (TI2V-5B), use `pipe.transformer` for every step |
| FPS argument | passed to transformer | Wan trained at 16 FPS; pipeline does not accept an FPS override; do not pass |
| Padding mask | `padding_mask` arg to transformer | Wan transformer does not take padding_mask |
| Second guidance scale | n/a | `guidance_scale_2` for the low-noise expert when `boundary_ratio is not None` |
| Latent layout | `(B, C, T, H, W)` | same |
| Scheduler step pattern | two-call (x0_pred, then advance) | check `WanPipeline.__call__` — Session 22 only confirmed `set_timesteps` + `timesteps` shape match |

`scripts/run_wan.py` already exists and drives the diffusers
`WanPipeline.__call__` path; Lane A adds the
`use_native_loop=True` branch.

Reference for the Wan pipeline's MoE handling:
`.venv/lib/python3.12/site-packages/diffusers/pipelines/wan/pipeline_wan.py`
— `__call__` at line 383, MoE branch at lines 577-595 (the
`current_model = self.transformer` / `self.transformer_2` swap is
exactly what `denoise_wan_video` must replicate).

#### A2. Extend `WanConfig` (in `src/mirage/models/wan.py`)

Add fields parallel to `CosmosConfig`:
- `cache_mode: str = "none"` (`"none" | "fixed" | "adaptive"`)
- `cache_adaptive_threshold: float = 0.1`
- `cache_force_full_every: int = 8`

`cache_skip_every` and `use_native_loop` already exist as
forward-compat hooks.

#### A3. Wire `WanEngine.generate` to drive `denoise_wan_video`

When `self._config.use_native_loop`, call `denoise_wan_video(self._pipe, ...)`
instead of the current `self._pipe(**pipe_kwargs)` (in `wan.py` ~line 212).
Plumb the cache config through.  Drop the `guidance_scale_2` injection
into `pipe_kwargs` — the native loop applies it directly on the
low-noise expert call.

#### A4. Unit tests in `tests/test_denoise_wan.py`

Mirror `tests/test_denoise.py` (which is Cosmos-only).  Required:

- `test_denoise_wan_video_runs_under_no_grad_gate` (source inspection
  for `inference_mode`).
- `test_denoise_wan_rejects_unknown_cache_mode`.
- A stub-pipe MoE-boundary test: 8 steps, `boundary_ratio=0.5`,
  verify transformer (high-noise) called for steps 0-3, transformer_2
  (low-noise) called for steps 4-7.
- Adaptive cache full/skip accounting (mirror
  `test_adaptive_cache_logs_full_and_skipped`).

CPU-only stubs as in `test_denoise.py::_make_stub_pipe`; no
GPU needed.

#### A5. Smoke validate on Wan TI2V-5B (already downloaded)

```bash
# Smoke: 17 f / 8 step / native loop / no cache yet
sg render -c "sg video -c 'PYTHONPATH=$(pwd)/src \
    .venv/bin/python -u scripts/run_wan.py \
        --small --frames 17 --steps 8 \
        --native-loop --cache-mode none'"

# Then: same shape with adaptive cache
sg render -c "sg video -c 'PYTHONPATH=$(pwd)/src \
    .venv/bin/python -u scripts/run_wan.py \
        --small --frames 17 --steps 8 \
        --native-loop --cache-mode adaptive \
        --cache-adaptive-threshold 0.30 --cache-force-full-every 4'"
```

Expect smoke wall in the same regime as the diffusers path
(currently ~45 s on the H100 pod's Wan TI2V smoke per
`docs/WAN_ON_MI300X.md`; MI300X TI2V smoke isn't separately
measured yet but should be in the same ballpark).  `--small` in
`scripts/run_wan.py` is the flag that routes to TI2V-5B; check
the script's argparse if unsure.

#### A6. Headline at Wan's reference shape (TI2V-5B first, A14B is stretch)

TI2V-5B has no MoE boundary, so it exercises the cache logic without
the boundary-swap branch.  Once it lands, A14B at 81f/40/1280×720
(`docs/WAN_ON_MI300X.md` § "Available reference numbers") is the
real headline target:

```bash
# A14B headline — needs ~118 GiB download first
sg render -c "sg video -c 'PYTHONPATH=$(pwd)/src \
    .venv/bin/python -u scripts/run_wan.py \
        --frames 81 --steps 40 --height 720 --width 1280 \
        --native-loop --cache-mode adaptive \
        --cache-adaptive-threshold 0.30 --cache-force-full-every 8'"
```

Target: Wan A14B headline drops from 1700 s steady-state (the
projected Session 11 number) to ~700–900 s.  That's a structural
result; the absolute number depends on how aggressive the
threshold can be while preserving quality.  Per
`COSMOS_ON_MI300X.md` § "Quantitative cache quality" the
threshold-vs-quality curve is flat (LPIPS shifts by 0.14 across
0.05 → 0.50); 0.30 is the doc-recommended mid-point.

### Lane A risks

- **The Wan MoE boundary swap interacts with adaptive cache
  bookkeeping.**  The accumulated rel-L1 distance reset on a full
  forward — does it reset across the boundary handoff, or carry?
  Currently undefined.  Recommendation: reset on boundary, treat the
  first low-noise step as if it were a warmup step.  Document the
  choice in `denoise_wan_video`'s docstring.
- **Per-expert cache state.**  Cosmos has one cached `noise_pred`;
  Wan needs to decide whether high-noise and low-noise share or
  split the cache slot.  Same MoE-step inputs flow through a *
  different* transformer once the boundary crosses, so reusing the
  high-noise cached output on a low-noise step is wrong by
  construction.  Recommendation: hard reset on boundary crossing.
- **`pipe.scheduler` step pattern.**  Cosmos's two-call x0 +
  advance pattern may not match Wan.  Read `pipeline_wan.py
  __call__` carefully before writing the loop.

---

## FVD N≥1000

### Goal

Replace COSMOS_ON_MI300X.md's current N=5 multi-prompt FVD (166.3
single number, "preliminary" caveat) with an N≥1000 number — the
literature-standard bar.  This is the credibility piece every
external reviewer will probe first per POSITIONING.md.

### Budget reality (re-state from Session 22 close)

Per-pair generation cost on this MI300X VF using
`COSMOS_ON_MI300X.md` warmup-separated numbers:

- no-cache: ~465 s
- adaptive thr=0.30: ~151 s
- **per prompt (one of each side)**: ~10.3 min

| N | GPU-hours | Wall (100% util) |
|---:|--:|---|
| 100 | 17 | overnight |
| 250 | 43 | 2 days |
| **1000** | **172** | **~7 days continuous** |
| 10000 | 1700 | not single-VF achievable |

Session 23 should target **N=100 inside this session's window,
N=1000 as a continuous batch ridden across the week.**

### Concrete scope, in execution order

#### F1. Read the existing FVD harness

`scripts/compute_fvd.py` (452 lines), `scripts/verify_quality.py`
(163 lines), `scripts/eval_cpu_quality.py`.  Session 22 listed them
but did not read them in detail.  Two questions to answer before
coding the batch harness:

- Does `compute_fvd.py` accept a list of reference clips + a list
  of candidate clips and emit a single FVD number?  (Skim
  argparse + main.)
- How does the I3D feature extractor handle different clip
  resolutions / lengths?  Cosmos outputs are 121 f @ 1280×704;
  the I3D backbone wants 8-frame windows.  How does the harness
  chunk?

#### F2. Generate the 1000-prompt corpus

Save to `docs/eval/fvd_prompts_v1.jsonl` (or `eval/` at repo
root if `docs/eval/` doesn't fit the layout):

```json
{"id": 0, "prompt": "...", "seed": 0, "generator_version": "fvd_prompts_v1"}
{"id": 1, "prompt": "...", "seed": 1, "generator_version": "fvd_prompts_v1"}
...
```

Generate deterministically: write a small `scripts/generate_fvd_prompts.py`
that takes a single seed + count + saves to JSONL.  Use a published
template structure (e.g., "[subject] [action] in [setting], [style]"
across 1000 (subject, action, setting, style) tuples drawn from
fixed lists).  The corpus must be reproducible from
`scripts/generate_fvd_prompts.py --seed 0 --count 1000`.

Do **not** use MSR-VTT or external corpora directly — license
ambiguity and reviewer-noise risk per Session 22's reasoning.
Synthetic + reproducible is the right tradeoff for a quality
distribution metric.

#### F3. Build `scripts/fvd_batch.py` — the resumable batch harness

Required behaviour:

- Reads `eval/fvd_prompts_v1.jsonl`.
- For each prompt, generates two clips (no-cache + adaptive)
  to `eval/clips/no_cache/{id:04d}.mp4` and
  `eval/clips/adaptive_thr30/{id:04d}.mp4`.
- After each clip writes, **fsync** and write a sibling
  `.done` marker.  On startup, skip any (prompt, mode) pair where
  the marker exists.
- Survives `SIGTERM` between clips; never leaves a half-written mp4
  with a `.done` marker.
- Logs per-clip wall time + peak HBM to `eval/run_log.jsonl` for
  the eventual variance analysis.
- Exit code 0 on completion of the prompt range; non-zero on
  unrecoverable error.

This is ~150-200 lines of Python.  Pattern-match the structure of
`scripts/run_cosmos.py`.

#### F4. Kick off the batch and ride it

```bash
sg render -c "sg video -c 'PYTHONPATH=$(pwd)/src \
    HF_TOKEN=$HF_TOKEN \
    nohup .venv/bin/python -u scripts/fvd_batch.py \
        --prompts eval/fvd_prompts_v1.jsonl \
        --range 0:1000 \
        --output eval/clips/ \
        >eval/run.log 2>&1 &'"
```

Or use a `tmux` / `screen` session if `nohup`-with-`sg` proves
fragile.  The batch will eat the GPU for ~7 days at full
utilization; no other GPU work should run on this pod during that
window.

#### F5. Incremental FVD at milestones

Compute FVD over completed clips at N=10, 50, 100, 250, 500, 1000.
The FVD trajectory is itself interesting: at what N does the number
stabilize?  Answer goes into a new `docs/FVD_ON_MI300X.md` or
extends `docs/COSMOS_ON_MI300X.md` § "Quantitative cache quality".

```bash
# Example for the N=100 milestone
.venv/bin/python scripts/compute_fvd.py \
    --reference 'eval/clips/no_cache/{0000..0099}.mp4' \
    --candidate 'eval/clips/adaptive_thr30/{0000..0099}.mp4' \
    --output eval/fvd_N100.json
```

(Wire exact CLI to whatever `compute_fvd.py` accepts.)

### FVD risks

- **HF cache disk usage.**  At 121 f / 1280×704 mp4, each clip is
  ~2 MiB.  N=1000 × 2 sides = 2000 clips × 2 MiB = ~4 GiB of mp4.
  Pod has 607 GiB free; comfortable.  But if `eval/clips/` lands on
  a small root partition, redirect to `~/.cache/mirage/eval/` or
  similar.
- **Pod preemption / disconnects.**  The 7-day batch must survive
  agent restarts.  `nohup` + the `.done` checkpoint design handles
  this; document the resume command in `eval/RUN.md` (or wherever).
- **GPU contention.**  No other work on this pod's GPU during the
  batch.  Lane A's smoke runs (a few minutes) can interrupt at
  prompt boundaries by stopping `fvd_batch.py`, doing the smoke,
  and re-launching — the `.done` checkpoint makes this safe.

---

## Sequencing recommendation

Lane A and FVD share the GPU; sequence to maximize Lane A's
chance of landing inside a normal session window.

1. **Lane A first, end-to-end, including A5 TI2V-5B smoke.**
   ~half-day of CPU code work + a few minutes of GPU smoke.  Land
   Lane A scaffolding + tests + TI2V-5B validation.
2. **FVD scaffold next** (F1-F3, ~half-day CPU only — no GPU
   contention with Lane A).  Land
   `scripts/generate_fvd_prompts.py`, `eval/fvd_prompts_v1.jsonl`,
   `scripts/fvd_batch.py`.
3. **Kick off FVD batch (F4)** at the end of the session window;
   commit the scaffolding and let the GPU run.
4. **A14B headline (A6) is *not* on the same pod as the FVD batch**
   — they'd compete for GPU.  Either: (a) defer A14B to a sister
   pod with ~118 GiB free + A14B downloaded, or (b) pause the FVD
   batch for the ~30 min A14B run + resume.  Recommendation: (a) if
   a second pod is available; (b) otherwise.

---

## Lower-priority items inherited from Session 21

These do not block Session 23's commitment but are still open and
should be visible:

5. **Install `flash-attn` on the H100 pod + re-run F45 trace.**
   Counter flips from `→ SDPA = 960` to `→ FA-3 = 960`.  Half-day.
6. **Wan-2.2-A14B 81f/40 re-verification on H100 stack.**
   Session 17's 1552.8 s / 72.6 GiB number — not re-verified on
   current torch+diffusers.  118 GiB download.
7. **F40 fix-path coverage for VAE + text encoder.**  Defer until
   flash-attn lands on H100.
8. **Bare-metal AMX measurement.**  Still hardware-blocked.  This
   pod's AMX-masked KVM does the CI compile-smoke (Session 22
   re-confirmed), not headline numbers.
9. **Ada FP8 wiring decision.**  Deprioritized per user direction
   (Session 20).
10. **Production serving driver hardening.**  Different axis per
    POSITIONING.md.

---

## Quick reproducers (Session 22's verified set)

```bash
cd /home/mshah/Mirage
export HF_TOKEN=$(grep '^export HF_TOKEN' ~/extra.sh | sed 's/.*HF_TOKEN=//')

# Cosmos cold e2e on this pod (420.2 s / 52.5 GiB on first invocation)
sg render -c "sg video -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$(pwd)/src HF_TOKEN=$HF_TOKEN \
    .venv/bin/python -u scripts/run_cosmos.py \
        --frames 121 --steps 36 --native-loop \
        --cache-mode adaptive --cache-adaptive-threshold 0.30 \
        --cache-force-full-every 16 \
        --out benchmark-results/cosmos_warm_check.mp4'"

# AMX CI smoke
MIRAGE_AMX_FORCE_BUILD=1 make kernels-cpu
.venv/bin/python -m pytest tests/test_attention_cpu.py \
    tests/test_amx_int8.py tests/test_amx_fp16.py -q

# Wan TI2V-5B diffusers path (no native loop yet — baseline before Lane A lands)
sg render -c "sg video -c 'PYTHONPATH=$(pwd)/src HF_TOKEN=$HF_TOKEN \
    .venv/bin/python -u scripts/run_wan.py --small --frames 17 --steps 8'"
```

---

## When you resume — start here

1. Read this doc.
2. Skim `docs/SESSION_22_CLOSE.md` for the pod state.
3. **Lane A:** start with A1 (`denoise_wan_video`).  Template is
   `denoise_cosmos_video` in `src/mirage/runtime/denoise.py`; the
   Wan-specific deltas are in the table above.
4. **FVD:** F1 first (read `compute_fvd.py` + `verify_quality.py`),
   then F2 (generator script + corpus), then F3 (batch harness).
5. Kick off F4 at the end of the session; let it ride.
6. A14B stretch (A6) only if a sister pod is available or the FVD
   batch can be paused cleanly.

Have a good week.
