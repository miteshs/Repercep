# Session 13 close — next-session pickup

**Date:** 2026-05-24 · **HEAD on `origin/main`:** `a14faf7` ·
**Branch:** `main` · **Working tree:** clean (modulo gitignored
`.claude/` worktrees)

Read this *before* `docs/HANDOFF.md`. This doc is the focused "what
happened today, what to do next" cut; `HANDOFF.md` is the full
orientation layer.

---

## What landed today (Sessions 12 + 13)

The verification campaign is **closed**. The numbers we ship in
`docs/COSMOS_ON_MI300X.md` and `docs/HANDOFF.md` are independently
reproduced on a clean GPU and characterised across multiple prompts.

### Headline numbers (settled — won't move materially without a different stack)

| Config | Wall (s) | vs H100 ref (~380 s) | Peak HBM |
|---|--:|--:|--:|
| 121 f / 36 / no-cache (baseline) | **469.84 ± 0.32** (N=5 prompts) | 0.81× | 52.5 GiB |
| 121 f / 36 / adaptive thr=0.30 | **154.48 ± 5.96** (N=5 prompts) | **2.46×–2.51×** | 52.5 GiB |
| 121 f / 36 / adaptive thr=0.30 + tuned FP8 | **142.0** (seed 0; 141.7 in worktree) | **2.68×** | 52.5 GiB |
| Wan-2.2-T2V-A14B 17 f / 8 step smoke | 326 s cold / 45 warm | — | 84.3 GiB |
| Wan-2.2-T2V-A14B 81 f / 40 step quality (projected steady-state) | ~1700 s | ~1.6× behind H100's 1041 s (FP8 + offload) | 85.1 GiB |

### Quality framing (settled, honest)

- **Adaptive caching produces trajectory-divergent valid Cosmos
  generations**, not pixel-equivalent ones vs no-cache.
- Multi-prompt LPIPS vs no-cache (5 pairs): **mean 0.616 ± 0.069**,
  range [0.53, 0.71]. Range is small across prompts; the divergence is
  consistent, not catastrophic on any single prompt.
- Multi-prompt FVD (5 pairs × 8 clips/video = 40 features/side, I3D
  R50): **166.3** preliminary, with the loud `N < 50` warning. FVD
  literature uses N ≥ 1000; we have N=5. Pattern (monotone with
  threshold, stable across prompts) is defensible; the absolute number
  is preliminary.
- Threshold curve: no setting recovers no-cache pixel-equivalence; the
  knob tunes magnitude of divergence within the cached regime. See the
  table in `docs/COSMOS_ON_MI300X.md`.

### Determinism (settled)

- Seed-0 adaptive output is **MD5-identical across 4 sessions** (`94852d9d…`).
- FP8 wiring transition is reflected in the bytes (Session 9 = same
  MD5 as no-FP8 baseline because the env var was a no-op then;
  Session 10–13 = distinct MD5 because FP8 actually routes now).

---

## OSS announce — drafts are committed, ready to ship

In this session you initially descoped OSS announce, then asked for the
drafts to be committed. They are at:

- `docs/RELEASE_NOTES_v0.1.md` — GitHub release notes
- `docs/ANNOUNCEMENT.md` — X / Twitter thread, HN angle, blog post
  outline, publication checklist

**Neither is published.** To ship:

1. Bump `Cargo.toml` workspace `version` and `pyproject.toml`
   `[project] version` from `0.0.1` → `0.1.0`.
2. `git tag -a v0.1.0 -m 'Repercep Runtime v0.1.0' && git push origin v0.1.0`.
3. Cut the GitHub release from the tag; paste `RELEASE_NOTES_v0.1.md`.
4. Post the X thread first; let it settle 24 h; then HN; then blog if
   long-form is wanted.
5. When the predictable "but H100 with the same stack…" pushback
   appears, the right link to drop is `docs/METHODOLOGY.md`.

---

## What's open after today (priority-ranked)

1. **FVD at N ≥ 1000.** The harness (`scripts/compute_fvd.py`) is
   ready. What's missing is a held-out Cosmos eval set — needs ~1000
   no-cache generations across diverse prompts/seeds. At 470 s/gen on
   one MI300X that's 130 GPU-hours; a real eval campaign. This is the
   right "is the cache distribution-equivalent" test; without it the
   166.3 number stays preliminary.

2. **HIP FP8 kernel correctness.** Triton wins on perf today (1.16×
   over SDPA at the Cosmos production shape after autotune); the HIP
   `v_mfma_f32_16x16x32_fp8_fp8` scaffold compiles + loads but the
   operand-register layout is incomplete (output values are wrong).
   Triton is the right path until shapes Triton can't tune for show up;
   HIP is the long-term option.

3. **Continuous batching** (Phase 2 plan residual). The Stage 4 v2
   serving path lays the substrate: `Router → Scheduler → engine
   driver`. What's missing is multi-instance engine multiplexing —
   the driver currently consumes one request at a time. A real
   continuous-batching implementation would need temporal-dependency-
   aware grouping, which is non-trivial for diffusion.

4. **Action conditioning hooks** (Phase 2 plan residual). Robotics-
   OEM-facing surface. Deferred per the plan until a robotics design
   partner is in the pipeline.

5. **Bare-metal MI300X validation.** Our measurements are on a VF
   slice. We *believe* single-tenant VF performance matches bare-
   metal; not independently verified.

6. **OSS announce push** (drafts committed; user's call when to ship).

---

## Quick reproducers (for next session)

```bash
# Smoke that the environment still works
make lint typecheck test
sg render -c "sg video -c 'make check-gpu'"

# Headline (the 142-s, 2.68× claim)
sg render -c "sg video -c '\
    REPERCEP_FP8_ATTENTION=1 .venv/bin/python scripts/run_cosmos.py \
        --frames 121 --steps 36 --native-loop \
        --cache-mode adaptive --cache-adaptive-threshold 0.30 \
        --cache-force-full-every 16'"
# Expect: generate_seconds ≈ 142, peak_hbm_gib = 52.5

# Quality: LPIPS adaptive vs no-cache (the load-bearing comparison)
.venv/bin/python scripts/verify_quality.py \
    benchmark-results/cosmos_no_cache_clean.mp4 \
    benchmark-results/cosmos_adaptive_fp8_tuned_clean.mp4 \
    --device cpu
# Expect: LPIPS ≈ 0.64 "substantially different"

# FVD on the 5-pair set
.venv/bin/python scripts/compute_fvd.py \
    --reference benchmark-results/verify_baseC_no_cache_p*.mp4 \
    --candidates benchmark-results/verify_baseB_p*.mp4 \
    --num-clips 8 --device cpu
# Expect: FVD ≈ 166 (with loud N < 50 warning)

# Multi-prompt timing variance (the verify_timing harness)
sg render -c "sg video -c '\
    .venv/bin/python scripts/verify_timing.py --skip-phase-a --prompts 5'"
# Expect: mean ≈ 154 s, std ≈ 6 s

# Threshold sweep (any single threshold)
for thr in 0.05 0.10 0.20 0.30 0.50; do
    sg render -c "sg video -c '\
        .venv/bin/python scripts/run_cosmos.py \
            --frames 121 --steps 36 --native-loop \
            --cache-mode adaptive --cache-adaptive-threshold $thr \
            --cache-force-full-every 16 \
            --out benchmark-results/cosmos_thr${thr}.mp4'"
done
# Expect: 291, 228, 177, 151, 126 s respectively
```

---

## Commits this session (Session 13, all on `origin/main`)

```
a14faf7  docs: add OSS announcement drafts (RELEASE_NOTES + ANNOUNCEMENT)
67067df  Session 13 — threshold sweep + multi-prompt variance + 5-pair FVD
a5c8d22  verification: scripts/compute_fvd.py — FVD harness with small-N caveat
c43f22e  docs/HANDOFF: refreshed for next-session pickup            ← (session-12 close)
```

Plus what carried over from Session 12: the timing-verification commit
`231ee3e` + the original verification scripts (`a24a2dc`, `2cbf88d`).

---

## When you resume — start here

1. Read this doc (you're doing it).
2. Read `docs/HANDOFF.md` §"TL;DR for a new session" (it's already
   pointing at the right open items; this doc supersedes it on the
   ranking).
3. Pick **one** of the open items above. None require a context-warming
   ramp; each can be a self-contained session.
4. If you publish the OSS announce, drop `docs/METHODOLOGY.md` as the
   link to anyone who raises the "but H100 with the same stack..."
   pushback — it's the most rigorous part of the writeup and pre-empts
   the predictable critique.

Have a good Monday.
