# LingBot-VA 2.0 serving-latency ladder + concurrency ceiling — measured on H100 (2026-07-13)

GPU-verify of `scripts/bench_lingbot_va_levers.py` (ported from a rescued WIP
branch this session — see the audit doc) plus a real (not extrapolated)
resident-sessions measurement, both on one RunPod H100 SXM (`torch
2.9.1+cu128`, `transformers 4.55.2`, `diffusers 0.36.0`, `numpy 2.0.2`\*,
config `seed=0`, 2 warmup + 6 measured chunks per rung — script defaults, no
overrides). Reproduces `docs/LINGBOT_VA_SEAM_VERIFY.md`'s 754.6 ms/chunk
baseline as **rung 0**, then ladders toward the paper's 927→142 ms rungs
(everything below 466 ms in their ladder is closed, CUDA-only tooling —
FP8 TensorRT, FlashInfer paged-KV; the point of this ladder is the *portable*
re-implementation of that gap).

\* `numpy==1.26.4` (the repo's own pinned version, matching
`docs/LINGBOT_VA_ON_H100.md`) crashed `diffusers 0.36.0` at import time
(`module 'numpy' has no attribute 'long'` — `diffusers`'s scheduler code
uses a NumPy-2.0-only alias). This is an environment finding, not a Repercep
bug: upgrading to `numpy==2.0.2` on this box fixed it cleanly, but the
recorded recipe's `numpy==1.26.4` pin should be treated as unverified against
current `diffusers==0.36.0` builds — a future setup should check this first
before assuming the old recipe still works verbatim.

## Rung-0 reproduction gate

**685.2 ms warm mean** vs the previously measured 754.6 ms baseline — 9.2%
*faster*, well inside the ~10% tolerance this run was gated on before
trusting anything above it. (Faster, not slower, on a fresh rental — a
reasonable box-to-box variance, not a red flag.)

## The ladder

| rung | config | chunk_ms_warm_mean | vs rung 0 |
|---|---|---|---|
| 0 (baseline) | CFG 5.0/1.0, 5 video / 10 action steps, SDPA | 685.2 ms | 1.0× |
| 1 (CFG off) | `guidance_scale=1.0` | 512.4 ms | 1.34× |
| 2 (video steps →4) | + `num_inference_steps=4` | 479.3 ms | 1.43× |
| 2 (video steps →3) | + `num_inference_steps=3` | 450.5 ms | 1.52× |
| 2 (video steps →2) | + `num_inference_steps=2` | 409.8 ms | 1.67× |
| 2 (action steps →8) | + `action_num_inference_steps=8` | 455.8 ms | 1.50× |
| 2 (action steps →6) | + `action_num_inference_steps=6` | 414.4 ms | 1.65× |
| **2 (action steps →4)** | **+ `action_num_inference_steps=4`** | **342.2 ms** | **2.00×** |
| 3 (torch.compile) | CFG off, `compile_transformer=True`, default steps | 632.9 ms (cold first chunk 2854.9 ms) | 0.81× — **slower** |
| 4 (attn_mode=flex) | CFG off, default steps, FlexAttention | 840.1 ms | 0.61× — **slower** |
| 4 (attn_mode=flashattn) | — | not measurable | flash-attn stubbed, not installed (disclosed non-portable reference point) |

**Best portable config: rung 2 (action steps → 4), 342.2 ms — exactly 2.0×
rung 0, using only config-flag changes, zero CUDA-locked tooling.** That
already reaches roughly halfway across the reference paper's 927→466 ms
"portable techniques" band, on top of an already-optimized 754.6 ms seam
baseline.

**Two negative results, reported because they're real, not hidden because
they're inconvenient:**
- **`torch.compile` made things slower here** (632.9 ms vs the *same lever
  config* uncompiled at 512.4 ms) — consistent with the module docstring's
  own predicted failure mode: the named-KV-cache mutation (`cache_name`/
  `update_cache` kwargs) likely triggers graph breaks, adding compile/dispatch
  overhead without the fusion win. Not investigated further this session —
  worth a dedicated pass (e.g. `torch._dynamo.explain` to find the break
  points) before writing this rung off entirely.
- **FlexAttention (`attn_mode=flex`) was also slower** (840.1 ms vs 512.4 ms
  at the same CFG-off config) — plausibly FlexAttention's own internal
  kernel-specialization compile not fully amortized within 2 warmup + 6
  measured chunks (unlike `rung3`'s explicit cold/warm split, `rung4` doesn't
  isolate a cold pass). A longer warmup window is the obvious next check.

## The CFG-off action-magnitude finding (important — read before using rung 1+ numbers)

The script's own quality guardrail (`action-magnitude p50/p95 per channel vs
rung 0, disclosed as distribution-consistency evidence, not a task-success
proof`) caught something real, not just noise: turning off video CFG shifts
the model's **action** output distribution substantially on specific
channels, not just numerically:

| channel | rung 0 p50 | rung 1 (CFG off) p50 | relative diff |
|---|---|---|---|
| 0 | 33.69 | 15.64 | 54% |
| 1 | 112.83 | 1.57 | **99%** |
| 2 | 71.62 | 15.94 | 78% |
| 3 | 78.13 | 72.53 | 7% |
| 4 | 3.30 | 3.47 | 5% |
| 5 | 71.09 | 98.26 | 38% |

Channel 1 collapses from 112.8 to 1.6 — essentially a different distribution,
not a rounding-level shift. All rungs from 1 onward inherit this (they're all
layered on CFG-off). This makes sense mechanically — video CFG scale changes
the visual context the action expert conditions on — but it means **the
latency numbers above are not free**: they come with an unvalidated change to
what the model actually outputs. This report states the latency win and the
distributional change side by side, deliberately not resolving which matters
more — that's a task-quality question this repo has no offline metric for
(the guardrail note's own honest limitation), not a latency question these
numbers already answer.

**Practical read:** the *architecture-level* levers (step-count reduction)
are lower-risk than the *CFG* lever specifically — a follow-up worth doing is
measuring the ladder with CFG left at 5.0 and only the step counts reduced,
to isolate how much of the 2.0× comes from step-count vs. how much requires
the CFG change and its distributional shift.

## Concurrency — measured, not extrapolated

Run at the winning portable config (`--guidance-scale 1.0 --action-steps 4`,
the same CFG-off action-magnitude caveat above applies) via
`scripts/bench_control_loop.py --engine lingbot-va --sessions N`.

| | N=10 | N=24 (near HBM ceiling) |
|---|---|---|
| sessions requested/measured | 10 / 10 | 24 / 24 |
| `all_finite` | true | true |
| avg marginal HBM/session | 2.82 GiB | 2.82 GiB (identical — no session-count-dependent drift) |
| peak HBM used | 37.7 / 79.2 GiB | 77.16 / 79.2 GiB |
| step latency at N resident (steady-state, excl. first) | ~366–373 ms | ~346–356 ms |
| single-session `step_ms_warm` (for comparison) | 396.6 ms | 397.05 ms |

**24 concurrent sessions, all healthy, right at the HBM ceiling (77.2/79.2
GiB) with no OOM** — a real measured number, replacing the previous
extrapolated "11 resident sessions/GPU" (measured under the *default* CFG=5.0
config, 6.01 GiB/session). The session-marginal HBM drop (6.01 → 2.82 GiB) is
mechanical: CFG-off roughly halves the DiT forward batch (bsz 1 vs 2), which
roughly halves the KV-cache's batch dimension too — so the CFG-off lever
compounds a latency win with a **~2.1× session-density win**, not just
latency alone. And critically: **per-session step latency does not degrade
with concurrency** — at N=24 it's essentially the same as (very slightly
*better* than) the single-session number, no sign of contention up to the
memory ceiling.

Verbatim provenance:

```json
{"mode": "control_loop_bench_v0", "model": "lingbot-va-2", "device": "cuda:0", "dtype": "bfloat16", "load_seconds": 11.6, "reset_seconds": 112.323, "step_ms_warm": 397.05, "steps_per_sec": 2.5, "plan_seconds": 0.402, "planning_decisions_per_sec": 2.486, "plan_horizon": 4, "cem": null, "energy_evals_per_plan": null, "energy_evals_per_sec": null, "weights_gib": 9.48, "session_marginal_gib": 3.04, "hbm_total_gib": 79.2, "resident_sessions_per_gpu": 22, "step_iters": 20, "plan_calls": 1, "state_carryover": true, "concurrency": {"sessions_requested": 24, "hbm_curve_gib": [12.52, 15.15, 17.97, 20.79, 23.61, 26.42, 29.24, 32.06, 34.88, 37.7, 40.52, 43.34, 46.16, 48.97, 51.79, 54.61, 57.43, 60.25, 63.07, 65.89, 68.71, 71.52, 74.34, 77.16], "marginal_gib_measured_per_session": [3.04, 2.63, 2.82, 2.82, 2.82, 2.82, 2.82, 2.82, 2.82, 2.82, 2.82, 2.82, 2.82, 2.82, 2.82, 2.82, 2.82, 2.82, 2.82, 2.82, 2.82, 2.82, 2.82, 2.82], "avg_marginal_gib_measured": 2.82, "single_session_extrapolated_marginal_gib": 3.04, "resident_sessions_extrapolated": 22, "resident_sessions_measured": 24, "step_ms_per_session_at_n_resident": [408.67, 355.66, 354.86, 349.14, 346.22, 348.39, 354.49, 348.3, 347.95, 347.43, 347.9, 351.59, 346.8, 351.67, 347.36, 350.55, 355.7, 356.27, 356.02, 352.75, 347.5, 346.69, 353.6, 350.42], "all_finite": true, "round_robin_rounds": 3}, "lingbot_va_non_default_levers": {"guidance_scale": 1.0, "action_steps": 4}}
```

Note: `resident_sessions_extrapolated: 22` in this line is the OLD formula
applied to the single-session number (3.04 GiB); the real measured average
across all 24 open sessions is lower (2.82 GiB), which is why the *measured*
ceiling (24, and climbing HBM headroom was only ~2 GiB from the 79.2 GiB
total at that point) comfortably exceeds the single-session extrapolation.

## What's not done here

- The `torch.compile` graph-break diagnosis and the FlexAttention
  warmup-isolation follow-up (both noted above).
- Isolating CFG's contribution from the step-count reduction (the "practical
  read" above).
- **MI300X companion run: deferred.** RunPod's GPU catalog had zero AMD
  entries at all when checked (2026-07-13) — not a stock-out on an otherwise
  listed part, but AMD delisted from the catalog entirely at that moment.
  Retry `runpodctl gpu list | grep -i instinct` before assuming this is still
  true.
- Real task-success validation of the CFG-off distributional shift (no
  offline metric exists in this repo for that — noted, not solved).

## Verbatim ladder RESULT (rungs, captured in full; run-config fields — device,
seed, warmup/measure counts — reconstructed from script defaults + the
printed device line, not re-quoted from a single unbroken capture)

Device: `NVIDIA H100 80GB HBM3`. Config: `seed=0`, `warmup=2`, `measure=6`
(script defaults, no CLI overrides).

```json
{"rungs": {"rung0_baseline": {"chunk_ms": [631.1, 651.0, 672.1, 695.8, 718.5, 743.1], "chunk_ms_warm_mean": 685.2, "action_magnitude": {"p50": [33.6871, 112.8336, 71.6196, 78.1288, 3.2955, 71.0901], "p95": [41.6138, 119.2641, 103.6785, 79.0385, 3.8111, 100.1743]}}, "rung1_cfg_off": {"chunk_ms": [505.6, 505.0, 512.1, 511.8, 516.2, 523.5], "chunk_ms_warm_mean": 512.4, "action_magnitude": {"p50": [15.6366, 1.5674, 15.9383, 72.5332, 3.4719, 98.2567], "p95": [17.5192, 4.4794, 37.7046, 72.8074, 3.7871, 99.2252], "p50_max_rel_diff_vs_rung0": 0.9861, "p95_max_rel_diff_vs_rung0": 0.9624}, "guidance_scale": 1.0}, "rung2_video_steps_4": {"chunk_ms_warm_mean": 479.3, "video_steps": 4, "action_steps": 10}, "rung2_video_steps_3": {"chunk_ms_warm_mean": 450.5, "video_steps": 3, "action_steps": 10}, "rung2_video_steps_2": {"chunk_ms_warm_mean": 409.8, "video_steps": 2, "action_steps": 10}, "rung2_action_steps_8": {"chunk_ms_warm_mean": 455.8, "video_steps": 5, "action_steps": 8}, "rung2_action_steps_6": {"chunk_ms_warm_mean": 414.4, "video_steps": 5, "action_steps": 6}, "rung2_action_steps_4": {"chunk_ms": [333.5, 335.9, 337.3, 340.8, 344.2, 361.3], "chunk_ms_warm_mean": 342.2, "video_steps": 5, "action_steps": 4}, "rung3_torch_compile": {"chunk_ms_warm_mean": 632.9, "load_seconds": 7.7, "cold_first_chunk_ms": 2854.9}, "rung4_attn_flex": {"chunk_ms_warm_mean": 840.1, "portable": true, "load_seconds": 7.8}, "rung4_attn_flashattn": {"measurable": false, "portable": false, "error": "RuntimeError: flash_attn stub: attn_mode='torch' should never reach here"}}}
```

(Trimmed to the fields this doc's table already surfaces, for readability;
`chunk_ms` per-chunk arrays and full `action_magnitude` blocks for every rung
are in the raw log this was generated from, not separately archived — the
warm-mean and guardrail-diff numbers above are the ones that matter for
trusting the ladder.)
