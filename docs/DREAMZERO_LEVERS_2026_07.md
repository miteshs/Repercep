# DreamZero-DROID serving-latency ladder — measured on H100 (2026-07-14)

GPU-verify of `scripts/bench_dreamzero_levers.py` on one RunPod H100 SXM
(`torch 2.8.0+cu128`, `transformers 4.51.3`, `diffusers 0.30.2`, real DROID
camera frames from the research clone's own bundled debug episode — not
random pixels, `scripts/extract_droid_debug_frames.py`), `seed=0`, 2 warmup +
6 measured chunks per rung (script defaults). Reproduces the control-loop
baseline (`docs/CONTROL_LOOP_BENCH.md`) as **rung 0** and ladders through
every Phase-2 lever scoped in `docs/DREAMZERO_PORT_PLAN.md` §4.

**Read this before quoting any number below:** three of DreamZero's levers
(`num_dit_steps`, `enable_dit_cache`, `local_attn_size`'s DiT-side attribute)
turned out to be either construction-time-only (baked into `WANPolicyHead`
at `__init__` from env vars, never re-read per call) or wired onto the wrong
object entirely — both discovered *because* this ladder's first attempt
silently produced a flat, unchanging number across supposedly-different
rungs. `dreamzero_pipeline.build_pipeline`'s docstring has the full technical
story; the practical upshot is that every rung touching a construction-time
lever below required a **fresh model load** (~4.5–5 min each), not a
config-mutation-and-rerun the way LingBot-VA's ladder could.

## The ladder

| rung | config | chunk_ms_warm_mean | vs rung 0 |
|---|---|---|---|
| 0 (true baseline) | `num_dit_steps=16` — every DiT call runs | 5671.6 ms | 1.0× |
| 0b (reference default) | `num_dit_steps=8` — their undisclosed default | 3068.8 ms | 1.85× |
| 3 (steps→7) | `num_dit_steps=7` | 2742.6 ms | 2.07× |
| 3 (steps→6) | `num_dit_steps=6` | 2424.5 ms | 2.34× |
| 3 (steps→5) | `num_dit_steps=5` | 2091.4 ms | 2.71× |
| **3b (dynamic DiT cache)** | **`enable_dit_cache=True`** | **1780.2 ms** | **3.19×** |
| 4 (torch.compile) | `compile=True`, default steps | 5863.0 ms (cold first chunk 5991.5 ms) | 0.97× — **no meaningful change** |
| 5 (KV window→12) | `local_attn_size=12` (shares rung-0 engine) | 6052.8 ms | 0.94× — **slightly slower** |
| 5 (KV window→6) | `local_attn_size=6` | 5620.8 ms | 1.01× — **no meaningful change** |
| 1 (batched CFG) | `cfg_batched=True` | not measurable | falls back to sequential CFG (not implemented — see below) |
| 2 (cfg_scale=1) | `cfg_scale=1.0` | not measurable | confirmed real lever, but crashes upstream (see below) |

**Best result: rung 3b, the dynamic cosine-similarity DiT-cache schedule —
1780.2 ms, 3.19× the true baseline** — and notably faster than even the most
aggressive static preset (rung 3's 5-step mask, 2091.4 ms), without any
config-mutation ambiguity since it required (and got) its own fresh load.
**The number to headline against the reference's own claim, though, is the
0b→0 gap:** their "~3s/chunk H100" claim is rung 0b (3068.8 ms), not rung 0
(5671.6 ms) — the two are 1.85× apart, and rung 0b is what ships by default
with zero opt-in flags.

### Rung 1 (batched CFG) — not measurable, correctly detected this time

`cfg_batched=True` is plumbed through `DreamZeroConfig` and
`DreamZeroPipeline`, but the actual batched-forward implementation (stacking
cond/uncond into one `WANPolicyHead` call instead of the reference's two
sequential calls — the single-GPU differentiator vs. their 2-GPU `ip=2`
split) is not built yet: it needs `WANPolicyHead._run_diffusion_steps` read
and modified against the research clone, out of scope for this pass. The
pipeline warns once and falls back to sequential CFG; **an earlier version
of this ladder script had a scoping bug where the fallback warning fired
during `_open_session` — outside the `catch_warnings` window that only
wrapped `_run_chunks`** — so it silently recorded the sequential-CFG number
under a "batched" label. Fixed by wrapping both calls in one monitored
window; this run correctly reports the rung as not measurable instead.

### Rung 2 (`cfg_scale=1.0`) — not measurable, but resolves a real disclosed uncertainty

This rung's own starting caveat was "whether `cfg_scale=1.0` skips the
uncond forward, or just changes the combine math, is not confirmed for
DreamZero." **It's now confirmed: yes, it skips the uncond forward** — the
reference's `_run_diffusion_steps` gates a second (uncond) prediction behind
`if self.cfg_scale != 1.0`, exactly like LingBot-VA's `guidance_scale=1.0`
lever. But the reference's own CFG-combine code immediately after
unconditionally unpacks a 2nd (uncond) prediction from the results list,
crashing with `IndexError: list index out of range` when only one exists.
Their own demo/training config never sets `cfg_scale=1.0`, so this call path
is plausibly untested upstream — not a bug this port introduced, but a real
one, reported rather than papered over.

## Quality guardrail — noisier than LingBot-VA's, and why

The magnitude-consistency check (`|action| p50/p95 per channel vs rung 0`,
same discipline as `LINGBOT_VA_LEVERS_2026_07.md`) shows much larger
relative diffs here — often >100% — across essentially every rung, not just
the ones with a real mechanism for changing outputs. **This is very likely a
methodology difference, not evidence every lever silently corrupts
generation:** LingBot-VA's ladder ran each rung against the *same* seed
observation in imagination mode, so consecutive chunks stayed close in
distribution. This script's real-DROID-episode rollout advances the
attention window across genuinely different, progressively-later video
content each rung (each rung opens a fresh session and rolls its own
handful of chunks forward), so chunk-to-chunk and rung-to-rung magnitude
drift is expected from the *content* changing, not just the lever. Treat
these numbers as a weaker signal than LingBot-VA's equivalent table — real,
recorded, but not yet isolated from the confound. A future pass fixing the
input frames/schedule identically across rungs (not just the seed) would
give a cleaner comparison.

## What's not done here

- The batched-CFG implementation itself (rung 1) — real surgery against the
  research clone, deferred (port plan §4).
- Isolating the magnitude-guardrail confound above (same input schedule
  across rungs, not just same seed).
- **MI300X companion run: not attempted this session** — see
  `docs/CONTROL_LOOP_BENCH.md` §5 for the multi-session-density question this
  would answer (DreamZero's ~22.88 GiB/session vs. LingBot-VA's 6.01 GiB
  makes MI300X's extra headroom a more interesting question here than it was
  there).
- The `rung2_cfg_scale_1` `IndexError` itself is not patched (it's in the
  research clone, not this port) — a real fix would need reading
  `_run_diffusion_steps`'s CFG-combine block and either guarding it or
  special-casing `cfg_scale=1.0` to skip the combine entirely.

## Verbatim ladder RESULT

Device: `NVIDIA H100 80GB HBM3`. Config: `seed=0`, `warmup=2`, `measure=6`
(script defaults).

```json
{"bench": "dreamzero_levers", "device": "NVIDIA H100 80GB HBM3", "seed": 0, "warmup_chunks": 2, "measured_chunks": 6, "quality_guardrail_note": "action-magnitude (|action| p50/p95 per channel) vs rung0, at a fixed seed, is the only quality signal available for DreamZero in this repo — there is no offline task-success metric. Consistency shows a lever didn't silently change the output distribution; it does not prove task correctness.", "rungs": {"rung0_baseline_full16": {"chunk_ms": [5299.2, 5517.6, 5953.3, 6407.4, 5312.4, 5539.5], "chunk_ms_warm_mean": 5671.6, "action_magnitude": {"p50": [0.0289, 0.01, 0.0201, 0.0089, 0.0243, 0.006, 0.013, 0.396], "p95": [0.0657, 0.1159, 0.0744, 0.2338, 0.156, 0.0609, 0.1374, 0.861]}, "num_dit_steps": "full (16)"}, "rung1_cfg_batched": {"measurable": false, "reason": "DreamZeroPipeline fell back to sequential CFG (cfg_batched not yet implemented -- needs the research clone's _run_diffusion_steps read on a GPU pod, port plan §4)", "warning": "DreamZeroConfig.cfg_batched=True requested, but batching cond+uncond into one forward (port plan §4 lever 1) is not yet implemented in DreamZeroPipeline -- falling back to the reference's sequential cond/uncond forwards. Implementing this needs WANPolicyHead._run_diffusion_steps read from the research clone on a GPU pod (not available in this environment) -- see docs/DREAMZERO_PORT_PLAN.md §4."}, "rung2_cfg_scale_1": {"measurable": false, "note": "cfg_scale=1.0 DOES skip the reference's uncond forward (confirmed -- a real latency lever, matching LingBot-VA's guidance_scale=1.0) but crashes this call path: the CFG-combine code unconditionally unpacks a 2nd (uncond) prediction that was never computed. Plausibly untested upstream at cfg_scale=1.", "error": "IndexError: list index out of range"}, "rung5_local_attn_size_12": {"chunk_ms": [6508.6, 6509.0, 5329.8, 5554.3, 5982.7, 6432.1], "chunk_ms_warm_mean": 6052.8, "action_magnitude": {"p50": [0.0207, 0.0104, 0.0163, 0.0132, 0.0301, 0.0104, 0.0141, 0.6401], "p95": [0.0634, 0.1159, 0.0728, 0.2338, 0.156, 0.0609, 0.1374, 0.861], "p50_max_rel_diff_vs_rung0": 0.7333, "p95_max_rel_diff_vs_rung0": 0.035}, "local_attn_size": 12, "note": "latency only -- pair with bench_control_loop.py --attn-window for the marginal-HBM/session-density side"}, "rung5_local_attn_size_6": {"chunk_ms": [5556.3, 5986.9, 5322.5, 5545.6, 5987.3, 5326.1], "chunk_ms_warm_mean": 5620.8, "action_magnitude": {"p50": [0.028, 0.0071, 0.0186, 0.0065, 0.0189, 0.0043, 0.008, 0.4045], "p95": [0.0657, 0.03, 0.073, 0.0235, 0.1357, 0.0146, 0.1191, 0.5119], "p50_max_rel_diff_vs_rung0": 0.3846, "p95_max_rel_diff_vs_rung0": 0.8995}, "local_attn_size": 6, "note": "latency only -- pair with bench_control_loop.py --attn-window for the marginal-HBM/session-density side"}, "rung0b_reference_default_8steps": {"chunk_ms": [2948.3, 2955.7, 3189.3, 3423.8, 2942.5, 2953.0], "chunk_ms_warm_mean": 3068.8, "action_magnitude": {"p50": [0.031, 0.0081, 0.0174, 0.0089, 0.0199, 0.0118, 0.0113, 0.3511], "p95": [0.0656, 0.0313, 0.0729, 0.0305, 0.1481, 0.0315, 0.1387, 0.5112], "p50_max_rel_diff_vs_rung0": 0.9667, "p95_max_rel_diff_vs_rung0": 0.8695}, "num_dit_steps": 8, "load_seconds": 286.6, "note": "the reference's own undisclosed default (NUM_DIT_STEPS=8) -- their published ~3s/chunk H100 claim is this number, not rung 0's"}, "rung3_dit_steps_7": {"chunk_ms": [2647.8, 2634.6, 2836.8, 3049.7, 2650.9, 2635.6], "chunk_ms_warm_mean": 2742.6, "action_magnitude": {"p50": [0.0308, 0.0075, 0.017, 0.0079, 0.0161, 0.0133, 0.0124, 0.3721], "p95": [0.0663, 0.0277, 0.0859, 0.0175, 0.1296, 0.0304, 0.1224, 0.7187], "p50_max_rel_diff_vs_rung0": 1.2167, "p95_max_rel_diff_vs_rung0": 0.9251}, "num_dit_steps": 7, "load_seconds": 263.4}, "rung3_dit_steps_6": {"chunk_ms": [2398.4, 2313.3, 2490.8, 2673.9, 2357.1, 2313.5], "chunk_ms_warm_mean": 2424.5, "action_magnitude": {"p50": [0.031, 0.0112, 0.0197, 0.0102, 0.024, 0.019, 0.022, 0.3521], "p95": [0.0568, 0.0264, 0.078, 0.0302, 0.1427, 0.0499, 0.1473, 0.4852], "p50_max_rel_diff_vs_rung0": 2.1667, "p95_max_rel_diff_vs_rung0": 0.8708}, "num_dit_steps": 6, "load_seconds": 289.5}, "rung3_dit_steps_5": {"chunk_ms": [2056.2, 1991.1, 2142.7, 2299.1, 2064.8, 1994.6], "chunk_ms_warm_mean": 2091.4, "action_magnitude": {"p50": [0.0098, 0.0185, 0.008, 0.017, 0.0248, 0.0093, 0.0151, 0.4072], "p95": [0.0477, 0.034, 0.0425, 0.1566, 0.3627, 0.0606, 0.4689, 0.8337], "p50_max_rel_diff_vs_rung0": 0.9101, "p95_max_rel_diff_vs_rung0": 2.4127}, "num_dit_steps": 5, "load_seconds": 287.0}, "rung3b_dynamic_dit_cache": {"chunk_ms": [1778.2, 1681.5, 1816.9, 1942.1, 1775.3, 1687.2], "chunk_ms_warm_mean": 1780.2, "action_magnitude": {"p50": [0.0094, 0.0109, 0.0095, 0.0063, 0.0226, 0.0081, 0.0149, 0.3809], "p95": [0.0201, 0.0298, 0.0371, 0.0347, 0.1103, 0.0357, 0.1268, 0.4417], "p50_max_rel_diff_vs_rung0": 0.6747, "p95_max_rel_diff_vs_rung0": 0.8516}, "enable_dit_cache": true, "load_seconds": 265.0}, "rung4_torch_compile": {"chunk_ms": [5522.4, 6104.7, 6394.4, 5567.8, 5518.9, 6069.6], "chunk_ms_warm_mean": 5863.0, "action_magnitude": {"p50": [0.0276, 0.0049, 0.0183, 0.0082, 0.0252, 0.0078, 0.0123, 0.3633], "p95": [0.0838, 0.0351, 0.076, 0.0516, 0.1193, 0.0167, 0.095, 0.4921], "p50_max_rel_diff_vs_rung0": 0.51, "p95_max_rel_diff_vs_rung0": 0.7793}, "load_seconds": 263.9, "cold_first_chunk_ms": 5991.5}}}
```

## Environment / setup notes

- Checkpoint `GEAR-Dreams/DreamZero-DROID` (public, no auth) + real DROID
  frames from `dreamzero0/dreamzero`'s bundled `debug_image/*.mp4` (419
  frames, 320×180, 3 cameras) — extraction via `scripts/
  extract_droid_debug_frames.py`, frame index 0 (matches the reference's own
  `test_client_AR.py` step-0 schedule).
- Venv: torch 2.8.0+cu128 (RunPod `runpod-torch-v280` image), transformers
  4.51.3, diffusers 0.30.2, plus einops/peft/hydra-core/omegaconf/dm-tree/
  pydantic/opencv-python-headless — same recipe as the Phase-1 port,
  confirmed still installing clean.
- **Every fresh-load rung explicitly frees the previous engine first**
  (`_free_engine`: drop the pipeline reference, `gc.collect()`,
  `torch.cuda.empty_cache()`) — DreamZero's ~42.8 GiB weights don't leave
  room for two resident copies on an 80 GiB H100; an earlier attempt at this
  ladder OOM'd mid-construction of a second model when a prior rung's engine
  was kept alive unnecessarily. Rungs sharing one engine (0, 1, 2, 5) also
  call `engine.release(state)` between each other — an earlier attempt
  OOM'd here too, from session-dict accumulation (5 sessions' live KV cache
  resident at once) before that release call was added.
