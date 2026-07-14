# Control-Loop Serving Benchmark — v0 (2026-07-11)

The leaderboard for the workload robots actually run: **closed-loop world-model
serving** — persistent state, action in, next state out, plan under a latency
budget. Generation benchmarks (frames/sec, denoise steps/sec) do not measure
this regime; as of v0, nobody publishes it. This doc defines the metrics, the
rules, and seeds the tables with everything measured so far. Harness:
`scripts/bench_control_loop.py` (one verbatim `RESULT` JSON line per run —
the provenance convention of `docs/METHODOLOGY.md`).

Voice check (same as METHODOLOGY): every number below says whether it is
**measured-by-us** or an **external claim**, and cold vs warm. Where a
comparison is apples-to-oranges we say so in place.

---

## 1. The four metrics

| # | Metric | Definition | What does NOT count |
|---|---|---|---|
| 1 | **Closed-loop step latency** (ms, warm) | Wall time of one `step(state, action)` on a *persistent session* — the state carries across calls (no re-encode, no session rebuild). Mean over ≥20 steps after ≥3 warmup steps. | Re-encoding the observation each step (that's a fresh-session benchmark); cold first-step kernel compiles (report separately if notable). |
| 2 | **Planning-decisions/sec** | Full `plan()` calls per second: everything from "state + goal in" to "next action out", including all internal rollouts/denoise loops. | Amortizing across pre-computed plans; returning a cached action. |
| 3 | **Energy-evals/sec** | For energy-based planners (V-JEPA-AC class): candidate-rollout energy evaluations per second inside `plan()` (= samples × CEM-iters / plan seconds). `null` for policy-mode engines (LingBot-VA class) — they don't search, they generate. | Counting predictor forwards (that's samples × iters × horizon — a different number; we report ms/forward separately where measured). |
| 4 | **Resident sessions/GPU** | Measured marginal HBM of one live session (peak − post-load baseline), extrapolated: `(HBM_total − weights) // session_marginal`. Weights counted once (they're shared). | Dividing total HBM by total peak (double-counts weights); assuming sessions share KV/state (v0 sessions are independent). |

**State-carryover rule (the regime gate):** a submission must keep one session
alive across the timed steps. This is what request/response generation servers
structurally don't express, and it is the entire point of the leaderboard.

**Warm vs cold:** all headline numbers are warm (post-warmup). ROCm first-forward
kernel compiles are ~3× the warm cost (measured, `BENCH_2026_06_RUNPOD.md`
gotcha 3); cold numbers go in footnotes, not headlines.

---

## 2. Leaderboard — measured by us

All rows: RunPod, single GPU, bf16 unless noted, verbatim RESULT provenance in
the source docs.

### V-JEPA 2-AC (300M AC predictor, ViT-g encoder, 8-frame ≈ 2048-token context) — energy-MPC regime

Source: `docs/BENCH_2026_06_RUNPOD.md` (2026-06-26). Predictor runs fp32 (RoPE
upcast; the bf16 path is the active optimization workstream).

| metric | H100 | MI300X |
|---|---|---|
| closed-loop step latency (warm) | **70.8 ms** (14.1 steps/s) | **75.8 ms** (13.2 steps/s) |
| planning-decisions/sec (H=4, 64×3 CEM, sequential fp32 baseline) | 0.018 (55.6 s/plan) | 0.015 (67.2 s/plan) |
| planning-decisions/sec (H=4, candidate-batched, fp32) | 0.029 (35.1 s/plan, 1.6×) | 0.031 (32.1 s/plan, 2.1×) |
| **planning-decisions/sec (H=4, candidate-batched + bf16)** | **0.106 (9.42 s/plan, 7.3×)** | **0.093 (10.81 s/plan, 6.3×)** |
| energy-evals/sec (H=4, batched fp32) | 5.5 | 6.0 |
| session marginal HBM | ~3.4 GiB | ~3.7 GiB |
| resident sessions/GPU (est.) | ~20 | **~50** |

**Lever ladder, same box** (`docs/LEVERS_2026_07_H100.md`, 2026-07-11 — a
different H100 rental/stack than the row above, so its own sequential-fp32
baseline (68.84 s) differs in absolute terms from the row's 55.6 s; the
*ratios* are the result and are what the bf16 row above carries forward):
sequential fp32 68.84 s → **+batching** 40.58 s (1.7×) → **+bf16** (parity
exact to bf16 resolution) **9.42 s (7.3×)**. Warm-start (1-iter steady-state
replan) reaches lower energy (30.25) than a cold 3-iter plan (45.0) at ⅓ the
per-plan work, ≈3.1 s/replan estimated (not yet timed directly — see KV-reuse
work below).

**MI300X companion run (2026-07-11, `docs/LEVERS_2026_07_H100.md` §"MI300X
companion run"):** sequential fp32 (June baseline, reused) 68.44 s →
**+batching** 29.77 s (2.3×) → **+bf16** (parity 0.0059 rel, bf16-resolution
noise) **10.81 s (6.3×)** — bf16 parity is architecture-general as expected,
though the batching/bf16 split of the total win differs by vendor (MI300X
gets more from batching alone, less incremental from bf16, than H100).
**One real discrepancy, not smoothed over:** MI300X's warm-start energy
ordering did *not* reproduce H100's — cold-1-iter (50.25) beat warm-1-iter
(58.0) here, the opposite of H100 (warm 30.25 < cold 40.25). Same default
seed, different box; treat as an open question (RNG-sensitive vs. a real
per-vendor effect) rather than a confirmed cross-silicon finding, pending a
multi-seed rerun.

The remaining gap to sub-second cold / ~100 ms-class warm planning is
KV/latent reuse across rollout steps and CEM candidates — designed and
GPU-verified for the growing-window case, sliding-window needs an explicit
approximation decision (`docs/adr/0009-kv-latent-reuse.md`); productization is
the active workstream, next row lands here once measured.

### LingBot-VA 2.0 base (MoT DiT over Wan2.2 latents, 32 actions/chunk) — policy regime

Two sources, both H100, both 2026-07-11: the reference `wan_va` stack through
our own timed driver (`docs/LINGBOT_VA_ON_H100.md`) and — since the Phase-1
port landed same day — the **shared `bench_control_loop.py` harness** driving
the real weights through the Repercep seam (`docs/LINGBOT_VA_SEAM_VERIFY.md` has
the raw imagination-mode rollout this corroborates). "Step" for a chunked
video-action model = one chunk (4 latent frames, 32 actions).

| metric | H100 (reference stack) | H100 (Repercep seam, shared harness) | MI300X (Repercep seam, shared harness) |
|---|---|---|---|
| chunk latency (warm) | 1384.7 ms (CFG 5.0, SDPA) | **754.6 ms** (1.3/s) | **1198.1 ms** (0.8/s) |
| planning-decisions/sec (1 chunk = 1 decision) | 0.72 (derived) | **1.09** (0.92 s/plan, measured) | **0.77** (1.29 s/plan, measured) |
| energy-evals/sec | n/a (policy regime) | n/a | n/a |
| one-time weight load | — | 9.48 GiB | 9.48 GiB |
| session marginal HBM | 38.8 GiB (undifferentiated) | **6.01 GiB** | **6.05 GiB** |
| resident sessions/GPU | **1** (est., undifferentiated) | **11** (measured: `(79.2 − 9.48) // 6.01`) | **30** (measured: `(192.0 − 9.48) // 6.05`) |

MI300X run (2026-07-11/12, RunPod, torch 2.9.1+rocm6.3, `scripts/bench_control_loop.py
--engine lingbot-va --backend rocm`): per-session HBM cost is nearly identical
to H100 (6.05 vs 6.01 GiB) — the 30-vs-11 resident-sessions gap is a capacity
story (MI300X's 192 GiB vs H100's 80 GiB), not a per-session efficiency win.
Chunk latency is 1.59× slower than the H100 seam number, consistent with the
MI300X-vs-H100 gap seen elsewhere in this doc. **Caveat:** `reset_seconds`
measured 76.5 s on this run (vs H100's sub-second reset) — almost certainly
ROCm cold first-forward kernel compilation (§1 "Warm vs cold": ROCm first-
forward compiles run ~3× warm cost elsewhere in this repo's numbers; this one
is larger, plausibly compiling the full VAE+T5+transformer pipeline for the
first time), not a steady-state cost — reported as measured, not corrected,
pending a warm-reset rerun to isolate it. Verbatim provenance:

```json
{"mode": "control_loop_bench_v0", "model": "lingbot-va-2", "device": "rocm:0", "dtype": "bfloat16", "load_seconds": 67.1, "reset_seconds": 76.532, "step_ms_warm": 1198.07, "steps_per_sec": 0.8, "plan_seconds": 1.291, "planning_decisions_per_sec": 0.7747, "plan_horizon": 4, "cem": null, "energy_evals_per_plan": null, "energy_evals_per_sec": null, "weights_gib": 9.48, "session_marginal_gib": 6.05, "hbm_total_gib": 192.0, "resident_sessions_per_gpu": 30, "step_iters": 20, "plan_calls": 1, "state_carryover": true}
```

The seam column isn't "faster because different work" — same imagination-mode
rollout, matching action-magnitude distribution (`LINGBOT_VA_SEAM_VERIFY.md`
§"why"). The gap is a design choice (T5/VAE kept CPU-resident, no reference-
server debug I/O in the hot loop) that also **separates one-time weight cost
from marginal session cost** — which is *why* resident-sessions jumps from an
undifferentiated "1" to a measured "11": the reference number conflated
weights (9.5 GiB) with session state into one 38.8 GiB blob per session.

**Lever ladder + measured concurrency ceiling, same H100 seam** (2026-07-13,
`docs/LINGBOT_VA_LEVERS_2026_07.md` — full ladder, negative results, and an
important caveat, read before quoting these in isolation): portable
config-flag levers alone (no CUDA-locked tooling) take the 754.6 ms row above
to **342.2 ms (2.0×)** — CFG off + reduced action-denoise steps. That same
lever config also **roughly halves session memory** (6.01 → 2.82 GiB
measured), which compounds into a **measured (not extrapolated) 24 resident
sessions on one H100**, up from the 11 above, with per-session step latency
holding steady (no concurrency penalty) all the way to the HBM ceiling.
**The catch, stated plainly:** the CFG-off lever measurably shifts the
model's action-output distribution on specific channels (one channel's p50
drops ~99%, others 5–78%) — the latency and memory wins are real and
measured; whether the resulting policy is still task-correct is not
validated (no offline task-success metric exists in this repo). Two rungs
tried and reported as **negative results**, not hidden: `torch.compile` and
FlexAttention both came out *slower* than the CFG-off baseline at the same
config, plausibly compile/graph-break overhead not amortized in this run's
warmup window — a follow-up, not resolved here.

### DreamZero-DROID (16.5B autoregressive world-action model, Wan2.1 DiT + action/state registers) — policy regime

H100 only so far (MI300X deferred — see §5), 2026-07-14, real DROID camera
frames (extracted from the research clone's own bundled 419-frame debug
episode via `scripts/extract_droid_debug_frames.py`, not random pixels —
`docs/DREAMZERO_PORT_PLAN.md`'s standing synthetic-input caveat is now
resolved). "Step"/chunk = one block (2 latent frames, 24 actions,
`used_action_dim=8` wire width). All numbers below are **true full-16-step
compute** (`num_dit_steps=16`, the config default) — see the levers doc
(`docs/DREAMZERO_LEVERS_2026_07.md`) for why this is NOT what the reference's
own "~3s/chunk H100" claim measures.

| metric | H100 (Repercep seam) |
|---|---|
| chunk latency (warm, full 16-step) | **5890.6 ms** (0.2/s) |
| planning-decisions/sec | **0.165** (6.06 s/plan) |
| energy-evals/sec | n/a (policy regime) |
| one-time weight load | 42.78 GiB |
| session marginal HBM (full attention window) | **22.88 GiB** |
| resident sessions/GPU | **1** — confirmed both by extrapolation (`(79.2−42.78)//22.88`) AND empirically: a live 2-session run OOM'd mid-forward-pass at true full compute (§ below) |

Verbatim provenance (single-session baseline, `--warmup 12` to fill the
~10-block/21-frame attention window before measuring so
`session_marginal_gib` reflects steady state, not an under-warmed number):

```json
{"mode": "control_loop_bench_v0", "model": "dreamzero-droid", "device": "cuda:0", "dtype": "bfloat16", "load_seconds": 286.1, "reset_seconds": 5.882, "step_ms_warm": 5890.61, "steps_per_sec": 0.2, "plan_seconds": 6.057, "planning_decisions_per_sec": 0.1651, "plan_horizon": 4, "cem": null, "energy_evals_per_plan": null, "energy_evals_per_sec": null, "weights_gib": 42.78, "session_marginal_gib": 22.88, "hbm_total_gib": 79.2, "resident_sessions_per_gpu": 1, "step_iters": 20, "plan_calls": 1, "state_carryover": true, "concurrency": null}
```

**The "H100 tops out at ~1 session" prediction (port plan §4/§5) is
confirmed, not just arithmetic:** a `--sessions 2` run at this same full-16-
step config OOM'd (`CUDA out of memory`, 79.14/79.18 GiB in use) partway
through the round-robin phase — two live sessions' KV caches genuinely do not
fit alongside the 42.78 GiB weights on an 80 GiB H100 once both approach the
full attention window. This is the ~5× LingBot-VA per-session memory cost the
port plan's back-of-envelope (~30 GB/session) flagged as a risk, landing
close to the estimate (22.88 GiB measured, same order of magnitude). MI300X
(192 GiB) is the natural next data point for whether this becomes a
multi-session story there — deferred, see §5.

**`release()`/`close()` leak check: passed** (3 open/release cycles,
`engine._sessions`/`pipeline._sessions` both empty after every release) —
with one nuance worth carrying forward: GPU memory does **not** drop
immediately at `release()` time (stayed flat across all 3 cycles in the
check) — `close()` only clears bookkeeping; the model's own live cache
attributes (`WANPolicyHead.kv_cache1` etc., plain mutable instance attributes,
no per-session native keying) are only overwritten — and thus freed — the
next time a session's `_swap_in` runs. Not a leak (confirmed no growth across
cycles), but reclaim is deferred to next-use, not eager.

**Serving-latency ladder: `docs/DREAMZERO_LEVERS_2026_07.md`** — the full
16-step baseline above is 3.2× the reference's own undisclosed 8-step
default (3068.8 ms), and the dynamic DiT-cache schedule reaches **1780.2 ms**
(3.2× the true baseline, beating even the most aggressive static preset).

---

## 3. External reference — LingBot-VA 2.0 paper, Table 3 (NOT measured by us)

Their acceleration ladder (GPU **unspecified** in the paper; async scheme, K=32
actions/chunk, control Hz = (1000/t_chunk)×32):

| rung | ms/chunk | async Hz | portable? |
|---|---|---|---|
| bf16 PyTorch async baseline | 927 | 35 | yes (eager PyTorch) |
| + consistency distillation | 466 | ~69 | yes (fewer denoise steps) |
| + FP8 TensorRT engines | 369 | ~87 | **no — CUDA-locked** |
| + paged/ragged KV + FlashInfer | 272 | ~118 | **no as shipped** (FlashInfer is CUDA); the *technique* is portable |
| + runtime-overhead reduction | 142 | 225 | technique portable |

Apples-to-apples note: our measured 1384.7 ms vs their 927 ms baseline differs
by (a) CFG 5.0 — the released demo config doubles every transformer forward;
their deployment path is distilled/CFG-free, (b) SDPA vs flash-attn, (c) sync
vs async accounting. The open-source release ships **only the eager bf16
rung**; every rung below 466 ms is closed, CUDA-only tooling. **The portable
re-implementation of those rungs is the vendor-neutral serving gap this
project exists to fill — roughly a 10× window on this model.**

---

## 4. Submission rules (v0)

1. Run `scripts/bench_control_loop.py` (or an equivalent driver for stacks the
   engine seam doesn't wrap yet) and publish the verbatim `RESULT` line.
2. State carryover required (§1); warmup ≥3, timed steps ≥20.
3. Disclose: GPU + stack versions, dtype, attention path, CFG scale, denoise
   step counts, sync/async accounting, cold-start costs if >2× warm.
4. Cross-vendor rows must disclose the version matrix (see
   `BENCH_2026_06_RUNPOD.md` for the MI300X torch/ROCm traps).

## 5. Open items

- ~~MI300X LingBot-VA + V-JEPA rows~~ — done 2026-07-11/12 (LingBot-VA row
  above; V-JEPA row + ladder in `docs/LEVERS_2026_07_H100.md` §"MI300X
  companion run"). One open thread from that run: the MI300X warm-start
  energy ordering did not reproduce H100's (see that doc) — needs a
  multi-seed rerun before treating either ordering as settled.
- V-JEPA rows re-measured on the batched+bf16 path (done —
  `docs/LEVERS_2026_07_H100.md`, 7.3x combined) and on KV-reuse
  (design done — `docs/adr/0009-kv-latent-reuse.md`; real-predictor GPU-verify
  still open) — update this leaderboard's V-JEPA table with those rows next.
- ~~LingBot-VA through the Repercep seam~~ — done 2026-07-11
  (`docs/LINGBOT_VA_SEAM_VERIFY.md`); both regimes now run under the same
  `bench_control_loop.py` binary (§2 table above).
- ~~Multi-session concurrency measurements~~ — done for LingBot-VA/H100
  2026-07-13 (`docs/LINGBOT_VA_LEVERS_2026_07.md`): 24 real sessions measured
  via `bench_control_loop.py --sessions N`, not extrapolated, with a
  per-session HBM curve and round-robin step latency showing no concurrency
  penalty up to the HBM ceiling. V-JEPA-AC concurrency (this metric for the
  energy-MPC regime) is still extrapolated only — open.
- LingBot-VA serving-latency ladder — done 2026-07-13
  (`docs/LINGBOT_VA_LEVERS_2026_07.md`): portable levers alone reach 2.0× the
  H100 seam baseline (754.6→342.2 ms), with an important caveat (the winning
  CFG-off lever measurably shifts the action-output distribution, not
  validated for task correctness) and two disclosed negative results
  (`torch.compile`, FlexAttention both came out slower here). MI300X
  companion run for this ladder deferred — RunPod's GPU catalog had zero AMD
  entries at all when checked (delisted, not merely out of stock).
- **DreamZero-DROID H100 row + levers ladder + release-check — done
  2026-07-14** (row above; ladder in `docs/DREAMZERO_LEVERS_2026_07.md`).
  Real DROID camera frames (no longer synthetic). Confirmed empirically (not
  just extrapolated) that H100 tops out at 1 resident session at full
  compute. **MI300X row: not yet attempted** — the natural next question is
  whether DreamZero's much larger per-session HBM cost (22.88 GiB vs
  LingBot-VA's 6.01 GiB) turns MI300X's extra headroom (192 vs 80 GiB) into
  a multi-session story the way it wasn't quite for LingBot-VA (30 vs 11, a
  capacity story either way) — check `runpodctl gpu list | grep -i instinct`
  for current AMD availability first.
- **DreamZero parity vs. the reference server — attempted, deferred.**
  The reference's own `GrootSimPolicy` wrapper reads the same checkpoint
  config our pipeline does and calls the same `WANPolicyHead.
  lazy_joint_video_action`, so a construction probe was attempted
  (2026-07-14, capped ~30 min): it requires `groot.vla.data.transform.
  ComposedModalityTransform`, exactly the GR00T-N1.5 dataset-schema stack
  the port plan (§5) deliberately chose not to vendor. Construction hit a
  real multi-step dependency chain (`tianshou` API mismatch — pin `0.5.1`,
  not the PyPI-default `2.0.1` which removed `tianshou.policy.BasePolicy` —
  then a missing `albumentations`, likely more beyond that) rather than
  resolving in the capped budget. Confirms the original vendoring boundary
  was the right call, not just caution; a future session's parity check
  should budget for standing up that transform stack properly rather than
  attempting it as a quick add-on.
