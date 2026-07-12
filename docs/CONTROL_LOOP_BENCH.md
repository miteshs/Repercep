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
| **planning-decisions/sec (H=4, candidate-batched + bf16)** | **0.106 (9.42 s/plan, 7.3×)** | *pending* |
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
work below). MI300X pending; bf16 parity is architecture-general so the ROCm
run is expected to land the same multiplier.

The remaining gap to sub-second cold / ~100 ms-class warm planning is
KV/latent reuse across rollout steps and CEM candidates — designed and
GPU-verified for the growing-window case, sliding-window needs an explicit
approximation decision (`docs/adr/0009-kv-latent-reuse.md`); productization is
the active workstream, next row lands here once measured.

### LingBot-VA 2.0 base (MoT DiT over Wan2.2 latents, 32 actions/chunk) — policy regime

Two sources, both H100, both 2026-07-11: the reference `wan_va` stack through
our own timed driver (`docs/LINGBOT_VA_ON_H100.md`) and — since the Phase-1
port landed same day — the **shared `bench_control_loop.py` harness** driving
the real weights through the Mirage seam (`docs/LINGBOT_VA_SEAM_VERIFY.md` has
the raw imagination-mode rollout this corroborates). "Step" for a chunked
video-action model = one chunk (4 latent frames, 32 actions).

| metric | H100 (reference stack) | H100 (Mirage seam, shared harness) | MI300X |
|---|---|---|---|
| chunk latency (warm) | 1384.7 ms (CFG 5.0, SDPA) | **754.6 ms** (1.3/s) | *pending (RunPod stock)* |
| planning-decisions/sec (1 chunk = 1 decision) | 0.72 (derived) | **1.09** (0.92 s/plan, measured) | *pending* |
| energy-evals/sec | n/a (policy regime) | n/a | — |
| one-time weight load | — | 9.48 GiB | *pending* |
| session marginal HBM | 38.8 GiB (undifferentiated) | **6.01 GiB** | *pending* |
| resident sessions/GPU | **1** (est., undifferentiated) | **11** (measured: `(79.2 − 9.48) // 6.01`) | *pending* |

The seam column isn't "faster because different work" — same imagination-mode
rollout, matching action-magnitude distribution (`LINGBOT_VA_SEAM_VERIFY.md`
§"why"). The gap is a design choice (T5/VAE kept CPU-resident, no reference-
server debug I/O in the hot loop) that also **separates one-time weight cost
from marginal session cost** — which is *why* resident-sessions jumps from an
undifferentiated "1" to a measured "11": the reference number conflated
weights (9.5 GiB) with session state into one 38.8 GiB blob per session.

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

- MI300X LingBot-VA + V-JEPA rows: RunPod stock flapped fully unavailable
  most of 2026-07-11, then returned same day — a run against the existing
  scripts (this section's rows do not yet reflect it; check the session's
  handoff doc for the latest MI300X numbers before assuming this row is
  current).
- V-JEPA rows re-measured on the batched+bf16 path (done —
  `docs/LEVERS_2026_07_H100.md`, 7.3x combined) and on KV-reuse
  (design done — `docs/adr/0009-kv-latent-reuse.md`; real-predictor GPU-verify
  still open) — update this leaderboard's V-JEPA table with those rows next.
- ~~LingBot-VA through the Mirage seam~~ — done 2026-07-11
  (`docs/LINGBOT_VA_SEAM_VERIFY.md`); both regimes now run under the same
  `bench_control_loop.py` binary (§2 table above).
- Multi-session concurrency measurements (metric 4 is extrapolated from a
  single session's marginal HBM today; measure N live sessions when the
  serving layer exposes it).
