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
| planning-decisions/sec (H=4, 64×3 CEM, sequential) | 0.018 (55.6 s/plan) | 0.015 (67.2 s/plan) |
| planning-decisions/sec (H=4, candidate-batched) | 0.029 (35.1 s/plan, 1.6×) | 0.031 (32.1 s/plan, 2.1×) |
| energy-evals/sec (H=4, batched) | 5.5 | 6.0 |
| session marginal HBM | ~3.4 GiB | ~3.7 GiB |
| resident sessions/GPU (est.) | ~20 | **~50** |

The 55→32 s plan numbers are the honest state of energy-MPC serving today —
far from a 10–100 ms control budget. That gap **is the roadmap** (KV/latent
reuse across rollout steps, bf16 predictor, CEM warm-start), and this
leaderboard is where each lever's gain lands as a measured row.

### LingBot-VA 2.0 base (MoT DiT over Wan2.2 latents, 32 actions/chunk) — policy regime

Source: `docs/LINGBOT_VA_ON_H100.md` (2026-07-11, reference `wan_va` stack
through our timed driver; Mirage-seam rows land with the Phase-1 port).
"Step" for a chunked video-action model = one chunk (4 latent frames).

| metric | H100 | MI300X |
|---|---|---|
| chunk latency (warm, CFG 5.0, SDPA) | **1384.7 ms** | *pending (RunPod MI300X stock)* |
| synchronous actions/sec | 23.1 | *pending* |
| planning-decisions/sec (1 chunk = 1 decision) | 0.72 | *pending* |
| energy-evals/sec | n/a (policy regime) | — |
| session HBM (30-chunk KV window, CFG batch 2) | **38.8 GiB** | *pending* |
| resident sessions/GPU (est.) | **1** | ~4 (extrapolated from H100; unverified) |

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

- MI300X LingBot-VA rows (blocked on RunPod MI300X availability, 2026-07-11).
- V-JEPA rows re-measured on the bf16 predictor path + KV-reuse when those
  levers land (the workstream this leaderboard exists to score).
- LingBot-VA through the Mirage seam (Phase-1 port) so both regimes run under
  the same harness binary.
- Multi-session concurrency measurements (metric 4 is extrapolated today;
  measure N live sessions when the serving layer exposes it).
