# Session 25 handoff — Reactor response: LingBot-VA port, latency levers, KV-reuse (2026-07-11)

- **Branch:** `feat/lingbot-va-port` (9 commits, pushed to origin, not merged to
  `main`, no PR opened yet).
- **Trigger:** a VC asked about reactor.inc; this session executed the dated
  action plan from `docs/COMPETITIVE_RESPONSE_REACTOR_2026_07.md`.
- **Spend:** ~4 RunPod H100 sessions today, all terminated at session end
  (see §6). No MI300X spend (blocked on stock all day).

---

## 1. What shipped (all committed, all GPU-verified where numbers are quoted)

### Reactor competitive response
`docs/COMPETITIVE_RESPONSE_REACTOR_2026_07.md` — verified teardown (their
docs, not press): hosted real-time-video API, zero action/robotics surface
today, convergence risk = video-action models (Ant/Robbyant's LingBot-VA,
announced the same week). Dated 6-week commitments made here are what this
session executed against.

### LingBot-VA 2.0 on the Repercep seam
- `src/repercep/models/lingbot_va.py` + `lingbot_va_pipeline.py` — second model
  on `InteractiveWorldModel`, policy-regime planning (generates its own
  actions) vs. V-JEPA's CEM search — two regimes, one Protocol.
- **H100 numbers, three independent measurements, all in docs/:**
  - Reference `wan_va` stack (their server code): 1384.7 ms/chunk
    (`LINGBOT_VA_ON_H100.md`).
  - Through the Repercep seam, raw driver: 801.5 ms/chunk, 15.4 GiB/session
    (`LINGBOT_VA_SEAM_VERIFY.md`) — faster/lighter by design (CPU-resident
    T5/VAE, no reference-server debug I/O), not a shortcut — same
    imagination-mode rollout, matching action-magnitude distribution.
  - **Formal shared-harness number (the leaderboard row):** 754.6 ms/chunk,
    1.09 plans/sec, **11 resident sessions/GPU** (`CONTROL_LOOP_BENCH.md`) —
    far better than the reference stack's undifferentiated "1 session"
    because the harness separates one-time weight load (9.48 GiB) from
    marginal session cost (6.01 GiB).
- Two real bugs found and fixed along the way (both documented in commit
  messages / `LINGBOT_VA_SEAM_VERIFY.md`): a `flash_attn` import stub needs a
  real `importlib.machinery.ModuleSpec` (bare `ModuleType` breaks diffusers'
  lazy loader); the model has **two action dimensionalities** (30 internal
  padded channels vs. 6 executed/wire channels for the demo task) —
  `LingBotVAConfig.used_action_dim` carries the latter.
- MI300X row: **deferred**, not attempted — RunPod had zero MI300X stock all
  day. Recipe for when it returns is in `repercep-lingbot-va-port` memory.

### V-JEPA 2-AC latency levers (batching + bf16)
`docs/LEVERS_2026_07_H100.md` — GPU-verified same-box ladder: sequential fp32
68.8s → batched fp32 40.6s (1.7×) → **batched bf16 9.42s (7.3× total)**. bf16
energy parity with fp32 exact to bf16 resolution. Warm-start: a 1-iteration
steady-state replan beats a cold 3-iteration plan (lower energy, ⅓ the work).
Code: `src/repercep/models/vjepa2_ac.py` (`plan_batched`, `predictor_compute_dtype`,
`plan_warm_start` config knobs + `_rollout_energy_batched`, the SDPA dtype
harmonizer).

### Control-Loop Serving Benchmark v0 (the leaderboard)
`scripts/bench_control_loop.py` + `docs/CONTROL_LOOP_BENCH.md` — defines the
four metrics (step latency under state carryover, planning-decisions/sec,
energy-evals/sec, resident sessions/GPU), drives either engine through one
binary. Fixed a real bug mid-session: the harness's goal-computation step was
silently making LingBot-VA's timed `plan()` calls free cache hits instead of
real chunk denoises — caught with a test that fails without the fix
(`tests/test_bench_control_loop.py`).

### KV/latent reuse (ADR-0009) — the deepest work, a real finding not a clean win
`docs/adr/0009-kv-latent-reuse.md` — engine-side seam (`_CachedPredictor`
Protocol, `step()`'s cache-aware path, a found-and-fixed CEM-branch-safety
hazard) is implemented and CPU-tested against a fake predictor. **Then wrapped
the real torch-hub V-JEPA 2-AC predictor against real pretrained weights on
H100** and found, with rigorous layer-by-layer verification (not assumed):
1. The model's own RoPE has a maintainer-acknowledged bug (non-composable
   rotation) that invalidates the ADR's original "free O(1) shift" design.
   Fixable — cache pre-rotation K, re-derive rotation fresh each step. Still
   O(window), verified bit-exact for the growing-window case.
2. **Eviction (the sliding-window case) is structurally unrecoverable from a
   K/V cache alone** for a full-depth causal transformer — verified
   layer-by-layer: layer 0 matches a fresh recompute exactly, layer 1+
   diverges ~11× immediately, at every sampled layer through 23. This is the
   well-known hard problem StreamingLLM/attention-sinks exist to solve, not a
   bug in this implementation.

**Net:** growing-window KV reuse (session/plan ≤ `context_frames`) is real
and delivers the O(window²)→O(window) win. Sliding-window reuse (indefinitely
long control loops — the ADR's original target use case) needs a separate,
explicit approximation decision not made this session.

---

## 2. What's NOT done / open

- **MI300X rows** (LingBot-VA control-loop bench, V-JEPA lever ladder) —
  blocked on RunPod stock. No watcher currently armed (stopped per user
  request 2026-07-11; re-arm or check `runpodctl gpu list | grep MI300X`
  manually next session).
- **KV-reuse sliding-window case** — needs an explicit design decision
  (attention-sinks, accept-and-measure drift, or full-recompute-on-evict)
  before any real implementation. Not started. See ADR-0009 §"Revisit if".
- **LingBot-VA grounded recondition** — only imagination-mode rollout has
  been exercised through the seam; real mid-session observation recondition
  (the pipeline supports it) is untested.
- **Multi-session concurrency** — resident-sessions numbers are extrapolated
  from one live session's marginal HBM, not load-tested with N concurrent
  sessions.
- **`_rollout_energy_batched` KV-reuse** (the CEM-candidate shared-prefix
  dimension) — design-only, needs a harder paged-cache design; not attempted.
- **PR / merge to main** — branch is pushed, not merged. No PR opened.
- **VC reply** — not drafted this session; the ammunition (this branch's
  docs) is ready whenever that's wanted.

---

## 3. Where to pick this back up

**If continuing the Reactor response:** the 6-week commitments in
`COMPETITIVE_RESPONSE_REACTOR_2026_07.md` are substantially met (latency
levers measured, second model on the seam, benchmark published) except the
MI300X row and the KV-reuse sliding-window story. Consider whether the VC
reply should go out now (strong numbers already) or wait for MI300X.

**If continuing KV-reuse:** read ADR-0009 in full before touching code — it
has the exact verification methodology (SDPA-capture trick to get bit-exact
ground truth, the layer-by-layer isolation technique) that would be needed
again for any attention-sinks or approximate-eviction design.

**If picking up MI300X:** `repercep-lingbot-va-port` and
`repercep-cem-batching-finding` memories have the recipes (image, torch/ROCm
version pins, known traps).

---

## 4. Key docs index (this session's output)

| Doc | What it is |
|---|---|
| `docs/COMPETITIVE_RESPONSE_REACTOR_2026_07.md` | The plan this session executed |
| `docs/LINGBOT_VA_PORT_PLAN.md` | LingBot-VA port scoping |
| `docs/LINGBOT_VA_ON_H100.md` | Reference-stack H100 numbers |
| `docs/LINGBOT_VA_SEAM_VERIFY.md` | Repercep-seam H100 numbers + why faster |
| `docs/LEVERS_2026_07_H100.md` | V-JEPA batching+bf16 ladder |
| `docs/CONTROL_LOOP_BENCH.md` | The leaderboard (both engines, all rows) |
| `docs/adr/0009-kv-latent-reuse.md` | KV-reuse design + GPU-verify findings |

## 5. Memory index pointers

`repercep-lingbot-va-port.md`, `repercep-cem-batching-finding.md`,
`reactor-inc-competitor.md` all updated this session — read those first for
fast context reload.

## 6. Infrastructure state at pause

All RunPod pods terminated this session (confirmed below). No background
watchers armed. No cron/scheduled jobs running.
