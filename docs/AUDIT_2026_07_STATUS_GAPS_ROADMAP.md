# Repercep — Full Project Audit, Decart Gap Analysis, Market Paths (2026-07-12)

- **Scope:** complete status audit of the runtime (code, measurements, process),
  a deep-dive on **Decart** (the closest new convergent competitor, and the
  source of the name collision that triggered today's Mirage→Repercep rename),
  and a prioritized roadmap along three axes: **versatile**, **solid**,
  **highest-performing** — plus the revenue/engagement question: *who wants
  this, and how do they engage.*
- **Voice:** same discipline as `METHODOLOGY.md` / `REVISED_STRATEGY.md` —
  measured-by-us vs external claim is marked everywhere; gaps are stated as
  gaps, not spun.

---

## 1. Status audit — what is real today

### 1.1 Measured and on `main` (all measured-by-us, verbatim provenance in the named docs)

Two serving regimes run through one `InteractiveWorldModel` seam, benchmarked
by one harness (`scripts/bench_control_loop.py`, `docs/CONTROL_LOOP_BENCH.md` —
the Control-Loop Serving Benchmark v0, which nobody else publishes):

| Regime / model | H100 | MI300X | Doc |
|---|---|---|---|
| **Energy-MPC** (V-JEPA 2-AC, 300M AC head): plan H=4, 64×3 CEM | 9.42 s batched-bf16 (7.3× vs sequential fp32) | 10.81 s (6.3×) | `LEVERS_2026_07_H100.md` |
| — closed-loop step (warm) | 70.8 ms fp32 / 84.9 ms bf16 | 75.8 ms fp32 / 73.6 ms bf16 | `BENCH_2026_06_RUNPOD.md`, levers doc |
| — KV growing-window reuse | GPU-verified 3.8e-4 (algorithm) | GPU-verified 6.8e-4 (integrated adapter) | ADR-0009 |
| **Policy regime** (LingBot-VA 2.0, video-action): chunk latency (warm) | **754.6 ms** | **1198.1 ms** | `CONTROL_LOOP_BENCH.md` |
| — resident sessions/GPU (extrapolated from measured marginal HBM) | 11 | 30 | same |
| **Generation camp** (kept, not headline): Cosmos-Predict-7B | 99.6 s (3.81× NVIDIA ref) | 142 s (first published AMD Cosmos number) | `COSMOS_ON_*.md` |

Plus: Wan-2.2-A14B both-experts-resident on 192 GiB (the H100-can't-hold-it
config), CPU/AMX path, FP8 Triton kernels (gfx942/Hopper/Ada) for the
generation path, Rust cache/scheduler/router crates with Python fallbacks.

Verification discipline is a real asset: ADR-0009 contains two *negative*
GPU-verified findings (non-composable RoPE; eviction provably non-exact at
K/V level) written up as findings rather than buried — this is the
credibility card `REVISED_STRATEGY.md` §2 names.

### 1.2 Solidity gaps (found in this audit, 2026-07-12)

Ranked by how cheaply they can bite:

1. **No CI whatsoever.** `.github/workflows/` does not exist. PR #2 (16
   commits, 4,152 insertions) merged with zero checks. Everything below
   compounds from this.
2. **`make lint` and `make typecheck` FAIL on `main`** — 6 ruff errors, 58
   mypy errors across 12 files (worst: `tests/test_quantize.py` 13,
   `tests/test_bench_control_loop.py` 11, `serving/app.py` 10) — on a codebase
   whose README claims `mypy --strict` discipline. The claim is currently
   false; either fix or stop claiming.
3. **2 env-dependent test failures** on a CPU-only dev box:
   `test_wan.py` guidance-scale tests construct a CUDA generator without CUDA
   libs instead of skipping. Test-portability bug, not a product bug.
4. **The serving surface doesn't serve the wedge.** `/v2/world/session`
   exposes `reset`/`step` only — **`plan()` is not reachable over the wire.**
   The strategy's Tier-1 item 1 ("planning service — energy-MPC as a served
   endpoint, not just a script path") is not done. No session caps, no
   idle-eviction, no backpressure, no auth on the WebSocket.
5. **Metric 4 is extrapolated, not load-tested.** "11 / 30 resident
   sessions/GPU" comes from one session's marginal HBM. Nobody has run N
   concurrent sessions through the serving layer.
6. **The Rust crates and FP8/AMX kernels serve the generation path only.**
   The control-regime hot loop (`vjepa2_ac.py`, `lingbot_va*.py`) touches none
   of them — the "paged latent-cache manager" crate is exactly what the
   CEM-shared-prefix KV design needs, and it's shelf-ware for that workload.
7. Staleness debris: `Cargo.toml` header comment says "no members yet"
   (false); `ANNOUNCEMENT.md` still says "Open source. Apache-2.0" (relicensed
   proprietary 4b70cfb); two stale `worktree-agent-*` branches/worktrees.

### 1.3 Strategy-tier scorecard (against `REVISED_STRATEGY.md` §4, set 2026-06-24)

| Tier item | Status |
|---|---|
| T1.1 Harden closed-loop serving (multi-session, planning service, streaming energy) | **Partial** — WS session exists; `plan()` unserved; no multi-session mgmt |
| T1.2 Novel measured action-loop mechanism (batching, KV/latent reuse, energy-eval batching) | **Done and exceeded** — batching+bf16 measured both vendors; KV growing-window designed+GPU-verified; two publishable negative findings; CEM shared-prefix still open (the remaining half) |
| T1.3 192 GiB → "runs what H100 can't" for control | **Partial** — 30-vs-11 sessions/GPU measured (capacity story is real); not load-tested, no headline demo |
| T1.4 Cosmos 3 action model on the seam | **Not started** — single-model-bet hedge is now LingBot-VA (done) instead; Cosmos 3 path still open |
| T2.5 Real proprioceptive pose through `WorldState` (the credible robot demo) | **Not started** (zeros; `REFINE` note in `vjepa2_ac.py`) |
| T2.6 FVD at N≥100 (generation credibility) | **Not done** |
| T2.7 Bare-metal MI300X control run | **Softened** — multiple independent containerized MI300X runs now exist (RunPod); literal bare-metal still unrun |

Sessions 24→25→today executed T1.2 hard and added a second regime the strategy
didn't anticipate (LingBot-VA / policy regime, which turned out to be the
*deployable-today* one). The serving/product layer (T1.1) is the lagging tier.

---

## 2. Decart — the deep-dive ("study Decart and find the gaps")

### 2.1 Company snapshot (external claims, web-verified 2026-07-12)

Israeli AI lab, founded 2023 (Leitersdorf/Shalev, Unit-8200 alumni). May 2026:
**$300M at ~$4B valuation** (Radical Ventures lead; NVIDIA, Adobe, Toyota
Ventures, Sequoia, Benchmark, eBay Ventures; angels incl. Karpathy). Revenue
today: **licensing DOS** (their inference stack) to "cloud providers, AI labs,
hyperscalers" — Amazon is a named customer — plus an API platform
(platform.decart.ai).

Three product lines:

- **DOS 2.0** — full-stack inference/training platform: hardware-aware model
  design, proprietary kernels/compilers, C++/CUDA LLM engine. Claims: 1,600
  tok/s agentic inference; full-HD world-model inference at 100 fps; 80% MFU
  on **Trainium3**. Hardware: **NVIDIA + Amazon Trainium + Google TPU — no
  AMD.**
- **Lucy 2.5** — real-time video transformation/world-editing (1080p live);
  robotics-relevant use: real-time data augmentation for sim.
- **Oasis 3** (launched **2026-06-10**) — "the first interactive world model
  for **physical AI**": action-conditioned, autoregressive, geometry-aware,
  synchronized multi-camera; **22 fps at 512×768, <200 ms end-to-end**;
  hosted API only (NVIDIA + CoreWeave infra), **$0.02/sec** (~$72/hr/stream);
  AV simulation first, "extends to drones, off-road, maritime, humanoid
  manipulation." Marketing claim: *"the first production-grade,
  action-conditioned API."*

Known Oasis 3 limitations (TechCrunch, 2026-06-10): thematic drift over
minutes (world doesn't persist), **no collision physics** (cars pass through
cars — CEO acknowledges), weak control responsiveness ("dream-like"), ~8k
tokens/frame → context window fills fast. These are exactly the autoregressive
context/memory problems ADR-0009 hit from the serving side.

### 2.2 What changed strategically

`REVISED_STRATEGY.md` (2026-06-24) filed Decart under "real-time video, wrong
lane, don't pitch into it." **That read is stale — Oasis 3 shipped two weeks
before that doc was written and pivoted Decart into the physical-AI lane**,
claiming the "action-conditioned API" phrase for themselves. The strategy's
"the control lane is unclaimed" sentence now needs a footnote:

- Decart claims the **world-model-as-simulator** side: actions in → *pixels
  out*, for **training** (synthetic environments, online RL, policy training
  loops). Hosted, closed models, their silicon partners.
- Repercep serves the **world-model-as-planner/policy** side: actions in →
  *latents/actions out*, for **deployment** (energy-MPC planning, VA policy
  chunks in a real control loop). Self-host, open third-party models, any
  silicon.

Same eventual buyers (robotics/AV/embodied-AI teams), different artifact.
Convergence risk is real but not head-on yet — and their DOS business proves
the "world-model inference runtime" category has revenue in it *today*.

### 2.3 Gap table — both directions

**They have, we don't:**

| Decart asset | Honest impact on Repercep |
|---|---|
| Own frontier models (Oasis/Lucy) + $300M + ~$4B brand | Can't compete on models; don't try. We serve *open* models. |
| Hosted API with public pricing ($0.02/s) | They own the easy-adoption motion; we have no hosted product. |
| DOS on NVIDIA + Trainium + TPU at hyperscaler scale | The "vendor-neutral runtime" phrase is no longer uniquely ours — **except AMD, which DOS conspicuously skips.** |
| Edge distribution (Comcast partnership) | No edge story on our side yet beyond CPU/AMX. |
| 22 fps interactive pixel streaming | We deliberately don't compete here (strategy Tier-3 still right). |

**We have, they don't:**

| Repercep asset | Why it holds against them |
|---|---|
| **AMD Instinct path** (only measured MI300X world-model serving numbers anywhere) | DOS supports TPU/Trainium but not ROCm; AMD's physical-AI push (below) has no serving story. Structural, not incidental — their kernel investment is CUDA-first. |
| **Self-host** | Robotics/AV/defense/industrial teams with data-sovereignty or on-prem fleets can't use a hosted-only API. Oasis 3 has no self-host option. |
| **Open third-party models on one seam** (V-JEPA 2-AC, LingBot-VA, Cosmos, Wan) | Their API serves only their models. Anyone deploying Meta/Ant/NVIDIA-ecosystem open world models needs a runtime, not a rival model. |
| **The control/planning regime** (energy-MPC, actions-out, latent-only — no decode in the loop) | Oasis's loop decodes pixels every frame (that's the product). Our regime is structurally cheaper per decision and is the *deployment* loop, not the training loop. |
| **Published closed-loop serving benchmark + honest methodology** | Nobody else (Decart included) publishes planning-decisions/sec, step-latency-under-state-carryover, resident-sessions/GPU. First-mover on the *leaderboard* is cheap and durable. |
| Per-session economics story | 6.0 GiB marginal/session measured (11/GPU on H100, 30 on MI300X) vs their $72/hr/stream hosted pricing — a self-host TCO argument writes itself for multi-robot fleets. |

Name collision: their **MirageLSD** vs our old name — **resolved today**
(project renamed Repercep, commit `80d5da5`).

### 2.4 Decart-specific risks to watch

1. **DOS adds ROCm.** Kill-signal for the AMD wedge's uniqueness; watch their
   job posts/releases. Mitigation: be so established on AMD (benchmarks,
   AMD-ecosystem co-marketing) that we're the incumbent by then.
2. **Oasis grows a policy-evaluation API** (world model as *evaluator* of
   robot policies, not just environment) — that crosses into our regime.
3. Their context/persistence problems get solved (the 8k-tokens/frame,
   world-drift issues) — raises the ceiling on what hosted generation covers.

---

## 3. Market & revenue — who engages, and how

Context signals (web-verified today): embodied-AI funding projected **>$20B in
2026**; Physical Intelligence (π0.7), Skild, Figure lead the robot-FM camp —
Skild already uses NVIDIA Cosmos for data-gen + Isaac for sim validation.
**AMD is pushing physical AI hard** — CES 2026 robotics hackathon (240 devs,
Lisa Su keynote visibility), π0/VLA fine-tuning on MI300X + Ryzen AI
edge-to-cloud robotics reference stacks, ROCm now first-class in vLLM — **but
AMD has no world-model serving story, and Decart's DOS skips AMD.** That
vacuum is the near-term commercial opening.

Segments, nearest-revenue first:

### 3.1 AMD ecosystem (nearest engagement, indirect revenue, highest leverage)

- **Who:** AMD ROCm dev-rel / AI ecosystem team, AMD Dev Cloud, AMD Ventures
  adjacency.
- **Why us:** we hold the only published world-model serving numbers on
  Instinct (Cosmos, V-JEPA-AC, LingBot-VA — all MI300X-measured), and the
  physical-AI story they're telling (hackathons, π0-on-MI300X) has a
  training story but no *serving* story.
- **How to engage:** pitch a ROCm-blog guest post ("Closed-loop world-model
  serving on MI300X: the control-loop leaderboard") + AMD Dev Cloud
  credits/partnership + ISV program entry. Target: co-marketing + hardware
  access + warm intros to their robotics design partners.
- **Revenue shape:** credits/NRE/co-marketing first; converts to §3.3 pilots.

### 3.2 MI300X neoclouds (TensorWave, Vultr, Hot Aisle, Crusoe-class)

- **Why us:** they need differentiated workloads that pull GPU-hours beyond
  LLM serving. "Physical-AI serving on your fleet" is a new hours category.
- **How:** a preinstalled "Repercep world-model serving" image + co-published
  benchmark ("N resident robot sessions per MI300X node") using the existing
  Repercep benchmark-report motion.
- **Revenue shape:** fixed marketing/benchmark engagements ($10–50k) +
  usage-linked referral.

### 3.3 Robotics / embodied-AI teams deploying open world models (the real product motion)

- **Who specifically:** teams deploying **LingBot-VA-class video-action
  models** (the class is public, Apache-2.0, shipping post-trains — and its
  own paper's serving ladder below 466 ms is **closed, CUDA-locked tooling**);
  labs in the V-JEPA/JEPA camp productionizing energy-MPC planners; AV-sim
  teams needing closed-loop model serving off NVIDIA allocation.
- **The pain we remove, measured:** open releases ship only the slow eager
  rung. We already beat the reference stack 1384.7→754.6 ms on H100 through
  design (not shortcuts), run 11–30 sessions/GPU where the reference manages
  1, and the portable re-implementation of the paged-KV/overhead rungs (their
  927→142 ms ladder) is scoped as the next performance workstream (§4).
- **How:** 2–3 paid **design-partner pilots**: "we port + serve your world
  model on your silicon under the control-loop metrics; you get the numbers
  and the runtime license." $25–100k per pilot, 4–8 weeks each.
- **This is also the kill-criteria clock** from `REVISED_STRATEGY.md` §5: if
  no design partner engages within ~2 quarters of the working demo, the
  strategy says treat this as a research artifact. The demo exists now; the
  clock is running.

### 3.4 Benchmark-report buyers (already-live motion)

- Repercep already has the productized motion: benchmark request form live +
  `Repercep_Benchmark_Report_2026-07.pdf`. The Control-Loop Serving Benchmark
  is the unique inventory — silicon-selection guidance for world-model
  workloads (H100 vs MI300X vs B200/MI355X) that no analyst has.
- **Revenue shape:** per-report ($5–25k), then subscription. Cheapest test of
  willingness-to-pay; every report is also design-partner lead-gen.

### 3.5 Later (explicitly not now)

Hosted control-loop **policy-evaluation** API (the counter-position to Oasis:
evaluate policies against world models rather than generate worlds) and
DOS-style runtime licensing at scale. Both capital-gated; revisit after two
pilots.

**Timing honesty:** the deployed energy-MPC-planner market is still early (the
2027–28 risk from `REVISED_STRATEGY.md` stands). What is deployable *today* and
buyable *today*: policy-regime VA serving (§3.3), benchmarks (§3.4), and the
AMD vacuum (§3.1). Sequence revenue in that order.

---

## 4. Roadmap — versatile, solid, highest-performing

### P0 — Solid (this week; prerequisites for everything credible)

1. **CI** (GitHub Actions): ruff + mypy + pytest (CPU) + `bench_control_loop
   --fake` smoke on every PR; branch protection on `main`. *(No more
   zero-check merges.)*
2. **Fix the 6 ruff + 58 mypy errors** (or explicitly downgrade the README's
   `--strict` claim — the honest-docs discipline demands one or the other).
3. Make `test_wan.py` env-safe (skip/mock CUDA generator off-GPU).
4. Serving hardening minimum: per-process session cap, idle-session eviction,
   WS auth token, graceful backpressure — the things a design partner hits in
   hour one.
5. Debris: fix stale `Cargo.toml` header, mark `ANNOUNCEMENT.md` superseded
   (license changed), delete stale agent worktrees/branches.

### P0 — Versatile & credible (2–3 weeks)

6. **Serve `plan()`** — the planning-service endpoint (WS message +
   `/v2/world/plan`), streaming energy/latency telemetry per decision. This
   closes Tier-1 item 1 and makes the wedge *demoable over the wire*.
7. **Goal-image → goal-embedding endpoint** (the missing API for real
   energy-MPC use; walkthrough Part 5 names it).
8. **N-session load test** — run N live sessions to failure on one GPU;
   converts metric 4 from extrapolated to measured (and likely a headline:
   "30+ resident robot sessions on one MI300X, measured").
9. **LingBot-VA grounded recondition test** (real observations mid-session) —
   the untested half of the policy regime.

### P1 — Highest-performing (the publishable wedge work)

10. **Portable LingBot-VA serving rungs** (the 927→142 ms ladder, reimplemented
    vendor-neutral): paged/ragged KV on SDPA (+ ROCm AITER path), HIP/CUDA-graph
    runtime-overhead elimination, async chunk pipeline. Target: **754.6 →
    <400 ms/chunk on both vendors** without TensorRT/FlashInfer. Each rung is a
    leaderboard row + a blog post + a design-partner selling point.
11. **CEM shared-prefix paged KV** (ADR-0009's scoped-out second dimension) —
    wire the Rust `repercep-cache` crate into the control path (it was built
    for exactly this and currently serves nothing on the wedge). This is the
    remaining "novel mechanism" of strategy T1.2.
12. **Sliding-window decision** (ADR-0009 open item): measure
    attention-sinks-style approximation vs reinit **on planning quality**
    (energy achieved, not just latency) and pick, explicitly documented.
13. **Multi-seed warm-start study** — resolve the H100/MI300X warm-start
    energy-ordering discrepancy (currently an honest open flag in
    `CONTROL_LOOP_BENCH.md`).
14. MI355X/B200 columns when access allows (MLPerf 6.0 shows the MI355X
    ecosystem is live; the FP8 attention kernels already exist for gfx942).

### P1 — Versatility (hedge + market surface)

15. **Cosmos 3 action-module on the seam** (strategy T1.4, still open) — the
    "we serve the open physical-AI models, whoever makes them" proof point,
    and the direct counter to single-model-bet risk.
16. **Proprioceptive pose through `WorldState`** (strategy T2.5) — turns the
    planner demo from "energy decreases" into "reaches an image goal," the
    demo a robotics design partner actually evaluates.

### GTM actions (parallel, from §3)

17. AMD ROCm-blog pitch + Dev Cloud application (this month).
18. Benchmark Report v2 with the control-loop leaderboard as the centerpiece;
    push through the live Repercep form funnel.
19. The pending **VC reply on Reactor** (`COMPETITIVE_RESPONSE_REACTOR_2026_07.md`
    ammunition is ready) — now add the Decart section from this doc: the
    category has a $4B validator and an unclaimed AMD/self-host/control flank.
20. Rename follow-through: ~~GitHub repo rename~~ (done 2026-07-12 —
    `miteshs/Repercep`, old URLs redirect), site copy, local dir.

### Explicitly still don't (unchanged from REVISED_STRATEGY Tier-3)

Real-time pixel-streaming race (Oasis/Genie lane), generation kernel/FP8
parity races, multi-GPU sequence-parallel. Decart's $300M just re-proved how
capitalized that lane is.

---

## 5. Kill-criteria check-in

`REVISED_STRATEGY.md` §5 set the clock: design-partner engagement within ~2
quarters of a working demo. The demo now exists on two regimes and two
vendors, with a public leaderboard. Decart's Oasis 3 launch is double-edged
evidence: it **validates the category** (a $4B company now sells
action-conditioned world-model serving) and it **compresses the window** (the
"first production-grade action-conditioned API" phrase is taken; "first
*self-host, any-silicon, open-model, control-regime*" is still free). The §3
engagement sequence is the test. If §3.3 produces zero paid pilots by
~2027-01, the strategy's own fallback (research artifact / portfolio) applies.
