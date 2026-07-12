# Repercep — Revised Strategy (2026-06-24)

- **Date:** 2026-06-24
- **Supersedes the *roadmap* in:** `docs/STRATEGIC_ASSESSMENT.md` §5 and
  `docs/POSITIONING.md` §"Strategic options matrix". It does **not** supersede their
  *analysis* — both still hold; this doc updates the conclusion for (a) what we built
  since (the interactive seam is now real) and (b) what the market did since
  (NVIDIA Cosmos 3, the real-time-interactive land-grab).
- **Voice:** same as the assessment — the version that survives diligence, not the
  pitch. Where this doc is optimistic it says why; where the moat is thin it says so.

---

## 0. TL;DR — the one-paragraph revision

The May-28 assessment said: stop hardening the commoditizing wedge-opener (fast batch
Cosmos/Wan on AMD) and go build the deferred moat (action-conditioned, closed-loop,
*control-regime* serving). **Two things have happened since.** (1) We built the first
real piece of it — the `InteractiveWorldModel` seam + a **GPU-verified V-JEPA 2-AC
engine** (latent rollout + energy-MPC planner, real weights on an L4, energy
decreases under CEM). (2) The market validated the *direction* and then crowded the
*adjacent* lane: **NVIDIA Cosmos 3** (2026-06-05) shipped an **open** omni world model
with native **action generation** + a NIM serving stack, and the real-time interactive
frontier (Genie 3, Decart, Odyssey, World Labs Marble, Matrix-Game 3) filled in — all
NVIDIA-anchored, mostly diffusion/autoregressive *generation*. The net: the
"fast batch Cosmos-7B on AMD" wedge is now **dead** (Cosmos-Predict-7B is two
generations old; Cosmos 3 is open and NIM-served). The defensible position narrows to
one sharp wedge nobody is serving: **vendor-neutral serving of the *control* regime of
world models — energy-based / action-conditioned closed-loop planning (V-JEPA-AC class)
— where the workload is latent rollout + energy-MPC, not pixel generation.** That is a
different leaderboard, a different compute profile, and a different (LeCun/JEPA) camp
than the one NVIDIA just took.

---

## 1. What changed since 2026-05-28

### 1.1 On our side — the moat layer is no longer "on the shelf"

The assessment's central finding was that the moat (action-conditioned closed-loop
serving) "sits on the shelf." It doesn't anymore:

- `runtime/interactive.py` — the `InteractiveWorldModel` seam (`reset → step → plan`),
  distinct from the one-shot `WorldModelEngine`.
- `models/vjepa2_ac.py` — a **real, GPU-verified** V-JEPA 2-AC engine. The four
  weight-load `NotImplementedError`s are filled; on an L4 it loads the
  `facebook/vjepa2-vitg-fpc64-256` encoder + the `vjepa2-ac-vitg` AC predictor,
  rolls the latent world state forward block-causally, and the CEM/energy-MPC planner
  **measurably reduces terminal energy** toward a goal (the documented
  `run_vjepa2_ac.py --backend cuda --plan` smoke test passes end-to-end).
- `serving/app.py` — a `/v2/world/session` bidirectional WebSocket for closed-loop
  driving.

This matters strategically: the pivot the assessment *recommended* now has a working
artifact, not just an argument. The cost-to-credibility of the "control-regime serving"
story dropped from "a plan" to "it runs."

### 1.2 On the market's side — the lane got crowded, and NVIDIA took the center

| Development (verified 2026-06) | Strategic read |
|---|---|
| **NVIDIA Cosmos 3** (2026-06-05): open omni world model, Mixture-of-Transformers, modalities = text/image/video/audio/**action**; scales Edge-4B / Nano-16B / Super-64B; **NIM** serving. | NVIDIA now ships **open, action-conditioned** world models *with a serving stack*. This is the lane the assessment named — NVIDIA is now in it. But it is **CUDA/TensorRT/NIM-locked** and **generation-first** (MoT diffusion), not control-first. |
| **Genie 3** (DeepMind): 24 fps, persistent, real-time — **closed API only**. | The flagship interactive WM is not serveable by anyone but Google. Not a competitor to an OSS runtime; a demand signal. |
| **Decart / Odyssey / World Labs Marble / Matrix-Game 3**: real-time interactive video, 20–30 fps, 35–50 ms, edge (Decart+Comcast on NVIDIA), Marble commercial. | The real-time **autoregressive video** frontier is busy and NVIDIA-edge-anchored. Repercep's diffusion/JEPA specialization is **not** this lane — do not pitch into it. |
| **V-JEPA 2-AC** (Meta): zero-shot Franka control, energy-MPC planning, **outperforms diffusion models in control efficiency**. | Validates Repercep's lead-model choice and the **energy-based camp** as a real, distinct, cheaper-for-control paradigm. This is the camp NVIDIA's Cosmos is *not* in. |
| Serving infra reality: 720p real-time = H200; 1080p/multi-stream = B200. | The *generation* serving cost-center is huge and NVIDIA-silicon-shaped. Competing there on AMD is the old, losing wedge. The *control* serving cost-center (latent rollouts, no decode) is small per-step and **unclaimed**. |

### 1.3 The two camps (the map that now matters)

The field has visibly forked into two paradigms with different economics:

- **Generation camp (diffusion / autoregressive, pixel-output):** Cosmos 3, Genie 3,
  Wan, HunyuanVideo, Decart, Odyssey, Marble, Matrix-Game. Output is video; the cost
  center is denoise/decode; the buyers are synthetic-data pipelines, media, games.
  **Crowded, NVIDIA-anchored, commoditizing on the serving side** (vLLM-Omni,
  SGLang-Diffusion, xDiT, FastVideo, NIM all absorb the caching/parallelism tricks).
- **Control / energy-based camp (JEPA, representation-space, plan-output):**
  V-JEPA 2-AC and successors. Output is *actions/plans*; the cost center is **latent
  rollout + energy minimization (CEM/MPC)**, *no pixel decode*; the buyers are
  robotics/embodied agents. **Sparse, LeCun/Meta-led on the model side, and on the
  *serving* side essentially nobody.**

Repercep already serves **both** camps through one Backend Protocol seam (Cosmos/Wan in
the first, V-JEPA-AC in the second). That dual-paradigm coverage is the rare position.

---

## 2. Re-evaluated moat (delta from the assessment)

The assessment's moat table mostly holds and, post-Cosmos-3, several rows get *worse*.
The new column is the honest update.

| Candidate | May-28 verdict | 2026-06-24 update |
|---|---|---|
| Fast batch Cosmos-7B on AMD | Eroding wedge-opener | **Dead.** Cosmos 3 is open + NIM-served; Cosmos-Predict-7B is two gens old. Stop. |
| Adaptive caching (TeaCache-style) | Not a moat | Still not. Now table-stakes (NIM/vLLM-Omni/xDiT all have it). |
| "Only WM serving on AMD" | Eroding | Eroding faster — but **vendor-neutral serving of *new* models still has a real lag**: Cosmos 3 / NIM are CUDA-locked, and there is no AMD/edge path for the control camp. The breadth asset is *more* valuable as model count explodes, even though it is not a deep moat. |
| 192 GB memory physics | Real, narrow, time-boxed | Same — and **re-aimed**: the structural win is now "hold many resident world-states for N-stream interactive control," not "fit Wan's experts." |
| Multi-target runtime + WM breadth | Asset, not deep moat | **Upgraded to the core position.** With two paradigms and a Cambrian explosion of WMs, "one engine, any silicon, generation *and* control" is the durable *position* (not a patent-moat, but a real one for design-partner trust). |
| Action-conditioned closed-loop serving (the recommended moat) | Unbuilt | **Built (v0).** Now defensible to *demonstrate*, not just claim. NVIDIA entered the *generation* side of this; the **control/energy-MPC serving side is still open**. |

**Updated honest verdict.** There is still **no durable, patent-style technical moat**.
What is available, in descending order of defensibility:

1. **A genuinely unoccupied niche:** serving-optimized **energy-MPC / latent-rollout**
   for the control camp (V-JEPA-AC class). Different leaderboard
   (planning-decisions/sec, energy-evals/sec, closed-loop control latency under
   state carryover), different compute profile (batched short rollouts, no decode),
   and **nobody — including Cosmos 3's NIM — is optimizing it as a serving problem.**
2. **A novel, publishable mechanism** living in that niche: reuse across the **action
   loop and across CEM candidates** (batched-rollout latent/KV reuse, energy-landscape
   eval batching), benchmarked head-to-head. The assessment pointed at this; the
   interactive engine now makes it concrete to build and measure.
3. **The dual-paradigm, vendor-neutral position** (generation + control, AMD/CUDA/CPU/
   edge). Replicable by a funded team in a quarter, but real today and increasingly
   valuable as WMs proliferate.
4. **The execution + honesty asset** (the F-ledger, METHODOLOGY §3, this very doc).
   Unchanged and still the strongest *credibility/hiring/design-partner* card.

---

## 3. The revised thesis

> **Repercep is the vendor-neutral serving runtime for the *control* regime of world
> models** — energy-based, action-conditioned, closed-loop planning (V-JEPA-AC class)
> — where the workload is latent rollout + energy-MPC, not pixel generation, served on
> any silicon (AMD/NVIDIA/CPU/edge). It keeps first-class support for the *generation*
> camp (Cosmos/Wan) through the same Backend Protocol, so a robotics/agent customer
> runs perception-generation and control-planning on one engine — but the **wedge** is
> the control regime nobody else is serving.

Why this is the defensible framing post-Cosmos-3:

- It is **orthogonal to where NVIDIA just planted its flag.** Cosmos 3 + NIM is
  generation-first, CUDA-locked, pixel-output. Energy-MPC control serving is a
  different leaderboard and a different camp (JEPA), on silicon NIM doesn't target.
- It is **where the compute is structurally cheap but the *serving* is unsolved.**
  A V-JEPA-AC planning step is latent-only (no VAE decode) and tiny per-call (we
  measured 3.4 GiB peak on an L4) — but a *planner* fires hundreds of rollouts per
  decision under a latency budget. That is a scheduling/caching/batching problem that
  general engines (built for one big generation per request) don't express. **That gap
  is the product.**
- It **matches where embodied-AI money is heading** (robot/agent policies in the loop),
  and it **doesn't require us to win the generation race** we can't win.

Repercep's one-line pitch becomes: *"the runtime for world models that act, not just
render — on any silicon."*

---

## 4. Revised roadmap

### Tier 1 — own the control-serving regime (the moat work)

1. **Harden the closed-loop serving surface** (build on what landed today). Multi-
   session world-state management, action-batched `step`, streaming `LatentStep`/energy
   over `/v2/world/session`, and a **planning service** (energy-MPC as a served
   endpoint, not just a script path). Metric: **closed-loop control latency and
   planning-decisions/sec under state carryover** — publish this leaderboard; nobody
   owns it.
2. **Invent + *measure* one novel serving mechanism in the action loop.** Concretely
   now reachable: (a) **CEM-candidate batching** — the planner rolls N action
   sequences; batch them through the predictor as one forward instead of the current
   per-candidate loop; (b) **latent/KV reuse across rollout steps and across CEM
   iterations** (the context prefix is shared across candidates); (c) **energy-eval
   batching**. Benchmark head-to-head vs the naive loop *and* vs TeaCache/DiCache
   framing (which are denoise-step caches, not action-loop caches). Win → a paper and
   IP; lose → learned cheaply. This is the line between "we ported V-JEPA-AC" and
   "we have the action-loop serving primitive."
3. **Convert the 192 GB physics into "runs what H100 can't" for control:** N-stream
   interactive serving with many resident world-states, or long-horizon rollouts with
   large resident latent context. "Runs the workload the other silicon physically
   can't hold" is the wedge; "faster" is a race.
4. **Bring the *generation* camp's new center of gravity onto the same seam:** a
   **Cosmos 3 action-model** path through `InteractiveWorldModel` (it has a native
   action modality). This hedges the V-JEPA bet, keeps the dual-paradigm story honest,
   and positions Repercep as the *vendor-neutral* place to serve Cosmos 3's action model
   off NVIDIA silicon — exactly the gap NIM leaves.

### Tier 2 — credibility (minimum, then stop)

5. **Real proprioceptive control loop for V-JEPA-AC**: thread the 7-DoF pose
   (`compute_new_pose`) through `WorldState` so the rollout is the *real* robot
   dynamics, not the zero-state plumbing smoke test. This is what turns "the planner
   reduces energy" into "the planner reaches an image goal," the credible demo.
6. **FVD at N≥100** for the generation paths (kept from the old roadmap — closes the
   "every skeptic probes this first" gap). Table-stakes, not a moat.
7. **One bare-metal MI300X control-serving run** to retire the VF-slice asterisk on the
   regime that now matters.

### Tier 3 — explicitly don't

- **Don't** harden fast-batch Cosmos-7B/Wan as a flagship. Cosmos 3 ended that lane.
  Keep them as the generation-camp coverage, not the headline.
- **Don't** enter the real-time **autoregressive video** race (Genie 3/Decart/Odyssey/
  Matrix-Game). Wrong architecture (bidirectional diffusion / JEPA), NVIDIA-edge-
  anchored, and a different product. The assessment's load-bearing caveat still holds:
  *do not pitch "real-time interactive video" while serving control-regime latents.*
- **Don't** chase generation kernel/FP8 parity or multi-GPU sequence-parallel. Races
  others lead.

---

## 5. Risks and kill-criteria (be honest)

- **Market timing risk (biggest).** Control-regime *serving* is real but the *buyers*
  (robotics/embodied agents deploying learned world-model planners at scale) are early.
  This may be a 2027–28 market. Kill-criterion: if no design partner or research
  collaborator engages on energy-MPC serving within ~2 quarters of a working demo,
  treat Repercep as a portfolio/research artifact (assessment option A), not a company.
- **NVIDIA-extends-down risk.** Cosmos 3 already has an "action" modality and reasoning;
  NVIDIA could add control-loop serving to NIM. Mitigation: the **vendor-neutral +
  energy-based (non-Cosmos)** angle is the part they're structurally unlikely to
  prioritize; lead with V-JEPA-AC on AMD/edge.
- **Single-model-bet risk.** V-JEPA-AC is Meta *research*, not a deployed product base.
  Mitigation: Tier-1 item 4 (Cosmos 3 action model on the same seam) + keep the seam
  model-agnostic (it already is).
- **"Breadth ≠ moat" risk (unchanged from assessment).** The dual-paradigm runtime is
  a *position*, copyable by a funded team in a quarter. The only durable differentiator
  is Tier-1 item 2 (a measured, novel action-loop serving mechanism) — so that item is
  the real bet, not the breadth.

---

## 6. What to claim / not claim (post-Cosmos-3)

- **Claim:** "Repercep runs V-JEPA 2-AC end-to-end — real weights, latent rollout,
  energy-MPC planning — on AMD/NVIDIA/CPU through one vendor-neutral seam." True today;
  GPU-verified.
- **Claim:** "The runtime for world models that *act* — control-regime serving (latent
  rollout + energy-MPC), not just generation." True as a position; honest about being
  early.
- **Do not claim:** any "faster than Cosmos / NVIDIA" headline on generation. Cosmos 3
  is open + NIM-served; that race is lost and was never the moat.
- **Do not claim:** "real-time interactive world model." We serve control latents and
  bidirectional diffusion, not 24 fps autoregressive video. A sharp reviewer will catch
  it (the assessment's standing caveat).
- **Lead with, still:** the honesty and the dual-paradigm vendor-neutral execution.
  Post-Cosmos-3 that is *more* valuable, not less — the field needs a neutral serving
  layer precisely because the model layer is consolidating under one vendor.

---

## 7. The one-line decision

Stop defending the generation wedge Cosmos 3 just ended. **Spend the next two quarters
turning today's V-JEPA-AC port into the first serving-optimized energy-MPC runtime —
one novel, measured action-loop mechanism + a closed-loop control leaderboard nobody
owns — kept vendor-neutral and dual-paradigm so it serves Cosmos 3's action model too,
off NVIDIA silicon.** That is the only path from "an impressive, honest artifact" to
"the runtime for world models that act" that survives both diligence *and* the
post-Cosmos-3 market.

---

## Appendix — sources (web-verified 2026-06-24)

- NVIDIA Cosmos 3 launch (2026-06-05): https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai ·
  technical report https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf ·
  developer blog https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/ ·
  HF https://huggingface.co/blog/nvidia/cosmos-3-for-physical-ai
- Genie 3: https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/
- V-JEPA 2 / 2-AC: https://arxiv.org/abs/2506.09985 · https://ai.meta.com/research/vjepa/ ·
  https://www.emergentmind.com/topics/v-jepa-2-ac
- Real-time interactive landscape: Decart https://decart.ai/ · World Labs Marble
  (TechCrunch 2025-11-12) · Matrix-Game 3 https://arxiv.org/html/2604.08995v2 ·
  GPU infra guide https://www.spheron.network/blog/gpu-infrastructure-world-models-2026/
- Serving stacks absorbing video diffusion (still true): vLLM-Omni, SGLang-Diffusion,
  xDiT, FastVideo (see `docs/STRATEGIC_ASSESSMENT.md` Appendix A).
