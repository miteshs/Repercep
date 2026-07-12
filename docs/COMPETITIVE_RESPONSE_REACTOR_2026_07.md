# Competitive Response — Reactor.inc (2026-07-11)

- **Trigger:** a VC flagged Reactor.inc to us; this doc is the CTO-level read and the
  action plan we answer them with.
- **Relationship to other docs:** extends `REVISED_STRATEGY.md` (2026-06-24); does not
  change the thesis — it *sharpens* it and adds dated commitments. Same voice: the
  version that survives diligence.
- **Sources (web-verified 2026-07-11):** reactor.inc, docs.reactor.inc (+ llms.txt
  index), Lightspeed's Series-A memo, PRNewswire stealth-exit release (2026-05-29),
  Robbyant LingBot-World 2.0 / LingBot-VA 2.0 announcements (2026-07-08/11).

---

## 0. TL;DR

Reactor is a **hosted inference API for real-time generative video** — the *generation*
camp. Today it is **not** a control-regime competitor: its docs expose exactly one
model (Helios, interactive text/image-to-video), WebRTC streaming, `set_prompt`-class
commands, and **zero** action, world-state, robotics, or planning surface. But its
press explicitly names **physical AI and robotics** as target markets, it already hosts
Ant/Robbyant's LingBot-World 2, and Robbyant just shipped **LingBot-VA 2.0** — a
video-**action** model for robot manipulation at **142 ms/chunk, 225 Hz async control**.
The convergence point between Reactor's lane and ours is **video-action models**, and it
is now visible on a ~2-quarter horizon. Our answer is not to race their media API; it is
to (1) **close our control-loop latency gap** (55 s/plan today → sub-second, then
~100 ms-class replanning — the levers are already identified and measured), (2) **put a
second and third action model on our seam** (LingBot-VA 2.0, Cosmos 3 action) so we are
the *vendor-neutral, self-hostable* runtime for the exact model class Reactor would
host, and (3) **publish the control-serving benchmark** we keep saying nobody owns,
before someone else defines the metrics.

---

## 1. What Reactor actually is (verified, not vibes)

| Dimension | Finding | Source |
|---|---|---|
| Product | Hosted API/SDK for real-time interactive **video** streaming; JS/React/Python SDKs, WebRTC sessions, JWT auth, browser playground; beta, usage-based billing | docs.reactor.inc |
| Catalog (docs) | **One model documented: Helios** (interactive real-time video, infinite streaming). Commands: `set_prompt`, `schedule_prompt`, `set_image`, `set_seed`, playback control. Site also lists LingBot World 2 (Ant) | docs.reactor.inc/llms.txt |
| Latency claims | "sub-50 ms streaming" (marketing); docs say "<1 s round-trip" | reactor.inc vs docs |
| Infra | AWS preferred partner, "1,000s of GPUs", custom CUDA + torch optimizations — **NVIDIA/CUDA-locked, cloud-only** | PRNewswire, Lightspeed memo |
| Funding/team | $59M Seed+A led by Lightspeed (May 2026); WndrCo, Amplify, Sky9, FPV. Founders: Alberto Taiuti (ex-Luma AI CTO), Bryce Schmidtchen (ex-Apple Vision Pro) | Variety, PRNewswire |
| Positioning | "Every major world model on one API"; buyers today = media/entertainment, games (Overworld partnership); press names **physical AI & robotics** as target markets | PRNewswire |
| What's absent | No actions, no world-state API, no planning, no robotics features, no on-prem/self-host, no non-NVIDIA silicon, no pricing page | docs.reactor.inc/llms.txt |

**Read:** strong team, real money, real product velocity — in the **generation camp**
we already declared dead-for-us in `REVISED_STRATEGY.md` §Tier-3. Their "world models"
umbrella and robotics press language creates **narrative collision** with us in a VC's
head, which is exactly why we got the question. The narrative collision is a marketing
problem; the *technical* collision is not here yet — but it has a date.

## 2. The convergence point: video-action models

The new fact since our June strategy doc: **Robbyant LingBot-VA 2.0** (2026-07-11) — a
causal DiT that outputs **video frames + robot actions**, sparse-MoE (2.5B active of
15.3B), 10–15-demo adaptation, and — the number that matters — **927 ms → 142 ms/chunk
after serving optimization, 225 Hz asynchronous control**, 93.6% on RoboTwin 2.0.

Three implications:

1. **The control lane's model supply is broadening beyond Meta research.** Our
   single-model-bet risk (V-JEPA-AC) now has an obvious hedge that is *shipping*,
   open(ing)-source, and from a vendor Reactor already hosts.
2. **Reactor's path into robotics is now concrete:** Ant relationship + LingBot-VA =
   a hosted video-action endpoint. If they ship that, "Reactor does robotics world
   models" becomes true *in the hosted-API sense* within a quarter or two.
3. **142 ms/chunk is the new public reference point** for action-model serving. Our
   measured CEM plan is **~55 s on H100**. Different workload (full energy-MPC plan vs
   one forward chunk) — but no VC or design partner will grant us that nuance. We must
   close the optics gap with our own measured numbers.

## 3. Where we win / where they win (honest)

**Structurally ours, hard for Reactor to copy:**
- **Self-host / on-prem / edge.** A robot's control loop cannot round-trip to a cloud
  WebRTC session — latency, safety certification, and data gravity all say the runtime
  ships to the robot's site. Reactor's entire business model is the opposite bet.
- **Vendor-neutral silicon.** We are the only measured H100 **and** MI300X world-model
  serving numbers in public (our July benchmark report). Reactor is AWS+CUDA.
- **The control workload itself.** Energy-MPC / latent-rollout serving (batched short
  rollouts, resident world-state, no pixel decode) is a scheduler/cache problem their
  per-session video-streaming architecture doesn't express. This is `REVISED_STRATEGY`
  §3 and it survives Reactor unchanged.
- **Measured honesty.** F-ledger, methodology, published raw JSON. Against a company
  whose marketing says "sub-50 ms" while its docs say "<1 s", this is a real
  diligence-room weapon.

**Theirs, honestly:**
- $59M, ~about-to-be-large team, brand-name investors, media lighthouse customers.
- Developer experience: self-serve signup, playground, three SDKs, credits. We have a
  WebSocket endpoint and scripts.
- Headline latency **today** on their workload; our headline number on ours is 55 s.
- Distribution/mindshare: "every major world model on one API" is a better-sounding
  story than ours to anyone who doesn't know the two-camps map.

## 4. The plan

### Sprint horizon (now → ~6 weeks) — close the number, claim the metric

1. **Latency assault on the planner** — execute the already-ranked levers from
   `BENCH_2026_06_RUNPOD.md` in order:
   (a) **KV/latent reuse across rollout steps + CEM iterations** (the structural win;
   the shared context prefix is recomputed every step today);
   (b) **bf16 predictor path** (fix the RoPE fp32 upcast → unlock SDPA/flash-attn,
   including on ROCm — the dominant per-forward cost);
   (c) **CEM warm-start** (receding-horizon shift-reuse);
   (d) re-tune the 64×3 sample/iter budget on an energy-vs-samples curve.
   **Target: 55 s → <1 s per plan on H100/MI300X; stretch: ~100 ms-class replanning
   with warm state.** This is the whole ballgame — every other item is framing.
2. **Publish the Control-Loop Serving Benchmark v0.** Define the metrics we keep
   saying nobody owns — planning-decisions/sec, closed-loop step latency under state
   carryover, energy-evals/sec, resident world-states per GPU — as an open harness +
   a repercep.ai page, seeded with our H100/MI300X numbers and the naive-loop
   baseline. Include LingBot-VA's published 142 ms/chunk as an external reference row.
   First mover on metric *definitions* owns the leaderboard.
3. **LingBot-VA 2.0 port scoped** (and started if weights land as announced): second
   model on the `InteractiveWorldModel` seam. This kills the single-model-bet risk,
   and it is the precise counter-position: *the self-host, any-silicon runtime for the
   model class Reactor will host in the cloud.* Cosmos 3 action modality stays third.
4. **Answer the VC** (see §6) with this doc's table + dated commitments, not adjectives.

### Quarter horizon (Q3 2026) — credibility and surface

5. **Real proprioceptive control demo:** thread the 7-DoF pose through `WorldState` so
   the planner *reaches an image goal* on real robot dynamics (Tier-2 item 5, now
   urgent — Reactor demos games; we must demo a robot).
6. **Minimum developer experience:** `pip install`-able client, a 20-line quickstart
   against `/v2/world/session`, and one hosted demo instance (single MI300X) for
   design partners to hit. Not a hosted business — a proof.
7. **Bare-metal MI300X control-serving run** (retire the VF-slice asterisk) and extend
   the benchmark matrix: V-JEPA-AC + LingBot-VA × H100 × MI300X.
8. **2–3 design-partner conversations** in robotics/embodied-AI, armed with the demo +
   leaderboard. The `REVISED_STRATEGY` kill-criterion stays live: no engagement within
   ~2 quarters of the working demo → reassess.

### Two-quarter+ horizon (Q4 2026 → 2027)

9. **Multi-session world-state serving:** N resident world-states per GPU (the 192 GB
   MI300X physics re-aimed), planning-as-a-service endpoint, action-batched `step`.
10. **Publish the mechanism** (action-loop KV/latent reuse) as a paper/tech report —
    the durable-differentiator bet from `REVISED_STRATEGY` §2, now with a competitor
    to benchmark against.
11. **Edge story** (Jetson/consumer silicon for latent-only planning — no decode means
    it plausibly fits). This is the lane neither Reactor (cloud) nor NIM (NVIDIA-DC)
    serves.
12. **Optional, only if pulled by customers:** a thin hosted control-serving tier.
    Never a hosted *media* API.

### Explicitly still don't (unchanged, now with a face on it)

- Don't race Reactor on real-time interactive **video** or sub-50 ms pixel streaming.
- Don't build a hosted media playground/credits business.
- Don't claim "faster than X" on generation workloads.

## 5. Investment areas, ranked

1. **Serving optimization for the action loop** (items 1, 9, 10) — the moat bet.
2. **Model breadth on the seam** (LingBot-VA, Cosmos-3-action) — kills concentration
   risk, rides others' model marketing.
3. **Benchmark/leaderboard publishing** (items 2, 7) — cheapest credibility per dollar;
   also our VC-facing artifact.
4. **Demo + DX** (items 5, 6) — the minimum to convert attention into design partners.
5. **GPU spend:** RunPod H100/MI300X stays the platform (proven); budget one
   bare-metal MI300X engagement.

## 6. What we tell the VC

> Reactor validates the market thesis and is not in our lane. They host real-time
> generative **video** (media/games buyers, AWS, CUDA-only, cloud-only); we are the
> vendor-neutral, self-hostable runtime for world models that **act** — latent rollout
> + energy-MPC for robotics control loops, on any silicon. Their own docs contain no
> action, world-state, or robotics surface. The lanes converge at **video-action
> models** (e.g. Ant's LingBot-VA 2.0, announced this week) — and that is exactly
> where we're moving first: [a] sub-second energy-MPC planning (measured levers
> already identified from our June H100/MI300X benchmarks), [b] LingBot-VA and Cosmos
> 3 action models on our seam next to V-JEPA 2-AC, [c] the industry's first
> control-loop serving benchmark, published — the leaderboard for the workload robots
> actually run, which a cloud video API structurally can't serve on-prem. We'll show
> the latency result and the benchmark page within 6 weeks.

---

*Owner: CTO. Review trigger: Reactor ships any action/robotics endpoint, or
LingBot-VA weights land — whichever first.*
