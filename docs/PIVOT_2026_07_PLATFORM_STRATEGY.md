# Repercep — Platform Pivot Strategy (2026-07-26)

- **Date:** 2026-07-26
- **Supersedes the *conclusion* of:** `docs/REVISED_STRATEGY.md` (2026-06-24) and the
  revenue sequencing in `docs/AUDIT_2026_07_STATUS_GAPS_ROADMAP.md` §3. It does **not**
  supersede their *analysis* — the two-camp map, the Decart read, and the moat honesty
  all still hold and are reused below. What changes is the shape of the company: from a
  **runtime you license** to a **cloud you buy inference from**, and from **world models
  only** to **every model class**.
- **Voice:** unchanged discipline — the version that survives diligence, not the pitch.
  Measured-by-us is marked. Where the moat is a hypothesis it is labeled a hypothesis
  with the experiment that would settle it.
- **Siblings:** `docs/BUSINESS_PLAN_2026.md` (market, pricing, unit economics, GTM),
  `docs/INFERENCE_MOAT_TECHNIQUES.md` (the technique ledger behind the cost claim).

---

## 0. TL;DR — the pivot in one paragraph

Repercep stops being a world-model runtime that hopes for a 2027–28 market and becomes
a **full-service inference cloud that serves every model class** — CNNs, encoders,
diffusion, transformers, LLMs, VLMs, VLAs, world models — on partner GPU capacity, with
**cost leadership as the commercial moat**. The near-term revenue is LLM and general
model serving, where demand exists today. The durable differentiation is the hard tail
the incumbents serve badly: **stateful, deadline-bound, non-text model serving**, which
is exactly what we spent a year proving we can do better than anyone publishing numbers.
The pivot is not a retreat from world models — it is the discovery that **the serving
primitives we built for the control regime are the same primitives the fastest-growing
LLM workload shape needs** (shared-prefix candidate batching, state-resident sessions,
deadline-aware scheduling). We monetize those primitives on LLMs now, and we are the
incumbent when the physical-AI wave lands. For physical AI specifically we sell a
**hybrid**: the edge processor runs the tight loop, our cloud runs the heavy and
strategic reasoning, and we partner with robotics/automation companies rather than
building edge silicon.

**The one-line pitch:** *Fireworks for the whole model zoo — priced off silicon nobody
else can use well.*

---

## 1. What actually changes

| # | Before (2026-06 strategy) | After (this doc) |
|---|---|---|
| 1 | Sell a **runtime** (license / design-partner pilots). No hosted product. | Sell **inference as a service** on our own managed cloud, built on partner GPU capacity. Runtime remains a product for self-host/BYOC, not the only one. |
| 2 | **World models only** — three regimes, one seam. | **Every model class.** World models become the proof of unusual depth, not the boundary of the offering. |
| 3 | Moat = "the control regime nobody serves." | Moat = **cost per unit of customer work**, from two measured levers: software efficiency and alt-silicon arbitrage. The control-regime work is *where the efficiency levers came from*. |
| 4 | Physical AI = we serve the model, someone else deploys the robot. | Physical AI = an explicit **hybrid edge↔cloud** product. Edge does the tight loop; we do heavy/strategic inference. We **do not build edge execution** — we partner. |
| 5 | Revenue timing = wait for the embodied-AI market (2027–28 risk). | Revenue timing = **LLM/general serving builds the run-rate now**; world-model demand is the option we already own. |

The kill-criteria clock in `REVISED_STRATEGY.md` §5 ("no design partner within two
quarters → treat as a research artifact") is what forced this. That clock is honest and
it is running. This pivot is the answer to it: stop waiting for a market to arrive and
go sell into one that exists, using assets built for the one that's coming.

---

## 2. The comparables — how the model actually works

Studied: **Fireworks AI, Baseten, Together AI, Modal, Anyscale.** All figures
web-verified 2026-07-26; sources in §10. Treat every third-party revenue figure as
reported-not-audited.

### 2.1 The five, side by side

| | Fireworks AI | Baseten | Together AI | Modal | Anyscale |
|---|---|---|---|---|---|
| **What they sell** | Fast serving of open models, per token | Production inference for *your* model, per GPU-minute + a token catalog | Open-model API **plus** GPU clusters | Python-native serverless compute, per second | Managed Ray; distributed inference graphs |
| **Primary meter** | per token | per GPU-minute (+ per token) | per token (~30–40% of rev) + per GPU-hour | per GPU-second | per compute-hour, sales-led |
| **Reported scale** | >$1B annualized (Jul 2026) | ~$600M annualized (Q1 2026), $13B val. (Jun 2026) | ~$1B annualized (Feb 2026), $8.3B post (Jul 2026) | ~$1.1B val. (Sep 2025), ~1,000 paying customers | ~$220M ARR (2025) |
| **Capacity posture** | Mixed rented/reserved | **Multi-cloud: 87 clusters, 18 providers** | Owns/operates clusters — sells them too | Rented, abstracted away | Runs in *customer's* cloud (incl. Azure tenancy) |
| **Anchor prices** | Serverless in $0.18–0.90/MTok; dense >16B $0.90/MTok flat; MoE ≤176B $1.20/MTok. Dedicated A100 $2.90/hr, H100/H200 $6.00/hr, B200 $9.00/hr. Batch = 50% | Dedicated H100 80GB $0.10833/min = **$6.50/hr**; B200 $0.16633/min; A100 $0.06667/min. Model APIs median $0.60 in / $2.20 out per MTok | $0.10–$9.00/MTok; HGX H100 $1.76–2.39/GPU-hr, H200 $3.15–3.79, B200 $4.00–5.50 (commitment-dependent) | H100 $0.001097/s = **$3.95/hr**; H200 $4.54/hr; B200 $6.25/hr; A100-80 $2.50/hr; L40S $1.95/hr. Starter $0/mo, Team $250/mo | Consumption + enterprise contracts; no public rate card |
| **Silicon** | NVIDIA | NVIDIA (multi-cloud) | NVIDIA (Hopper + Blackwell) | NVIDIA only — **no AMD on the rate card** | Whatever the customer's cloud has |

### 2.2 What the five have in common — the playbook to copy

1. **Two meters, one platform.** A *serverless* meter for zero-commitment adoption
   (per token or per second) and a *dedicated* meter for production (per GPU-minute/hour).
   Serverless is the funnel; dedicated is where the revenue and margin live. Every one of
   the five runs both. **We will too.**
2. **The model catalog is the storefront.** Adoption comes from "the model I want is
   already there, one API call away." Nobody adopts an inference platform by reading
   about its compiler.
3. **They are arbitrage businesses and they say so.** Together's own positioning is that
   falling GPU prices *lower their cost basis* while per-token prices hold. The margin is
   the spread between capacity cost and value-priced output. **This is the business we
   are entering, and it is a real one.**
4. **Multi-cloud capacity is a competitive weapon, not plumbing.** Baseten's 87 clusters
   across 18 providers is the clearest statement of it: route each workload to the
   cheapest capable capacity, minute by minute. This is directly copyable at our scale
   and is half of our cost lever.
5. **Benchmarks are the marketing.** All five publish latency/throughput comparisons
   constantly. We already have the rarest asset in this genre: a published methodology
   with disclosed negative results.
6. **Nobody wins on model quality.** They serve the same open weights. The entire
   competitive surface is **speed, cost, reliability, breadth, and developer experience.**
   That is a surface we can actually compete on.

### 2.3 Where the incumbent price stack leaves room

Public anchors, same GPU, July 2026:

| Layer | H100 $/GPU-hr | Note |
|---|---|---|
| Cost to serve at ~70% utilization | **~$1.45** | Third-party decomposition; hardware + power + DC |
| Budget neocloud on-demand | **$1.35 – $2.45** | Market median ≈ $2.45 across 50+ configs |
| Together (committed cluster) | $1.76 – $2.39 | Commitment-dependent |
| Modal (serverless, per-second) | **$3.95** | ~1.6× median on-demand |
| Fireworks (dedicated) | **$6.00** | ~2.4× |
| Baseten (dedicated) | **$6.50** | ~2.7× |
| Hyperscaler on-demand | $6.88 (AWS) – $12.29 (Azure) | The price customers flee |

And on AMD: **MI300X rents $1.99–$3.00/GPU-hr** (DigitalOcean at the low end), with
192 GB against H100's 80 GB.

Read that table as a business plan. **The platform layer charges 1.6–2.7× the cost of
the capacity underneath it, and none of the five prices off AMD at all.** A platform
that (a) runs well on MI300X-class parts and (b) needs fewer GPU-seconds per unit of
customer work has two independent, compounding routes to undercut all of them — which is
exactly the two levers chosen as our moat.

### 2.4 The gap none of them fills

Every one of the five is **an LLM company that grew a container service for everything
else.** The per-token abstraction, the model catalog, the autoscaler, the routing — all
of it is shaped around a stateless autoregressive text decoder. Bring them a CNN, a
segmentation model, a video diffusion model, a VLM pipeline, a VLA, or a world model and
the answer is the same: *here is a GPU-hour and a Dockerfile, good luck.* No per-call
economics, no workload-shaped scheduling, no benchmark leadership, no expertise.

That tail is not small. It is every production computer-vision system, every industrial
inspection line, every AV perception stack, every robotics fleet, every media pipeline,
and every multi-modal agent — and it is growing faster than text-only inference because
it is where physical AI lives.

**Repercep's position: the inference cloud that starts from the hard tail and grows into
LLMs, instead of the reverse.** We are credible on that claim in a way a Fireworks
engineer is not, because we already published cross-vendor serving numbers on three
world-model regimes and six models when nobody else published any.

---

## 3. The thesis, and why the world-model year was not a detour

> **Repercep is a full-service inference cloud for every model class, priced off a cost
> base the incumbents can't reach — because we get more customer-work per GPU-second on
> the workloads that aren't plain text, and we run on silicon they don't support.
> World models are where we proved it and where the market is going; LLMs and general
> model serving are how we build the run-rate on the way there.**

The load-bearing claim in that sentence is that the world-model work *transfers*. It
does, and the transfer is specific, not hand-waved. Three primitives:

### 3.1 Primitive 1 — shared-prefix candidate batching

**What we measured (OpenVLA-7B, both vendors, exact parity):** decoding N action-token
candidates as one batched pass sharing the prompt+vision prefix, instead of a
per-candidate loop, is **5.4→8.8× on H100 and 5.3→9.9× on MI300X** for N = 8→32.
Root cause, stated in the benchmark report: each candidate decode is only 7 tokens, so
batch-of-one badly underuses the GPU, and batching the shared prefix recovers the idle
capacity.

**Why it transfers:** that is a structural description of *every* parallel-sampling LLM
workload — best-of-N, self-consistency, agentic tree search, reranked tool-call
proposals, speculative fan-out. Long shared prefix, short divergent decodes, N of them,
under a latency deadline. The 2026 agentic workload shape **is** the VLA candidate-decode
shape. We optimized it first because robotics forced us to.

**Status: measured on VLA, hypothesis on LLM.** The experiment that settles it is in
`docs/INFERENCE_MOAT_TECHNIQUES.md` §T1 — benchmark best-of-N against stock vLLM
`n>1` and against Fireworks/Together's published per-token rates. Until that runs we
claim the VLA number and describe the LLM case as expected, not proven.

### 3.2 Primitive 2 — state-resident sessions

**What we measured:** closed-loop serving with state carryover as a *requirement*, not
an optimization — resident sessions per GPU as a first-class metric (LingBot-VA 11 on
H100 vs 30 on MI300X, 24 measured under load; DreamZero **1 on H100 vs 6 on MI300X** at
~23 GiB/session). Large-memory silicon converts a hard capacity ceiling into serving
headroom.

**Why it transfers:** a long-lived agent session with a large KV cache is the same
resource problem — per-session memory is the binding constraint on density, density is
the denominator of cost per session, and 192–288 GB parts change the arithmetic. The
incumbents' per-token meter actively hides this; their pricing assumes statelessness.
**Long-session agentic serving is a memory-density business, and memory density is where
AMD wins.**

### 3.3 Primitive 3 — deadline-aware scheduling

**What we built:** a serving stack whose scheduling target is a control deadline — the
answer is worth nothing after it passes — rather than aggregate throughput. That framing
is throughout `docs/CONTROL_LOOP_BENCH.md` and it is not how vLLM/SGLang schedule.

**Why it transfers:** interactive agents, voice, and real-time multi-modal pipelines have
exactly this property, and the industry currently expresses it as a crude p99 target.
This is the least-proven of the three primitives on LLM workloads and is labeled as such.

### 3.4 The honest version of the transfer claim

On **vanilla single-turn chat completions we have no efficiency advantage.** Our LLM path
today is a co-located reverse proxy to vLLM/SGLang (`feat/llm-proxy`, `docs/LLM_PROXY.md`)
— we ride the same open engine everyone else rides. On that workload our only cost lever
is silicon arbitrage, worth tens of percent, not multiples.

That is fine, and it is the correct segmentation:

- **Do not fight Fireworks on plain chat tokens.** They are at >$1B annualized with scale
  economics we cannot match in 2026. We match on price via silicon and win the deal on
  breadth, not on being 5× cheaper at something they've already optimized.
- **Do win the workload shapes where our primitives bite:** parallel-sampling/agentic,
  long-lived stateful sessions, and every non-text model class.

---

## 4. The moat: cost per unit of customer work

The chosen moat is *"we run it cheap so we can sell it cheap."* Stated that way it is a
race to the bottom. Stated correctly it is a structural position. The correct statement:

> **Price floor = (GPU-seconds per unit of customer work) × ($ per GPU-second) ÷
> utilization.** We own levers on the first two terms that our competitors do not, so our
> floor is below theirs *at the same margin*. We are not choosing to be cheap — we are
> able to be.

### 4.1 Lever A — software efficiency (fewer GPU-seconds per unit of work)

All measured by us, both vendors, archived JSON, caveats in the benchmark report:

| Lever | Measured win | Workload |
|---|---|---|
| Shared-prefix candidate batching | **5.3 – 9.9×** | Token-VLA action decode (OpenVLA-7B) |
| Candidate batching (energy-MPC) | 1.6 – 2.1× | V-JEPA 2-AC CEM planning |
| Adaptive step-skip cache | **3.2 – 3.8×** | Video diffusion (Cosmos) |
| Dynamic DiT-cache schedule | 3.19× | Policy diffusion (DreamZero) |
| Config-level portable levers | 2.0× | LingBot-VA chunk latency |
| Session density on large-memory silicon | **6×** | DreamZero resident sessions |

These do not all multiply, they apply to different workloads, and two of them trade
quality for speed in ways we have disclosed rather than buried. **The claim is not "we
are 10× cheaper." The claim is that on the workload classes we target we have a measured
2–10× efficiency lever and our competitors publish none.**

### 4.2 Lever B — alt-silicon arbitrage (cheaper GPU-seconds)

Measured: **26–39% lower cost per unit of work on MI300X than H100 at current rental
prices, across regimes.** That is our own number and it already includes the performance
gap, not just the sticker price.

Why this is not trivially copied:

- **None of the five comparables prices AMD at all.** Modal's rate card has no AMD line.
  Fireworks, Baseten and Together are NVIDIA shops. Decart's DOS supports NVIDIA,
  Trainium and TPU — and conspicuously skips ROCm.
- **We have a year of production-grade ROCm work no competitor has**: six models, three
  regimes, first published Cosmos benchmark on any AMD GPU, exact cross-vendor parity
  gates, and a ledger of the ROCm traps (autotune cost, torch/ROCm version pinning,
  attention-backend dispatch) that cost us months to learn.
- **The memory story compounds it.** 192 GB (MI300X) and 288 GB (MI355X) change session
  density, and density is the denominator of per-session cost.

**Honest limits, stated up front:**

- AMD supply and neocloud availability is genuinely constrained — our own build log
  records RunPod's AMD catalog being empty when checked, and the AMD Developer Cloud
  workaround. Capacity sourcing is an execution risk, in §9.
- The advantage is **time-boxed**. If Fireworks ships ROCm, Lever B compresses to the raw
  price delta. Estimated durability: 12–24 months, and the answer is to be the entrenched
  AMD-native inference platform before it closes.
- Several of our MI300X numbers were taken on **virtualized slices**, disclosed in the
  report. Bare-metal confirmation remains open.

### 4.3 Why the two levers compound

Lever A applies on *both* vendors. Lever B applies to the capacity underneath. Against a
customer's real alternative — a stock serving stack on NVIDIA on-demand — the combined
delta on our target workloads is multiplicative in the numerator and additive-in-percent
in the denominator. Against the same stack on the same NVIDIA GPU, Lever A alone still
holds. **Neither lever depends on the other, which is what makes the cost position
robust rather than a single bet.**

### 4.4 What is emphatically *not* the moat

Being cheap is not a moat if the only mechanism is thinner margin. We will not
price below cost to buy logos. Every price in `docs/BUSINESS_PLAN_2026.md` §5 is derived
from a cost stack with a stated gross margin, and the discount versus incumbents comes
out of the cost base, not the margin.

---

## 5. Product architecture

Three products, one engine. The current site's four-product ladder
(Runtime/Studio/Forge/Worlds) is replaced.

### 5.1 Repercep Cloud — *the business*

Managed inference on partner GPU capacity. Two meters, per §2.2:

- **Serverless** — per token (LLM/VLM) or per second/per call (vision, diffusion, VLA,
  world models). Zero commitment. The funnel.
- **Dedicated** — per GPU-minute, autoscaled, scale-to-zero. Where production and margin
  live.

Covers **every model class**: CNNs and classical vision, encoders/embeddings,
speech, transformers, LLMs, VLMs, diffusion, VLAs, world models. Breadth is the product,
and it is credible because the engine already runs six models across three regimes and
four silicon targets.

### 5.2 Repercep Runtime — *the engine, sold to those who can't use a cloud*

Self-host and BYOC licensing for teams with data-sovereignty, on-prem fleet, air-gapped,
or defense constraints — an entire segment the hosted-only incumbents (and Decart's
Oasis) structurally cannot serve. Same code as Cloud; that is the point.

### 5.3 Repercep Link — *the hybrid edge↔cloud product for physical AI*

The user-specified item 4, made concrete. **We do not build or sell edge execution.** We
provide the split:

- **On the edge processor** (Jetson Thor-class, Ryzen AI, Qualcomm Dragonwing) the
  partner runs the tight, safety-critical loop — reflexes, low-level control, the
  policy chunk that must not depend on a network.
- **In Repercep Cloud** runs the heavy, strategic, latency-tolerant half — long-horizon
  world-model rollouts, multi-candidate planning, large VLM reasoning, fleet-wide
  learning and replay.
- **Repercep Link** is the protocol and scheduler between them: what runs where, under
  what deadline, with what fallback when the link degrades. Graceful degradation to
  edge-only is a requirement, not a feature.

Go-to-market is **through robotics and automation companies**, not direct to end users —
they own the edge silicon choice and the customer. Jetson Thor's T3000/T2000 launch
(July 2026) and its adoption list (1X, Boston Dynamics, FANUC, Amazon Robotics, Hitachi,
Agile Robots) is the partner surface.

**Status: this is the least-built of the three.** Link is a design and a set of design
partners, not shipped code. It is a 2027 product on this plan and is labeled that way in
the deck.

### 5.4 Explicitly dropped: Repercep Worlds (owned models)

Training our own world models contradicts the neutral-platform position. Fireworks,
Baseten, Together and Modal all deliberately do not compete with their customers'
models; Decart, which does, has to sell a closed hosted API as a result. **Recommend
removing it from the site and the roadmap.** Flagging rather than deciding unilaterally —
it is currently public messaging.

---

## 6. Sequencing — how the run-rate gets built

The user's framing is exactly right: *start LLM revenue now, be ready when world models
arrive.* Concretely:

| Phase | Window | What ships | What it earns |
|---|---|---|---|
| **P0 — Make it real** | now → 2026-10 | Harden `feat/llm-proxy` into a metered multi-tenant gateway: auth, quotas, per-token/per-second metering, billing, a model catalog, an OpenAI-compatible API. CI, which still does not exist. | Nothing. Prerequisite. |
| **P1 — LLM run-rate** | 2026-10 → 2027-01 | Public serverless + dedicated on MI300X-first capacity. 20–40 open models. Price at/below Fireworks on the strength of Lever B. | First paying customers. Target logos over ARR. |
| **P2 — Own the tail** | 2027-01 → 2027-06 | The non-LLM catalog nobody else prices per call: vision, embeddings, diffusion, VLM pipelines. This is the differentiated wedge and the higher-margin half. | The ARR that justifies the A. |
| **P3 — Physical AI** | 2027-06 → | Repercep Link with 2–3 robotics partners; world-model serving productized on top of the same cloud. | The option we've been holding. |

The world-model assets are **not shelved** during P1–P2. They are the benchmark
leadership that makes us credible, the source of the efficiency levers, and the reason a
robotics buyer takes the meeting. But they stop being the thing we ask a customer to buy
before it is ready to be bought.

---

## 7. What we claim / what we don't

**Claim (all measured by us, archived, caveated):**
- Six models, three regimes, two vendors, one unmodified engine.
- 5–10× shared-prefix candidate batching on token-VLA decode, exact parity, both vendors.
- 3.2–3.8× adaptive-cache generation speedup, reproduced across sessions and stacks.
- 6× resident session density on MI300X vs H100 for a 16.5B policy model.
- 26–39% lower cost per unit of work on MI300X at current rental prices.
- First publicly reported Cosmos benchmark on any AMD GPU (a claim about the state of
  public disclosure on the date made — we catalog where we searched).

**Do not claim:**
- **Any LLM performance or cost number we have not run.** Today's LLM path is a proxy in
  front of vLLM/SGLang. Until §3.1's experiment lands, LLM efficiency is a hypothesis.
- **"Cheapest inference"** as an unqualified superlative. Claim it per workload class,
  with the cost stack shown.
- **Named cloud partners** in any public artifact. Contact is informal; named partners
  appear only in the confidential deck and business plan, marked by status.
- **A hosted product that exists.** Repercep Cloud is in build. Every external artifact
  must say design-partner/private-beta, not GA.
- **Repercep Link as shipping.** It is a 2027 design.
- The standing caveats from `REVISED_STRATEGY.md` §6 — no "real-time interactive world
  model," no generation-kernel superiority claims — all still apply.

---

## 8. What this pivot costs us

Stated plainly, because a deck that omits it fails diligence:

1. **We enter a market with three ~$1B-revenue incumbents.** Fireworks, Together and
   Baseten are all at or near $1B annualized with $8–13B valuations. We are not going to
   out-execute them on plain LLM serving. The plan depends on the tail being real and on
   AMD staying uncontested for 12–24 months.
2. **We take on operating risk we did not have.** A runtime that ships has no uptime SLA.
   A cloud does. Multi-tenancy, billing correctness, capacity management, on-call — none
   of it exists today, and the repo has no CI at all.
3. **Focus dilution is the real danger.** "Every model class" can mean "nothing served
   well." The mitigation is P2: the tail is the *differentiated* wedge, LLMs are the
   *volume*, and world models are the *proof* — not three equal priorities.
4. **The narrative gets harder to hold.** "World-model runtime" was sharp. "Inference
   cloud" is crowded. The shared-primitive argument in §3 is what keeps it coherent, and
   it must be told well or we read as a team chasing revenue.

---

## 9. Risks and kill criteria

| Risk | Severity | Mitigation / kill signal |
|---|---|---|
| **AMD capacity sourcing fails.** Not enough MI300X/MI355X at the prices Lever B assumes. | **High** — it is half the moat | Multi-provider from day one (Baseten's 18-provider posture). Kill signal: cannot secure committed MI300X at <$2.20/hr by 2026-11 → re-plan on NVIDIA and lean entirely on Lever A. |
| **Lever A doesn't transfer to LLMs.** Best-of-N batching gives us nothing vLLM `n>1` doesn't. | **High** | Run the experiment *before* the deck goes out (§3.1). Kill signal: <1.5× over stock vLLM on best-of-N → drop the LLM efficiency claim, compete on breadth + silicon only. |
| **A major platform ships ROCm.** Lever B compresses to raw price delta. | Medium-high | Speed. Be the entrenched AMD-native platform first; deepen into MI355X. Watch competitor job posts. |
| **The tail doesn't pay.** Non-LLM serving stays a Docker-and-a-GPU-hour business because customers don't want a per-call meter. | Medium | P2 is explicitly the test. Kill signal: <5 paying tail customers by 2027-06 → the company is an AMD-native LLM cloud, which is a smaller but real business. |
| **Physical-AI demand stays 2028.** | Medium | This pivot is precisely the hedge — the run-rate does not depend on it. |
| **We are outspent on GTM.** Incumbents have $800M–$1.5B rounds. | Medium | Never compete on brand spend. Compete on published, reproducible numbers — the one asset money can't instantly buy and the one we already have. |
| **Operating immaturity burns a customer.** Billing bug, outage, data incident. | Medium | P0 gates GA. No paying multi-tenant traffic before CI, quotas, and auth exist. |

**Company-level kill criterion, replacing the one in `REVISED_STRATEGY.md` §5:** if
Repercep Cloud has **no paying customers by 2027-03** (roughly two quarters after P1
opens), the hosted thesis is wrong and the fallback is the runtime-licensing + benchmark
business, which is smaller and does not need a seed round.

---

## 10. Sources (web-verified 2026-07-26)

- **Fireworks:** https://sacra.com/c/fireworks-ai/ · https://www.morphllm.com/fireworks-ai-pricing · https://pricepertoken.com/pricing-page/provider/fireworks
- **Baseten:** https://sacra.com/c/baseten/ · https://valueaddvc.com/blog/baseten-valuation-revenue-2026-13b-ai-inference · https://www.morphllm.com/comparisons/fireworks-vs-baseten · https://costbench.com/software/ai-model-hosting/baseten/
- **Together:** https://sacra.com/c/together-ai/ · https://www.together.ai/ · https://www.cloudzero.com/blog/together-ai-pricing/
- **Modal:** https://modal.com/pricing · https://sacra.com/c/modal-labs/
- **Anyscale:** https://aimatrixmap.com/companies/anyscale · https://www.beri.net/article/anyscale-azure-sovereign-ai-90-percent-cost-savings-2026
- **GPU price stack:** https://siliconanalysts.com/analysis/gpu-rental-premium-decomposition-2026 · https://gpufinder.dev/gpu/h100 · https://gpufinder.dev/gpu/mi300x · https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/
- **Inference techniques:** https://www.morphllm.com/llm-inference-optimization · https://llm-d.ai/blog/llm-d-v0.5-sustaining-performance-at-scale · https://arxiv.org/html/2606.17104v1
- **Physical AI / edge:** https://www.globenewswire.com/news-release/2026/07/01/3320809/28124/en/physical-ai-market-set-to-surpass-430-billion-by-2030-driven-by-nine-key-vertical-sectors.html · https://blogs.nvidia.com/blog/jetson-thor-robotics-edge-ai-agent/ · https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/
- **Internal:** `docs/REVISED_STRATEGY.md` · `docs/AUDIT_2026_07_STATUS_GAPS_ROADMAP.md` · `docs/CONTROL_LOOP_BENCH.md` · `docs/METHODOLOGY.md` · `docs/VLA_ON_{H100,MI300X}.md` · `docs/LLM_PROXY.md` · `reports/benchmark-report-2026-07.html`
