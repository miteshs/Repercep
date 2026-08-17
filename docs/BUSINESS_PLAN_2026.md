# Repercep — Business Plan (2026-07-26)

**CONFIDENTIAL.** Contains partner names, pricing strategy, and financial modelling.
Never publish to `repercep-site` or any public artifact.

- **Companion to:** `docs/PIVOT_2026_07_PLATFORM_STRATEGY.md` (why), this doc (how much,
  from whom, at what price), `docs/INFERENCE_MOAT_TECHNIQUES.md` (the technical basis of
  the cost claim).
- **Convention:** every number is tagged **[M]** measured by us, **[V]** externally
  verified with a source, or **[A]** assumption/model input that is not yet measured.
  No number appears untagged. Assumptions that carry the plan are called out in §6.4.

---

## 1. The business in five lines

1. Repercep sells **inference as a service** on partner GPU capacity, for **every model
   class** — not just LLMs.
2. We win on **cost per unit of customer work**, from two independent levers: **software
   efficiency** (measured 2–10× on our target workloads) and **alt-silicon arbitrage**
   (measured 26–39% cheaper per unit of work on AMD).
3. **LLM and general serving builds the run-rate now.** The non-LLM tail — vision,
   diffusion, VLM, VLA, world models — is the differentiated, higher-margin wedge that no
   incumbent prices per call.
4. For **physical AI** we sell a hybrid: the partner's edge processor runs the tight
   loop, our cloud runs the heavy reasoning. We do not build edge execution.
5. We are raising a **$8–10M seed** to convert a proven runtime into a metered,
   multi-tenant cloud and get to first revenue.

---

## 2. Market

### 2.1 Where the money is, today

The inference-platform layer is proven and large. Three direct comparables crossed ~$1B
annualized inside eighteen months **[V]**:

| Company | Reported annualized revenue | Valuation | Date |
|---|---|---|---|
| Fireworks AI | >$1B | ~$17.5B implied | Jul 2026 |
| Together AI | ~$1B (bookings >$1.15B) | $8.3B post | Feb / Jul 2026 |
| Baseten | ~$600M (from $200M one quarter earlier) | $13B | Q1 / Jun 2026 |
| Anyscale | ~$220M ARR | — | 2025 |
| Modal | not disclosed | ~$1.1B | Sep 2025 |

**[V]** The category question — "will anyone pay for a layer between the model and the
GPU?" — is answered. We are not funding market creation; we are funding entry.

### 2.2 Where the money is going

- **Physical AI to surpass $430B by 2030** across nine verticals **[V]**; physical-AI
  *software platforms* ~$17.2B by 2030 **[V]**. VLA models are named as the core
  cognitive engine of that stack **[V]**.
- **Jetson Thor** shipped mainstream robotics modules (T3000/T2000) in July 2026, with
  1X, Boston Dynamics, FANUC, Amazon Robotics, Hitachi, Agile Robots and Techman
  building on it **[V]**. The edge compute layer for physical AI now exists at volume —
  which is precisely what makes a hybrid edge↔cloud product buildable, and what makes
  the cloud half of it purchasable.
- Embodied-AI funding was projected >$20B in 2026 **[V]**, and Goldman projects $50B
  cumulative humanoid investment by 2030 **[V]**.

**The timing read:** the world-model/physical-AI serving market is real but still
early — the honest 2027–28 risk from `REVISED_STRATEGY.md` §5 stands. This plan does not
depend on it arriving early. It builds revenue on demand that exists and holds the
physical-AI position as a paid-for option.

### 2.3 Segments, nearest-revenue first

| # | Segment | What they buy | Why us | Cycle | Deal size (yr 1) |
|---|---|---|---|---|---|
| **S1** | AI-native startups & product teams burning LLM spend | Serverless tokens, then dedicated | Price, at parity DX. Fast, self-serve, no procurement | days | $1–20k/mo |
| **S2** | **Production CV / industrial / media teams** (the tail) | Per-call serving of CNNs, detectors, encoders, diffusion, VLM pipelines | **Nobody prices this per call.** They are renting GPU-hours and eating idle | 4–10 wks | $5–50k/mo |
| **S3** | Robotics / AV / embodied teams | VLA + world-model serving; later Repercep Link | Only published cross-vendor closed-loop numbers anywhere; self-host option | 2–5 mo | $25–100k pilot |
| **S4** | Regulated / sovereign / defense / on-prem fleets | **Repercep Runtime** license, BYOC | Hosted-only incumbents structurally cannot serve them | 3–9 mo | $50–250k/yr |
| **S5** | GPU neoclouds (esp. AMD fleets) | Co-sell; our stack pulls workloads onto their idle capacity | We make their AMD inventory sellable for inference | 1–3 mo | rev-share / NRE |

**S1 is volume. S2 is the wedge. S3 is the future. S4 is margin. S5 is the channel.**

Note the asymmetry that makes S5 unusually valuable: neoclouds with AMD fleets have a
demand problem, not a supply problem. We are one of very few teams who can credibly move
production inference onto their idle MI300X inventory — which is the same fact that makes
Lever B cheap for us. **The channel and the cost moat are the same relationship.**

---

## 3. Competitive position

Full teardown in `docs/PIVOT_2026_07_PLATFORM_STRATEGY.md` §2. The one-table version:

| | Fireworks | Baseten | Together | Modal | **Repercep** |
|---|---|---|---|---|---|
| LLM serving | ●●● | ●●● | ●●● | ●● | ●● (rides vLLM/SGLang) |
| Non-LLM model classes | ○ | ● | ● | ● (generic compute) | **●●●** |
| World models / VLA | ○ | ○ | ○ | ○ | **●●● (only published numbers)** |
| AMD / alt-silicon | ○ | ○ | ○ | ○ (no AMD on rate card) | **●●●** |
| Self-host / BYOC | ○ | ● | ● | ○ | **●●●** |
| Edge↔cloud hybrid | ○ | ○ | ○ | ○ | **●● (2027)** |
| Scale, brand, capital | ●●● | ●●● | ●●● | ●● | ○ |

We lose the last row decisively and win five of six others. **The plan is to never
compete where the last row decides the outcome** — no brand-spend war, no fight for
plain chat tokens at the head of the distribution.

---

## 4. Products and packaging

### 4.1 Repercep Cloud

- **Serverless** — per token (LLM/VLM), per second or per call (vision, diffusion,
  VLA, world models). No commitment. Free tier with credits, matching Modal's Starter
  model **[V]**.
- **Dedicated** — per GPU-minute, autoscaling, scale-to-zero idle (Baseten's model
  **[V]** — customers strongly prefer not paying for idle replicas).
- **Batch** — 50% of serverless, matching Fireworks **[V]**. Cheap for us: it fills
  utilization troughs, which is exactly what the margin model in §6 needs.

### 4.2 Repercep Runtime

Annual license, self-host or BYOC, priced per node or per GPU. Serves S4, and doubles as
the fallback business if the hosted thesis fails (§9).

### 4.3 Repercep Link (2027)

Hybrid edge↔cloud split for physical AI. Priced per robot-session/month plus cloud
consumption. Sold **through** robotics/automation partners. Design stage, not code.

### 4.4 Benchmark reports

Already-live motion (`reports/`, the request form on the site). $5–25k per report, and
every report is S2/S3 lead-gen. Keep it — it is the cheapest willingness-to-pay test we
have and it is running today.

---

## 5. Pricing

### 5.1 The anchors we price against **[V]** (July 2026)

| | Serverless | Dedicated H100 |
|---|---|---|
| Fireworks | $0.18–0.90/MTok in; dense >16B $0.90/MTok flat; MoE ≤176B $1.20/MTok | $6.00/hr (A100 $2.90, B200 $9.00) |
| Baseten | Model APIs median $0.60 in / $2.20 out per MTok | **$6.50/hr** ($0.10833/min) |
| Together | $0.10–$9.00/MTok | $1.76–2.39/hr committed cluster |
| Modal | — (compute, not tokens) | **$3.95/hr** ($0.001097/s) |

Underneath them: H100 on-demand **$1.35–2.45/hr**, MI300X **$1.99–3.00/hr**, and a
cost-to-serve floor around **$1.45/GPU-hr at 70% utilization** **[V]**.

### 5.2 Our price posture

| Workload | Posture | Why |
|---|---|---|
| Vanilla LLM tokens (S1) | **10–25% under Fireworks/Baseten** | Silicon arbitrage only. Enough to win a switch, not enough to start a price war we'd lose |
| ~~Agentic / best-of-N LLM~~ | **row deleted 2026-07-27** | There is no separate agentic price posture. The 1.5× lever is real but not ours to capture or to differentiate on (§6.3) — agentic traffic prices like any other LLM traffic, off silicon |
| Non-LLM tail (S2) | **Per-call pricing where the alternative is a rented GPU-hour** | Customers today pay for idle. Per-call at 60–70% margin still reads as a large saving to them |
| Dedicated | **$3.50–4.50/hr H100-class, $2.75–3.50/hr MI300X-class** | Under Modal, well under Fireworks/Baseten |
| Batch | 50% of serverless | Fills troughs |

**Never** price below the cost stack in §6 to buy a logo. The discount comes out of the
cost base, not the margin — that is the entire point of the moat.

---

## 6. Unit economics

### 6.1 The formula

```
price floor per unit of work  =  GPU-seconds/unit  ×  $/GPU-second  ÷  utilization
                                 └── Lever A ──┘     └── Lever B ──┘   └ operations ┘
```

### 6.2 Reverse-derived incumbent margin **[V]** inputs, **[A]** throughput

Nobody publishes their cost stack, so we derive it from public prices. **The one
unmeasured input is aggregate serving throughput; 2,000 tok/s per H100 on a dense
>16B model is the [A] assumption, and §6.4 shows the sensitivity.**

| | Fireworks on H100 | Repercep on MI300X |
|---|---|---|
| Capacity cost/hr | $2.45 **[V]** median on-demand | $2.20 **[A]** committed target (observed range $1.99–3.00 **[V]**) |
| At 70% utilization | $3.50 | $3.14 |
| Throughput | 2,000 tok/s **[A]** | 2,000 tok/s **[A]** — parity assumed, not measured |
| Tokens/hr | 7.2M | 7.2M |
| Price | $0.90/MTok **[V]** | $0.675/MTok (**25% under**) |
| Revenue/hr | $6.48 | $4.86 |
| **Gross margin** | **46%** | **35%** |

Read honestly: **on plain tokens, silicon arbitrage alone buys a 25% price cut at a
gross margin 11 points thinner than the incumbent's.** That is a real but modest
position — it wins switches, it does not win a war. It is why §5.2 caps the vanilla-LLM
discount at 25% and why the plan does not rest on it.

### 6.3 ~~The same math where Lever A applies~~ — **WITHDRAWN 2026-07-27**

**This table claimed an advantage that is not ours. It is withdrawn, not restated at a
smaller number.** Full evidence: `docs/LLM_BESTOFN_RESULT.md` (three parts).

The 1.5× is real. It reproduced twice on independent pods and measured *larger* on
SGLang (1.67×). But two later measurements removed it from the plan:

1. **We cannot capture it for the customer.** Fusing concurrent requests into one `n=N`
   call at our gateway was measured end-to-end and came out **slower than doing
   nothing** — 1.02× of plain fan-out at concurrency 1, 1.23× slower at 16. A 30×
   longer coalescing window never got the mean batch past 4.45 out of 16, because
   requests a client issues simultaneously do not *arrive* simultaneously.
2. **Even if we could, it would not differentiate us.** `n=N` is a stock feature of
   every serving engine. A customer who sends `n=N` gets the same efficiency from
   Fireworks, Baseten or Together as from us. The advantage was never in having the
   lever — it was in capturing it for callers who *don't* send `n=N`, and that is the
   part that failed.

**Consequence for pricing.** There is no separate agentic-workload margin story. §6.2
— silicon arbitrage, ~25% under at ~35% gross margin — is now the *whole* LLM economic
case, and §5.2's agentic row is deleted rather than reduced.

**What survived, and it is small but ours.** SGLang served this workload shape 9–17%
faster than vLLM at identical settings. Engine selection *is* a lever we control as the
operator — the customer never sees which engine runs under the API. That is a real
9–17%, not a 1.5×, and it belongs in the deployment playbook rather than the pitch.

> **The honest one-line version for any investor conversation:** *we measured a 1.5×
> serving lever on agentic LLM traffic, then measured that we cannot deliver it, and
> removed it from the plan. Our LLM cost position is silicon, plus a 9–17% engine-choice
> effect.*

### 6.4 Sensitivity — what breaks the plan

| Input | Plan value | Breaks if | Consequence |
|---|---|---|---|
| MI300X committed rate | $2.20/hr modelled; **$1.99/hr observed [V]** | >$2.80/hr | Lever B gone; §6.2 margin → ~19% |
| ~~MI300X vs H100 token throughput~~ | **MEASURED 2026-08-09: `R = 1.02–1.38×` [V]** | ~~<0.75×~~ **did not fire** | **Criterion cleared.** See `docs/LLM_SILICON_GATE_RESULT.md` |

**The throughput criterion is no longer an assumption.** Measured on
2026-08-09 under a pre-registered plan (`docs/LLM_SILICON_GATE_PLAN.md`): a
single MI300X matched or beat a single H100 on output throughput at **every
shape and both models** tested — Qwen2.5-7B and -32B, interactive at
concurrency 1 and 32, and prefill-heavy batch — giving **1.7–2.3× lower cost
per million output tokens** at observed prices ($1.99/hr vs $3.29/hr). The
result is a *lower bound*: the AMD leg ran an older vLLM (0.23.1 vs 0.26.0),
a handicap that favours H100.

> **MEASURED LIMIT, 2026-08-16.** The "no MoE" caveat below is no longer a
> precaution. `docs/LLM_MOE_GATE_RESULT.md` ran DeepSeek-V2-Lite on both
> vendors at matched engine versions: **the comparison inverts on MoE** —
> MI300X reached only `R = 0.53 / 0.74 / 0.60` of H100, failing the 0.75
> threshold at every shape. The dense result stands exactly as measured; its
> scope is now a boundary we have tested rather than assumed, and **"dense"
> must appear in any external use of the silicon claim.** One operator-side
> gain came out of it: ROCm's *default* kernel selection leaves 22–35% on the
> table versus AMD's AITER library, though AITER's output divergence is
> unresolved and it is not yet safe to deploy on that basis.

**Three limits on that claim, which belong in any external use of it:** it
covers **dense Qwen2.5-class models only** (no MoE — now measured as a real
boundary, above; no long-context, no FP8);
it is **one box per vendor, one session**; and the cost half is
**provider-confounded** — $1.99 is AMD Developer Cloud and $3.29 is RunPod, so
part of the advantage is procurement rather than engineering, and procurement
advantages are less durable. The throughput half is a clean silicon comparison;
the cost half is not.
| ~~Lever A on LLM~~ | **withdrawn 2026-07-27** | — | **Not a differentiator.** The 1.5× is real but available to every provider running the same engine; we could not capture it for customers who don't already send `n=N`. §6.3 rewritten below |
| Utilization | 70% **[A]** | <50% | All margins ~20 points thinner. **Most likely early failure mode** |
| Tail willingness to pay per call | assumed **[A]** | customers insist on GPU-hours | S2 wedge collapses to commodity hosting |

**Utilization is the quiet one.** Every margin above assumes 70%; a young platform with
lumpy demand runs far below that. Mitigations: batch tier as trough-filler, dedicated
deployments as a utilization floor, and multi-tenant packing across model classes — the
last being a genuine advantage of breadth, since vision and LLM demand curves are not
correlated.

---

## 7. Go-to-market

### 7.1 Motions, in dependency order

1. **Self-serve (S1).** Public rate card, free credits, OpenAI-compatible API, model
   catalog. No sales touch. This is the funnel and it must exist before anything else.
2. **Benchmark-led content (S2, S3).** Our differentiated asset. Publish what nobody
   else does: cost-per-unit-of-work across model classes and vendors, with methodology
   and disclosed negatives. Every report is lead-gen and every number is already the way
   we work.
3. **AMD ecosystem (S5).** ROCm-blog guest post, AMD Dev Cloud, ISV program, AMD
   Ventures adjacency. We hold the only published world-model serving numbers on
   Instinct; AMD's physical-AI push has a training story and no serving story. Carried
   over from `AUDIT` §3.1 — still the highest-leverage unpaid channel we have.
4. **Neocloud co-sell (S5).** Preinstalled Repercep image + co-published benchmark
   ("N resident sessions per MI300X node"). They get demand for idle AMD inventory; we
   get capacity economics. Both sides are motivated.
5. **Design partners (S3, S4).** 2–3 paid pilots, $25–100k, 4–8 weeks.
6. **Robotics/automation partnerships (Link, 2027).** Through the Jetson Thor / Ryzen AI
   / Dragonwing ecosystem, not direct.

### 7.2 Partner pipeline

> **Status legend:** `USING` = we have an active account and have run production
> benchmarks · `CONTACT` = informal conversation, nothing signed · `TARGET` = identified,
> not yet approached. **Nothing here is public. Nothing here may be named on the website
> or in any external artifact without a signed agreement** (`PIVOT` §7).

| Provider | Silicon | Status | Why them |
|---|---|---|---|
| RunPod | H100, MI300X | **USING** — proven benchmark platform, H100 + MI300X rows in the report | Fast provisioning; AMD catalog availability is intermittent (documented in build log) |
| AMD Developer Cloud (DigitalOcean-backed) | MI300X | **USING** — DreamZero + OpenVLA MI300X rows run here | Cheapest observed MI300X ($1.99/hr class); direct line into AMD's ecosystem |
| TensorWave | MI300X/MI325X | `TARGET` | AMD-only fleet — structurally the best-aligned capacity partner |
| Hot Aisle | MI300X | `TARGET` | AMD-only, bare-metal — would retire the VF-slice caveat in the report |
| Vultr | MI300X, NVIDIA | `TARGET` | Broad regions, AMD inventory |
| Crusoe | NVIDIA, AMD | `TARGET` | Scale + energy cost position |
| Nebius / Lambda / Voltage Park | NVIDIA | `TARGET` | NVIDIA-side capacity for parity work and NVIDIA-preferring customers |
| **[CONFIRM]** | | | **Mitesh: add the providers you have actually spoken to and correct any status above. I marked only what the repo and build logs evidence.** |

### 7.3 Positioning message by segment

- **S1:** "Same models, same API, 25% less."
- **S2:** "You're renting a GPU-hour to run a 40 ms model. Pay per call."
- **S3:** "The only inference platform that publishes closed-loop numbers — on the silicon
  you can actually get."
- **S4:** "The same engine, inside your walls."
- **S5:** "We make your AMD inventory sell."

---

## 8. Plan of record — 18 months

| Phase | Window | Milestones | Exit criteria |
|---|---|---|---|
| **P0 Foundation** | now → Oct 2026 | Metering, auth, quotas, billing, model catalog, OpenAI-compatible API. Harden `feat/llm-proxy` to multi-tenant. ~~Run the §6.3 gate experiment~~ — **done 2026-07-27, claim withdrawn**. Secure committed MI300X capacity. **Default the LLM path to SGLang** (§6.3, 9–17% faster on this shape). | Private beta serving real traffic; capacity contracted |
| **P1 Run-rate** | Oct 2026 → Jan 2027 | Public serverless + dedicated. 20–40 open models. Self-serve billing. First benchmark report under the new positioning. | **10 paying customers; $25–50k MRR** |
| **P2 The tail** | Jan → Jun 2027 | Non-LLM catalog priced per call: vision, embeddings, diffusion, VLM. 2 design-partner pilots. 1 Runtime license. | **$150–250k MRR; ≥5 paying S2 customers** |
| **P3 Physical AI** | Jun → Dec 2027 | Repercep Link with 2–3 robotics partners. World-model serving productized on Cloud. | **$400k+ MRR; Series A** |

Revenue targets are **[A]** and deliberately modest against the comparables' curves —
they are what a seed-stage team should be held to, not what makes a deck look good.

---

## 9. The raise

**Seeking $8–10M seed.** Use of funds over ~24–30 months:

| Line | Monthly (steady) | Notes |
|---|---|---|
| Engineering (6–8) | ~$165k | Serving/platform, ROCm/kernels, one SRE hire early |
| GTM (2) | ~$45k | DevRel first — this is a benchmark-led, developer-adopted business |
| Compute (dev, bench, pre-revenue capacity) | $50–80k, rising | Also the capacity commitments Lever B needs |
| G&A, legal, tooling | ~$30k | |
| **Total** | **~$300–320k/mo** | ~$9M ⇒ **28–30 months** |

**What the money buys, in one sentence:** it converts a runtime that is proven on the
hardest workloads in the market into a metered cloud with paying customers, while the
AMD window is open.

**Why a seed and not bootstrapping:** capacity commitments (Lever B) and multi-tenant
operations both need capital ahead of revenue. Neither is optional.

**Fallback if the hosted thesis fails** (no paying customers by 2027-03, `PIVOT` §9):
Runtime licensing (S4) + benchmark reports (§4.4) is a real, smaller, capital-light
business that the same assets support. That fallback is why the downside is bounded.

---

## 10. Why this team

- **A working runtime**, not a deck: six models, three serving regimes, four silicon
  targets, one unmodified engine **[M]**.
- **The only published cross-vendor closed-loop serving benchmarks** we can find,
  including on AMD **[M]**.
- **A year of production ROCm experience** that competitors do not have and cannot
  shortcut — including the failure ledger.
- **A documented culture of publishing negative results** (ADR-0009's two GPU-verified
  negative findings; the disclosed CFG-distribution-shift caveat; the "the model's own
  published latency isn't full compute" disclosure). In a category where everyone
  benchmarks and nobody discloses, this is a commercial asset, not just hygiene.

---

## 11. Open items before this plan is investor-ready

**Updated 2026-08-16.** Three of the six are closed. What remains is
uniformly commercial, which is itself the finding — see
`docs/BENCHMARK_PROGRAM.md`.

1. ~~**Run the §6.3 gate experiment.**~~ **Done 2026-07-27** — measured 1.5×, table
   rebuilt, price posture revised down (`docs/LLM_BESTOFN_RESULT.md`). The SGLang
   follow-on also ran (`6ca6889`): the lever measured *larger* on SGLang (1.67×), and
   the claim was withdrawn anyway because we cannot capture it for the customer.
2. ~~**Measure MI300X LLM throughput vs H100.**~~ **Done 2026-08-09** — `R = 1.02–1.38×`
   at every shape, 1.7–2.3× cheaper per Mtok (`docs/LLM_SILICON_GATE_RESULT.md`).
   §6.2's parity assumption is retired; the surviving scope limit is dense
   Qwen2.5-class only, and the MoE counter-example is pre-registered but unrun
   (`docs/LLM_MOE_GATE_PLAN.md`).
3. ~~**Fix `AUDIT` §1.2 P0 items** — no CI, failing lint/typecheck.~~ **Done** — CI runs
   ruff, `mypy --strict` and pytest plus a CPU end-to-end smoke on every PR and every
   push to `main` (`.github/workflows/ci.yml`); all three are green. The metered
   multi-tenant gateway that §8's P0 asks for now exists in part
   (`docs/METERED_GATEWAY.md`): per-customer keys, exact token metering, an append-only
   usage ledger, and quotas. Still missing from P0: self-serve signup, payments,
   autoscaling, and a model catalog.
4. **Confirm the partner list** in §7.2 — replace `[CONFIRM]` with real status.
5. **Get a committed MI300X quote** to replace the $2.20/hr **[A]** with a real number.
6. ~~**Decide on Repercep Worlds**~~ **Done 2026-08-16** — dropped from the public site
   per `PIVOT` §5.4; the platform-neutrality contradiction is resolved.

**The remaining two are both commercial, and so are all five of
`decks/VC_QA.md` §6's losing questions.** No further measurement moves them.
That is the explicit basis on which `docs/BENCHMARK_PROGRAM.md` closes the
gate program until the round is done.
