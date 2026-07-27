# Repercep — The Technique Ledger

### What we actually do that makes inference cheaper

- **Date:** 2026-07-26
- **Purpose:** this is the document behind the sentence *"our moat is that we run it
  cheap, so we can sell it cheap."* It is the answer to a technical diligence question
  and to a customer's "why are you cheaper?" — so every entry states **what the technique
  is, what it costs us to run, what we measured, and what we have not.**
- **Companions:** `docs/PIVOT_2026_07_PLATFORM_STRATEGY.md` §4 (why cost is the moat),
  `docs/BUSINESS_PLAN_2026.md` §6 (the money arithmetic).
- **Rule of this doc:** techniques are sorted by **whether we have measured them**, not
  by how impressive they sound. Tier 0 is what everyone has and we do not claim. Tier 1
  is ours and measured. Tier 2 is committed. Tier 3 is a bet.

---

## 0. The cost identity everything maps to

```
cost per unit of customer work  =  GPU-seconds/unit  ×  $/GPU-second  ÷  utilization
```

Every technique below moves exactly one of those three terms. If a technique cannot be
placed in this table, it does not belong in a customer conversation.

| Term | What moves it | Our tiering |
|---|---|---|
| **GPU-seconds per unit** | batching, caching, quantization, scheduling, kernels | Tier 0 + Tier 1 + Tier 2 |
| **$ per GPU-second** | which silicon you can actually run on | Tier 1 (parity gating), Tier 2 (MI355X) |
| **Utilization** | packing, session density, trough-filling, routing | Tier 1 (density), Tier 2 (routing) |

---

## Tier 0 — Table stakes we ride, and do not claim

We serve LLMs through a co-located reverse proxy in front of **vLLM / SGLang**
(`feat/llm-proxy`, `docs/LLM_PROXY.md`). That means we inherit the following for free —
and so does every competitor. **We do not present any of these as differentiation.**

| Technique | What it does | Who has it |
|---|---|---|
| PagedAttention / continuous batching | Non-fragmented KV, rolling admission | Everyone |
| Prefix / prompt caching | Reuse of shared prompt prefixes across requests | Everyone; Fireworks discounts cached input |
| Speculative decoding | Draft model proposes, target verifies in parallel — ~2–3× in production | Everyone |
| FP8 / INT8 weight + activation quantization | Fewer bytes moved per token | Everyone |
| Tensor / pipeline parallelism | Big models across GPUs | Everyone |
| Chunked prefill | Prevents long prompts from stalling decode | Everyone |

**Stating this plainly is deliberate.** A diligence conversation that discovers we were
implying credit for PagedAttention ends badly. A conversation where we volunteer it and
then show Tier 1 goes well.

---

## Tier 1 — Ours, shipped, and measured

Every number here is wall-clock, from a scripted harness, with the raw JSON result line
archived in-repo, and with its caveat stated in `reports/benchmark-report-2026-07.html`.

### T1.1 — Shared-Prefix Candidate Batching (SPCB) ★ *the biggest lever we own*

**What:** when a caller needs N candidate outputs that share a long prefix and diverge
only over a short tail, decode all N as one batched pass over the shared prefix instead
of looping. Sounds obvious; is not implemented in the models' own reference code, and in
OpenVLA's case is actively blocked by two guards its authors labelled *"simplified for
batch size = 1."* We lifted them and proved the batched output bit-identical.

**Measured [M]** — OpenVLA-7B, both vendors, greedy and batched-vs-batch-1 parity gates
both exactly **0.0**:

| N candidates | H100 loop → batched | MI300X loop → batched |
|---|---|---|
| 8 | 985 → 182 ms (**5.4×**) | 1042 → 198 ms (**5.3×**) |
| 16 | 1963 → 264 ms (**7.4×**) | 2088 → 278 ms (**7.5×**) |
| 32 | 3906 → 443 ms (**8.8×**) | 4204 → 427 ms (**9.9×**) |

**Why it works, stated mechanically:** each candidate decode is only 7 action tokens, so
batch-of-one badly underuses the GPU; batching over the shared prompt+vision prefix
recovers the idle capacity. The win **grows** with N.

**Why it matters far beyond robotics:** that is a structural description of every
parallel-sampling LLM workload — best-of-N, self-consistency, agentic tree search,
reranked tool-call proposals. Long shared prefix, short divergent tails, N of them, under
a deadline. **The 2026 agentic workload has the same shape as the VLA workload we already
optimized.**

**Not measured [A]:** the transfer to LLM decode. Stock vLLM `n>1` already shares some
prefix work, so the delta there will be smaller than 5–10×. **This is the gate in
`BUSINESS_PLAN_2026.md` §6.3 and the single most important experiment we owe ourselves.**

**Also not claimed:** the candidate *scorer* is a modeling choice, not ours. SPCB is a
throughput result, not a better policy.

### T1.2 — Topology-Gated Adaptive Caching

**What:** step-skipping in a diffusion denoise loop, gated on the *block topology of the
specific transformer* rather than a generic delta threshold. TeaCache the idea is public;
the Cosmos-shaped gate that knows `CosmosTransformer3DModel`'s structure is ours.

**Measured [M]:** **3.2–3.8×** on Cosmos-Predict-7B, reproduced on both vendors, across
sessions, driver stacks and months (154.0 s vs 154 s on a re-run; ±0.5% run-to-run;
same-seed outputs MD5-identical).

**Caveat, stated up front:** this is a quality/speed trade, not a free lunch. Cached
output is a *different valid generation* of the same prompt — LPIPS 0.645 vs uncached,
~28–30% reduced inter-frame motion at the speed-biased default. FVD at N≥50 is the gate
before we publish the speed-vs-quality story externally. **Customer-facing consequence:
it is opt-in with a threshold dial, not a silent default.**

### T1.3 — Similarity-Scheduled Denoise Caching

**What:** instead of a static "skip every k-th step" schedule, decide per-step from
cosine similarity between successive latent states.

**Measured [M]:** **3.19×** on DreamZero-DROID (5890.6 → 1780.2 ms/chunk) — **faster
than every static skip level we tested**, including the model's own undisclosed default
8-of-16 skip.

**Finding worth telling:** DreamZero's reference implementation silently skips half its
DiT steps with no opt-in flag. We measured both the true full-compute baseline and the
silently-skipped one and published which is which. This is the kind of thing that makes a
benchmark trustworthy, and it is why our numbers are worth more than a competitor's blog
post.

### T1.4 — Growing-Window Latent/KV Reuse

**What:** across a closed-loop rollout, the context prefix is shared between steps and
across planning candidates. We reuse the latent/KV state on a growing window instead of
recomputing, with an exactness gate.

**Measured [M]:** GPU-verified to **3.8e-4** (algorithm, H100) and **6.8e-4** (integrated
adapter, MI300X) in fp32. CEM's lockstep-uniform rollouts turned out not to need paged or
ragged machinery at all — a batch-dimension generalization of the growing-window cache
was sufficient (ADR-0009 §"CEM-batched resolution").

**Two negative results published rather than buried** (ADR-0009): RoPE is **not
composable** the way the first design assumed, and eviction is **provably non-exact** at
the K/V level. Both are GPU-verified negatives. See §4.

### T1.5 — State-Resident Session Packing

**What:** treat a long-lived session's resident state as the primary capacity unit and
schedule for density, with state carryover as a *requirement* of the benchmark rather
than an optimization. Resident-sessions-per-GPU is a first-class metric in our harness;
it is not a metric the incumbents publish at all.

**Measured [M]:**

| Model | H100 | MI300X | Note |
|---|---|---|---|
| DreamZero-DROID (16.5B, ~23 GiB/session) | **1** | **6** | H100 OOM'd mid-forward on a live 2-session run — empirical, not extrapolated |
| LingBot-VA 2.0 (~6 GiB/session) | 11 (24 under levers, measured) | 30 | No latency penalty found up to the memory ceiling |

**Why it is a cost technique:** density is the denominator of cost-per-session. A 6×
density difference is a 6× difference in the fixed-cost share every session carries.
**And it is the mechanism that makes large-memory silicon a business advantage rather
than a spec-sheet bullet.**

**Transfer:** a long-context agentic LLM session with a large KV cache is the same
resource problem. Per-token pricing structurally hides this; long-session serving is a
memory-density business.

### T1.6 — Cross-Vendor Parity Gating ★ *the technique that makes arbitrage sellable*

**What:** before any workload moves to cheaper silicon, we gate on **numerical parity
against the reference implementation**, and we publish the gate result.

**Measured [M]:** OpenVLA-7B reproduces its own `predict_action` to **0.0e+00 maximum
error — bit-identical — on both H100 and MI300X.** Both parity gates (greedy, and
batched-vs-batch-1) measured exactly 0.0 on both machines.

**Why this is a moat item and not hygiene:** silicon arbitrage is only sellable if the
customer believes their outputs do not change. "AMD is 30% cheaper" is a procurement
argument; **"AMD is 30% cheaper and here is the bit-identical parity gate on your model"
is a product.** No incumbent offers this because no incumbent runs AMD.

**Honest scope:** parity is exact for OpenVLA. It is **open work** for DreamZero
(reference-server comparison not completed) and the diffusion models trade determinism
for speed by design when caching is on. We state which models have a closed parity gate
and which do not.

### T1.7 — Portable Config-Level Levers

**What:** unglamorous but real — CFG reduction, denoise-step counts, precision selection
— tuned per model, using **zero CUDA-locked tooling**, so the win is portable across
vendors.

**Measured [M]:** **2.0×** on LingBot-VA (754.6 → 342.2 ms/chunk on H100), roughly
halfway across the model's own published "portable techniques" band, on top of an
already-optimized baseline; and nearly doubled session density (11 → 24) with no latency
penalty.

**Caveat we disclose every time:** the winning lever (CFG off) **measurably shifts the
action-output distribution** — one channel's p50 drops ~99%. The latency and memory wins
are real and measured; task-correctness of the resulting policy is not validated. This
lever is therefore opt-in and flagged, never a silent default.

### T1.8 — Multi-Target Backend Protocol

**What:** one engine, one unmodified codebase, four verified silicon targets — AMD
MI300X (ROCm), NVIDIA H100 (Hopper), NVIDIA Ada (sm_89), Intel Sapphire Rapids (CPU/AMX).
Each new target is one backend class behind a stable protocol.

**Why it is a cost technique, not an architecture bullet:** it is the *precondition* for
Lever B. You cannot arbitrage silicon you cannot run on. Six models × three regimes ×
two vendors, all on identical engine code, is the evidence that we can follow the cheapest
capable capacity instead of being locked to one vendor's price.

---

## Tier 2 — Committed, next two quarters

These are the techniques the pivot depends on. Each has an owner-shaped scope and a
measurable gate.

| # | Technique | Cost term | Why now | Gate |
|---|---|---|---|---|
| **T2.1** | **SPCB on LLM decode** — best-of-N / agentic sampling vs stock vLLM `n>1` | GPU-s/unit | Decides whether `BUSINESS_PLAN` §6.3 exists | ≥1.5× over stock vLLM or the LLM efficiency claim is dropped |
| **T2.2** | **Heterogeneous prefill/decode disaggregation** — compute-bound prefill on high-end parts, memory-bound decode on cheaper large-memory parts | $/GPU-s **and** GPU-s/unit | Published results show this wins for most practical workloads even over 25 GbE **[V]**; it is *the* technique that turns a mixed NVIDIA+AMD fleet from a compromise into an advantage — **prefill on H100, decode on MI300X** | Beat single-silicon cost-per-token on a real model |
| **T2.3** | **KV cache quantization to 3–4 bit** | utilization (density) | 2026 results show ~6× KV memory reduction at negligible accuracy loss **[V]**. Compounds directly with T1.5 | Density gain at a stated quality bound |
| **T2.4** | **Cost-aware multi-cloud routing** | $/GPU-s + utilization | Baseten runs 87 clusters across 18 providers **[V]** — routing to cheapest capable capacity is a first-class competitive weapon, not plumbing | Live price-aware placement across ≥3 providers |
| **T2.5** | **Cross-model-class multi-tenant packing** | utilization | Vision and LLM demand curves are uncorrelated; breadth becomes a utilization advantage nobody with an LLM-only fleet can copy | Measured utilization lift vs single-class packing |
| **T2.6** | **MI355X (288 GB) qualification** | $/GPU-s + density | Extends T1.5's density lever; FP8 attention kernels for gfx942 already exist | A benchmark column |
| **T2.7** | **Rust cache/router/scheduler wired into the hot path** | GPU-s/unit | The crates exist and currently serve only the generation path — documented shelf-ware for the control path (`AUDIT` §1.2.6) | Measured overhead reduction |

**T2.2 deserves emphasis.** Disaggregated prefill/decode is the technique that makes our
*heterogeneous* fleet strictly better than a homogeneous one: the two phases want
different hardware, and we are the only platform in this comparison set willing to run
both vendors. It converts "we support AMD" from a coverage claim into an architectural
cost advantage.

---

## Tier 3 — Bets, labelled as bets

Not on the plan of record. Listed so nobody mistakes them for commitments.

- **Deadline-aware admission control across model classes.** Our control-loop scheduler
  targets a deadline rather than aggregate throughput. Generalizing that to mixed
  interactive/batch multi-tenant traffic is genuinely novel and genuinely unproven.
- **Speculative decoding with a world-model draft.** Using a cheap latent predictor to
  draft for an expensive policy model.
- **Compiler/kernel synthesis** (the old "Forge" idea). Interesting; not a 2027 revenue
  path; do not put it in a customer conversation.

---

## 4. The negative-results ledger

Published rather than buried. This is a competitive asset in a category where every
vendor benchmarks and none disclose.

| Finding | Status |
|---|---|
| RoPE is **not composable** the way the first KV-reuse design assumed | GPU-verified negative, ADR-0009 |
| KV eviction is **provably non-exact** at the K/V level | GPU-verified negative, ADR-0009 |
| `torch.compile` measured **slower** on LingBot-VA | Disclosed, not yet root-caused |
| FlexAttention measured **slower** on LingBot-VA | Disclosed, not yet root-caused |
| FP8 attention was a **net loss on Hopper** for Cosmos | Disclosed in the report table |
| Our own no-cache Cosmos baseline is **~17% slower** than NVIDIA's bespoke pipeline | Disclosed in `POSITIONING.md` |
| CEM candidate batching bought only 1.6–2.1×, not the 64× the forward-count reduction suggested — the per-candidate forward was already compute-bound | Disclosed; it is what redirected us to KV reuse and bf16 |
| DreamZero MI300X measured *faster* than H100, contradicting LingBot-VA's ordering | One run each; flagged as unconfirmed rather than smoothed |

---

## 5. What we do not have

Stated so it is never discovered instead of disclosed:

- **No measured LLM performance advantage.** Today's LLM path is a proxy in front of
  vLLM/SGLang. T2.1 is the experiment that would change this.
- **No CI.** `.github/workflows/` does not exist; `make lint` and `make typecheck`
  currently fail on `main` (`AUDIT` §1.2). This is a P0 blocker for any diligence.
- **No multi-tenant production surface.** No auth, quotas, metering, or billing.
- **No FVD at N≥50**, so the generation speed-vs-quality story is not externally
  publishable yet.
- **No bare-metal MI300X confirmation** — several AMD numbers are virtualized slices.
- **No task-correctness validation** for the levers that trade latency for distributional
  shift (T1.7, and DreamZero's reference-server parity).

---

## 6. The one-slide version

| Lever | Technique | Measured | Where it applies |
|---|---|---|---|
| **Fewer GPU-seconds** | Shared-prefix candidate batching | **5.3–9.9×** | Agentic/parallel sampling, VLA |
| | Topology-gated adaptive caching | **3.2–3.8×** | Video/image diffusion |
| | Similarity-scheduled denoise caching | **3.19×** | Policy diffusion |
| | Portable config levers | **2.0×** | Policy models |
| **Cheaper GPU-seconds** | Multi-target backend + parity gating | **26–39%** lower cost/unit on AMD | Everything |
| **Better utilization** | State-resident session packing | **6×** density | Long-lived stateful sessions |

**The sentence that ties it together:** *we get more customer-work out of each
GPU-second, on GPU-seconds that cost less, and we can prove the outputs did not change.*
