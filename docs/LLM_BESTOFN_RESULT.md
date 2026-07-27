# Best-of-N Serving Lever — Measured Result

- **Date measured:** 2026-07-27
- **Plan (pre-registered, binding):** `docs/LLM_BESTOFN_PLAN.md`
- **Verdict: `L = 1.5×` at the pre-registered gate point → REBUILD
  `BUSINESS_PLAN_2026.md` §6.3 at the measured lever, not at 3×.** The claim
  survives; the number in the deck does not.

---

## 1. Environment

| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3, driver 580.126.09 (RunPod secure cloud) |
| Server | vLLM **0.26.0**, torch 2.11.0+cu130 |
| Model | `Qwen/Qwen2.5-7B-Instruct`, `--max-model-len 16384` |
| Prefix caching | **`prefix_caching=True`** (explicit `--enable-prefix-caching`, confirmed in engine log) |
| KV cache | 1,071,136 tokens · max concurrency 65.4× |
| Client | `scripts/bench_llm_bestofn.py`, **run on the pod against `127.0.0.1`** (plan §6) |
| Sampling | timed sweep `temperature=0.8`; correctness gate greedy and untimed |
| Repeats | 10 (stage 1) / 8 (stage 2), median reported, warm-up discarded |

**Correctness gates passed at every point in both stages** — request-shape
equivalence at `n=1`/greedy, and equal decode work across rungs (R1 and R2 both
reported exactly 512 completion tokens = 16 candidates × 32 tokens). Zero
mismatches. Raw result lines: `stage1.json`, `stage2.json`.

---

## 2. The gate — concurrency sweep

N = 16 candidates, 2,048-token shared prefix, 32-token decodes.

| Concurrent decisions | R0 (cache denied) | R1 (baseline) | R2 (`n=N`) | **L = R1/R2** |
|---|---|---|---|---|
| 1 | 357 ms | 346 ms | 270 ms | **1.28×** |
| **4 — gate point** | 623 ms | 638 ms | 421 ms | **1.52×** |
| 16 | 1697 ms | 1689 ms | 816 ms | **2.07×** |

Reproduced independently in stage 2 at the gate point: **1.49×** (vs 1.52×).

### 2.1 A prediction the plan got wrong

Plan §4 named the most likely killer in advance: *"at high concurrency the lever
should shrink or vanish"*, on the reasoning that continuous batching already
saturates the GPU. **That was wrong, and in the opposite direction** — the lever
*grows* with concurrency, 1.28× → 1.52× → 2.07×.

Recorded rather than quietly dropped, per the standing practice. The reason it
was wrong is in §4 below.

---

## 3. Mechanism — prefix × decode sweep

N = 16, concurrency 4. **L = R1/R2:**

| shared prefix ↓ / decode → | 8 tokens | 32 tokens | 200 tokens |
|---|---|---|---|
| **256** | 1.35× | 1.24× | **1.04×** |
| **2,048** | 1.74× | 1.49× | 1.12× |
| **8,192** | **2.48×** | 2.30× | 1.33× |

Two clean monotonic trends, both in the direction the stated mechanism predicts:

- **L rises with shared prefix length** (1.24 → 1.49 → 2.30 at decode 32). Plan
  §5 condition 3 made a flat prefix curve fatal to the claim — *"a number we
  cannot explain is not a lever, it is a coincidence."* The curve is steep. **The
  mechanism claim survives.**
- **L falls with decode length** (2.48 → 2.30 → 1.33 at prefix 8,192). The longer
  the decode, the more decode arithmetic dominates the shared prefill, and the
  less there is to win. At 256-token prefix with 200-token decodes the lever is
  **1.04× — effectively nothing.**

---

## 4. The finding that explains both, and is worth more than the number

**Denying the prefix cache changes almost nothing: R0 ≈ R1 at every single point
in both stages** (357/346, 623/638, 1697/1689, 1190/1158, …). Automatic prefix
caching is on and confirmed, and it is buying essentially zero on this workload —
while `n=N`, which shares the same prefill, wins by up to 2.5×.

The explanation is a scheduling race, and it is specific:

> R1 fans out 16–256 requests that share a prefix **simultaneously**. They all
> miss the prefix cache together, because none of them has finished prefilling
> to populate it yet. APC is a *temporal* optimization and a simultaneous
> fan-out defeats it. `n=N` shares the prefill **structurally inside one
> request** — it prefills once and forks — so it does not depend on cache timing
> at all and cannot lose that race.

This also explains §2.1: more concurrent decisions means more simultaneous
cache-missing requests, so the gap R1→R2 *widens* with concurrency instead of
being absorbed by continuous batching.

**Why this matters commercially.** It means the win is not "we beat vLLM" — it is
a real, mechanistically-explained gap that vLLM's own caching does not close for
simultaneous fan-out, which is exactly the shape agent frameworks emit. Fusing N
related calls into one `n=N` captures something APC alone does not deliver.

---

## 5. Consequences — what changes in which artifact

| Artifact | Change |
|---|---|
| `BUSINESS_PLAN_2026.md` §6.3 | **Rebuild at 1.5×, not 3×.** Margin at half-price pricing no longer holds; the table must be recomputed and the price posture in §5.2 revised down |
| Seed deck, unit-economics slide | Same. The `GATED` tag comes off and the real number goes on, with the workload caveat |
| `PIVOT_2026_07_PLATFORM_STRATEGY.md` §3.1 | The transfer argument holds *directionally* but the magnitude does not carry from VLA. Must state the measured LLM number, not the 5.3–9.9× VLA one |
| `INFERENCE_MOAT_TECHNIQUES.md` T1.1 | Move the LLM row from "hypothesis" to measured, at 1.5×, with the shape dependence |
| Benchmark report §8 | The transfer section can now cite a measurement instead of an expectation — including that it is much smaller than the VLA figure |

**The scope sentence that must travel with the number:** *1.5× at a 2k prefix
with 32-token decodes at moderate concurrency; 1.04× when prefixes are short and
decodes long; 2.5× when prefixes are long and decodes short.* Quoting the top of
that range without its shape is the same offence as quoting R0 as a baseline.

---

## 6. What this does not establish

- **Not measured on MI300X.** Vendor-neutrality of this lever is unverified.
- **Not measured against SGLang**, whose RadixAttention may close the fan-out gap
  that vLLM's APC leaves open. Plan §8 already named this as the adversarial
  next test, and §4's finding makes it the *most* important one — if RadixAttention
  wins the simultaneous-fan-out race, this lever shrinks against that engine.
- **R3 overhead unmeasured.** These numbers say a fused `n=N` is faster than a
  fan-out. They do not yet show Repercep's gateway can do the fusing cheaply
  enough to keep the win.
- **One model, one size.** 7B dense. Larger models and MoE shift the
  prefill/decode balance and therefore the lever.
- **No claim about output quality.** This is throughput only; the candidate
  *scorer* remains a modeling choice we do not ship.

---

# Part 2 — the gateway test (plan §10), measured 2026-07-27

Same H100, same vLLM 0.26.0, same model, same harness, client co-located. The
gateway under test is the **real** `LlmProxy` + `CandidateFuser`, hosted in a
bare FastAPI app; nothing on the completions path is stubbed, and `LlmProxy`
bypasses the world-model scheduler by design anyway.

## 7. The vLLM baseline reproduced

Before testing the gateway, the Part-1 numbers were re-measured on this fresh
pod, fresh model load, three days later:

| concurrency | Part 1 | Part 2 | |
|---|---|---|---|
| 1 | 1.28× | **1.22×** | |
| 4 (gate point) | 1.52× | **1.49×** | |
| 16 | 2.07× | **2.05×** | |

**The R1→R2 lever is real and reproducible.** So is the mechanism finding: R0 ≈ R1
again at c=1 and c=4 (357/354, 626/631), i.e. prefix caching still contributes
essentially nothing to a simultaneous fan-out. Nothing below undermines Part 1.

## 8. R3 — the product claim — **FAILS**

> Plan §10.2: *Fail if `R3 ≥ R1` — the gateway eats the lever, and the honest
> advice becomes "call `n=N` yourself," which needs no product from us.*

| concurrency | R1 direct | **R3 gateway** | R3/R1 | verdict |
|---|---|---|---|---|
| 1 | 354 ms | **360 ms** | **1.02×** | **FAIL** — slower than doing nothing |
| 4 | 631 ms | 622 ms | 0.99× | break-even (and see §10) |
| 16 | 1701 ms | **2087 ms** | **1.23×** | **FAIL** — 23% slower |

One point out of three at parity is not a product. **The gateway does not deliver
the lever.**

## 9. Why — and why it is *broken*, not mistuned

The fuser's own telemetry gives the diagnosis: at the 8 ms default it achieved a
**mean batch size of 2.75** against a target of 16, fusing only **29%** of
requests. Long enough to make everyone wait; too short to actually coalesce them.

Plan §10.3's window sweep was pre-registered precisely to separate "bad default"
from "bad idea". It says bad idea:

| window | mean batch | % fused | c=1 R3/R1 | c=4 R3/R1 | c=16 R3/R1 |
|---|---|---|---|---|---|
| 2 ms | 2.41 | 14% | 1.06× | 1.03× | 1.13× |
| 8 ms | 2.75 | 29% | 1.02× | 0.99× | 1.23× |
| 25 ms | 3.25 | 43% | 1.08× | 0.97× | 1.25× |
| 60 ms | 4.45 | 46% | **1.19×** | 0.90× | **1.31×** |

**A 30× longer window bought only 1.8× more batching, and never got past a mean
of 4.45 out of 16.** Meanwhile c=1 and c=16 degrade *monotonically* as the window
grows — everyone pays the wait, few get coalesced. There is no window that wins:
short windows don't fuse, long windows cost more than they save.

**Root cause.** Requests a client issues simultaneously do not *arrive*
simultaneously. Connection establishment, the accept/read loop and the event
loop spread them out by more than a coalescing window can absorb. The very
property that makes fusing valuable inside the engine — that a simultaneous
fan-out defeats temporal caching — also means our gateway never sees a
simultaneous fan-out to fuse.

## 10. The one favourable column is an artifact, and was flagged before the run

c=4 is the only concurrency where R3 beats R1, and it improves as the window
grows. It should not be believed, for a reason recorded **before** the data was
read:

> The harness's concurrency dimension issues C decisions that all use the *same*
> prefix. At C>1 every request in the run shares one fusion key, so the gateway
> fuses **across decisions** — 64 requests into 2 calls of n=32, against R2's four
> separate n=16 calls. A real agent's concurrent decisions carry *different*
> prompts and would not merge.

So c=4's win partly measures the harness handing the gateway a coalescing
opportunity that production traffic would not. **c=1 is the only clean
apples-to-apples row, and c=1 fails.**

## 11. Consequence — applying plan §10.4

> *If §10.2 fails, `INFERENCE_MOAT_TECHNIQUES.md` T1.1 and the deck's fusing claim
> describe a lever we measured but cannot deliver, and both must say so. A
> technique that works in the engine but not through our own gateway is a paper
> result, not a product.*

Applied. Specifically:

- **The 1.5× lever is still real** — measured twice, reproducibly. What fails is
  our mechanism for capturing it *on the customer's behalf*.
- **A customer can still have it** by calling `n=N` themselves. That is a
  one-line client change and needs no product from us.
- **"No client rewrite" — the entire product framing of this technique — is dead**
  until some mechanism other than time-window coalescing is found.
- `llm_fusing.py` **stays off by default** and its documentation now records this
  result rather than its aspiration. It is not deleted: the code is correct, the
  tests are green, and it remains the right shape if request arrival is ever
  made simultaneous (e.g. an explicit batch endpoint where the caller hands us N
  candidates in one request — which, note, is just `n=N` with extra steps).

**What would have to change for this to work.** Not a longer window. Either a
client-side SDK that batches before sending — at which point the client could
simply send `n=N` — or an explicit multi-candidate endpoint. Both move the work
to the caller, which is exactly the thing the claim promised to avoid.

---

# Part 3 — the SGLang adversarial test (plan §9), measured 2026-07-27

Same pod, same model, same harness, client co-located. SGLang **0.5.16**, with
`disable_radix_cache: false` read back from `/get_server_info` rather than
assumed (plan §9.4). vLLM was stopped first and its numbers banked, so the
cross-engine comparison is same-box, same-weights, same-grid.

## 12. RadixAttention does not close the gap — it widens it

§9.1 predicted this was the test most likely to kill the claim, because
RadixAttention is *designed* to share a prefix across requests in one batch
rather than depending on one request finishing first. **That prediction was
wrong.**

| concurrency | L on vLLM | **L on SGLang** |
|---|---|---|
| 1 | 1.22× | **1.33×** |
| 4 — gate point | 1.49× | **1.67×** |
| 16 | 2.05× | **3.70×** |

Plan §9.3: `L_sglang ≥ 1.5` at the gate point ⇒ *"the gap is a property of
simultaneous fan-out, not of vLLM. Claim strengthens — it survives an engine
purpose-built to close it."* **1.67 ≥ 1.5.** The underlying lever is engine-
independent and larger on the better engine.

## 13. The override did not fire

§9.3's override — the row that must not be skipped — asks whether a customer's
plain fan-out on SGLang already matches our fused `n=N` on vLLM, in which case
"switch engines" beats anything we sell.

| concurrency | SGLang R1 | vLLM R2 | | |
|---|---|---|---|---|
| 1 | 352 ms | 290 ms | 1.21× | no |
| 4 | 614 ms | 424 ms | 1.45× | no |
| 16 | 2561 ms | 831 ms | 3.08× | no |

Fan-out on a better engine is **not** a substitute for `n=N`. Changing engines
does not rescue a caller who fans out.

## 14. The finding that is actually worth money

SGLang is faster than vLLM on the `n=N` path at every concurrency:

| concurrency | vLLM R2 | SGLang R2 | SGLang advantage |
|---|---|---|---|
| 1 | 290 ms | 264 ms | **9%** |
| 4 | 367 ms | 424 ms → 367 ms | **13%** |
| 16 | 831 ms | 692 ms | **17%** |

**Engine selection is a lever we actually control**, unlike fusing. As the
operator we choose what runs under the API; the customer never sees it. A
9–17% cost reduction on this workload shape is smaller than the 1.5× we lost,
but it is *ours to capture* — which the 1.5× turned out not to be.

That is the one durable result of this session, and it is a deployment
decision, not a claim: **serve this workload shape on SGLang, not vLLM.**

## 15. Where the LLM story stands after all three parts

| Claim | Status |
|---|---|
| Candidate batching helps LLM parallel sampling | **True** — 1.5× at the gate point, reproduced twice, and 1.67× on SGLang |
| The mechanism is a cache-timing race, not prefill arithmetic | **True** — APC contributes <3%, and RadixAttention doesn't close it either |
| *We* can capture it for the customer | **False** — gateway fusing measured slower than plain fan-out (Part 2) |
| It is a Repercep differentiator | **False** — any provider on the same engine serves `n=N` equally well |
| Choosing the faster engine is a differentiator | **Modestly true** — 9–17%, and it is ours to control |

**Net:** the technique is real and we understand it better than the people
shipping it. It is not something we can sell. Demoted from the moat to the
deployment playbook.
