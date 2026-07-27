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
