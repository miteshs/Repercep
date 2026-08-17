# MI300X-vs-H100 LLM throughput gate — **COMPLETE** (2026-08-09)

- **Plan (pre-registered, binding):** `docs/LLM_SILICON_GATE_PLAN.md`, including
  the §6b amendment committed before the AMD leg ran.
- **Verdict: `R = 1.02–1.38` at all six points, against a `< 0.75` kill
  threshold. Lever B survives, and by a wider margin than the plan's own
  predictions allowed for.** MI300X was faster than H100 on **every shape and
  both models**.
- Raw: `docs/results/llm_silicon_gate_2026-08-09.json`,
  `..._accuracy_both_2026-08-09.json`. Harness:
  `scripts/bench_llm_silicon_gate.py`.

---

## 1. Result

`R = MI300X tok/s ÷ H100 tok/s`. Cost is per **million output tokens** at each
provider's actual hourly rate.

### Qwen2.5-7B-Instruct

| shape | H100 tok/s | MI300X tok/s | **R** | $/Mtok H100 | $/Mtok MI300X | cost adv |
|---|---:|---:|---:|---:|---:|---:|
| S1-interactive (c=1) | 164.03 | **216.11** | **1.32×** | 5.5715 | 2.5579 | **2.18×** |
| S2-interactive-load (c=32) | 4741.99 | **4816.44** | **1.02×** | 0.1927 | 0.1148 | **1.68×** |
| S3-batch (c=32) | 3126.59 | **3397.19** | **1.09×** | 0.2923 | 0.1627 | **1.80×** |

### Qwen2.5-32B-Instruct

| shape | H100 tok/s | MI300X tok/s | **R** | $/Mtok H100 | $/Mtok MI300X | cost adv |
|---|---:|---:|---:|---:|---:|---:|
| S1-interactive (c=1) | 40.84 | **56.44** | **1.38×** | 22.3773 | 9.7941 | **2.28×** |
| S2-interactive-load (c=32) | 1200.02 | **1319.48** | **1.10×** | 0.7616 | 0.4189 | **1.82×** |
| S3-batch (c=32) | 965.35 | **1065.77** | **1.10×** | 0.9467 | 0.5187 | **1.83×** |

**Equal-work check passed at all six points** — enforced structurally
(`min_tokens == max_tokens` + `ignore_eos`), so every request emitted exactly the
requested token count on both vendors. Run-to-run spread ≤0.5% on H100 and ≤1.2%
on MI300X.

### 1.1 Why the PASS is robust, not lucky

Plan §6b, written **before** the AMD run: the two legs could not be
version-matched (H100 on vLLM **0.26.0**, MI300X on **0.23.1** — the newest
available for gfx942 in AMD's official images). Three minor versions of engine
work accrue to the H100 side, so **the handicap runs against AMD**.

The amendment pre-committed the reading: *"If `R ≥ 0.75` despite the handicap →
the gate passes, robustly… Report it as a lower bound on AMD's true position."*

`R ≥ 1.02` everywhere. **These are lower bounds.** A matched-version re-run
would be expected to move the numbers in AMD's favour, not against, so it is no
longer needed to decide the gate — though it would sharpen the ratios.

One unplanned control helps: **torch is 2.11.0 on both sides** (cu130 vs
rocm7.14.0), so the framework layer is matched even though the engine is not.

---

## 2. What this settles for the business plan

`BUSINESS_PLAN_2026.md` §233–234 carried two kill criteria for **Lever B**
(silicon arbitrage), which since commit `6ca6889` has been the **entire** LLM
economic case:

| criterion | threshold | measured | status |
|---|---|---|---|
| MI300X committed rate | kill if `> $2.80/hr` | $1.99/hr observed | **passes** |
| MI300X vs H100 throughput | kill if `< 0.75×` | **1.02–1.38×** | **passes** |

**Lever B survives on measurement rather than assumption for the first time.**
After the best-of-N lever was withdrawn in July, this was the last untested
assumption holding up the LLM story; it is now tested and it held.

**The claim that is now supportable**, stated at the width the evidence
actually carries: *on dense Qwen2.5-class models under vLLM, a single MI300X
matched or beat a single H100 on output throughput at every workload shape
tested, at ~60% of the hourly price — a 1.7–2.3× advantage in cost per million
output tokens.*

**The claim that is NOT supportable** and must not be written anywhere: that
this generalises to all models, all engines, or all providers. See §5.

> **2026-08-16 — the "dense only" limit is now MEASURED, not precautionary.**
> The MoE gate (`docs/LLM_MOE_GATE_RESULT.md`) ran DeepSeek-V2-Lite on both
> vendors at matched engine versions and **the comparison inverts**: MI300X
> reached only `R = 0.53 / 0.74 / 0.60` of H100, failing the 0.75 threshold at
> every shape. Nothing in this document changes — but the word **"dense"** is
> now load-bearing and must appear in any external use of the claim above.

---

## 3. Predictions scored — two of four wrong, including the headline one

The plan recorded four predictions so being wrong would be visible. It was.

| # | prediction | outcome |
|---|---|---|
| 1 | `R ≥ 0.75` on S3-batch | **CORRECT** (1.09 / 1.10) |
| 2 | **`R < 0.75` on S1-interactive at c=1** | **WRONG, by the largest margin** — measured **1.32 / 1.38** |
| 3 | 32B favours MI300X more than 7B | **CORRECT** (1.38/1.10/1.10 vs 1.32/1.02/1.09) |
| 4 | outcome is shape-scoped survival, not a clean pass | **WRONG** — clean pass at all six points |

Prediction 2 is the one worth dwelling on. It was **SemiAnalysis's strongest
claim** — that NVIDIA wins interactive, low-latency, low-concurrency work — and
the plan adopted it at moderate confidence. The measurement came out **the exact
opposite, and by the widest margin of any shape**: single-stream decode is where
MI300X led H100 by the most (1.32× and 1.38×), not the least.

Being wrong here in the *favourable* direction is exactly as important to record
as July's unfavourable surprise. The purpose of pre-registration is not to be
right; it is to make the scorecard unavoidable either way.

---

## 4. Accuracy smoke test — no support for the ROCm accuracy claim

20 prompts, greedy (`temperature=0`), 64 tokens, exact string compare against
the stored H100 outputs.

| model | exact-identical |
|---|---|
| Qwen2.5-7B-Instruct | **16/20** |
| Qwen2.5-32B-Instruct | **16/20** |

The 4/20 divergences in each are mid-generation (first differing character at
78–202) and are **content-preserving**: e.g. `"a subset of artificial
intelligence"` vs `"a type of artificial intelligence"`, or different distractor
wording inside a generated multiple-choice list. **0 of 40 responses were
incorrect, degraded, or malformed.**

This is ordinary cross-vendor floating-point nondeterminism under greedy
decoding, which the plan pre-declared as expected and not the bar. It is
**not** evidence for SemiAnalysis's *"25% of tested models are failing accuracy
tests"* — that is a claim about benchmark scores across a model zoo, and a
20-prompt smoke test on two Qwen models neither confirms nor refutes it. What
can be said: on the two models we serve, ROCm produced correct output
throughout.

---

## 5. Caveats — read before quoting any number here

- **One box per vendor, one session.** No cross-box replication, no second
  region, no repeat on a different day. Within-run spread is tiny; between-box
  variance is unmeasured.
- **Engine versions differ** (§1.1). The PASS is robust because the handicap
  points the other way, but **the exact ratios are not a matched-version
  measurement** and should not be quoted to two decimal places as if they were.
- **The cost column is provider-confounded and is not a property of the
  silicon.** MI300X at $1.99/hr is AMD Developer Cloud; H100 at $3.29/hr is
  RunPod. A different pair of providers gives a different answer, and
  SemiAnalysis's contrary conclusion rests substantially on assuming a
  thinner, pricier AMD rental market (they cite MI300X > $2.50/hr). **Our
  advantage is partly a procurement result, not purely an engineering one — and
  procurement advantages are less durable.**
- **Narrow model coverage.** Qwen2.5 dense only. No MoE, no long-context, no
  FP8/FP4, no multi-GPU. SemiAnalysis's AMD-favourable findings were on large
  dense models, and their NVIDIA-favourable ones included sparse/MoE
  (DeepSeek-class) — which we did not test at all.
- **MI300X here is a VF** (virtualised function) instance, and still won.
  Bare-metal would be expected to do no worse.

---

## 6. What to do next

1. **Update `BUSINESS_PLAN_2026.md` §233–234** — mark the throughput criterion
   measured, replace `[A]` with `[V]`, and cite this document.
2. **Do not widen the claim to match the good news.** The single most likely way
   to lose a technical diligence conversation on this result is to state it
   without §5's scope, exactly as July's lesson about quoting a speedup without
   its workload.
3. **Highest-value follow-up: an MoE model** (DeepSeek/Mixtral class). That is
   where SemiAnalysis's data most favours NVIDIA and where our coverage is
   zero — the shape most likely to produce a counter-example, which makes it
   the one worth running next. **Pre-registered 2026-08-16 as
   `docs/LLM_MOE_GATE_PLAN.md`; not yet run.** It is the one measurement
   `docs/BENCHMARK_PROGRAM.md` holds open before the seed round, and its main
   design risk is that Mixtral bf16 does not fit a single 80 GB H100 — see
   that plan's §4 before running it.
4. Matched-version re-run (`vllm==0.23.0` on H100) is now optional rather than
   required; it would sharpen ratios, not change the verdict.
