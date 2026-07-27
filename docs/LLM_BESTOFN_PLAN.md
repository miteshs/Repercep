# Best-of-N Serving Lever — Pre-Registered Experiment Plan

- **Date:** 2026-07-26
- **Why this doc exists:** `docs/BUSINESS_PLAN_2026.md` §6.3 contains the best table in
  the company's materials — half the incumbent's price at 22 points better gross margin —
  and it rests on a **3× efficiency multiplier transferred from a VLA measurement that
  was never run on LLM decode.** This is the experiment that settles it.
- **This plan is written before any number is measured, and the decision rule in §5 is
  binding.** That ordering is the whole point: an experiment whose success criterion is
  chosen after seeing the result is not evidence. `docs/METHODOLOGY.md` §3.
- **Companions:** `docs/INFERENCE_MOAT_TECHNIQUES.md` §T1.1 (the technique),
  `docs/PIVOT_2026_07_PLATFORM_STRATEGY.md` §3.1 (the transfer argument),
  `docs/LLM_PROXY.md` (what we inherit from the upstream and deliberately did not build).

---

## 1. The finding that changes the experiment

The obvious plan is to port `scripts/bench_vla_levers.py` to an LLM: time a
per-candidate decode loop against one batched decode, report the ratio. **That would
produce a strawman number, and we would deserve to be caught.**

The reason the VLA measurement was legitimate is specific and does not carry over:

> OpenVLA's stock implementation **refuses `batch>1`** via two guards its authors label
> *"simplified for batch size = 1."* The per-candidate loop was not a weak baseline we
> constructed — it was the only thing the shipped code could do. Our 5.3–9.9× was
> measured against the real alternative.

vLLM has no such limitation. Out of the box it already:

- **shares the prompt KV across `n` samples** — `SamplingParams(n=N)` prefills once and
  forks the sequence, which is *precisely* the mechanism our VLA lever exploits;
- **shares prefixes across separate requests** via automatic prefix caching (APC);
- **packs concurrent short decodes** via continuous batching.

`docs/LLM_PROXY.md` already says this in its own words ("what you inherit for free … and
deliberately did not rebuild"). So the honest question is **not** "can batching beat a
loop" — it is:

> **Is there anything left on the table above a competently configured vLLM?**

The answer may well be *no*, and this plan is built so that a *no* is a publishable
result rather than a wasted GPU session.

---

## 2. The four-rung ladder

All four rungs run the same workload — one shared prefix, N candidate continuations,
identical sampling parameters — against an OpenAI-compatible endpoint.

| Rung | What it is | Who is in this state today | Role |
|---|---|---|---|
| **R0** | N separate requests, **APC off** | Nobody competent | **Strawman floor.** Reported and labelled as such, so the distance between R0 and R1 is visible rather than sellable |
| **R1** | N separate requests, **APC on** | **Most real agent code** — frameworks emit N independent calls | **The realistic baseline** |
| **R2** | **One** request, `n=N` | Anyone who knows to do this and can restructure their call sites | **The ceiling vLLM already offers** |
| **R3** | Repercep fuses N related calls into one upstream `n=N` | — | **The proposed product** |

### 2.1 The consequence nobody should skip past

**R3 and R2 are the same upstream call.** The difference is only *who does the fusing* —
the customer rewriting their agent, or us doing it transparently at the gateway.

So R3's performance is R2's performance plus our fusing overhead, and the experiment
reduces to measuring **R0, R1, R2** plus a separate, later check that our overhead is
negligible. That is a simplification, and it is also the honest framing of what we would
be selling:

> **We are not claiming to beat vLLM. We are claiming to deliver vLLM's own best path to
> workloads that are emitted as N separate calls — without the caller rewriting
> anything.** The lever is `R1 → R2`. The product is that it needs no client change.

That is a narrower claim than "3× on agentic workloads," and it is one we can defend in
a room with someone who knows how vLLM works.

---

## 3. What gets measured

Primary metric: **wall-clock time to obtain all N candidates**, median of `--repeats`,
warm-up discarded — matching `bench_vla_levers.py` so the numbers sit next to the VLA
row without an accounting asterisk.

Secondary: total GPU-seconds consumed (the cost-model input), TTFT, and tokens/s.

### 3.1 Sweeps, and what each one is for

| Sweep | Range | Why |
|---|---|---|
| **Candidate count N** | 4, 8, 16, 32 | The VLA win *grew* with N. If the LLM win does not, the mechanisms differ and the transfer argument is weaker than claimed |
| **Shared prefix length** | ~256, ~2k, ~8k tokens | The win should grow with prefix length — that is the mechanism. A flat curve falsifies the stated cause |
| **Decode length** | 8, 32, 200 tokens | **The VLA win came from 7-token decodes underusing the GPU.** Short decodes are where the prefix dominates. If the win vanishes by 200 tokens, say so — it bounds which workloads we can price |
| **Concurrency** | 1, 4, 16 concurrent decisions | **The most likely killer, see §4** |

---

## 4. The result most likely to kill this, stated in advance

**At high concurrency the lever should shrink or vanish.** Continuous batching already
saturates the GPU when there is enough independent traffic; fusing N candidates into one
request saves scheduling overhead but not arithmetic. The lever most plausibly lives at
**low-to-moderate load**, which is exactly where a young platform operates and exactly
where `BUSINESS_PLAN_2026.md` §6.4 already flags utilization as the quiet risk.

If that is what the data says, the honest consequence is **not** to drop the claim — it
is to scope it: *"the lever is worth X at the utilization we actually run at, and decays
to Y as the fleet fills."* That is a real and defensible statement, and it couples the
efficiency claim to the utilization assumption instead of letting them be argued
separately. It also *lowers* the ceiling on the §6.3 table, which should then be rebuilt
at the measured operating point rather than at the best point.

Second most likely killer: **APC already closes most of R1 → R2.** If so, there is no
product here, and §5 says what to do.

---

## 5. Decision rule — binding, chosen before the data

Let **L = R1 / R2** (the realistic baseline over the achievable ceiling), measured at the
sweep point that best represents an agentic workload: **N = 16, prefix ≈ 2k, decode ≈ 32,
concurrency = 4.**

| Measured L | Consequence |
|---|---|
| **L ≥ 3.0** | The §6.3 table stands as written. Publish the ladder |
| **1.5 ≤ L < 3.0** | **Rebuild §6.3 at the measured L**, not at 3×. Deck table survives with the real number |
| **L < 1.5** | **Delete §6.3.** Drop the LLM efficiency claim from the deck, business plan, site and report. Compete on breadth + silicon only. This is the `PIVOT` §9 kill signal, and it fires without further argument |

Additional binding conditions, regardless of L:

1. **The concurrency curve ships with the number.** No headline L may be quoted without
   the concurrency at which it was measured. A single-concurrency L is not a result.
2. **R0 is never used as the baseline in any external artifact.** It exists to show the
   distance we are *not* claiming.
3. **If the prefix-length curve is flat**, the stated mechanism is wrong; the claim is
   withdrawn pending a correct explanation even if L is large. A number we cannot explain
   is not a lever, it is a coincidence.

---

## 6. Method and controls

- **Correctness gate — redesigned on the box, 2026-07-27.** The plan originally called
  for the LLM analogue of the VLA parity gate: compare R1's and R2's greedy candidate
  sets and require equality. **That gate is impossible, and the impossibility is a
  property of the server, not a gap in the harness.** Two independent reasons, both
  confirmed against vLLM 0.26.0:

  1. vLLM **rejects greedy `n>1` outright** — `n must be 1 when using greedy sampling,
     got 16` (HTTP 400). The comparison cannot be executed at all.
  2. Under sampling, R2's `n` candidates come from **one shared RNG stream** while R1's N
     requests each seed their own. The two sets are different draws from the same
     distribution and will never be equal.

  So the gate is what is actually checkable, and — importantly — what a *throughput*
  claim actually needs:

  - **Request-shape equivalence:** at `n=1, temperature=0` the two code paths must return
    identical text, proving the harness builds equivalent requests rather than silently
    asking for different work.
  - **Equal decode work:** R1 and R2 must report the same completion-token total (±2%).
    **This is the load-bearing check.** If the rungs did not decode the same number of
    tokens, the wall-clock ratio is not a serving lever, it is an accounting error.

  Text equality was the wrong gate for a throughput comparison in any case. Equal decode
  work is the right one, and unlike the VLA parity gate it is achievable here.

- **The timed sweep runs at `temperature > 0` (default 0.8), never greedy.** Caught while
  validating the harness locally, and worth stating because the first version got it
  wrong: best-of-N only exists *because* the candidates differ. A greedy `n=N` is a
  configuration nobody runs, and one the server is free to special-case — timing it would
  have produced a number about an artificial workload. The harness refuses a timed run at
  `temperature=0`.
- **One server, one process, both rungs.** Rungs are flipped against the same loaded
  model in the same run — never across restarts, never across boxes.

- **The harness runs *on the serving box*, against `127.0.0.1`. This is not a
  convenience, it is a correctness requirement.** R1 issues N HTTP requests where R2
  issues one. Measured across a WAN — e.g. from a laptop through a cloud provider's HTTP
  proxy — R1 pays N network round-trips and R2 pays one, and the ratio would look like a
  large lever that is *entirely* per-request network overhead. That is precisely the
  strawman this plan exists to avoid, arriving through the back door. Any result whose
  client was not co-located with the server is void.
- **APC toggled explicitly**, not assumed. R0 vs R1 is only meaningful if we control it,
  and vLLM's default has changed across versions. Record the vLLM version and the flag.
- **Warm-up discarded**, median of repeats reported, ROCm first-run autotune excluded via
  warm-up separation — same convention as every other number we publish.
- **Cache state controlled between rungs.** APC persists across requests; R2 measured
  after an R1 run is measuring a warm cache. Reset or randomize the prefix per repeat and
  say which.
- Both vendors if the lever survives on the first: H100 and MI300X, same as every other
  row in the report.

---

## 7. Sequencing

1. **Local (no GPU):** harness written and validated against `httpx.MockTransport`, the
   pattern `tests/test_llm_proxy.py` already uses. Proves request construction, rung
   logic, parity checking and the result-line format without burning GPU time.
2. **One GPU session:** a mid-size open model (Llama-3.1-8B-Instruct class) on vLLM.
   Full sweep. This is a cheap session — no training, no large weights.
3. **Decide** per §5, and update every artifact the answer touches. If the answer is
   *drop it*, that edit is the deliverable.
4. **Only if it survives:** build the fusing into `serving/llm_proxy.py` and measure R3's
   overhead against R2.

---

## 8. What this experiment does not test

- **Cross-request fusing of *non-identical* prefixes** (candidates differing in tool
  schema or few-shot). Plausibly where the real product is, and explicitly out of scope
  here — this plan tests the simpler claim first.
- **Deadline-aware admission**, the third primitive in `PIVOT` §3.3. Untested and stays
  labelled as such.
- ~~**SGLang**~~ — **now in scope, pre-registered in §9 below.** It became the single most
  important open test once the vLLM result showed the lever comes from a *cache-timing
  race* that RadixAttention is specifically designed to win.
- **Any claim about model quality.** This is a serving-throughput experiment. The
  candidate *scorer* remains, as in the VLA case, a modeling choice we do not ship.

---

## 9. The SGLang test — pre-registered 2026-07-27, before measuring

Written before any SGLang number exists, same discipline as §5. **This test can kill the
commercial claim outright, and the rule below says so in advance.**

### 9.1 Why this is the test that matters

The vLLM result (`docs/LLM_BESTOFN_RESULT.md` §4) found that the lever does **not** come
from prefill arithmetic. It comes from a **cache-timing race**: N prefix-sharing requests
issued simultaneously all miss the prefix cache together, because none has finished
prefilling to populate it. `n=N` sidesteps the race by sharing the prefill structurally
inside one request.

**RadixAttention is designed to win exactly that race.** SGLang maintains a radix tree of
cached prefixes and can detect and share a common prefix *across requests in the same
batch*, rather than relying on one request finishing before another can hit the cache. If
that works as designed, SGLang's R1 should approach its R2 and our lever largely
disappears against that engine.

So this is not a footnote. It is the adversarial case.

### 9.2 The comparison that decides it

Same harness, same model, same shapes, same gate point. Two numbers matter, and the
second one matters more:

1. **`L_sglang = R1/R2` measured on SGLang.** Does fusing still buy anything on an engine
   built to handle fan-out?
2. **`SGLang R1` vs `vLLM R2` in absolute wall-clock.** *This is the real commercial
   question.* If a customer's plain fan-out on SGLang is already as fast as our fused
   `n=N` on vLLM, then **the honest advice is "switch engines," not "buy our fusing"** —
   and we must say so rather than sell a lever that a free config change matches.

### 9.3 Binding decision rule

| Outcome | Consequence |
|---|---|
| **`L_sglang ≥ 1.5`** | The gap is a property of simultaneous fan-out, not of vLLM. **Claim strengthens** — it survives an engine purpose-built to close it. Report both engines |
| **`1.2 ≤ L_sglang < 1.5`** | Real but engine-dependent. Keep the claim, scope it per engine, and quote the *weaker* of the two in any economics table |
| **`L_sglang < 1.2`** | **RadixAttention closes it.** The lever is a vLLM artifact. Drop "candidate fusing" as a differentiator, rebuild `BUSINESS_PLAN` §6.3 on **silicon alone**, and say plainly that on SGLang the right answer is to use SGLang |
| **`SGLang R1` ≤ `vLLM R2`** (regardless of `L_sglang`) | Same as above, and stronger: our fused path on vLLM is not better than a stock alternative that already exists. **This overrides a favourable `L_sglang`** |

That last row is the one that must not be quietly skipped. A large `L_sglang` measured
against a slow SGLang baseline would be a ratio that flatters us while the absolute
answer for the customer is "use the other engine."

### 9.4 Controls

Identical to §6, plus:

- **Same GPU model, same weights, same prefix/decode/concurrency grid**, harness
  unchanged and co-located.
- **RadixAttention explicitly enabled** and its state recorded, the same way
  `--enable-prefix-caching` was for vLLM. If SGLang's default changed, that is disclosed,
  not assumed.
- **Engine versions recorded.** A cross-engine comparison is only as good as its
  disclosure, and this one will be read adversarially by anyone who prefers the other
  engine.
- Absolute cross-engine timings are only comparable because both run on the same pod
  spec, same model and same harness — stated explicitly wherever they appear.
