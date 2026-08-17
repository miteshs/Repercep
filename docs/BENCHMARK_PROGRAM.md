# The benchmark program — closed for the raise (2026-08-16)

**Decision: the pre-registered gate program has answered its questions and is
closed until the seed round is done.** One exception is pre-registered and
scoped to a single day (§4). Everything else on the measurement backlog is
deferred to post-raise, deliberately, and this document records why so the
decision is auditable later.

This is a change of *activity*, not of standards. Nothing here withdraws a
claim, relaxes a caveat, or reopens a settled result.

---

## 1. Why close it

The program existed to stop the company from selling assumptions. It worked.
Every load-bearing technical assumption in `BUSINESS_PLAN_2026.md` has now
been measured, and two of them came back in ways that changed the plan:

| Gate | Question | Verdict | Effect on the plan |
|---|---|---|---|
| **Best-of-N** (2026-07-27) | Does the VLA candidate-batching lever transfer to LLM serving at ~3×? | **1.5×, then withdrawn entirely** — the gateway could not capture it, and `n=N` is a stock feature every competitor ships | §6.3 deleted, agentic price posture deleted |
| **Fusing / R3** (2026-07-27) | Can we capture the 1.5× for callers who don't send `n=N`? | **No** — coalescing measured *slower* than doing nothing | Code left on its branch as the record |
| **Silicon** (2026-08-09) | Is MI300X within 0.75× of H100 on LLM throughput? | **1.02–1.38×, passed at every shape** | Lever B measured rather than assumed; two of four predictions scored wrong |
| **Cosmos 3 ROCm** (2026-08-09) | Does the Cosmos 3 Nano policy path run on ROCm at usable latency? | **Yes** — 3610 ms/chunk, real-time on AMD | Sixth model, third regime |

**The binding constraint on the raise is no longer technical.** It is the five
questions `decks/VC_QA.md` §6 says we lose: no revenue, no signed capacity
partner, no team beyond one founder, no validation that the tail pays per
call, and no committed silicon price. Not one of those is answerable with a
GPU. Another benchmark is the most comfortable possible way to avoid them, and
comfort is the reason to name the trap in writing.

A second, blunter reason: **the AMD window is time-boxed at 12–24 months in
our own strategy** (`PIVOT_2026_07_PLATFORM_STRATEGY.md` §4.2), and it is
consumed by calendar time whether or not we are benchmarking. Measurement that
does not change a decision is spending that window for nothing.

---

## 2. What the program produced, and what it is worth

Kept and quotable, with their scope attached (never without):

- Six models, three serving regimes, two vendors, one unmodified engine.
- **5.3–9.9×** shared-prefix candidate batching on token-VLA action decode,
  exact parity, both vendors.
- **3.2–3.8×** adaptive-cache generation speedup, reproduced across sessions.
- **6×** resident session density on MI300X vs H100 for a 16.5B policy model.
- **1.02–1.38×** MI300X/H100 LLM output throughput; **1.7–2.3×** cheaper per
  million output tokens at observed prices.
- First publicly reported Cosmos benchmark on any AMD GPU.

The rarer asset is the practice rather than the numbers: pre-registered plans
with kill thresholds committed before the run, predictions scored in public
(the silicon gate got two of four wrong, including its headline), and a
withdrawn claim left visible instead of quietly restated smaller. In a
category where everyone benchmarks and nobody discloses, that is the part a
competitor cannot buy.

**Standing rule, unchanged:** every speedup travels with the workload it was
measured on. The 5.3–9.9× is token-VLA action decode. The LLM figure is 1.5×
and it is withdrawn from the plan. Quoting either without its shape is the
fastest way to lose a diligence conversation.

---

## 3. Deferred to post-raise — and why each is not a blocker

Each of these is a real gap. None of them changes a decision the round
depends on.

| Deferred | Why it is genuinely open | Why it waits |
|---|---|---|
| **Bare-metal MI300X** | Several MI300X numbers are on virtualised slices, disclosed in the report | The VF instance *still won*. Bare metal would be expected to do no worse, so it sharpens a number rather than deciding anything |
| **Matched-version re-run** (`vllm==0.23.0` on H100) | The silicon gate's legs ran different engine versions | The handicap runs *against* AMD, so the result is already a lower bound. Pre-declared as optional in the plan's own §6b |
| **Long-context / FP8 / multi-GPU LLM shapes** | Zero coverage | Widens the claim. We are not currently making a claim that needs widening — §5.2 caps the vanilla-LLM discount at 25% regardless |
| **Per-second / per-call metering for non-LLM classes** | The S2 tail wedge will need it | It is P2 product work, and the P2 hypothesis is *whether the tail pays*, which is a customer conversation before it is an engineering one |
| **FVD at N≥100** | Generation-quality credibility | The generation camp is explicitly not the headline post-pivot |
| **Real proprioceptive pose through `WorldState`** | Would make the robot demo credible | Physical AI is P3 (2027). Building the demo before a robotics partner asks for it is the same mistake the pivot corrected |
| **RoboLab success-rate harness** | Would turn the levers ladder into something quotable | Nice-to-have for a benchmark report, not for a term sheet |

If a specific investor or design partner asks for one of these by name, that
is a decision-changing reason and it gets run. Curiosity is not.

---

## 4. The one exception: an MoE row — **RUN 2026-08-16, and it FAILED**

Pre-registered in **`docs/LLM_MOE_GATE_PLAN.md`**; result in
**`docs/LLM_MOE_GATE_RESULT.md`**. Scope held: one day, one model pair, ~$8.40.

**The counter-example is real.** At matched engine versions and default
configuration, MI300X reached `R = 0.53 / 0.74 / 0.60` of H100 on
DeepSeek-V2-Lite — failing the 0.75 threshold at every shape, where the dense
gate had passed at 1.02–1.38. The silicon claim is now bounded by measurement
rather than by caution, and "dense" is load-bearing in every external use of it.

This is the outcome the §4 argument below said was *worth paying for*, and it
paid: we found it ourselves, in writing, before a partner's engineer did. The
reasoning that follows is left exactly as written before the run.

The reasoning is adversarial rather than curious. Our silicon result covers
**dense Qwen2.5 only**. SemiAnalysis's data most favours NVIDIA on
**sparse/MoE (DeepSeek-class)** models — the exact shape we did not test.
That makes an MoE counter-example the single most likely technical objection
in diligence, and the asymmetry is stark:

- **We run it and it passes** → the claim widens for one day of box time.
- **We run it and it fails** → we scope the claim to dense models ourselves,
  in writing, before anyone else finds it. Which is precisely the move that
  made the withdrawn best-of-N claim an asset rather than a liability.
- **We don't run it** → a partner's technical diligence finds the gap, and we
  are answering it live with no data.

The third outcome is strictly worse than the second. That is what makes this
the only measurement worth spending time on before the raise.

---

## 5. When the program reopens

Explicitly, so this does not drift into a permanent stop:

0. **Already reopened once, narrowly:** the MoE gate ran on 2026-08-16 and
   failed (§4). It also surfaced a variable no gate had named — *kernel backend
   selection*, worth 22–35% on ROCm — which is now a required pre-registered
   variable for any future architecture gate.
1. **After the round closes** — the deferred list in §3 becomes P1/P2 work.
2. **On request from a named investor or design partner** — a question from
   someone deciding is decision-changing by definition.
3. **If a competitor publishes ROCm serving numbers** — Lever B's durability
   is the assumption most exposed to someone else's action, and their
   publication is the signal to re-measure ours.
4. **Before any new external claim.** No number goes into a deck, report or
   site that has not been measured under a pre-registered plan. That rule
   never lapses, closed program or not.

Related: `docs/METHODOLOGY.md` (how numbers are measured),
`docs/POSITIONING.md` (what is defensible to claim),
`decks/VC_QA.md` §6 (the five questions this document says to go answer
instead).
