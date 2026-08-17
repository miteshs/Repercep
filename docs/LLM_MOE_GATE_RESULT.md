# MoE silicon gate — **COMPLETE. The gate FAILS.** (2026-08-16)

- **Plan (pre-registered, binding):** `docs/LLM_MOE_GATE_PLAN.md`, including the
  §2.1a model decision committed before any box was created.
- **Verdict: under matched engine versions and each side's default
  configuration, `R = 0.53 / 0.74 / 0.60` against a `< 0.75` kill threshold —
  MI300X loses at every shape. Outcome 3 fires. The dense-model silicon claim
  does NOT extend to MoE.**
- Raw: `docs/results/moe_gate_h100_2026-08-16.json`,
  `..._h100_v23_2026-08-16.json`, `..._h100_v23_flashinfer_2026-08-16.json`,
  `..._mi300x_2026-08-16.json`, `moe_aiter_mi300x_2026-08-16.json`.
  Harness: `scripts/bench_llm_silicon_gate.py`.

This is the counter-example the gate was built to look for, and it was found on
the first serious attempt. **`LLM_SILICON_GATE_RESULT.md` §5 already restricted
that claim to "dense Qwen2.5-class models only." This result converts that
restriction from a precaution into a measured boundary.**

---

## 1. Result

`deepseek-ai/DeepSeek-V2-Lite` (15.7B total / 2.4B active, MLA + 64 routed
experts), bf16, TP=1, one GPU per side. Equal work enforced structurally
(`min_tokens == max_tokens` + `ignore_eos`) — verified at every point of every
run below.

### 1.1 The gate: version-matched, default configuration both sides

Both legs on vLLM 0.23.x, both selecting the TRITON MoE backend by default.

| shape | H100 tok/s | MI300X tok/s | **R** | vs 0.75 |
|---|---:|---:|---:|:--:|
| S1-interactive (c=1) | 239.21 | 126.21 | **0.528** | **FAIL** |
| S2-interactive-load (c=32) | 3290.11 | 2433.32 | **0.740** | **FAIL** |
| S3-batch (c=32) | 2262.05 | 1356.72 | **0.600** | **FAIL** |

**`R < 0.75` at every shape → §3 outcome 3.**

### 1.2 All five configurations measured

| # | config | S1 | S2 | S3 |
|---|---|---:|---:|---:|
| 1 | H100, vLLM 0.26.0, TRITON | 248.74 | 3632.64 | 2282.68 |
| 2 | H100, vLLM 0.23.0, TRITON *(gate)* | 239.21 | 3290.11 | 2262.05 |
| 3 | H100, vLLM 0.23.0, flashinfer installed → **still TRITON** | 238.99 | 3292.56 | 2252.35 |
| 4 | MI300X, vLLM 0.23.1, TRITON *(gate)* | 126.21 | 2433.32 | 1356.72 |
| 5 | MI300X, vLLM 0.23.1, **AITER** | 154.06 | 3295.08 | 1822.18 |

---

## 2. The version confound is measured, and it is small

The dense gate's §6b required that a *failing* result not be declared until the
H100 leg was re-run on the AMD-matched engine version. That was done (config 2),
and it is why this document can carry a verdict at all.

| shape | H100 0.23.0 ÷ H100 0.26.0 |
|---|---:|
| S1 | 0.962 |
| S2 | 0.906 |
| S3 | 0.991 |

Three minor versions of engine work are worth **4%, 9% and 1%** on H100. The
MI300X deficit is **26–47%**. **The version gap cannot account for the result**,
which is exactly what the pre-registered re-run existed to establish. Had it
been skipped, the honest reading would have been "inconclusive."

---

## 3. The finding that matters operationally: ROCm's default kernel is not its best

vLLM on MI300X offered `['ROCm AITER', 'TRITON', 'BATCHED_TRITON']` and
**selected TRITON**. Setting `VLLM_ROCM_USE_AITER=1` selects AMD's own kernel
library instead and is worth a lot:

| shape | TRITON | AITER | gain |
|---|---:|---:|---:|
| S1 | 126.21 | 154.06 | **1.22×** |
| S2 | 2433.32 | 3295.08 | **1.35×** |
| S3 | 1356.72 | 1822.18 | **1.34×** |

Against the version-matched H100, that moves `R` to **0.644 / 1.002 / 0.806** —
parity at S2, comfortably over threshold at S3, still losing S1.

**This is not scored as a gate pass, for two reasons.**

**(a) It is not the pre-registered configuration.** The plan fixed engine,
model, precision, client and repeats, and both legs ran their defaults. Changing
one side's kernel selection after seeing a failing number is precisely the
freedom pre-registration exists to remove. It is reported as an exploratory
result, which is what it is.

**(b) It is not demonstrated to be free.** See §4.

**A control was run so this is not an unfair comparison.** If AMD gets its
optimized kernels, NVIDIA must get its own. `flashinfer` was installed on the
H100 and vLLM **still selected TRITON** (config 3), with throughput unchanged
within noise (238.99 vs 239.21). The FlashInfer TRTLLM/CUTLASS paths are
quantized routes and are not chosen for unquantized bf16. **TRITON genuinely is
H100's best available backend for this workload**, so AITER-vs-TRITON is a fair
each-vendor's-best comparison rather than a handicap on NVIDIA.

---

## 4. AITER changes the outputs, and that is unresolved

Accuracy smoke: 20 greedy prompts, compared against the version-matched H100.

| config | exact-identical to H100 | contains correct answer (13 checkable) |
|---|---:|---:|
| MI300X TRITON | **11/20** | **11/13** — same as H100 |
| MI300X AITER | **5/20** | **10/13** |

TRITON on MI300X tracks H100 closely and answers exactly what H100 answers.
AITER diverges from H100 more than twice as often, frequently **from the first
generated token**, and lost one otherwise-correct answer: *"Who wrote Pride and
Prejudice?"* is answered "Jane Austen" by both H100 and MI300X-TRITON, while
AITER emits `, 1813` repeatedly and never names the author.

**What must not be concluded from this.** Thirteen checkable prompts with a
one-answer difference is *not* an accuracy evaluation and cannot support a
claim that AITER is less accurate. Degenerate repetition is **equally common
under TRITON and AITER (3/20 each)** — that is a property of a base,
non-instruct model under greedy decoding at 64 tokens, not an AMD defect, and
the first draft of this analysis got that wrong before counting it.

**What can be concluded:** the 1.22–1.35× is **not demonstrated to be free**,
and a numerical path that diverges from the reference this much belongs behind
a real eval before it goes anywhere near production or a benchmark claim. That
is the same standard applied to the adaptive-cache lever, which is disclosed as
trading a measured quality dimension rather than presented as a pure win.

---

## 5. What this settles for the business plan

`BUSINESS_PLAN_2026.md` §6.4 and `LLM_SILICON_GATE_RESULT.md` carry the silicon
claim. **Nothing about the dense result changes** — it stands exactly as
measured. What changes is that its stated scope is now load-bearing:

- **Supportable, unchanged:** on dense Qwen2.5-class models under vLLM, a single
  MI300X matched or beat a single H100 at every shape tested, at 1.7–2.3× lower
  cost per million output tokens.
- **Now measured, and must be stated wherever the above is used:** on a
  **DeepSeek-class MoE** model, the same comparison **inverts** — MI300X reached
  only 0.53–0.74 of H100 in the default configuration a customer would actually
  get.
- **The honest one-line version:** *we went looking for the counter-example to
  our own strongest claim, on the model family most likely to produce one, and
  we found it.*

Commercially this is smaller than it sounds and worth stating without drama.
Lever B was never claimed to be architecture-independent, MoE serving is not the
segment the plan leads with, and the operator-side finding in §3 — that ROCm's
default kernel selection leaves 22–35% on the table — is a genuine deployment
lever we now own and competitors publishing nothing do not. But **any future
external use of the silicon claim must carry "dense" in the sentence**, and the
deck and site must not imply otherwise.

---

## 6. Caveats — read before quoting anything here

- **One box per vendor, one session, one model.** No cross-box replication.
- **MI300X is a VF (virtualised) instance**, as in the dense gate.
- **S2's H100 run-to-run spread was 16.5%** (vs ≤0.5% throughout the dense
  gate). S2's `R = 0.740` sits within noise of the 0.75 threshold and should be
  read as "at the line", not as a precise value. S1 and S3 fail by margins far
  outside their spread (2.6–4.4%) and are what carry the verdict.
- **Engine versions are matched but not identical**: H100 on PyPI `0.23.0`,
  MI300X on AMD's image `0.23.1.dev1+g9ddef7117`. Closer than the dense gate's
  three-minor-version gap, and §2 shows the residual is small.
- **This is a 16B/2.4B MoE.** It answers "does the advantage survive an MoE
  *architecture*", not "does it survive a *large* MoE" — the latter needs
  multi-GPU and was out of scope by the plan's one-day rule.
- **The accuracy smoke is a smoke test, not an eval** (§4).

---

## 7. Predictions scored — three of four right, and the miss is instructive

| # | prediction | outcome |
|---|---|---|
| 1 | `R < 0.75` on S1 at c=1 | **CORRECT** — 0.528, the worst shape, as predicted |
| 2 | `R` on MoE lower than on dense at every matched shape | **CORRECT** — 0.53–0.74 vs the dense gate's 1.02–1.38 |
| 3 | Outcome 2 (shape-scoped survival) | **WRONG** — clean fail at all three |
| 4 | The fit asymmetry would be the hardest thing to report honestly | **WRONG** — §2.1a's model choice removed it entirely; the hardest thing was §3/§4, which the plan did not anticipate at all |

Prediction 1 is worth dwelling on. The *identical* prediction on the dense gate
was wrong by the largest margin of any — MI300X led H100 by 1.32–1.38× at c=1
there. Here the same prior was right. Low-concurrency interactive decode is
where the architecture-dependence bites hardest, in both directions.

Prediction 4's miss is the useful one: the plan spent its foresight on the
memory-fit trap, solved it in §2.1a, and was then blindsided by kernel
*selection* — a variable it never named. **The next gate on any new
architecture should treat backend selection as a first-class pre-registered
variable, on both vendors.**

---

## 8. Cost and infrastructure

Three boxes, ~$8.40 total: H100 @ $3.29/hr ×2 (~50 min, ~65 min) and MI300X @
$1.99/hr (~63 min). **All terminated; zero pods and zero droplets confirmed via
both APIs.**

Findings worth not re-deriving are in the RunPod and AMD Dev Cloud memory notes.
The two that cost the most here:

- **Killing a vLLM run does not free the GPU.** The engine renames its process
  title to `VLLM::EngineCore`, survives a `pkill` matching the launching script,
  and holds ~76 GiB; the next run then fails with a misleading *"reduce GPU
  memory used by other processes"*, and `nvidia-smi` reports **host** PIDs that
  cannot be killed from inside the container. Use `pkill -f 'VLLM::EngineCor[e]'`.
- **A DigitalOcean API token does not remove the manual step.** It authenticates
  and can list/inspect/**delete** droplets — which does solve cleanup — but
  MI300X creation is console-gated: `gpu-mi300x1-192gb-devcloud` returns *"This
  size is unavailable"* across all 13 regions tried, while other AMD parts
  (MI325X `tor1`, MI350X `ric1`, MI355X `mem1`) **are** API-orderable.
