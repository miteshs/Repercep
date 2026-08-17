# Pre-registration: the MoE silicon gate (written 2026-08-16, unrun)

**Binding.** Written and committed before any measurement, same discipline as
`LLM_BESTOFN_PLAN.md` and `LLM_SILICON_GATE_PLAN.md`. The point of writing it
first is that the analysis cannot be chosen after seeing the data — which is
why the best-of-N result was believable when it came back negative and why the
silicon result was believable when it came back better than predicted.

**Status: not yet run.** This is the single measurement
`docs/BENCHMARK_PROGRAM.md` holds open before the seed round. Everything else
is deferred.

## 1. What is under test

`LLM_SILICON_GATE_RESULT.md` established `R = 1.02–1.38×` (MI300X ÷ H100
output throughput) at every shape tested — but **on dense Qwen2.5 models
only**. Its own §5 names the gap and its §6 names the follow-up:

> Highest-value follow-up: an MoE model (DeepSeek/Mixtral class). That is
> where SemiAnalysis's data most favours NVIDIA and where our coverage is
> zero — the shape most likely to produce a counter-example, which makes it
> the one worth running next.

This gate runs exactly that. The adversarial hypothesis is handed to us again:
**SemiAnalysis's AMD-favourable findings were on large dense models, and their
NVIDIA-favourable ones included sparse/MoE.** If our claim breaks anywhere, it
breaks here.

MoE is also mechanistically the most plausible place for it to break: expert
routing is scatter/gather-heavy and latency-sensitive to kernel dispatch,
which is historically where ROCm has lagged — the opposite of the
bandwidth-bound regime where MI300X's 192 GB and HBM bandwidth carry it.

## 2. Design

Deliberately a re-run of the existing harness with one variable changed. No
new code, no new metrics, no new shapes.

| | |
|---|---|
| Harness | `scripts/bench_llm_silicon_gate.py` + a `--tensor-parallel-size` flag (§2.2) |
| Engine | vLLM, same rationale as the dense gate (most mature ROCm path) |
| Model | **`deepseek-ai/DeepSeek-V2-Lite`** (~31 GB bf16) — decided 2026-08-16, see §2.1a |
| ~~Primary~~ | ~~`mistralai/Mixtral-8x7B-Instruct-v0.1`~~ — **rejected**, does not fit one H100 |
| ~~Fallback~~ | ~~`Qwen/Qwen3-30B-A3B`~~ — **rejected on arithmetic**, §2.1a |
| Precision | bf16 both sides. No FP8 on either, even where supported |
| Client | co-located on the pod, against `127.0.0.1` |
| Repeats | 5 timed, median reported, warm-up discarded |

**Model choice, stated in advance so it is not reverse-engineered later.**
Mixtral-8x7B at ~87 GB bf16 **fits MI300X's 192 GB and does not fit a single
80 GB H100.** That is not an accident and it must not be quoted as a throughput
win: the H100 leg therefore needs either tensor-parallel across 2× H100 or a
smaller MoE. See §4 — this confound is the main design risk in the gate.

### 2.1a Model decision — settled on arithmetic, before any box was created

The instruction was "the smaller MoE that fits both sides single-GPU." Working
out which model that actually *is* eliminated the plan's own named fallback.

Config values read from each model's `config.json`; KV per token is
`2 × kv_heads × head_dim × layers × 2 bytes` (MLA models instead carry
`kv_lora_rank + rope_dim` per layer). The binding shape is **S3-batch**, whose
working set is `32 × (4096 + 128) = 135,168` tokens. H100 budget is
`80 GB × 0.90 = 72 GB` minus weights.

| candidate | weights bf16 | KV/token | S3 KV need | H100 KV free | verdict |
|---|---:|---:|---:|---:|---|
| Mixtral-8x7B | ~87 GB | 24 KiB | 3.1 GiB | **negative** | **does not fit at all** |
| Qwen3-30B-A3B | ~61 GB | 96 KiB | **12.4 GiB** | **~11 GB** | **does not fit** |
| Qwen1.5-MoE-A2.7B | ~29 GB | 192 KiB | 24.8 GiB | ~43 GB | fits, but a 2024-era model |
| **DeepSeek-V2-Lite** | **~31 GB** | **30 KiB** | **4.0 GiB** | **~41 GB** | **fits, large headroom** |

**Why Qwen3-30B-A3B was rejected, and why it matters more than it looks.** It
*loads* on an 80 GB H100 — 61 GB of weights leaves ~11 GB for KV — but the
S3-batch working set needs ~12.4 GiB. vLLM would not crash; it would preempt
and queue, quietly serving fewer concurrent requests. MI300X, with ~111 GB of
KV headroom after the same weights, would run the full batch.

That produces an MI300X win **on S3-batch that is a memory-capacity result
wearing a throughput result's clothes** — and it would inflate our own number,
which is the direction we have the least licence to be wrong in. It is the
same failure mode as the Mixtral fit asymmetry (§4), one level less obvious,
and it would have been invisible in the output JSON.

**Why DeepSeek-V2-Lite is the right answer rather than merely a working one.**
It is the **DeepSeek architecture** — the exact family §1 names as the source
of SemiAnalysis's NVIDIA-favourable findings, so it probes the counter-example
head-on instead of near it. Its MLA attention is also a genuinely distinct
kernel path where ROCm maturity is most likely to differ from CUDA's, which is
the highest-information place to look. And at 30 KiB/token of KV it clears
every shape on both GPUs with room to spare, so **nothing in the result can be
attributed to memory capacity.**

Recorded risk, in advance: **MLA on ROCm under vLLM 0.23.x is the least
certain part of this gate.** If it will not serve, that is not a failed
experiment — a DeepSeek-class model failing to run on ROCm is a first-class
finding and gets published as one under §3's outcome 3.

The 16B/2.4B size is smaller than the dense gate's 32B. Stated plainly so it
is not discovered later: this gate answers "does the advantage survive an MoE
*architecture*", not "does it survive a *large* MoE." The latter needs
multi-GPU and is out of scope by §6's one-day rule.

### 2.1 Workload shapes — identical to the dense gate, reported separately

| tag | prompt | decode | concurrency |
|---|---:|---:|---:|
| **S1-interactive** | 512 | 512 | 1 |
| **S2-interactive-load** | 512 | 512 | 32 |
| **S3-batch** | 4096 | 128 | 32 |

Reusing the shapes verbatim is the point: the dense result is the control, and
a shape change would make the two ungraphable together.

### 2.2 The one harness change, made before the run

The dense gate ran TP=1 on both vendors, so `bench_llm_silicon_gate.py` never
had a tensor-parallel option — it took vLLM's default of 1. **The H100 leg of
a Mixtral run could therefore not be expressed at all.** Added 2026-08-16,
before any measurement:

- `--tensor-parallel-size` (default 1, so every dense-gate invocation is
  byte-identical in behaviour to before).
- `tensor_parallel_size` and `gpus_used` are written into the result JSON.

The second half is the part that matters. A TP=1-vs-TP=2 run is a **per-node**
comparison, not a per-GPU one, and recording it in the artifact means the
number cannot later be quoted as a silicon ratio by someone reading the JSON
without the plan. §4's first confound is the reason.

Recorded here rather than in a commit message alone because the plan is the
binding document: the capability was added, the default is unchanged, and no
metric or shape moved.

### 2.3 Metrics

Primary: **median output tokens/sec**, per shape, per model. Secondary: TTFT,
end-to-end latency, completion-token totals (the equal-work check, enforced
structurally with `min_tokens == max_tokens` + `ignore_eos` as before).

## 3. Kill criteria — decided now

Let `R = median MI300X throughput ÷ median H100 throughput`, per shape.

1. **`R ≥ 0.75` at every shape** → the silicon claim **widens** from "dense
   Qwen2.5-class" to "dense and sparse, at the shapes tested." Report with
   the same §5-style caveat block as the dense result.
2. **`R ≥ 0.75` on some shapes, `< 0.75` on others** → the claim stays
   **scoped to dense models**, and the MoE result is published as the
   scoping evidence. Expected outcome (§5).
3. **`R < 0.75` at every shape** → **the MoE counter-example is real.** The
   claim is explicitly restricted to dense models in
   `BUSINESS_PLAN_2026.md`, `LLM_SILICON_GATE_RESULT.md` and any external
   artifact, *and the negative result is published*, exactly as the best-of-N
   claim was withdrawn rather than shrunk.

**Pre-committed:** outcome 3 is not a reason to suppress or delay. A published
MoE negative costs one day and one paragraph. Being shown it by a partner's
engineer in a diligence call costs the round.

## 4. Confounds pre-declared

- **The fit asymmetry is the big one.** Mixtral bf16 does not fit one H100.
  Whatever is done about it — 2× H100 tensor-parallel, or a smaller MoE —
  **changes what is being compared, and the result must say so in its first
  paragraph.** If the H100 leg runs on two GPUs, then `R` is a per-*node*
  comparison and the per-GPU cost arithmetic must be redone accordingly; a
  1-GPU-vs-2-GPU throughput ratio quoted as a silicon ratio would be a fake
  number of exactly the kind the Cosmos 3 batching gate already caught once.
- **Provider confound on cost, not throughput.** Same as the dense gate:
  MI300X on AMD Developer Cloud, H100 on RunPod. The throughput ratio is a
  clean silicon comparison; the cost ratio mixes silicon with procurement.
  Report separately. Never present the cost ratio as a property of silicon.
- **Engine-version asymmetry persists.** MI300X tops out at vLLM 0.23.x in
  AMD's official gfx942 images; H100 runs newer. As established in the dense
  gate's §6b, this handicap runs **against** AMD, so a pass is a lower bound
  and a fail is *not* decisive on its own — a failing result requires an H100
  re-run pinned to the matching version before outcome 3 is declared.
- **MoE expert placement / autotune.** ROCm pays autotune per tensor shape
  (established in the Cosmos 3 run). Every shape must be warmed on both GPUs
  before timing, or the AMD side eats a penalty unrelated to steady-state
  throughput. This trap has already produced wrong numbers twice.
- **Payload assertion, not status codes.** Readiness asserts the model id in
  `/v1/models`, never a 2xx — RunPod's nginx returns its own 200s.

## 5. Predictions on record

1. **`R < 0.75` on S1-interactive at concurrency 1.** Expert-routing dispatch
   overhead should hurt most where there is no batching to hide it.
   Confidence: moderate. *(Note: the identical prediction on the dense gate
   was wrong by the largest margin of any — MI300X led by 1.32–1.38× there.)*
2. **`R` on MoE is lower than `R` on dense at every matched shape.**
   Confidence: moderate-high. This is the directional claim actually being
   tested.
3. **Outcome 2 — shape-scoped survival.** Confidence: moderate. *(The same
   prediction was wrong on the dense gate, which produced a clean pass.)*
4. **The fit asymmetry, not the throughput, will be the hardest thing to
   report honestly.** Confidence: high.

Predictions 1 and 3 are recorded knowing their dense-gate counterparts both
failed. Repeating a prior that was already falsified once, and saying so, is
the correct way to keep the scorecard meaningful.

## 6. Scope and stopping rule

**One day. One model pair. Both boxes terminated at the end of the session
regardless of outcome.**

If the fit asymmetry cannot be resolved cleanly within that day — no 2× H100
available and no suitable smaller MoE that fits both sides — **stop and
publish the attempt as inconclusive** rather than extending. An honest
"we tried, here is why it did not produce a comparable number" is worth more
than a number produced by an unequal comparison, and `BENCHMARK_PROGRAM.md`
closes the program either way.
