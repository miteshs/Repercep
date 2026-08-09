# MI300X-vs-H100 LLM throughput gate — **H100 leg only** (2026-08-09)

> **This is half an experiment and carries no verdict.** The gate is a *ratio*,
> `R = MI300X / H100`, and only the denominator exists. Nothing here decides
> anything about `BUSINESS_PLAN_2026.md` Lever B. The MI300X leg needs a manual
> AMD Developer Cloud instance (browser-blocked for automation), so the
> comparison completes in a second sitting.
>
> Plan (pre-registered, binding): `docs/LLM_SILICON_GATE_PLAN.md`.
> Raw: `docs/results/llm_silicon_gate_h100_2026-08-09.json`,
> `..._accuracy_2026-08-09.json`.

## 1. Environment

| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3, driver 580.126.09 |
| Host | RunPod SECURE, `US-GA-2`, **$3.29/hr** |
| Engine | **vLLM 0.26.0** (same version as the July best-of-N gate) |
| torch | 2.11.0+cu130 |
| Attention backend | vLLM default (not overridden) |
| Models | `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`, bf16, `max_model_len=8192` |

**Recorded deviation from the plan.** §2 specified a co-located client against
`127.0.0.1`; this used vLLM's **in-process `LLM` API** instead. That is a
tightening of the same concern — the plan wanted the network out of the
measurement, and in-process removes it entirely — and it deletes HTTP
client/server variance, which matters when the whole point is comparing two
machines. Continuous batching is still exercised: all `conc` prompts go into one
`generate()` call and the scheduler batches them as it would online. Noted here
rather than quietly done.

## 2. H100 baseline

Equal-work enforcement: `min_tokens == max_tokens` with `ignore_eos=True`, so
every request emits **exactly** the requested token count. The plan's equal-work
check is therefore structural rather than checked after the fact —
`equal_work_ok: true` at all six points.

### Qwen2.5-7B-Instruct

| shape | prompt | decode | conc | median s | **output tok/s** |
|---|---:|---:|---:|---:|---:|
| S1-interactive | 512 | 512 | 1 | 3.1214 | **164.03** |
| S2-interactive-load | 512 | 512 | 32 | 3.4551 | **4741.99** |
| S3-batch | 4096 | 128 | 32 | 1.3101 | **3126.59** |

### Qwen2.5-32B-Instruct

| shape | prompt | decode | conc | median s | **output tok/s** |
|---|---:|---:|---:|---:|---:|
| S1-interactive | 512 | 512 | 1 | 12.5373 | **40.84** |
| S2-interactive-load | 512 | 512 | 32 | 13.6531 | **1200.02** |
| S3-batch | 4096 | 128 | 32 | 4.2430 | **965.35** |

Run-to-run spread across 5 repeats was **≤0.5% at every point** (e.g. 7B/S1:
3.1204–3.1266 s). That tightness is the useful part — it means any MI300X
difference above ~1% is signal, not noise, and the gate will not hinge on
whether a run was lucky.

### 2.1 One observation that does not need the second leg

At concurrency 32 the 7B model is **28.9× the throughput** of concurrency 1
(4742 vs 164 tok/s) while per-request latency rises only 11% (3.12 → 3.46 s).
The 32B shows the same shape: 29.4× throughput for 8.9% more latency. Batching
is close to free in this regime on H100. That is a deployment-playbook fact
about serving posture, not a silicon claim, and it is vendor-independent enough
that it will be worth re-checking on the AMD side as a sanity signal.

## 3. Accuracy smoke test — H100 half captured

20 greedy prompts, `temperature=0`, 64 tokens, both models, outputs stored in
`..._accuracy_2026-08-09.json`. The comparison against MI300X is what makes this
meaningful (plan §6 tests SemiAnalysis's "25% of models fail accuracy on ROCm"
claim), so this leg is only a stored baseline. Pre-declared reading stands:
identical output is expected and unremarkable; divergence beyond the first few
tokens is reportable even though it does not move the gate.

## 4. What still has to happen

1. **The MI300X leg** — same script, same models, same shapes. `scripts/` copy is
   `llm_gate.py` as uploaded; it is vendor-agnostic (reads `torch.version.hip`).
2. **Compute `R` per shape per model** and apply the §3 kill criteria **as
   written**, including the uncomfortable branch.
3. **Cost ratio separately and labelled** — the H100 leg cost **$3.29/hr** on
   RunPod against MI300X's **$1.99/hr** on AMD devcloud. Per the plan's §4
   confound, the throughput ratio is clean silicon; **the cost ratio mixes
   silicon with two different providers' pricing and must never be presented as
   a property of the chip.**

## 5. Prediction check — deferred, not quietly dropped

The plan recorded four predictions. **None can be scored yet**, and they are
restated here so they are scored later rather than forgotten: (1) `R ≥ 0.75` on
S3-batch; (2) `R < 0.75` on S1-interactive at concurrency 1; (3) the 32B favours
MI300X relatively more than the 7B; (4) the outcome is shape-scoped survival
rather than a clean pass or kill.

**One early and unofficial data point, from a different workload.** The Cosmos 3
policy chunk measured on both GPUs today (`COSMOS3_ON_MI300X.md` §4.7) gives
MI300X/H100 = **0.83×** — above the 0.75 line, but that is a *diffusion*
workload, not token generation, and it is not the gate. It is mentioned only
because it is the sole cross-vendor number currently in hand, and it should not
be allowed to set expectations for the LLM result.
