# Latest Mirage benchmarks — H100 vs MI300X on RunPod (2026-06-26)

Fresh end-to-end measurements of the Mirage runtime on **NVIDIA H100 SXM**
and **AMD MI300X**, both rented on RunPod, for the two live models:
**Cosmos-Predict-7B** (diffusion world model, generation regime) and
**V-JEPA 2-AC** (energy-based action-conditioned world model, control regime —
the strategic lead per `docs/REVISED_STRATEGY.md`).

These numbers re-confirm the prior `docs/COSMOS_ON_H100.md` /
`docs/COSMOS_ON_MI300X.md` results on current driver/library stacks and add the
**first measured V-JEPA 2-AC latency profile** on both vendors, plus a measured
prototype of the **CEM-candidate-batching** optimization the revised strategy
names as the core serving mechanism.

## Setup

| | NVIDIA | AMD |
|---|---|---|
| GPU | H100 SXM 80GB HBM3 | MI300X 192GB HBM3 (gfx942) |
| RunPod cost | $3.29/hr | $2.19/hr |
| Base image | `runpod/pytorch:…cu1281-torch280-ubuntu2404` | `runpod/pytorch:2.4.0-…rocm6.1.0` |
| Python | 3.12 | 3.12 (conda base) |
| torch | 2.8.0+cu128 | **2.3.1+rocm6.0** (see gotcha 2) |
| transformers / diffusers | 4.56.2 / 0.38.0 | 4.56.2 / 0.38.0 |

## Cosmos-Predict-7B (121 frames @ 1280×704, 36 steps)

| config | H100 (this run) | H100 (prior doc) | MI300X (this run) | MI300X (prior doc) |
|---|---|---|---|---|
| baseline, no cache | **449.9 s** | 446.3 s | _measuring_ | 470 s |
| adaptive cache (thr 0.30) | **139.4 s** (3.23×) | 138.4 s | _measuring_ | 154 s |
| peak HBM | 52.5 GiB | 52.5 GiB | _measuring_ | 52.5 GiB |

Smoke (17 f / 8 step): H100 10.0 s gen, 28.4 GiB peak. The adaptive-cache
speedup and peak HBM reproduce the prior H100 doc within run-to-run variance —
the TeaCache-style loop-level cache is still the dominant Cosmos optimization
and is vendor-neutral by construction.

## V-JEPA 2-AC (300M AC predictor, ViT-g encoder, 8-frame context = 2048 tokens)

First measured latency profile. The rollout `step` and the CEM `plan` are the
numbers that matter for closed-loop control. The AC predictor runs in **fp32**
on both vendors (its RoPE attention upcasts q/k → bf16 SDPA dtype mismatch), so
the per-forward cost is an fp32-attention cost on both.

| metric | H100 | MI300X | MI300X / H100 |
|---|---|---|---|
| rollout step | **70.8 ms** (14.1/s) | **75.8 ms** (13.2/s) | 1.07× |
| CEM plan, H=4 (768 fwds) | 55.6 s | 67.2 s | 1.21× |
| CEM plan, H=8 (1536 fwds) | 110.9 s | 129.7 s | 1.17× |
| encoder reset | 0.55 s | 0.41 s | 0.75× |
| peak HBM | 3.4 GiB | 3.7 GiB | — |

**The silicon gap is small** (step within 7%, plan 17–21%) — consistent with the
~5–11% Cosmos silicon delta in `docs/METHODOLOGY.md`. MI300X is a credible host
for the control regime, and its 192 GB HBM leaves enormous headroom (3.7 / 192
GiB used).

## The headline analysis: how to make it better

The V-JEPA 2-AC profile exposes the real bottleneck. A single CEM `plan` call at
H=4 takes **55.6 s on H100** — far too slow for a closed-loop controller (robot
control loops want 10–100 ms). The cost is structural: CEM does
`samples × iters × horizon` predictor forwards (64 × 3 × 4 = **768**), and the
shipped planner (`VJepa2ACEngine._plan_sequence`) runs them **sequentially at
batch=1**, with only **3.4 GiB of 80 used**.

### 1. CEM-candidate batching — measured, 1.6× (not 64×)

`scripts/bench_cem_batched.py` rolls all 64 candidates out in **one batched
predictor forward per timestep** (768 → 12 forwards), identical CEM math:

| horizon | sequential | batched | speedup | forwards | energy (seq/bat) |
|---|---|---|---|---|---|
| H=4 | 54.5 s | 35.1 s | **1.6×** | 768 → 12 | 110 / 117 |
| H=8 | 110.1 s | 70.2 s | **1.6×** | 1536 → 24 | 146 / 148 |

**The surprise — and the most important finding:** 64× fewer forwards yields only
**1.6× wall-time**, with energy parity. The per-candidate forward over the
2048-token context is **already compute-bound**, so candidate-batching buys GPU
efficiency, not launch-overhead elimination. The naive "batch the candidates and
get Nx" intuition is wrong at this context size. This redirects the optimization
roadmap:

### 2. Where the real wins are (next, un-measured)

- **Latent / KV reuse across rollout steps.** Within one rollout the predictor
  reprocesses the full ~2048-token context every timestep; the context is a
  sliding window, so most K/V is recomputed. KV-caching the carried context
  across the H steps removes redundant attention — the single biggest structural
  win, and the one the strategy bets on.
- **A bf16/fp16 predictor path.** The fp32 requirement (RoPE upcast) is the
  dominant per-forward cost and the reason ROCm has no flash path here. An
  attention op that keeps q/k fp32 only where RoPE needs it, with bf16
  v/score/output, should roughly halve the forward — on *both* vendors, and it
  re-enables flash-attention on ROCm.
- **CEM amortization (receding-horizon warm-start).** In MPC the next control
  step's CEM can warm-start from the previous solution (shift-and-reuse the
  elite mean), cutting `iters` or `samples` for steps after the first — a
  serving-loop win orthogonal to the per-forward cost.
- **Fewer, smarter samples.** 64 samples × 3 iters is a generic default;
  measuring the energy-vs-samples curve likely allows a smaller budget at equal
  plan quality.

## Reproduction gotchas (new this run)

1. **`[models]` extra clobbers ROCm torch.** `torchvision>=0.22` in the `models`
   extra resolves a CUDA torch from PyPI, silently replacing the HIP build (the
   backend then falls back to CPU). Install the package deps, then
   `--force-reinstall --no-deps` the ROCm `torch`/`torchvision`.
2. **MIOpen Conv3d is broken on torch 2.6+rocm6.1.** The V-JEPA 2 encoder's
   tubelet patch-embed (`Conv3d`) hard-crashes with
   `munmap_chunk(): invalid pointer → Aborted` (heap corruption in MIOpen's algo
   search) — even an isolated tiny Conv3d. **torch 2.3.1+rocm6.0 runs it fine.**
   Pin the rocm6.0 wheel on MI300X until a newer ROCm fixes MIOpen Conv3d.
3. **First ROCm predictor forward is ~3× the warm cost** (MIOpen kernel compile);
   benchmark with warmup or the cold number misleads (222 ms cold → 75.8 ms warm).
4. Cosmos is a **gated** HF repo; a **fine-grained** token needs the
   "public gated repos" permission (a classic *Read* token just works).

## Scripts

- `scripts/bench_vjepa2_ac.py` — V-JEPA 2-AC latency/throughput harness (load /
  reset / rollout-step / CEM-plan). Fills the gap that `run_vjepa2_ac.py` is only
  a wiring demo.
- `scripts/bench_cem_batched.py` — sequential-vs-candidate-batched CEM, with the
  energy-parity check.
