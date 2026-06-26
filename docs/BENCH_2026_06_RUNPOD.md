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
| torch | 2.8.0+cu128 | **2.9.1+rocm6.3** (Cosmos), 2.3.1+rocm6.0 (V-JEPA) — see gotcha 2 |
| transformers / diffusers | 4.56.2 / 0.38.0 | 4.56.2 / 0.38.0 |

## Cosmos-Predict-7B (121 frames @ 1280×704, 36 steps)

| config | H100 (this run) | H100 (prior doc) | MI300X (this run) | MI300X (prior doc) |
|---|---|---|---|---|
| baseline, no cache | **449.9 s** | 446.3 s | **578.2 s** | 470 s |
| adaptive cache (thr 0.30) | **139.4 s** (3.23×) | 138.4 s | **154.0 s** (3.75×) | 154 s |
| peak HBM | 52.5 GiB | 52.5 GiB | 52.5 GiB | 52.5 GiB |

Smoke (17 f / 8 step): H100 10.0 s gen / 28.4 GiB; MI300X 55.5 s (cold MIOpen
kernel compile on first generation). The adaptive-cache result reproduces the
prior docs within variance on **both** vendors — the MI300X adaptive 154.0 s
matches the prior 154 s exactly. The TeaCache-style loop-level cache is still
the dominant Cosmos optimization and is vendor-neutral by construction.

**Cross-silicon:** the H100/MI300X gap is **1.10× on the deployable adaptive
path** and 1.29× on the untuned baseline. The wider baseline gap is *stack*, not
silicon — the prior MI300X 470 s used a tuned rocm7.2 / FP8 aotriton path; this
run is untuned rocm6.3 SDPA. The cache-dominated adaptive path is where the two
converge, consistent with `docs/METHODOLOGY.md` §3. MI300X also gets a *bigger*
cache speedup (3.75× vs 3.23×) because its uncached baseline is slower.

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

| horizon | GPU | sequential | batched | speedup | energy (seq/bat) |
|---|---|---|---|---|---|
| H=4 | H100 | 54.5 s | 35.1 s | **1.6×** | 110 / 117 |
| H=8 | H100 | 110.1 s | 70.2 s | **1.6×** | 146 / 148 |
| H=4 | MI300X | 68.4 s | 32.1 s | **2.1×** | 106 / 104 |
| H=8 | MI300X | 129.7 s | 60.8 s | **2.1×** | 146 / 140 |

(768 → 12 forwards at H=4, 1536 → 24 at H=8; energy parity preserved everywhere.)

**The surprise — and the most important finding:** 64× fewer forwards yields only
**1.6× (H100) / 2.1× (MI300X)** wall-time, with energy parity. The per-candidate
forward over the 2048-token context is **already compute-bound**, so
candidate-batching buys GPU efficiency, not launch-overhead elimination. The
naive "batch the candidates and get Nx" intuition is wrong at this context size.
MI300X gains more (2.1×) because its batch=1 forward was further from saturating
the device — i.e. the optimization is *more* valuable on the AMD part. This
redirects the optimization roadmap:

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
2. **MI300X torch/ROCm version matrix — the central setup trap.** The two models
   have opposite needs, and only one ROCm version satisfies both:
   - V-JEPA 2 encoder uses `Conv3d` (tubelet patch-embed). **MIOpen Conv3d is
     broken on torch 2.6+rocm6.1** — hard-crashes with
     `munmap_chunk(): invalid pointer → Aborted` (heap corruption in MIOpen's
     algo search), even an isolated tiny Conv3d.
   - Cosmos's DiT calls SDPA with `enable_gqa=True`, which needs **torch ≥2.5**;
     its VAE decode *also* uses `Conv3d`, so it hits the same MIOpen bug on
     rocm6.1.

   | torch / ROCm | MIOpen Conv3d | `enable_gqa` SDPA | runs |
   |---|---|---|---|
   | 2.3.1 + rocm6.0 | ✅ | ❌ | V-JEPA only |
   | 2.6.0 + rocm6.1 | ❌ (crash) | ✅ | neither fully |
   | **2.9.1 + rocm6.3** | ✅ | ✅ | **both** |

   **Use torch 2.9.1+rocm6.3 on MI300X** — MIOpen Conv3d is fixed by rocm6.3 and
   `enable_gqa` is present. (The V-JEPA numbers above were taken on 2.3.1+rocm6.0
   before this was found; re-running them on 6.3 is a minor follow-up.)
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

## Raw measured results (provenance)

Verbatim JSON `RESULT` lines from the harnesses, RunPod 2026-06-26.

```jsonl
# V-JEPA 2-AC (scripts/bench_vjepa2_ac.py)
{"model": "vjepa2-ac-300m", "device": "NVIDIA H100 80GB HBM3", "dtype": "bf16", "load_seconds": 798.7, "reset_seconds": 0.552, "step_ms": 70.8, "steps_per_sec": 14.1, "cem_config": {"samples": 64, "iters": 3}, "plans": {"4": {"plan_seconds": 55.622, "predictor_forwards": 768, "ms_per_forward": 72.42}, "8": {"plan_seconds": 110.874, "predictor_forwards": 1536, "ms_per_forward": 72.18}}, "peak_hbm_gib": 3.4, "context_shape": [2048, 1408]}
{"model": "vjepa2-ac-300m", "device": "AMD Instinct MI300X", "dtype": "bf16", "load_seconds": 65.7, "reset_seconds": 0.412, "step_ms": 75.8, "steps_per_sec": 13.2, "cem_config": {"samples": 64, "iters": 3}, "plans": {"4": {"plan_seconds": 67.171, "predictor_forwards": 768, "ms_per_forward": 87.46}, "8": {"plan_seconds": 129.653, "predictor_forwards": 1536, "ms_per_forward": 84.41}}, "peak_hbm_gib": 3.7, "context_shape": [2048, 1408]}
# Cosmos-Predict-7B (scripts/run_cosmos.py; 17f/8 smoke, 121f/36 baseline, 121f/36 adaptive)
{"model": "cosmos-predict1-7b-text2world", "device": "NVIDIA H100 80GB HBM3", "frames": 17, "steps": 8, "generate_seconds": 10.0, "peak_hbm_gib": 28.4}
{"model": "cosmos-predict1-7b-text2world", "device": "NVIDIA H100 80GB HBM3", "frames": 121, "steps": 36, "generate_seconds": 449.9, "seconds_per_step": 12.5, "peak_hbm_gib": 52.5}
{"model": "cosmos-predict1-7b-text2world", "device": "NVIDIA H100 80GB HBM3", "frames": 121, "steps": 36, "generate_seconds": 139.4, "seconds_per_step": 3.87, "peak_hbm_gib": 52.5}
{"model": "cosmos-predict1-7b-text2world", "device": "AMD Instinct MI300X", "frames": 17, "steps": 8, "generate_seconds": 55.5, "peak_hbm_gib": 28.4}
{"model": "cosmos-predict1-7b-text2world", "device": "AMD Instinct MI300X", "frames": 121, "steps": 36, "generate_seconds": 578.2, "seconds_per_step": 16.06, "peak_hbm_gib": 52.5}
{"model": "cosmos-predict1-7b-text2world", "device": "AMD Instinct MI300X", "frames": 121, "steps": 36, "generate_seconds": 154.0, "seconds_per_step": 4.28, "peak_hbm_gib": 52.5}
# CEM-candidate batching (scripts/bench_cem_batched.py)
{"model": "vjepa2-ac-300m", "device": "NVIDIA H100 80GB HBM3", "cem": {"samples": 64, "elites": 8, "iters": 3}, "results": {"4": {"sequential_s": 54.458, "batched_s": 35.073, "speedup": 1.6, "energy_sequential": 110.0, "energy_batched": 117.0}, "8": {"sequential_s": 110.094, "batched_s": 70.187, "speedup": 1.6, "energy_sequential": 146.0, "energy_batched": 148.0}}, "peak_hbm_gib": 11.4}
{"model": "vjepa2-ac-300m", "device": "AMD Instinct MI300X", "cem": {"samples": 64, "elites": 8, "iters": 3}, "results": {"4": {"sequential_s": 68.442, "batched_s": 32.066, "speedup": 2.1, "energy_sequential": 106.0, "energy_batched": 104.0}, "8": {"sequential_s": 129.677, "batched_s": 60.846, "speedup": 2.1, "energy_sequential": 146.0, "energy_batched": 140.0}}, "peak_hbm_gib": 43.9}
```
