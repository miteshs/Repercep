# Mirage Methodology — measurement protocol and honest framing

This document is meant for a sceptical reader. The headline numbers we
quote (the **2.52× faster than NVIDIA's H100 Cosmos reference** in
particular) deserve scrutiny: there is a legitimate apples-to-oranges
hazard in comparing our optimized path to a reference that doesn't have
the same optimizations applied. We spell it out here so the comparison
can be re-litigated by anyone who picks the project up.

---

## 1. The H100 reference

The number **"~380 s"** that we cite for Cosmos-Predict1-7B-Text2World on
a single H100 GPU comes from NVIDIA directly:

| Source | Quote |
|---|---|
| [huggingface.co/nvidia/Cosmos-Predict1-7B-Text2World](https://huggingface.co/nvidia/Cosmos-Predict1-7B-Text2World) | *"The table below presents the end-to-end inference runtime on a single H100 GPU, excluding model initialization time."* — **~380 seconds**, **peak GPU memory 74.0 GB**, "offload prompt upsampler" mode. |

What NVIDIA does **not** disclose on that page:

- Exact denoising step count (the Cosmos-Predict1 codebase defaults to
  35–36 steps; we use 36, matching the diffusers pipeline default).
- Exact frame count + resolution (we use the model's documented default:
  121 frames @ 1280×704, BF16).
- Exact prompt (we use a single deterministic prompt with seed 0;
  documented below).
- Whether the H100 timing is one run or a mean of many; whether warmup
  passes are excluded; whether the autotuning storms NVIDIA's stack
  invariably hits on first-shape execution are included or excluded.

These omissions are not deal-breakers (Cosmos's default config is well-
known and the "default" interpretation is standard practice), but they
mean the H100 reference is a single published wall-time, not a
distribution we can compare ours against statistically.

The NVIDIA stack the reference implies (per Cosmos-Predict1's
`INSTALL.md`, verified 2026-05-23):

- `transformer-engine[pytorch]==1.12.0` (Hopper-specific BF16 mixed
  precision + FP8 paths)
- `apex` (NVIDIA's mixed-precision + fused optimizers)
- The research brief that informed `docs/COSMOS_ON_MI300X.md`
  additionally cites NATTEN and flash-attn-3 as components; they are
  not in `INSTALL.md` itself but are referenced by the model code.

**None of these run on ROCm.** The dependency-stack divergence is the
entire reason Mirage exists on AMD silicon — see ADR-0001 + ADR-0002.

---

## 2. Mirage measurements

### 2.1 Hardware

| Item | Value |
|---|---|
| GPU | AMD Instinct MI300X **VF** (virtualized; 1 of N partitions) |
| HBM | 192 GiB HBM3 |
| Compute Units | 304 (gfx942 / CDNA3) |
| Host RAM | 235 GiB |
| Host CPUs | 20 |
| Disk | 697 GiB total, ~400 GiB free during measurements |

**Caveat:** This is a *Virtual Function* slice of an MI300X, not a bare
metal card. The VF exposes the full 192 GiB and 304 CUs per `rocm-smi`.
We have not been able to compare to a bare-metal MI300X; the VF *should*
have the same per-device performance for a single-tenant workload
(no SR-IOV partition mate), but this is unverified.

### 2.2 Software stack

| Component | Version |
|---|---|
| ROCm runtime | 7.2.0 |
| HIP | 7.2.53211 |
| Python | 3.12.3 |
| `torch` | `2.12.0+rocm7.2` (from `download.pytorch.org/whl/rocm7.2`) |
| `torchvision` | `0.27.0+rocm7.2` (same index) |
| `diffusers` | 0.37.1 (cherry-picked verified at 0.34.0 — both equivalent for this measurement; see Session 8 / F18 diagnosis trail) |
| `transformers` | 5.9.0 (also verified at 4.57.6) |
| `accelerate` | 1.13.0 |
| `huggingface-hub` | 1.16.1 |
| `triton-rocm` | 3.7.0 |
| Mirage runtime | this repo, `HEAD` at the time of the headline |

### 2.3 Workload

- Model: `nvidia/Cosmos-1.0-Diffusion-7B-Text2World` (the diffusers
  variant of Cosmos-Predict1).
- 121 frames at 1280×704, 36 denoising steps, EDM Euler scheduler,
  guidance scale 7.0, dtype BF16, seed 0.
- Prompt: *"A sleek autonomous delivery robot rolls along a sunlit city
  sidewalk past glass storefronts, smooth forward motion, photorealistic,
  high detail."* (the default in `scripts/run_cosmos.py`).
- Cosmos safety guardrail **disabled** for benchmarking — same as
  NVIDIA's published reference. Production deployments per NVIDIA's
  Open Model License must enable it; the integration is in
  `mirage.models.cosmos`.

### 2.4 Timing protocol

- **Each reported number is wall time from inside `scripts/run_cosmos.py`,
  reported as the `generate_seconds` field in the JSON RESULT line.**
- Model load time (`load_seconds` ≈ 10–18 s for Cosmos, 19 s for Wan)
  is **excluded** from `generate_seconds`, mirroring NVIDIA's
  "excluding model initialization time" disclosure.
- "Warmup-separated" means: a warmup forward at the same shapes preceded
  the timed run. ROCm's kernel autotuning (aotriton, hipBLASLt, MIOpen)
  fires on first-shape execution and adds ~273 s to a cold run; the
  warmup-separated number measures the steady state after that.
- Headline measurements are single-run, single-prompt, single-seed
  unless explicitly N > 1.

### 2.5 What we have measured

The current headline (HEAD = `7cb23f0`+) on a quiet GPU, no other
processes contending:

| Config | `generate_seconds` | Peak HBM | Source MD5 |
|---|--:|--:|---|
| Cosmos 121f/36 `cache=adaptive thr=0.30` | **150.9 s** | **52.5 GiB** | `94852d9d` (Session 9 + 10 identical) |
| Cosmos 121f/36 `cache=fixed/4` (legacy) | 163.9 s | 52.5 GiB | `d7bf6972` |
| Cosmos 121f/36 adaptive + `MIRAGE_FP8_ATTENTION=1` | 154.7 s | 52.5 GiB | `8f88f5f8` |
| Cosmos 121f/36 no caching, native loop | 465 s baseline / 740 s cold | 52.5 GiB | (not archived) |
| Wan-2.2-T2V-A14B 17f/8 steps smoke | 326.2 s | 84.3 GiB | `63c8937a` |

Frame integrity was independently verified with `ffprobe`:

- All Cosmos artifacts are **real 121-frame 1280×704 24 FPS h264** mp4s.
- The Wan smoke artifact is a **real 17-frame 1280×720 16 FPS h264** mp4.

The MD5 column above is a verification artifact too: seed-0 determinism
**holds across sessions** — the same `(prompt, seed, config)`
deterministically produces the same mp4 bytes on this hardware/stack.
The FP8 row's distinct MD5 confirms that FP8 routing actually changes
the inference path (Session 9's identical MD5 to baseline confirmed F19
— pre-wiring, the env var was a no-op).

---

## 3. The apples-to-apples accounting

This is the section a skeptical reader should read most carefully.

**The 2.52× claim compares a cached path on MI300X to an uncached path
on H100.** That is not a hardware comparison. It's a system comparison
between two distinct configurations:

| | MI300X + Mirage adaptive cache | H100 + NVIDIA reference stack |
|---|---|---|
| Hardware | MI300X (192 GiB, ROCm 7.2) | H100 80 GB (CUDA, TransformerEngine 1.12) |
| Compute saved by caching | yes (~25 of 36 DiT forwards skipped) | no (full 36 forwards) |
| Generate-time | **150.9 s** | **~380 s** |
| Peak HBM | 52.5 GiB | 74 GB |

Two honest framings:

**Hardware-only (raw, same workload):**
- Mirage on MI300X with **no caching** (i.e. the same compute as the
  H100 reference): **465 s baseline** measured.
- H100 reference: ~380 s.
- **MI300X is 1.22× *slower* than H100 at the same compute.** This is
  the like-for-like hardware comparison. The MI300X has more peak math
  throughput on paper; the gap is real and is dominated by:
  - aotriton flash kernels not (yet) matching CUDA flash-attn-3's
    Hopper-specific micro-optimizations
  - hipBLASLt vs cuBLAS on the GEMM-bound paths
  - TransformerEngine's BF16/FP8 microkernels having no clean ROCm
    equivalent

**System-level (each side's best-published path):**
- Mirage on MI300X with adaptive caching: **150.9 s**.
- H100 with NVIDIA's published reference (no caching disclosed):
  ~380 s.
- **Mirage's MI300X system is 2.52× faster than NVIDIA's published
  H100 system** — true. With these caveats:
  - This is the published-vs-published comparison; we can't run TeaCache
    on H100 because we don't have one. The H100 + TeaCache combination
    is presumably also achievable and would close the gap. **We are
    not claiming MI300X is intrinsically 2.52× faster than H100.**
  - We are claiming that Mirage on AMD silicon, using Phase-2 work
    that is currently absent from NVIDIA's documented Cosmos serving
    path, delivers a 2.52× headline against that documented path. That
    is the legitimate framing.

### Why this framing is still meaningful

vLLM-style serving infrastructure compares to "what people actually
run" rather than "what an unrealized optimal would be." NVIDIA has not
shipped a TeaCache-style adaptive cache in the public Cosmos path on
H100; the AMD ROCm + diffusers + native-loop combination Mirage provides
is the only documented path with that cache at all. The fair
comparison reflects shipped-system vs shipped-system, not
hypothetical-vs-hypothetical.

Where this framing becomes misleading is if a reader thinks "MI300X is
2.52× the hardware of H100." That is wrong by our own measurement (the
465 vs 380 baseline shows MI300X is 1.22× slower on raw compute). We
should never let that misreading stand uncorrected.

---

## 4. Reproducibility envelope

| Property | Measured | Confidence |
|---|---|---|
| Seed-0 determinism (same config → same mp4 bytes) | ✓ across 3 sessions, MD5 match | high |
| Single-run wall-time variance | ±0.5 % across 3 fresh sessions (150.9 / 151.1 / 151.2) | high |
| Variance across prompts/seeds | not yet measured (open) | low |
| LPIPS / FVD between cached and uncached output | not yet measured (open) | low |
| Per-stage profile (DiT-dominance) | matches BUILD_LOG F4 (~99 % DiT) | medium — re-validate |

What we have measured supports the claim that **the 142–151 s numbers
are stable and reproducible** on this hardware/stack. The independent
verification on a clean GPU (Session 11+) measured:

  | Config                    | Run-1 wall | Run-2 wall (clean) | Δ |
  | ---                       | ---:       | ---:               | ---: |
  | adaptive baseline         | 150.9 s    | 151.4 s            | +0.5 s |
  | adaptive + tuned-FP8      | 141.7 s    | 142.0 s            | +0.3 s |
  | no-cache reference        | 465 s (Session 7) | 470.0 s (Session 11) | +5 s |

What we found in the cache-quality verification (this section is new
as of 2026-05-23 Session 11; previous wording was too soft):

- **Adaptive caching produces output that is substantially different
  from the no-cache reference at the same prompt+seed.** LPIPS = **0.645**
  ("substantially different") between `cosmos_no_cache_clean.mp4` and
  `cosmos_adaptive_clean.mp4`. PSNR 13.64 dB.
- **Inter-frame motion is reduced by ~28–30 %** under adaptive caching:
  no-cache mean |Δframe| = 6.48; adaptive = 4.64; adaptive+FP8 = 4.52.
  Brightness and per-frame intensity variance are preserved (107 vs 107,
  std 64 vs 62).
- The cached videos **are visually coherent Cosmos generations** — same
  prompt-scene-content, well-formed h264 mp4s, 121 frames each. They
  are not garbage. They are different valid generations.
- Earlier text in `docs/BUILD_LOG.md` Session 9 said "motion stat 4.65
  vs 4.66 matches the verified `skip=4` reference." That comparison was
  **between two cached outputs**, not against the no-cache truth.
  The cache trades trajectory equivalence for compute; subsequent
  pixel-level metrics should not be expected to match the uncached
  reference.

What this means for the headline:
- The **timing claims** (142 s tuned-FP8, 151 s adaptive, 2.68× / 2.52×
  vs the H100 reference) are independently verified and reproducible.
- The implicit **"same quality as no-cache"** claim is NOT supported by
  pixel-level metrics. The cached path is a quality/speed trade-off
  with a knob (`--cache-adaptive-threshold`): lower threshold = fewer
  skips = closer to no-cache, slower. The default 0.30 is the speed
  bias; serving users wanting closer-to-uncached fidelity should run
  at ~0.05–0.10 and re-measure.

Other items still open:
- **Variance across prompts.** Single (prompt, seed) so far; the
  reproducibility verified within that point.
- **FVD against held-out references.** LPIPS measures pixel-trajectory
  divergence, which is too strict for diffusion outputs that vary by
  trajectory while preserving distribution-level quality. FVD on a held-
  out Cosmos eval set would be the right "no-cache vs cache quality"
  arbiter.
- **Cold vs warm distribution.** Cold first-shape execution adds ~270 s
  for ROCm autotuning; warmup-separated steady-state numbers are what
  we report. The cold number isn't the published headline, but a
  serving-system user feels it on the first request.

---

## 4a. Held-out reference set for FVD

This section documents the workflow that closes the §4 open item
**"FVD against held-out references."** It also pins the directory layout
the eval scripts expect so that "I ran the eval" reproduces what we did.

### Why a held-out set, and why N ≥ 50

Pixel-LPIPS on a single `(prompt, seed)` pair tells you whether two
videos look like the same trajectory; it does **not** tell you whether
a cache preserves distribution-level fidelity. Diffusion outputs trade
trajectory equivalence for compute by design — see §4's note on the
LPIPS = 0.645 result. The right arbiter is FVD over a sample of outputs.

The classic FVD paper uses N ≥ 1000. We won't get there in a
single-GPU regime, but anything below N ≈ 50 makes the empirical
covariance over 2048-D I3D features rank-deficient and the
`scipy.linalg.sqrtm` term numerically fragile. `scripts/compute_fvd.py`
already documents this in its module docstring and emits a loud
small-N warning below N = 50 (and a `feature_l2` fallback below N = 2).
See also `docs/SESSION_17_CLOSE.md` §"What's open after today" item 3
for the framing: FVD at N ≥ 50 is the gate we want crossed before any
external publication of the speed-vs-quality story.

### Directory layout

Both the reference set and the candidate set live in flat directories,
one mp4 per `(prompt, seed)` pair. The two directories MUST share the
same set of `(prompt, seed)` pairs so the FVD compares like-for-like
generations — only the cache strategy differs:

```
held_out_refs/
  prompt000_seed000.mp4    # no-cache baseline
  prompt000_seed042.mp4
  prompt001_seed000.mp4
  ...
  prompt049_seed042.mp4    # >= 50 pairs total

adaptive_cache_outputs/
  prompt000_seed000.mp4    # same prompts + seeds, adaptive cache on
  prompt000_seed042.mp4
  ...
```

The filename convention is informational only — `scripts/compute_fvd.py`
sorts by name and treats each set as an unordered bag of clips. Keep
the naming parallel so it's diff-able by eye.

### Single-pair pixel LPIPS vs distribution-level FVD

The two metrics answer different questions:

| Metric | Script | What it measures | When to trust it |
|---|---|---|---|
| MSE / PSNR / LPIPS | `scripts/verify_quality.py` | per-frame pixel + perceptual distance between two specific mp4s | when you want to know whether `(prompt, seed)` deterministically reproduces; when comparing two byte-determined trajectories |
| FVD | `scripts/compute_fvd.py` | Fréchet distance between the I3D feature distributions of two video sets | when comparing two **generative strategies** (cache on vs off, FP8 vs BF16) over a held-out sample |

A cached path that scores LPIPS = 0.6 on a single pair can still score
low FVD if the cache preserves the output distribution; the pixel
metric is the tighter test and is over-strict for the "did caching
break the model?" question.

### How `scripts/eval_cpu_quality.py` chains them

`scripts/eval_cpu_quality.py` is the CPU-focused convenience runner
that wraps both. It forces `--device cpu` on both inner stages (this is
explicitly the CPU eval flow — the evaluator itself never uses GPU,
regardless of the host) and emits one combined `RESULT` JSON line:

```bash
# Pixel-only (just verify_quality, CPU LPIPS):
python scripts/eval_cpu_quality.py REF.mp4 CAND.mp4

# Pixel + distribution FVD (the full workflow):
python scripts/eval_cpu_quality.py REF.mp4 CAND.mp4 \
    --fvd-reference-set held_out_refs/ \
    --fvd-candidate-set adaptive_cache_outputs/
```

The combined JSON layout (`mode = "cpu_quality_eval"`) carries a
`pixel` block (MSE/PSNR/LPIPS means + frame count + backbone) and a
nullable `fvd` block (the raw `compute_fvd()` return shape — `mode`
will be `"fvd"` at N ≥ 2 and `"feature_l2"` at N = 1). The loud small-N
FVD warning from `compute_fvd` is forwarded verbatim to the user's
stderr — do not silence it.

### The held-out set itself is NOT in the repo

The reference videos are **not committed** to this repository. They
must be generated locally: pick ≥ 50 prompts (the project ships
`scripts/prompts/` candidates; pick a diverse slice), run
`scripts/run_cosmos.py` (or `run_wan.py`) with `--cache none` and
multiple seeds, and save the outputs into one flat directory. That
directory is then **re-used** across cache-strategy comparisons —
generate the candidate set with the strategy under test against the
exact same `(prompt, seed)` pairs, point `eval_cpu_quality.py` at both
directories, and compare strategies by their FVD against the shared
held-out reference. The cost (≥ 50 × no-cache runs) is paid once; every
subsequent cache-strategy comparison is cheap.

---

## 5. The "first publicly reported" claim — what we checked

`docs/COSMOS_ON_MI300X.md` claims "first publicly reported Cosmos
benchmark on any AMD GPU." Verifying the negative:

| Source | Hits for Cosmos-on-AMD/MI300X/Instinct |
|---|---|
| `nvidia-cosmos/cosmos-predict1` GitHub issues | 0 |
| `nvidia-cosmos/cosmos-predict2` GitHub issues | 0 |
| `huggingface/diffusers` issues, "cosmos rocm" | 0 |
| AMD's ROCm blog (HunyuanWorld, Wan-2.2, Micro-World published) | 0 for Cosmos |
| AMD xDiT supported model list | Cosmos absent |
| MLPerf v6.0 video-generation submissions | None for Cosmos on AMD |

We have not done an exhaustive search of cloud-provider blogs and
research papers; "first publicly reported" is a claim about the state of
public disclosure on 2026-05-22 when the result was first prepared, not
a global novelty claim. If a published prior result surfaces, the
correct response is to update the doc, not to argue.

For Wan-2.2-T2V-A14B on MI300X, the search is similar — see
`docs/WAN_ON_MI300X.md` (Agent J writeup, in progress) for the catalog.
AMD has published Wan-2.2-T2V on the newer MI355X but not MI300X.

---

## 6. What would harden the claims further

In priority order:

1. **N=10 timing distribution** at the headline config — mean + std.
2. **Multi-prompt variance** — same config across 5 prompts at seed 0,
   then 5 prompts at seed 42, then 5 prompts at seed 100. 15 runs ≈ 40
   GPU-minutes.
3. **Quantitative quality metrics** — LPIPS (and ideally FVD) between
   adaptive-cache and no-cache outputs at the same `(prompt, seed)`.
   Requires `pip install lpips` and a short eval script.
4. **Multi-config curve** — wall time at (49f/12), (49f/24), (81f/24),
   (121f/24), (121f/36) with and without caching. Fits a model.
5. **External replication** — anyone with an MI300X reproducing the
   number from `make install` to `scripts/run_cosmos.py` end-to-end.
   This is the strongest possible verification; we haven't yet
   distributed widely.
6. **Bare-metal MI300X (not VF)** comparison if/when access is
   available. Should be near-identical to VF for single-tenant
   workloads.

This document is the gate against "trust me, the numbers are real" —
the gate is "here's what we measured, here's how, and here's where
the uncertainty is." That distinction matters more than the headline.
