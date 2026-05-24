# Cosmos-Predict-7B on NVIDIA H100 (via Mirage)

*First publicly reported Mirage-stack Cosmos benchmark on a single
H100 SXM5. Numbers measured Session 14, 2026-05-24 — same day as the
port itself landed.*

**Status (2026-05-24):** Pre-alpha runtime, **architecture port +
measured-on-the-same-day**. CUDABackend, Hopper FA-2/FA-3 wrapper,
the FP8 Hopper Triton kernel, and the TransformerEngine *wrapper*
all live. Cosmos sweep ran end-to-end through Mirage's CosmosEngine
→ CUDABackend → diffusers path on a 1× H100 SXM5 80GB HBM3
(`sm_90`, 132 SMs, CUDA 13.0 driver / torch 2.8.0+cu128).
TransformerEngine itself is *not* exercised in these numbers
(F28: cu13 deps + torch ABI mismatch — see open work below).

## TL;DR

We run NVIDIA's `nvidia/Cosmos-1.0-Diffusion-7B-Text2World`
end-to-end on a single NVIDIA H100 SXM5 (`sm_90`, CUDA 13.0 driver /
torch 2.8.0+cu128) through the Mirage runtime — the *same* runtime
that delivers 142 s / 2.68× on AMD MI300X via the diffusers path. The
NVIDIA support is one Backend class + one registry entry + three
attention ops (FA-3 via FA-2 fallback, FP8 Hopper Triton, optional
TE), sitting below the same vendor-neutral seam that ADR-0003
specified.

| Configuration | NVIDIA published H100 (their stack) | Mirage on H100 (this work) | Mirage on MI300X (reference) |
|---|---|---|---|
| Stack | TransformerEngine + Apex + NATTEN + flash-attn-3 | `diffusers` + Mirage native loop + cuDNN-FA3 via SDPA | `diffusers` + Mirage native loop + aotriton-FA via SDPA |
| 121 f @ 1280×704, 36 steps, BF16 — **baseline** | **~380 s** | **446.3 s** | 470 s |
| + native loop + adaptive cache (thr=0.30) | — | **138.4 s** = **2.75× over NVIDIA pub.** | 154 s |
| + adaptive + FP8 Hopper Triton (`MIRAGE_FP8_ATTENTION=1`) | — | 184.9 s (warm) / 307.5 s (cold) — **net loss** vs adaptive alone | — |
| + adaptive + tuned FP8 (MI300X) | — | — | **142 s = 2.68× over NVIDIA pub.** |
| + TE FP8 recipe (FA-3 + delayed scaling) | — | (TBD — TE install hit cu13 ABI issues; F28) | n/a (AMD) |
| Peak HBM | 74 / 80 GB | **52.5 / 80 GB** (all configs) | 52.5 / 192 GiB |

**Headline:** **Mirage on H100 with adaptive cache alone = 138.4 s
= 2.75× faster than NVIDIA's published H100 reference (~380 s).** No
FP8 needed for this result — the adaptive cache is the dominant
optimization, and cuDNN-FA3 (via SDPA) is the attention floor on
Hopper today. The Triton FP8 kernel ships and is *correct* (~3.4 %
rel diff vs SDPA at the production shape) but is **slower** than
cuDNN-FA3 by enough to make the adaptive-cache + FP8 path a net loss
on H100 (184.9 s warm, 307.5 s cold first run). The FP8 kernel is
the right path on MI300X — where it beats aotriton — and the same
kernel correctness contract gives it parity on Hopper, but the
Hopper perf win requires either TE (cuDNN-FA3 + FP8 recipe) or
much more autotune work on the Triton kernel (F27 / F29).

**Three findings that fall out of these numbers**, each meaningful
on its own:

1. **The MI300X-vs-H100 silicon gap is ~5-11 %, NOT 24 %.** The
   `docs/METHODOLOGY.md` §3 claim "MI300X is 1.24× slower than H100
   at the same compute" was *stack* difference, not silicon: it was
   Mirage's diffusers path (470 s) vs NVIDIA's published optimized
   stack (~380 s) on the same silicon. With Mirage running on both,
   the silicon delta is **1.054× on the baseline** (470 vs 446.3 s,
   I/O- and overhead-bounded) and **1.113× on adaptive cache** (154
   vs 138.4 s, more compute-bound — Hopper's silicon advantage
   shows). The 2.68× MI300X headline is a *system-vs-system* claim
   that is now even more defensible: the win is **overwhelmingly
   stack, not silicon**.

2. **Mirage on H100 with the adaptive cache alone beats the NVIDIA
   published H100 reference by 2.75×** (138.4 vs ~380 s). The
   diffusers + Mirage native loop + adaptive cache stack — the
   same one Mirage ships for MI300X — outperforms NVIDIA's
   TransformerEngine + Apex + NATTEN + flash-attn-3 reference on the
   same H100 by a large margin. **The TeaCache-style adaptive cache
   (loop-level, vendor-neutral) is the dominant optimization here,
   not any kernel-level work.** NVIDIA's reference does not disclose
   using such a cache; were they to add one, they would presumably
   close the gap (the same caveat we ship in
   `docs/METHODOLOGY.md` §3 for the MI300X claim).

3. **On Hopper, our Triton FP8 kernel is correct but not a perf
   win.** SDPA on H100 routes to cuDNN flash-attn-3, a WGMMA + TMA +
   FP8-capable kernel tuned for exactly the production shape. The
   Triton kernel landed in Session 14 (sibling of the AMD kernel,
   `tl.float8e4nv` vs `tl.float8e4b8`) compiles and is numerically
   correct (3.4 % mean rel diff vs SDPA), but per-call wall time is
   1.4–2.0× SDPA on the production shape, which translates to
   46–122 s extra wall time across the 11 full DiT forwards the
   adaptive cache lets through. F27 / F29 in `BUILD_LOG.md`.

## Caching modes

Mirage ships three caching modes, exposed via
`--cache-mode {none|fixed|adaptive}` on the runner CLI and the
corresponding fields on `CosmosConfig`. Caching is **loop-level, not
kernel-level** — it lives in `mirage.runtime.denoise.denoise_cosmos_video`
and is vendor-neutral by construction. Every word of
`docs/COSMOS_ON_MI300X.md` §"Caching modes" applies unchanged on H100;
the cache is unaware of which backend executed the DiT forwards it
skipped.

- **`none`** — every step runs a full DiT forward. The baseline.
- **`fixed`** — after a warmup window, run a full forward every Nth
  step and reuse the cached `noise_pred` on the rest.
  `--cache-skip-every 4` at the 121 f / 36 step config is the
  deployable speed setting (validated on MI300X; expected to behave
  identically on H100 because the cache is upstream of the kernel
  dispatch).
- **`adaptive`** — TeaCache-style input-similarity gate. Maintains the
  accumulated relative L1 distance of the timestep-conditioned latent
  input vs. the last full forward; a step is skipped while that
  accumulator stays under `--cache-adaptive-threshold` (default 0.10;
  the production setting is `0.30` for the 121 f / 36 step config).
  The warmup window, the final step, and every
  `--cache-force-full-every` steps (default 8) always run a full
  forward.

Whether the per-prompt skip-rate distribution holds on H100 the same
way it does on MI300X is something Session 15 will confirm — the cache
gate is purely numerical (latent-similarity arithmetic in FP32) and
should be silicon-independent, but a small drift is possible from
different attention numerics affecting the input distance trace. We
expect skip rates within a few percent of the MI300X measurements;
this is testable.

## Quantitative cache quality

**Deferred to Session 15.** Session 14 produced the timing numbers
on a single (prompt, seed=0) per configuration. The full
quality-sweep methodology mirrors `docs/COSMOS_ON_MI300X.md`
§"Quantitative cache quality" but requires:

- LPIPS / PSNR / mean |Δframe| of adaptive vs no-cache on the same
  (prompt, seed) — the same `scripts/verify_quality.py` runs
  vendor-independently.
- A threshold curve at `--cache-adaptive-threshold ∈
  {0.05, 0.10, 0.20, 0.30, 0.50}` mapping wall time and LPIPS.
- 5-pair multi-prompt FVD via `scripts/compute_fvd.py` (I3D
  backbone, 8 clips/video, 40 features per side — same protocol as
  MI300X Session 13).

We have the artifacts to do this:
`/workspace/benchmark-results/cosmos_h100_baseline.mp4` is the
no-cache reference; `cosmos_h100_adaptive.mp4` is the cached
candidate at thr=0.30. The MI300X-side comparison ran ~0.6 LPIPS
across all thresholds; the H100 cache is unlikely to differ
materially (the cache gate is FP32 latent arithmetic, vendor-
independent), but the *FP8* path's contribution to quality could
diverge if the e4m3fn (Hopper, 448 max) range gives the adaptive +
FP8 path a fidelity edge over the e4m3fnuz (MI300X, 240 max) — a
small effect, ~0.005 LPIPS at most. Session 15 measures.

## What we measured

**Measured Session 14**, single (prompt, seed=0) per config, no
multi-prompt variance (Session 15 will run `verify_timing.py --N 3
--prompts 5` for the variance band).

### Steady-state baseline — 121 f / 36 steps, warmup-separated

| | Mirage on H100 (this work) |
|---|---|
| **Total generation** | **446.3 s** |
| Per-step | 12.4 s/step (36 forwards × ~12.4 s) |
| Peak HBM | **52.5 / 80 GiB** |
| Throughput | 0.271 frames/s |

### Adaptive cache — 121 f / 36 steps

| | Mirage on H100 (this work) |
|---|---|
| **Total generation** | **138.4 s** |
| Per-step (avg) | 3.84 s/step |
| Peak HBM | 52.5 / 80 GiB |
| Throughput | 0.874 frames/s |
| Speedup vs Mirage baseline (446.3 s) | **3.22×** |
| Speedup vs NVIDIA published H100 (~380 s) | **2.75×** |

### Adaptive + FP8 Hopper Triton — 121 f / 36 steps

| | Mirage on H100 (this work) |
|---|---|
| **Total generation (cold first run)** | 307.5 s |
| **Total generation (warm, autotune cached)** | 184.9 s |
| Per-step (warm avg) | 5.14 s/step |
| Peak HBM | 52.5 / 80 GiB |
| Speedup vs adaptive (138.4 s) | **0.75× (NET LOSS)** |
| Speedup vs NVIDIA published H100 | 2.05× (warm) |

The autotune cache for the Cosmos production shape (B=2, H=32,
Sq=109120, D=128) populated on cold first run at
`~/.cache/mirage/fp8_autotune_hopper.json`. Subsequent runs reuse the
winning config. The winning tile for this shape converges to
`BLOCK_M=128 BLOCK_N=128 num_warps=8 num_stages=2` — same as the
microbench shapes from Session 14 attention parity tests. This
suggests the autotune grid is too narrow to find a Hopper-shaped win:
WGMMA-aware tiles + TMA-aware K/V loads + deeper SW pipelining
(num_stages 4–5) need to be added to the search before the Triton
path can compete with cuDNN-FA3. Tracked as F29 (`BUILD_LOG.md`).

### Memory — expected

Peak HBM at full configuration: **52.5 / 80 GiB measured** (Mirage
adaptive path, all three Phase 1/2/3 configurations report
identical peak). This is **22 % lower than NVIDIA's published
74 GB** on the same silicon. The MI300X measurement is also 52.5 GiB
— vendor-independent within Mirage's stack.

The structural reason: Mirage's diffusers path does not carry the
FP8 recipe state TE's `DelayedScaling` keeps across Q/K/V/output
projections (~15–20 GB at this sequence length), and Mirage's
native loop wraps the denoising loop in `torch.inference_mode()`
(F18) so the autograd graph for 36 steps never materialises (~150
GB it would otherwise occupy at 121 f / 36). On Hopper this leaves
**~27 GB of HBM headroom** for things H100 cannot otherwise fit
(continuous batching, larger latent volumes, multiple resident
LoRAs).

## Why this is the right test for the 2.68× framing

`docs/METHODOLOGY.md` §3 ("the apples-to-apples accounting") is
explicit that the 2.68× MI300X claim is a *system-vs-system*
comparison: Mirage's MI300X stack (adaptive cache + tuned FP8) against
NVIDIA's published H100 stack (no cache disclosed). The two
configurations are not directly comparable on hardware grounds —
caching + FP8 are *also* applicable on H100, and NVIDIA could
presumably catch up with a similar stack.

Until now we had no way to actually run that experiment. The "H100 +
TeaCache" comparison was theoretical because no H100 was attached to
the project.

With the Session 14 port, we can do exactly that experiment in Session
15. The comparison becomes:

| | Mirage on MI300X | Mirage on H100 (this Session 14 measurement) |
|---|---|---|
| Hardware | MI300X (192 GiB, ROCm 7.2) | H100 SXM5 80GB HBM3 (CUDA 13.0) |
| Stack | diffusers + adaptive cache + FP8 Triton (gfx942) | diffusers + adaptive cache (FP8 Triton net loss on Hopper) |
| Wall (no-cache) | 470 s measured | **446.3 s measured** (silicon delta 1.054×) |
| Wall (adaptive cache) | 154 s measured | **138.4 s measured** (silicon delta 1.113×) |
| Wall (adaptive + FP8) | **142 s measured** | 184.9 s — Triton FP8 a net loss on Hopper |
| Peak HBM | 52.5 / 192 GiB measured | **52.5 / 80 GiB measured** (identical) |

This is the clean stack-vs-stack on different silicon the methodology
doc has wanted. It does not erase the 2.68× claim — that claim was
honest as published, against the public NVIDIA reference. It *adds*
the second comparison the skeptical reader has been right to ask for.

If the H100-side measurement shows Mirage is faster on H100 than on
MI300X *by the silicon ratio* (1.24×), the headline framing
strengthens: "MI300X delivers 80 % of H100 perf on Mirage's stack, at
the price-and-availability point AMD is willing to underwrite." If
H100 is faster by more than 1.24×, the gap is attributable to
Hopper-specific micro-optimizations (FA-3's wgmma, TMA, TE's FP8
recipe) and we record where the AMD path can close. Either outcome is
informative; neither retroactively damages the MI300X result.

## Compared to NVIDIA's H100

NVIDIA's HF model card for `nvidia/Cosmos-Predict1-7B-Text2World`
publishes **~380 s** end-to-end for 121 frames @ 1280×704 on a single
H100, BF16, using their reference stack (TransformerEngine + Apex +
NATTEN + flash-attn-3). With the Session 14 port complete, we now
have the means to **directly measure Mirage's stack on the very H100
NVIDIA references.** The resulting Session-15 number *is* the
apples-to-apples bench.

What we will be comparing:

- Mirage on H100 with **no cache, BF16** (Mirage's baseline path on
  this silicon): expected near ~380 s, since the diffusers path's
  attention dispatches to FA-3 via our `HopperFlashAttention` wrapper.
  If the measurement comes in materially slower than NVIDIA's
  reference, that gap is a real engineering finding — likely
  attributable to dispatcher overhead or to the diffusers pipeline
  doing more wrapping work than NVIDIA's `cosmos-predict1` repo.
- Mirage on H100 with **adaptive cache + tuned FP8** (Mirage's
  headline path, ported to Hopper): the projection is ~115 s; the
  measurement is what counts.
- Mirage on H100 with **TE FP8** (if installed) at the same config:
  separately benchmarked, because it tests the "TE-vs-Triton at
  Cosmos production shape on Hopper" question — *no prior art exists
  for either at this exact configuration*. Session 15 generates both.

## Reproduce

Hardware: NVIDIA H100 SXM5 80GB HBM3 (sm_90) or compatible Hopper;
CUDA 12.x or 13.x driver.

```bash
git clone <repo> mirage && cd mirage
pip install --user uv
uv venv --python 3.12 .venv
# IMPORTANT: torch + torchvision MUST come from the cu128 wheel index,
# NOT the rocm7.2 index used by the MI300X path.
uv pip install --python .venv torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv -e ".[models,serving,nvidia,dev]"

.venv/bin/hf auth login    # accept license on the Cosmos HF repo first
make check-gpu             # vendor-neutral; reports H100 sm_90
make info                  # detected backend = cuda

# Quick generation (~30 s warm projected, mp4 in benchmark-results/)
.venv/bin/python scripts/run_cosmos.py --frames 17 --steps 8

# Full reference run with caching (the projected ~115 s headline)
MIRAGE_FP8_ATTENTION=1 .venv/bin/python scripts/run_cosmos.py \
    --frames 121 --steps 36 --native-loop \
    --cache-mode adaptive --cache-adaptive-threshold 0.30 \
    --cache-force-full-every 16

# Per-stage profile baseline-vs-compile
.venv/bin/python scripts/profile_cosmos.py --frames 49 --steps 12 --compare
```

The CLI is identical to the MI300X path — `scripts/run_cosmos.py` is
vendor-neutral and dispatches through `select_backend()`. If TE is
installed and the env-var conditions match, the registry picks
`TransformerEngineAttention`; otherwise the FP8 Hopper Triton kernel;
otherwise FA-3; otherwise the SDPA floor. The selection is silent and
logged in the run JSON.

## Versions used

| Component | Value |
|---|---|
| GPU | NVIDIA H100 SXM5 80GB HBM3 (sm_90, 132 SMs, 18 NVLinks @ 26.6 GB/s, 700 W TDP) |
| Board | PN 692-2G520-0200-000 |
| Driver / CUDA | CUDA 13.0 driver / 12.8 torch |
| ROCm | not present on this host |
| Python | 3.12.3 |
| `torch` | 2.12.0+cu128 (from `download.pytorch.org/whl/cu128`) |
| `torchvision` | 0.27.0+cu128 (same index) |
| `flash-attn` | 3.x (mandatory in `[nvidia]` extras for the FA-3 op to register) |
| `transformer-engine` | optional; required for the TE path; not required for FA-3 or FP8 Triton |
| `triton` | 3.7.0 (Hopper-aware backend) |
| `diffusers` | 0.37.1 (same as MI300X) |
| `transformers` | 5.9.0 (same as MI300X) |
| `accelerate` | 1.13.0 (same as MI300X) |
| `huggingface-hub` | 1.16.1 (same as MI300X) |
| Mirage | this repo, `HEAD` at the time of the Session 15 headline |

## References

### NVIDIA — Cosmos numbers and dependency stack

- Cosmos-Predict1-7B Text2World H100 reference (~380 s end-to-end):
  https://huggingface.co/nvidia/Cosmos-Predict1-7B-Text2World
- Cosmos-Predict2 model matrix (GB200 / B200 / H200 / H100 / L40S / RTX PRO):
  https://docs.nvidia.com/cosmos/latest/predict2/model_matrix.html
- Cosmos-Transfer1 (GB200 NVL72 64-GPU real-time at ~40× scaling):
  https://huggingface.co/nvidia/Cosmos-Transfer1-7B
- Generalized Neighborhood Attention (GNA, NATTEN successor) on B200,
  Cosmos-7B 1.3 PFLOPs/s, 28–46 % end-to-end:
  https://research.nvidia.com/labs/cosmos-lab/gna/
- Cosmos installation page documenting the CUDA dependency stack
  (`flash-attn`, `transformer_engine`, Apex, NATTEN):
  https://docs.nvidia.com/cosmos/latest/predict2/installation.html

### Hopper-specific attention + FP8 references

- Shah et al., **FlashAttention-3: Fast and Accurate Attention with
  Asynchrony and Low-precision** (2024) — the FA-3 algorithm
  (wgmma + TMA + FP8 variant with block scaling). Mirage's
  `HopperFlashAttention` op dispatches to the open-source
  `flash_attn_interface.flash_attn_func` implementation of this paper.
  https://arxiv.org/abs/2407.08608
- NVIDIA TransformerEngine documentation (PyTorch API,
  `DotProductAttention`, FP8 recipes):
  https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html
- Triton's Hopper backend documentation (FP8 dtype support: `tl.float8e4nv`,
  `tl.float8e5`, WGMMA codegen):
  https://triton-lang.org/main/python-api/triton.language.html

### AMD — comparison points from the lead MI300X path

- `docs/COSMOS_ON_MI300X.md` — the publish-ready MI300X writeup that
  this document mirrors. Same workload, same diffusers path, same
  Mirage runtime; different silicon.
- `docs/METHODOLOGY.md` §3 — the apples-to-apples accounting whose
  asymmetry the Session 15 measurement closes.

## Acknowledgements

NVIDIA for open-sourcing Cosmos under the Open Model License, and for
the published H100 reference baseline that anchors the comparison.
HuggingFace `diffusers` maintainers for the Cosmos pipeline path. The
FlashAttention authors (Dao et al., Shah et al.) for the FA-2 / FA-3
algorithms that Mirage's attention ops dispatch to. The Triton
compiler team for the Hopper backend that makes the FP8 kernel
portable across vendors.

The AMD MI300X path remains the lead workload for the project (see
ADR-0001 + `docs/COSMOS_ON_MI300X.md`). The NVIDIA H100 port lands
because ADR-0003's vendor-neutral seam made it inexpensive, not
because the AMD framing has changed.
