# Cosmos-Predict-7B on NVIDIA H100 (via Mirage)

*Port-ready writeup of NVIDIA Cosmos on a single H100 SXM5 through the
Mirage runtime. Numbers below marked **TBD (Session 15)** are pending
the H100 benchmark sweep; Session 14 landed the architecture, kernels,
and tooling.*

**Status (2026-05-24):** Pre-alpha runtime measurement, **architecture
port complete; no benchmark numbers measured yet on this H100.**
Session 14 stood up the NVIDIA backend, the Hopper FA-3 path, the FP8
Hopper Triton kernel, and the optional TransformerEngine op. The
full-config 121 f / 36 step sweep is the Session 15 deliverable. Until
those numbers land, every wall-time entry in this doc carries a
**TBD (Session 15)** marker and the speedup framing is *projected*,
not measured. We are publishing the framework now so the measurement
slots into it cleanly.

## TL;DR

We now run NVIDIA's `nvidia/Cosmos-1.0-Diffusion-7B-Text2World`
end-to-end on a single NVIDIA H100 SXM5 (`sm_90`, CUDA 13.0 driver /
torch 2.12+cu128) through the Mirage runtime — the *same* runtime that
delivers 142 s / 2.68× on AMD MI300X via the diffusers path. The
NVIDIA support is one Backend class + one registry entry + three
attention ops (FA-3, FP8 Hopper Triton, optional TE), sitting below
the same vendor-neutral seam that ADR-0003 specified.

| Configuration | NVIDIA H100 — NVIDIA's published reference | Mirage on NVIDIA H100 (this work) |
|---|---|---|
| Stack | TransformerEngine + Apex + NATTEN + flash-attn-3 | `diffusers` + Mirage native loop + FA-3 / FP8 Hopper Triton |
| 121 frames @ 1280×704, 36 steps, BF16 — **baseline** | **~380 s** (NVIDIA HF model card) | **TBD (Session 15)** |
| same + native loop + adaptive cache (thr=0.30) | — | **TBD (Session 15)** |
| same + FP8 Hopper Triton (`MIRAGE_FP8_ATTENTION=1`) | — | **TBD (Session 15)** |
| same + TE FP8 recipe (FA-3 + delayed scaling) | — | **TBD (Session 15)** |
| Peak HBM | 74 / 80 GB | **TBD (Session 15)** |

**What this writeup IS today:** a vendor-neutral benchmark *path* on
NVIDIA H100 via Mirage's diffusers-path runtime — the same adaptive-
cache + FP8-Triton stack that lands 142 s on MI300X, now compiled and
selected for Hopper. All measured numbers in the table above are
placeholders until the Session 15 benchmark sweep lands.

**Headline number stub.** If Mirage on H100 measures within the raw-
hardware comparison bounds the MI300X-side methodology already
established (`docs/METHODOLOGY.md` §3: MI300X is **1.24× *slower***
than H100 at the same compute), the H100 path should land around:

- **No-cache baseline:** ~470 s / 1.24 ≈ **~380 s** (matching NVIDIA's
  published reference, as expected — the diffusers path is not
  intrinsically slower than `cosmos-predict1`).
- **Adaptive cache + tuned FP8:** ~142 s / 1.24 ≈ **~115 s**.

**Until measured these are projections, not claims.** The point of
landing the port is to *replace* those projections with measurements
in Session 15.

**What changes when Session 15 lands the numbers:**

- The "system-vs-system" framing in `docs/METHODOLOGY.md` §3 becomes
  *stack-vs-stack on the same silicon*. The question "what if you ran
  Mirage's stack on H100?" has an answer.
- The 2.68× headline gets a sibling: "Mirage on MI300X is 2.68× faster
  than NVIDIA's published H100 reference; Mirage on H100 is K× faster
  than the same reference." Whatever K turns out to be is the
  apples-to-apples bench the project has been missing.
- The MI300X claim sharpens, not softens — "MI300X is a credible
  alternative to H100 for Cosmos serving" stops being a counterfactual
  and becomes a measurement.

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

**TBD (Session 15).** The methodology mirrors
`docs/COSMOS_ON_MI300X.md` §"Quantitative cache quality":

- LPIPS / PSNR / mean |Δframe| of adaptive vs no-cache on the same
  (prompt, seed) — the same `scripts/verify_quality.py` runs on
  whichever GPU is present.
- A threshold curve at `--cache-adaptive-threshold ∈
  {0.05, 0.10, 0.20, 0.30, 0.50}` mapping wall time and LPIPS.
- 5-pair multi-prompt FVD via `scripts/compute_fvd.py` (I3D backbone,
  8 clips/video, 40 features per side — same protocol as Session 13).

**Prior on what we expect:** the cache is a loop-level optimization
unaware of the underlying kernel; it should be trajectory-divergent on
H100 just as on MI300X, with comparable LPIPS magnitude (mean ~0.6
across thresholds). The FP8 path's quality contribution is different
across vendors though — H100's FP8 e4m3fn has finer-grained values
than gfx942's e4m3 fnuz (the IEEE-ish vs finite-only distinction in
ADR-0006); a small quality delta in favor of H100 is plausible.
Session 15 measures.

## What we measured

**TBD (Session 15).** Per-stage profile, peak HBM, cold-vs-warm gap,
and the steady-state baseline all run through the same harness used
on MI300X:

- `scripts/profile_cosmos.py --frames 49 --steps 12 --compare` for the
  per-stage profile + torch.compile A/B.
- `scripts/run_cosmos.py --frames 121 --steps 36 --native-loop ...`
  for the full-config measurement.
- `scripts/verify_timing.py --N 3 --prompts 5` for multi-prompt
  variance.

The placeholder per-stage rows the MI300X writeup carries
(DiT-loop dominance ~99 % at steady state, VAE decode <1 s, etc.) are
*expected* on H100 because Cosmos's compute envelope is structural,
not silicon-specific. Session 15 confirms.

### Memory — expected

Peak HBM at full configuration: **TBD (Session 15)** measured.
*Expected:* close to NVIDIA's published 74 GB on H100 (within the
80 GB envelope, no headroom), since the diffusers path's memory
footprint is dominated by activations + KV state that scale with
sequence length and not by per-kernel scratch. On MI300X this run
peaks at 52.5 / 192 GiB; the gap to H100's 74 GB is mostly the FP8
recipe state in TE's path (~20 GB across Q/K/V/output projection
amax buffers) and aotriton scratch on ROCm vs cuBLAS scratch on CUDA.

Mirage on H100 *without* the TE path (i.e. our Triton FP8 + FA-3
combination) should land closer to the MI300X 52.5 GB number than to
NVIDIA's 74 GB, because we are not paying the TE recipe-state cost.
This is one of the structural deltas Session 15 measures.

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

| | Mirage on MI300X | Mirage on H100 |
|---|---|---|
| Hardware | MI300X (192 GiB, ROCm 7.2) | H100 SXM5 80GB HBM3 (CUDA 13.0) |
| Stack | diffusers + adaptive cache + FP8 Triton (gfx942) | diffusers + adaptive cache + FP8 Hopper Triton or TE |
| Wall (no-cache) | 470 s measured | **TBD (Session 15)** |
| Wall (adaptive + FP8) | **142 s measured** | **TBD (Session 15)** |
| Peak HBM | 52.5 / 192 GiB measured | **TBD (Session 15)** |

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
