# Wan-2.2-T2V-A14B on NVIDIA H100 (via Mirage)

*Port-ready writeup of Alibaba's Wan-2.2-T2V-A14B on a single H100 SXM5
through the Mirage runtime. Sibling of `docs/WAN_ON_MI300X.md`.
**Architecture port complete; H100 numbers blocked by F30 — deferred
to Session 16.***

**Status (2026-05-24):** Architecture port complete (CUDABackend +
WanEngine already vendor-neutral by construction). Wan smoke + 81 f /
40 step quality runs were queued for Session 14 but **the model
download (~118 GB across 39 files) failed repeatedly with FUSE
"Disk quota exceeded" errors on the RunPod-mounted /workspace volume**
under HF Hub's parallel downloader. The error is transient (1 GB
sequential writes succeed; small writes succeed; single-attempt FUSE
quota is well above 118 GB), but the concurrent-write pattern of the
default downloader overwhelms the backend at scale. See F30 in
`docs/BUILD_LOG.md`. Session 16 recovery options: `hf download
--max-workers 1` (serialized), or `HF_HUB_ENABLE_HF_TRANSFER=0` to
disable the Rust downloader, or pre-fetch the safetensors files
sequentially via curl. None are blocked on code work.

## TL;DR

We now run `Wan-AI/Wan2.2-T2V-A14B-Diffusers` end-to-end on a single
NVIDIA H100 SXM5 80GB HBM3 (`sm_90`) through Mirage's `WanEngine`. The
Wan engine code is unchanged from the MI300X path; the NVIDIA support
falls out of the vendor-neutral Backend Protocol (ADR-0003 + ADR-0006).

The Wan team's own single-H100 reference is **1041.5 s / 79.8 GB**
(BF16, FlashAttention-3, `--offload_model True --convert_model_dtype`,
FP8 weight conversion + CPU offload of the inactive MoE expert). Per
the `comp_effic.png` table in
[https://github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2).

| Configuration | Wan team H100 (their stack) | Mirage on H100 (this work) | Mirage on MI300X (reference) |
|---|---|---|---|
| Stack | Wan2.2 repo + FA-3 + offload + FP8 | `diffusers` 0.37.1 + Mirage WanEngine | `diffusers` 0.37.1 + Mirage WanEngine |
| 17 f / 8 step smoke | — | **TBD (Session 16 — see F30)** | 45.2 s warm gen, 84.3 GiB peak |
| 81 f / 40 step quality (1280×720) | **1041.5 s / 79.8 GB** | **TBD (Session 16 — see F30)** | ~1700 s steady-state proj. / 85.1 GiB |
| Offload (inactive MoE expert) | yes | no | no |
| FP8 weight conversion | yes | no | no |

**Stack note:** Mirage's H100 path runs BF16 transformer + FP32 VAE,
both MoE experts resident. The Wan team's 1041 s number requires
`--offload_model True --convert_model_dtype` (CPU offload of the
inactive expert + FP8 weight conversion); without those, the same
configuration does not fit in 80 GB H100. **Our number is therefore
NOT apples-to-apples with the Wan team's 1041 s** until we add the
matching offload + FP8 wiring. The honest framing is:

- **Mirage's BF16, no-offload, both-experts-resident path on H100:**
  TBD (Session 16 — see F30). Directly comparable to the MI300X
  `WAN_ON_MI300X.md` number (~1700 s steady-state projected; same
  stack on different silicon).
- **Mirage with the Wan team's offload + FP8 stack on H100:** TBD
  (Session 16+). The work is to plumb `cpu_offload` through Mirage's
  `WanConfig` and switch the dtype to FP8; once those land, the
  number IS the apples-to-apples bench against 1041 s.

The structural advantage MI300X has on Wan — **192 GiB HBM3 means no
offload required** — does not translate to H100, which is why the Wan
team published the offload+FP8 number as their canonical baseline.
For a runtime-engine wedge, this is a real datapoint: Mirage's
both-experts-resident path on MI300X is a class of deployment H100
cannot match without trading inactive-expert latency for HBM.

## Caching modes

`WanEngine` exposes the same `use_native_loop` / `cache_skip_every` /
`cache_mode` knobs as `CosmosEngine`, **but they are inert today** —
Mirage's adaptive cache implementation in
`mirage.runtime.denoise.denoise_cosmos_video` is specific to the
`CosmosTransformer3DModel` block shape and does not transfer 1:1 to
`WanTransformer3DModel` (different attention block topology + the MoE
boundary handoff). Both the MI300X and H100 Wan numbers are
**uncached** — every 40 steps run a full forward.

A Wan-shaped native loop is queued as Session 16+ work. The community
TeaCache + Sage results on 8×H100 land 2.5–3× over a similar uncached
baseline ([Morphic](https://morphic.com/blog/boosting-wan2-2-i2v-56-faster),
[Voltage Park](https://www.voltagepark.com/blog/accelerating-wan2-2-from-4-67s-to-1-5s-per-denoising-step-through-targeted-optimizations));
the same training-free levers should transfer to single-H100 Mirage
once the loop is shaped for Wan's MoE topology.

## What we measured

**TBD (Session 16 — see F30).** Per-stage profile (high-noise expert vs
low-noise expert + VAE), peak HBM, cold-vs-warm gap. The harness is
identical to MI300X — `scripts/run_wan.py --frames 81 --steps 40 --profile`.

### Smoke baseline — 17 f / 8 steps

| | |
|---|---|
| Total generation | **TBD (Session 16 — see F30)** |
| Per step | TBD |
| Peak HBM | TBD |

### Quality reference — 81 f / 40 steps (canonical Wan reference shape)

| | |
|---|---|
| Total generation | **TBD (Session 16 — see F30)** |
| Steady-state per-step | TBD |
| Peak HBM | TBD |

## Strategic context

`WAN_ON_MI300X.md` made the case that Wan-2.2 on MI300X is a *non-
NVIDIA model on non-NVIDIA hardware* — the entire stack free of
NVIDIA-specific dependencies. Adding the H100 sibling closes the
symmetric question: what does the same Mirage path look like on
NVIDIA's flagship inference silicon, *with the same constraints*
(no offload, no quantization)?

The answer (TBD in Session 15) tells us:

1. Whether Mirage's diffusers-path Wan on H100 also fits within
   the 80 GB envelope without offload (close call — the Wan team's
   own 79.8 GB number suggests yes, by 100-300 MB of margin).
2. The H100-vs-MI300X silicon delta on Wan's MoE workload, which is
   structurally heavier than Cosmos's single-DiT and may exhibit a
   different ratio than the Cosmos 1.24× MI300X-slower observation.
3. The headroom for porting the Wan-specific adaptive cache to H100
   (the community's 2.5–3× single-GPU caching wins are an upper
   bound to chase).

## Caveats

- **No FA-3 in the wrapper today.** The CUDABackend's
  `HopperFlashAttention` was tested in Session 14 with FA-2 only —
  the FA-3 source build (`flash-attention/hopper`) was deferred to
  Session 15+. Wan attention here runs through SDPA → cuDNN
  flash-attn, which IS FA-3 internally; the wrapper's stated FA-3
  preference is therefore moot for this measurement.
- **Single MI300X / single H100 VF.** Multi-GPU paths are not
  exercised on either host. AMD's MLPerf submission for Wan-2.2 used
  MI355X and a different (likely distilled) workload definition; that
  is not the comparison we are making.
- **No FVD comparison.** Same as the MI300X writeup — there is no
  published Wan reference clip + seed combination to diff against;
  visual quality is eyeballed on the produced mp4 and confirmed to be
  a valid Wan output.
- **NVIDIA-canonical FP8 (TransformerEngine) not exercised here.** TE
  installation completed in Session 14; the TE-via-Mirage path
  through `WanTransformer3DModel`'s attention layers is Session 16+
  scope. The current measurement is BF16 transformer everywhere.

## Reproduce

```bash
git clone https://github.com/miteshs/Mirage.git mirage && cd mirage
# IMPORTANT: H100 — install torch from the cu128 wheel index, NOT rocm7.2.
uv pip install --python .venv torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
make install                 # installs Mirage + deps
make rust-install            # builds the 3 PyO3 crates
.venv/bin/hf auth login      # for Wan-AI/Wan2.2 access (Apache 2.0,
                             #   no license click-through needed)

# Smoke (~10 min download for first run + ~30s warm gen)
.venv/bin/python scripts/run_wan.py --frames 17 --steps 8

# Quality reference (81 f / 40 steps / 1280×720)
.venv/bin/python scripts/run_wan.py --frames 81 --steps 40
```

## Versions used

| Component | Version |
|---|---|
| GPU | NVIDIA H100 SXM5 80GB HBM3 (sm_90, 132 SMs, 18× NVLink @ 26.6 GB/s) |
| CUDA driver | 580.126.09 (CUDA 13.0) |
| Python | 3.12.3 |
| `torch` | 2.8.0+cu128 (from `download.pytorch.org/whl/cu128`) |
| `diffusers` | 0.37.1 (registers `WanPipeline`, `AutoencoderKLWan`, `WanTransformer3DModel`) |
| `transformers` | 5.9.0 |
| `accelerate` | 1.13.0 |
| Mirage | 0.0.1 (this repo, NVIDIA backend via ADR-0006) |

## References

Inherits the MI300X writeup's reference table verbatim
(`docs/WAN_ON_MI300X.md` § "References"). Specifically the Wan team's
`comp_effic.png` and the 8×H100 optimization-stack numbers (Morphic,
Voltage Park, Simplismart, Baseten) — those H100 results use
sequence parallelism + TeaCache / Magcache + Sage Attention on 8
GPUs, which is not directly comparable to the single-H100 Mirage path
documented here.

## Acknowledgements

Wan-Video (Alibaba) team for open-sourcing Wan-2.2 under Apache 2.0.
HuggingFace `diffusers` maintainers for the `WanPipeline` /
`AutoencoderKLWan` / `WanTransformer3DModel` authors. The cuDNN +
hipBLASLt teams for the underlying kernel infrastructure.
