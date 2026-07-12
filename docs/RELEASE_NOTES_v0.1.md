# Repercep Runtime v0.1 — Release Notes

*Draft. Pre-publication. Update the date + clear this banner before publishing.*

**Date:** 2026-05-24 · **Repo:** https://github.com/miteshs/Mirage ·
**License:** Apache-2.0

Repercep Runtime is the first world-model-native inference engine to ship
a public Cosmos-Predict-7B benchmark on AMD silicon. v0.1 is the first
release; it is **pre-alpha software** but the numbers below are measured
and independently re-verified on a clean GPU.

---

## TL;DR

**Cosmos-Predict-7B Text2World, 121 frames @ 1280×704, 36 denoising
steps, BF16 — on a single AMD Instinct MI300X (192 GiB HBM3, ROCm 7.2):**

| Configuration | Wall time | vs NVIDIA H100 published reference (~380 s) |
|---|--:|--:|
| Baseline (no cache, no FP8) | 470 s | 0.81× |
| Native loop + adaptive caching | **151 s** | **2.52× faster** |
| **Native loop + adaptive caching + autotuned FP8** | **142 s** | **2.68× faster** |
| Peak HBM (all configs) | **52.5 GiB** | 30 % less than H100's 74 GB |

**Wan-2.2-T2V-A14B** (Alibaba's MoE flagship, 14 B active per step):
runs end-to-end on the same MI300X; 81 f / 40 steps at ~1700 s
steady-state, 85 GiB peak. First publicly reported Wan-2.2 number on
any AMD MI300X.

To our knowledge **this is also the first publicly reported Cosmos-Predict
benchmark on any AMD GPU as of 2026-05-22** — see the diligence catalog in
[`docs/COSMOS_ON_MI300X.md` §"Why no one has done this before"](COSMOS_ON_MI300X.md).

---

## What's in v0.1

### Runtime

- **`repercep.models.cosmos.CosmosEngine`** — Cosmos-Predict-7B via the
  HuggingFace `diffusers.CosmosTextToWorldPipeline` path. Bypasses
  NVIDIA's reference stack entirely (no TransformerEngine, no Apex, no
  NATTEN, no CUDA flash-attn). On ROCm, SDPA dispatches to aotriton
  flash kernels — see ADR-0002 for why.
- **`repercep.models.wan.WanEngine`** — Wan-2.2-T2V-A14B via the diffusers
  `Wan2_2Pipeline` path. Second world-model family in the same shape.
- **`repercep.runtime.denoise.denoise_cosmos_video`** — Repercep-native
  denoising loop with CFG batching, fixed step-skip caching, and
  TeaCache-style adaptive caching. `--cache-mode {none|fixed|adaptive}`.

### Caching

- **Adaptive caching (TeaCache-style):** skip a step when the
  accumulated relative L1 distance of the timestep-conditioned latent
  input is below `--cache-adaptive-threshold` (default 0.10; benched at
  0.30 for the 121 f / 36 step config). Warmup window, last step, and
  `--cache-force-full-every` (default 8) always run a full forward.
- **Fixed step-skip caching:** the prior simple cadence path is still
  shipped (`--cache-mode fixed --cache-skip-every {2,4}`).
- **Honest framing on cache quality:** the cached output is
  **trajectory-divergent** from the no-cache reference at the same seed.
  LPIPS 0.64 vs no-cache; inter-frame motion ~28 % lower. The output is
  a valid Cosmos generation, but it is *not* pixel-equivalent to the
  uncached one. The cache is a quality/speed dial, not free compute.
  See `docs/METHODOLOGY.md` for the full LPIPS / MSE / PSNR tables and
  `docs/COSMOS_ON_MI300X.md` §"Quantitative cache quality" for the
  serving-user guidance.

### Attention kernels

- **`repercep.attention.fp8_triton.FP8TritonAttention`** — FlashAttention-2
  in Triton with FP8 quantized GEMMs. `@triton.autotune` over a 19-config
  grid; cache keyed on (B, H, S, D, dtype). Best config at Cosmos
  production shape (B=2, H=32, D=128, S≈109k): `BLOCK_M=256, BLOCK_N=128,
  num_warps=4, num_stages=2/3`. **1.16× over SDPA→aotriton** at that
  shape; cross-over around S ≈ 4–8 k below which SDPA wins.
- **`repercep.attention.diffusers_backend`** — registers `"repercep_fp8"`
  with diffusers' `_AttentionBackendRegistry`. `REPERCEP_FP8_ATTENTION=1`
  flips the active backend in `CosmosEngine.load`.
- **HIP scaffold** (`kernels/hip/fp8_attn/`) — `hipcc` compiles a
  `v_mfma_f32_16x16x32_fp8_fp8` GEMM and the pybind layer loads
  end-to-end; the operand-register layout is incomplete so the output
  values are wrong. Toolchain proven; perf engineering open. Triton
  wins the perf path today.

### Serving

- **v1 endpoints** — the existing FastAPI `POST /generate` synchronous
  path. Unchanged from the pre-Stage-4 release.
- **v2 endpoints (new in v0.1)** — `POST /v2/generate/stream` (NDJSON),
  `POST /v2/generate/{id}/cancel`, `GET /v2/generate/{id}/state`. Routes
  through the Rust core: `Router.accept → Scheduler.submit → engine
  driver → Router.push_frame → frame stream`. Server-minted UUIDs;
  priority field (`"low" | "normal" | "high"`).

### Rust core

- Cargo workspace at the repo root with three crates:
  - **`crates/repercep-cache`** — paged latent cache; port of the prior
    pure-Python `PagedLatentCache`.
  - **`crates/repercep-scheduler`** — request scheduler; 3-bucket priority
    queue, cancellation skip-set, capacity backpressure.
  - **`crates/repercep-router`** — per-request state machine + bounded
    frame channels + scheduler-handle trait. Used by the v2 serving
    path; available standalone for other consumers.
- PyO3 0.25 bindings; abi3-py311; uv-managed venv; `maturin build` +
  `uv pip install --reinstall` is the dev loop (`make rust-install`).
  Per-crate Rust lib names are unique to avoid `target/release/lib_*.so`
  collisions — a real bug we caught in Stage 3 integration.

### Build / dev experience

- `make lint typecheck test` — ruff + mypy `--strict` + pytest. v0.1
  ships 115 pytest + 11 GPU-skip + 41 Rust tests, all green on
  ROCm 7.2 / MI300X / VF.
- `make rust-install` — builds all three crates' wheels, installs into
  `.venv`. The README + `crates/README.md` document the polyglot dev
  loop.
- `scripts/run_cosmos.py`, `scripts/run_wan.py` — end-to-end runners.
- `scripts/profile_cosmos.py` — per-stage forward-hook profiler.
- `scripts/bench_caching.py`, `scripts/bench_fp8.py`,
  `scripts/bench_cosmos_fp8.py` — head-to-head harnesses.
- `scripts/verify_timing.py` — multi-seed reproducibility campaign.
- `scripts/verify_quality.py` — LPIPS / MSE / PSNR between two mp4s.
- `scripts/autotune_fp8.py` — populates the autotune cache for shape
  signatures.
- `scripts/compute_fvd.py` *(landed in the same session)* — Fréchet
  Video Distance between two video sets via I3D.

---

## Honest framing — read this before you cite the 2.68× number

The 2.68× headline compares **Repercep with adaptive caching + autotuned
FP8** against **NVIDIA's published Cosmos-Predict-7B reference on H100**
(as documented on the HuggingFace model card at
`huggingface.co/nvidia/Cosmos-Predict1-7B-Text2World`, ~380 s,
excluding model init, peak 74 GB).

NVIDIA's reference is the published H100 path: the diffusers /
TransformerEngine / Apex / NATTEN / flash-attn-3 stack with no
caching disclosed. **Repercep applies adaptive caching + a tuned FP8 kernel
on top.** The same optimizations are presumably applicable on H100; we
do not have an H100 to run them and measure.

The **raw-hardware comparison** — both sides without caching — has
Repercep's MI300X baseline at 470 s vs NVIDIA's H100 at ~380 s, i.e.
**MI300X is 1.24× *slower* than H100 at the same compute**. The 2.68×
emerges from the *shipped optimization stack*, not from raw silicon
advantage. We are claiming a **shipped-system-vs-shipped-system** lead.

`docs/METHODOLOGY.md` covers this in detail and is the right document
to read before quoting any number.

---

## What's not yet shipped (open work)

- **FVD against a held-out reference set.** The right quality arbiter
  for trajectory-divergent diffusion outputs. Initial small-N FVD on
  N=5 pairs is in v0.1; a large-N (1000+) evaluation needs a real eval
  set that doesn't yet exist publicly for Cosmos.
- **Threshold-quality-speed curve.** We have point measurements at
  threshold 0.05 / 0.10 / 0.20 / 0.30 / 0.50; needs a real plot + a
  defensible "pick your threshold" recommendation. See the threshold
  sweep section in `docs/COSMOS_ON_MI300X.md` (added in v0.1).
- **HIP FP8 kernel correctness.** Triton wins on perf today; HIP is
  the long-term path for shapes Triton can't tune well.
- **Continuous batching + action conditioning hooks** — Plan Phase-2
  deliverables that are deferred to v0.2.
- **Bare-metal MI300X validation.** Our measurements are on a Virtual
  Function slice; we believe single-tenant VF performance matches
  bare-metal but have not verified externally.

---

## Hardware + software dependencies

- **GPU:** AMD Instinct MI300X (192 GiB HBM3, gfx942 / CDNA3). MI300A /
  MI325X almost certainly work; not tested.
- **ROCm:** 7.2.0 (we resolve `torch` from the ROCm wheel index at
  `download.pytorch.org/whl/rocm7.2`).
- **PyTorch:** 2.12.0+rocm7.2 from the ROCm wheel.
- **Python:** 3.11+.
- **`diffusers`:** 0.34.0 – 0.37.x (we verify both ends).
- **`transformers`:** 4.57+ or 5.x.
- **Rust:** 1.85+ (edition 2024).

See `README.md` §"Requirements" + `docs/METHODOLOGY.md` for the full
pinned dep table and the apples-to-apples accounting.

---

## License + acknowledgements

Apache-2.0. NVIDIA for open-sourcing Cosmos under the NVIDIA Open Model
License. Wan-AI / Alibaba for releasing Wan-2.2 weights under
Apache-2.0. HuggingFace `diffusers` maintainers for the Cosmos +
Wan pipelines. The AMD ROCm + aotriton + hipBLASLt teams for the
underlying kernel infrastructure. The TeaCache authors for the
input-similarity-gated caching technique.
