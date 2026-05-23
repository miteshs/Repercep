# Mirage — Optimization Strategy

**Workload:** Cosmos-Predict-7B Text2World · **Hardware:** AMD Instinct MI300X (gfx942)

## 1. Principle — measure, then optimize

Per the implementation plan: build the benchmark, establish the baseline,
optimize against *measured* cost. No optimization lands without a before/after
number from `mirage.bench` against the baseline below.

## 2. The measured baseline — 2026-05-22

Two measurements, both `CosmosEngine` on the naive `diffusers` path, MI300X, bf16.

**Cold full run** — 121 frames @ 1280×704, 36 steps: **738 s**, peak HBM
52.5 GiB. But a cold run pays one-time ROCm kernel autotuning (aotriton /
hipBLASLt / MIOpen) on every new shape — ~300 s of that 738 s is autotuning,
not compute.

**Warmup-separated per-stage profile** — 49 frames @ 1280×704, 12 steps, steady
state (`scripts/profile_cosmos.py`):

| Stage | Wall time | Share |
|-------|-----------|-------|
| Text encode (T5) | 0.06 s | 0.1% |
| **DiT denoising loop** (24 transformer forwards) | **42.0 s** | **95%** |
| VAE decode | 0.53 s | 1.2% |
| Other (latent prep, scheduler, postprocess) | 1.8 s | 4% |
| **Total** | **44.3 s** | 100% |

What this sets:

- **The DiT loop is the entire game** — 95% of steady-state time. VAE decode and
  text encode are rounding error; Tier 2 below is deprioritized accordingly.
- CFG is **unbatched**: 24 transformer forwards for 12 steps = 2 per step. The
  DiT processes a long latent-video sequence; attention is quadratic in it — the
  single most-leveraged kernel.
- A cold run wastes ~300 s on kernel autotuning — **warming the kernel cache at
  deploy time is itself a real latency win**.
- Peak HBM is 52 of 192 GiB. **~140 GiB sits unused** — headroom that converts
  directly into batching and resident-model throughput (the MI300X advantage).

Measured so far (49 f / 12 steps, warmup-separated):
- `torch.compile` on the DiT — **1.13× loop / 1.12× end-to-end**.
- Mirage-native loop with CFG batching — **1.02× loop / 1.05× end-to-end**.
  Smaller than projected: at Cosmos-7B scale each transformer call is
  compute-bound, so packaging two batch-1 forwards as one batch-2 forward
  doesn't reduce GEMM work — see BUILD_LOG F14.
- Native loop + **step-skip caching (`cache_skip_every=4`) — 2.00× loop /
  1.96× end-to-end at 49 f / 12 steps.** The biggest measured single lever;
  24 → 6 DiT calls. Quality dial — visual verification of the cached output
  is still pending.
- **At the full reference config (121 f / 36 steps), caching scales further:
  154 s warmup-separated → 3.02× over the MI300X baseline, 2.47× FASTER than
  NVIDIA's published H100 reference (~380 s).** Same caveat: quality
  verification pending.

## 3. Optimization tiers

### Tier 1 — The diffusion loop (target: the 435 s / 59%)

- **Step reduction.** 36 steps is the reference. Higher-order solvers
  (DPM-Solver++) cut to ~20 with no retraining. Consistency / step distillation
  reaches 4–8 steps — a 4–9× loop win, but needs a distilled checkpoint
  (training effort; Studio territory).
- **CFG batching / distillation.** Classifier-free guidance is 2 DiT forwards
  per step. Batch cond+uncond into one (built — `--native-loop`); measured
  **1.05× e2e at 49 f**, F14 explains why it's small at this scale.
  Guidance-distillation (drop the unconditional pass entirely) is the
  remaining lever here — up to 2× because it actually halves the work.
- **Attention.** ~56k-token sequence, head_dim 128. Today: SDPA → aotriton
  flash. Next: CK flash-attn tuned for gfx942; FP8 attention (CDNA3 MFMA);
  NATTEN-style neighborhood attention exploiting video locality → sub-quadratic.
- **`torch.compile`** the DiT forward (inductor + triton-rocm): operator
  fusion, removes Python and kernel-launch overhead.
- **Feature / step caching** (DeepCache / TeaCache style): DiT block outputs
  change slowly between adjacent steps — cache and skip. Training-free, ~1.5–2×.

### Tier 2 — VAE decode (deprioritized — see §2)

Steady-state VAE decode is ~1% of wall time, not the ~41% a cold run implied.
This tier is parked unless a workload moves the number (very long clips,
decode-heavy model variants). Techniques, if needed later:

- **Tiled / temporal-chunked decode.** The Cosmos VAE is causal-temporal;
  decode in temporal chunks (and spatial tiles) to bound memory traffic.
- **Streaming decode.** Decode chunks as the diffusion finishes them and emit
  frames — this is the plan's frame-level streaming; collapses time-to-first-frame.
- **Compile / fuse** the decoder convolutions; evaluate lower-precision VAE.
- **Overlap** VAE decode of chunk N with DiT denoising of chunk N+1 on separate
  HIP streams.

### Tier 3 — MI300X kernel layer

- **FP8 (e4m3) GEMMs** — CDNA3 has native FP8 MFMA; the DiT is GEMM-bound at
  4096 hidden. ~2× math throughput with quantization care. (Plan: Phase 2.)
- **hipBLASLt autotuning** for the exact DiT / VAE shapes.
- **CK / AITER** attention and GEMM kernels tuned for gfx942.
- Longer term: the **Kernel** product — RL-driven kernel synthesis (Series A).

### Tier 4 — Serving throughput

- **Continuous batching** where temporal dependencies allow (plan, Phase 2).
- **Stage pipelining** — T5 / DiT / VAE as pipeline stages: while the VAE
  decodes request A, the DiT denoises request B.
- **Paged latent cache** (already stubbed in `mirage.runtime.latent_cache`) —
  frame-aware reuse of latent tiles.
- **Exploit the ~140 GiB of free HBM** — large batches, multiple model
  variants / LoRAs co-resident, zero CPU offload. An H100 at 80 GiB must
  offload; the MI300X does not — a structural latency and throughput edge.

## 4. The MI300X angle

- 192 GiB HBM3 → everything resident, big batches, no offload.
- ~5.3 TB/s HBM bandwidth → directly helps the memory-bound VAE decode.
- Native FP8 (OCP e4m3 / e5m2) MFMA on CDNA3.
- Strategic: no production-grade world-model serving stack exists on AMD today
  — being fast here *is* the differentiation (implementation plan §5.4).

## 5. Sequencing & targets

1. **Profile properly** — per-stage, warmup-separated (one-time ROCm kernel
   autotuning vs. steady state). `mirage.bench` + torch profiler / `rocprof`.
2. **Tier 1 training-free wins** — `torch.compile`, CFG batching, solver swap,
   feature caching. Best near-term ratio, no new weights.
3. **Tier 2 VAE** — tiling + streaming decode.
4. **Tier 3 kernels** — FP8, CK attention, hipBLASLt.
5. **Tier 4 throughput** — batching + stage pipelining.

Plan targets: **2–3× over naive PyTorch + Diffusers in Phase 1**, **3–5× in
Phase 2** (with continuous batching + FP8). Every change is measured by
`mirage.bench` against the 738 s baseline in §2.

## 6. Explicitly NOT yet

- Hand-written HIP kernels from scratch — exhaust CK / aotriton / Triton /
  `torch.compile` first (plan: "use existing primitives, don't reinvent").
- The RL kernel synthesizer — Series A scope.
- Multi-GPU — only a single MI300X VF is available on this host.
- Distillation training — needs a training pipeline (Studio scope).
