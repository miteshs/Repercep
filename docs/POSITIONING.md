# Mirage — positioning, moat analysis, and what we should (not) claim

*This doc is the strategic-framing layer. It tells you which claims are
defensible, which aren't, and where the wedge actually is.  Sibling docs
handle adjacent layers:*

- `docs/METHODOLOGY.md` — **how** numbers are measured, the apples-to-
  apples accounting that prevents misreading.
- `docs/ANNOUNCEMENT.md` — **how** to communicate publicly (channel-
  specific copy).
- `docs/COSMOS_ON_H100.md`, `docs/COSMOS_ON_MI300X.md`,
  `docs/WAN_ON_H100.md`, `docs/WAN_ON_MI300X.md` — the **what** (the
  measured numbers themselves).
- `docs/BUILD_LOG.md` — the **why** (findings, root causes, F-numbers).

Read in that order if you're new.

---

## TL;DR — Mirage in one sentence

> Mirage is a **world-model-native inference engine** with a vendor-
> neutral backend protocol (AMD MI300X first; NVIDIA H100 + Intel SPR
> CPU also supported) that serves Cosmos-Predict-7B, Wan-2.2-T2V-A14B,
> and V-JEPA 2 through a single Python + Rust runtime.

What it **is**: a runtime + per-target attention kernels + Cosmos-shaped
adaptive caching, packaged as one engine serving three different
world-model families on three different silicon vendors.

What it **isn't** (yet): a kernel-level performance leader on raw
compute, a multi-GPU serving stack, or an FP8-everywhere production
path.  See § "Where Mirage doesn't differentiate yet."

---

## The three landed claims, ranked by external defensibility

### Claim 1 — Cosmos-Predict-7B 121f/36 on a single H100 in 99.6 s (3.81× NVIDIA's published reference)

- **Numbers**: `docs/COSMOS_ON_H100.md`, `docs/SESSION_16_CLOSE.md`
  §"TL;DR". 99.6 ± 3.9 s mean, 3.86 % spread across 5 prompts × 5 seeds.
- **Defensible**: yes, against NVIDIA's *published* ~380 s reference,
  with the caveat that this is **system-vs-system, not kernel-vs-kernel**
  (METHODOLOGY §3).
- **Indefensible misread to guard against**: "Mirage's kernels are 3.81×
  faster than NVIDIA's." That's wrong by our own measurement (see
  § "Where the 3.81× actually comes from" below).

### Claim 2 — only single-GPU path that runs Wan-2.2-A14B with both 14B MoE experts resident, no offload

- **Numbers**: `docs/WAN_ON_MI300X.md` (192 GiB MI300X VF, 85.1 GiB
  peak); `docs/WAN_ON_H100.md` (H100 80 GiB OOMs without `--vae-tiling`
  or offload; with tiling, 72.6 GiB peak; full MoE-resident BF16 path
  works).
- **Defensible**: structural, real, narrow. H100 80 GiB *physically
  cannot* fit Wan-2.2-A14B both-experts-resident BF16 + FP32 VAE peak
  decode without either offload (Wan team's approach) or VAE tiling
  (our addition this session).  MI300X 192 GiB has the headroom.
- **What this isn't**: a wall-time win. MI300X is ~9 % faster than H100
  here, modest; Wan team's H100 + offload + FP8 is 1.49 × faster than
  ours.  The claim is *structural* — about which silicon can deploy the
  model without architectural compromise — not *throughput*.

### Claim 3 — three world-model families on one engine, three silicon targets

- **Numbers**: Cosmos (diffusion video), Wan-2.2 (MoE diffusion video),
  V-JEPA 2 (non-diffusion encoder-predictor) all running through the
  same `mirage.backend.protocol.Backend` Protocol on AMD ROCm, NVIDIA
  CUDA, and Intel CPU (AMX where exposed). See ADR-0003 and
  `docs/TARGETS_AND_KERNELS.md`.
- **Defensible**: architectural, hard to replicate quickly. Most
  inference engines are single-vendor (vLLM is CUDA-first; SGLang
  similarly) or single-model-family (TRT-LLM is text-LLM specific).
- **What this isn't**: a benchmark headline.  Breadth doesn't show up
  on a leaderboard; it shows up when a customer needs to swap silicon
  or model family without rewriting the serving layer.

---

## Where the 3.81× actually comes from

The honest decomposition.  `docs/COSMOS_ON_H100.md` §"Compared to
NVIDIA's H100" has the full table; this is the abbreviated reading.

| step | wall time | factor vs NVIDIA pub. | what changed |
|---|---|---|---|
| NVIDIA published H100 reference | ~380 s | 1.00 × | TE + Apex + NATTEN + FA-3, **no cache** disclosed |
| Mirage **no-cache** on the same H100 | 446.3 s | 0.85 × (we're *slower*) | diffusers + native loop + FA-3 dispatch, BF16, no cache |
| Mirage + adaptive cache (thr=0.30) | 138.4 s | **2.75 ×** | TeaCache-style step-skip in `mirage.runtime.denoise.denoise_cosmos_video` |
| Mirage + adaptive cache + FA-3 source build | **99.6 s** | **3.81 ×** | minimal-config FA-3 wheel dispatching for Hopper WGMMA via `MIRAGE_FP8_ATTENTION=fa` |

Read the columns left-to-right and you can see exactly where the
speedup lives:

- **The cache is the dominant lever (~2.75×).**  Everything else is
  multiplicative on top.
- **FA-3 over the default cuDNN-FA dispatch is worth another ~1.39×.**
- **Mirage's own diffusers wrapping is slower than NVIDIA's bespoke
  cosmos-predict1 pipeline at the no-cache config** (446 vs 380 s).
  This is honest — our overhead at the framework layer costs us ~17 %.

What NVIDIA's published reference is **missing** that Mirage adds:

1. **Adaptive caching** — a TeaCache-style step-skip loop.  NVIDIA's
   reference doesn't disclose using one. Were they to add one,
   they would presumably get a similar 2-3 × speedup (`docs/
   COSMOS_ON_H100.md` line 73 says this explicitly).
2. **A Cosmos-shaped denoise loop** — `mirage.runtime.denoise.
   denoise_cosmos_video` knows the block topology of
   `CosmosTransformer3DModel` and can short-circuit when the latent
   delta is below threshold.

### What this means for the claim

- **"Mirage on H100 is 3.81× faster than NVIDIA's published H100
  reference"** — TRUE.  This is a system-vs-system, published-vs-
  published comparison.
- **"Mirage's kernels are faster than NVIDIA's kernels on Hopper"** —
  FALSE.  At the kernel level we're at parity or slightly behind. The
  no-cache baseline shows this directly.
- **"NVIDIA can match this if they ship caching"** — TRUE.  The
  underlying technique is public (TeaCache).  We should expect this gap
  to compress over time as published references catch up.

### The quality caveat — H100 measured this session

The cache is gated by `--cache-adaptive-threshold 0.30`.  Lower
thresholds skip more steps; higher thresholds skip fewer.  The
*quality* of cached output vs no-cache output is now measured on
**both** MI300X (LPIPS 0.645, F23) and **H100 (LPIPS 0.6067, this
session)** — same regime on both silicon.  Reading is per F23: the
cache is **trajectory-divergent, not pixel-preserving**, but
brightness and per-frame std are preserved (the output is a valid
Cosmos generation with ~30 % less inter-frame motion, not noise).
Details + full table in `docs/COSMOS_ON_H100.md` §"Quantitative cache
quality".

What's *still* open: FVD against a held-out eval set with N ≥ 50
prompts.  Pixel LPIPS is too strict for diffusion outputs that trade
trajectory for compute; FVD is the right distribution-level metric.
`scripts/compute_fvd.py` is in tree; the held-out reference set isn't.
For an external publication, this is the next vouchability item.

---

## What is and isn't a moat

### Real moats (defensible at 6-12 month timescales)

1. **Multi-target Backend Protocol with measured numbers on each.**
   ROCm + CUDA + CPU through one engine is hard to copy in a weekend.
   ADR-0003 is the architectural decision; the engines themselves are
   the proof.
2. **MI300X 192 GiB single-VF MoE-resident path.**  Structural; H100
   80 GiB cannot deliver this without offload, and the offload path
   has its own perf characteristics (~1041 s for Wan vs our 1552 s
   with no offload — they pay in expert-swap latency, we pay in HBM).
3. **World-model-family breadth on one engine.**  Cosmos +
   Wan-2.2 + V-JEPA 2 served by the same runtime; ports of new
   diffusers-shape video models or encoder-predictor models are
   1-2 day exercises, not multi-week ones.
4. **Cosmos-shaped adaptive cache implementation.**  TeaCache the
   *idea* is public; the Cosmos-specific block-topology gate is
   ours and isn't trivially portable to Wan (see F40 / Wan story).

### Not moats

- **Kernel-level Cosmos performance.**  At parity with cuDNN-FA3 / NVIDIA's
  reference stack on raw compute. NVIDIA could match in a release cycle.
- **The TeaCache idea itself.**  Public, well-documented by Morphic /
  Voltage Park / Simplismart blogs.
- **The single-target headline numbers.**  Any of them.  All published
  numbers compress over time.

### Adjacent things that *could* become moats with focused investment

- **A Wan-shaped adaptive cache.**  Doesn't exist anywhere on single
  H100 today.  Would replicate the Cosmos lever on the Wan workload.
  ~1-2 weeks of focused work; the F40 fix-path 1 in BUILD_LOG is the
  prerequisite for FA-3 + cache to compound.
- **TE-FP8 wired through Mirage's bridge.**  Would shift the Cosmos
  headline from 99.6 s toward ~115-130 s (different stack, possibly
  same wall but with FP8 quality preserved).  Doesn't widen the gap vs
  NVIDIA's published reference — narrows it if NVIDIA adds caching —
  but does extend coverage of the FP8 design space.
- **Multi-GPU Context-Parallel.**  Currently uncovered; lets us play
  in the 8 × H100 league (where Wan community numbers sit at ~50-110 s).
  ~2-3 weeks; serious engineering.

---

## Where Mirage doesn't differentiate yet

- **Wan-2.2 on H100 wall time.**  1552.8 s vs Wan team's 1041 s with
  offload + FP8.  We are slower because we don't yet replicate their
  stack.  Apples-to-apples (matching their offload + FP8 wiring)
  produces a *credibility* number ("we match their stack"), not a
  wedge.  Real wedge needs a Wan-shaped cache (next 1-2 weeks).
- **F40 — Mirage's FA-3 bridge is dead weight on Wan.**  Confirmed
  empirically this session: 0 dispatcher engagements vs 780 direct
  `torch.F.scaled_dot_product_attention` calls.  `WanTransformer3DModel`
  bypasses `_AttentionBackendRegistry` entirely.  Three fix paths
  ranked in BUILD_LOG F40; path 1 (custom `WanAttnProcessor` via
  `set_attn_processor`) is the right starting point.
- **TE-FP8 path uninvested.**  Installed and validated in isolation
  (Session 16 §3) but never benchmarked end-to-end.  The "NVIDIA-
  canonical FP8" comparison is therefore still TBD.
- **Multi-GPU sequence-parallel.**  Unimplemented.  The 8 × H100
  community numbers (Morphic 109.8 s, Voltage Park 60 s, Simplismart
  49 s on Wan) are a league we don't enter.
- **Production serving driver.**  `mirage.serving.driver` exists; the
  PyO3 crates `mirage-cache`, `mirage-router`, `mirage-scheduler`
  exist.  End-to-end FastAPI / gRPC has never been exercised under
  load.  Bench infrastructure is the headline; productionization is
  unflagged work.

---

## Strategic options matrix

Choose the *story* first, then the lane.  The lanes flow from the
story, not the other way around.

| Story you want to tell | Right lane | Effort | Risk | What it does to the headline |
|---|---|---|---|---|
| "Single-GPU performance leader" | Extend Cosmos lead with TE-FP8 + push toward sub-90 s | ~1-2 days | low | 99.6 s → ~115-130 s (with FP8 quality) or unchanged if cuDNN-FA3 still wins |
| "World-model-native breadth + MI300X structural advantage" | Fix F40, add Wan-shaped cache, document MI300X-only paths | ~1-2 weeks for cache + ~half-day for F40 | medium | Wan H100 1552 s → ~700-800 s; story becomes "two world-model families with cache on AMD silicon" |
| "Apples-to-apples credibility everywhere" | Wan offload + FP8 weight convert, FVD/LPIPS quality numbers, multi-prompt variance on everything | ~1 week | low | No wall-time wedge widens; numbers become bullet-proof against skeptical reading |
| "Production-ready serving engine" | Productionize `mirage.serving.driver`, benchmark batched throughput, build SLA characterization | ~2-3 weeks | medium | Different axis entirely — RPS/$, latency tails, not headline wall time |
| "Multi-GPU contender" | Context-Parallel sequence sharding, play in the 8 × H100 league | ~2-3 weeks | high | Lets us cite community-comparable numbers; no MI300X equivalent ships today |

Doing two stories in parallel is feasible; doing three is overcommit.
The dependency that quietly blocks the most others is FVD/LPIPS quality
measurement on H100 — every claim that includes the cache currently
rests on an unmeasured quality assertion, and that's the one thing
*every* skeptical external reader will probe first.

---

## What we recommend doing in the next 2 weeks if forced to choose

0. **~~Cosmos H100 quality measurement~~** (done this session).
   The 3.81 × headline now rests on a measured H100 quality result
   matching MI300X (LPIPS 0.61 vs 0.65; motion compression −34 %
   vs −28 %); the trajectory-divergent framing of F23 holds on
   Hopper unchanged.  Pixel quality vouchability closed.  FVD on
   a held-out eval set still open as an external-publication item.
1. **F40 fix-path 1** (~half-day).  Custom `WanAttnProcessor` via
   diffusers `set_attn_processor`.  Makes the FA-3 bridge actually
   engage on Wan; without it the bridge is documented dead weight.
2. **Wan-shaped adaptive cache** (~1-2 weeks).  The big lever.  Turns
   the Wan story from honest-but-modest to "two world-model families
   served with the cache lever, on three silicon targets."
3. **FVD with N ≥ 50 prompts on a held-out Cosmos eval set** (~1
   week).  The "is cache quality-preserved at distribution level"
   measurement.  Required before external publication of the 3.81 ×
   number; the pixel-LPIPS work that landed this session is
   necessary but not sufficient for that claim.

Skip in this window:
- Multi-GPU.  High-effort, doesn't help the single-silicon story.
- Wan offload + FP8.  Credibility-only.  Do after the cache lands so
  the comparison is "cache on Mirage vs offload on Wan team" — that's
  the interesting one.
- TE-FP8 on Cosmos as a *headline* play.  The cache is already the
  dominant lever; FP8 doesn't widen the cache-driven gap.  TE-FP8
  becomes interesting *after* the cache, as a path to FP8 quality.

---

## Cross-references

- Measured numbers: `docs/COSMOS_ON_H100.md`, `docs/COSMOS_ON_MI300X.md`,
  `docs/WAN_ON_H100.md`, `docs/WAN_ON_MI300X.md`, `docs/COSMOS_ON_CPU.md`
- Variance / verification: `scripts/verify_timing.py` (Cosmos),
  `scripts/verify_wan_timing.py` (Wan), `scripts/trace_wan_attention.py`
  (F40 evidence)
- Apples-to-apples accounting (read this if anything above looks like
  marketing): `docs/METHODOLOGY.md` §3
- Per-target architecture: `docs/TARGETS_AND_KERNELS.md`,
  `docs/architecture.md`
- Findings (root causes for everything above): `docs/BUILD_LOG.md`
  F-numbered entries; this session added F38-F41.
