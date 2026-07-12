# Repercep v0.1 — announcement drafts

*Pre-publication. Read `docs/RELEASE_NOTES_v0.1.md` first; this file is
the social / distribution layer. Pick a channel below and ship.*

---

## Option 1 — X / Twitter thread

**Tweet 1 (the hook):**

> First publicly reported Cosmos-Predict-7B benchmark on AMD silicon:
>
> **121 frames @ 1280×704, 36 steps, in 142 s** on a single AMD Instinct
> MI300X — 2.68× faster than NVIDIA's published H100 reference.
>
> Open source. Apache-2.0. github.com/miteshs/Repercep

**Tweet 2 (the how, briefly):**

> The stack: `diffusers` + ROCm 7.2 SDPA→aotriton + adaptive caching
> (TeaCache-style) + an FP8 Triton kernel autotuned for Cosmos's
> production shape (B=2, H=32, D=128, S≈109k). Zero CUDA-only deps —
> no TransformerEngine, no Apex, no NATTEN, no flash-attn-3.

**Tweet 3 (the honest framing):**

> Important: this is system-vs-system, not silicon-vs-silicon.
>
> Raw MI300X (no cache, no FP8) runs the same workload at 470 s vs
> H100's ~380 s — MI300X is 1.24× *slower* than H100 on raw compute.
> The 2.68× comes from the optimization stack we shipped, on cheaper
> non-NVIDIA hardware. Same tricks would also help H100.

**Tweet 4 (what we measured carefully):**

> We didn't just claim "same quality" — we measured.
>
> LPIPS adaptive-cache vs no-cache = 0.64 ("substantially different")
> at the pixel level; inter-frame motion is ~28 % lower. The cached
> output is a *valid* Cosmos generation, but trajectory-divergent.
> Cache is a quality/speed dial, not a free lunch. Docs say so.

**Tweet 5 (Wan):**

> Also runs Wan-2.2-T2V-A14B (Alibaba's MoE flagship, 14B active per
> step). ~1700 s steady-state at 81 f / 40 steps, peak 85 GiB HBM. To
> our knowledge, also the first publicly reported Wan-2.2 number on
> any AMD MI300X via the diffusers path.

**Tweet 6 (links):**

> Repo: github.com/miteshs/Repercep
> Cosmos writeup: github.com/miteshs/Repercep/blob/main/docs/COSMOS_ON_MI300X.md
> Wan writeup:    github.com/miteshs/Repercep/blob/main/docs/WAN_ON_MI300X.md
> Methodology:    github.com/miteshs/Repercep/blob/main/docs/METHODOLOGY.md
>
> Phase 2 of the implementation plan landed: WM-native, MI300X-first.

---

## Option 2 — Hacker News submission

**Title (≤80 chars):**
> Show HN: Repercep — Cosmos-Predict-7B on AMD MI300X, 2.68× faster than H100 reference

**First comment (anchors the discussion to the honest framing):**

> Author here. The 2.68× is a shipped-system-vs-shipped-system comparison:
> Repercep's adaptive caching + a tuned FP8 Triton kernel composed against
> NVIDIA's published Cosmos reference (~380 s on H100, no caching).
>
> Raw hardware-only: MI300X is ~1.24× slower than H100 at the same
> compute. The 2.68× is from the optimization stack on cheaper non-
> NVIDIA hardware. Same optimizations would help H100 too; we don't
> have one to measure.
>
> The cache is also not "lossless" in the pixel sense — LPIPS 0.64 vs
> no-cache. Inter-frame motion is ~28 % lower. The output is a valid
> Cosmos generation, just trajectory-divergent. `docs/METHODOLOGY.md`
> spells this out — happy to dig into specifics.

---

## Option 3 — Blog post (longer-form)

### Title (working)

> Cosmos-Predict-7B on AMD MI300X, 2.68× faster than NVIDIA's H100
> reference — and how we found out the cache wasn't lossless

### Outline (1500–2500 words)

#### 0. The headline (50 words)
Top: the table from `RELEASE_NOTES_v0.1.md`. 142 s / 2.68×. Repo link.

#### 1. Why no public Cosmos number on AMD until now (200 words)
The dependency stack. NVIDIA's reference Cosmos repo hard-depends on
`transformer_engine`, `apex`, NATTEN, flash-attn — all CUDA-only. A
naive port has to replace all four at once. We sidestepped this by
using the HuggingFace `diffusers.CosmosTextToWorldPipeline`, which on
ROCm dispatches through SDPA → aotriton flash kernels. Zero of the
NVIDIA-specific dependencies. (Reference: `docs/adr/0002` and the
"first publicly reported" diligence catalog in `COSMOS_ON_MI300X.md`.)

#### 2. The wedge (250 words)
Why MI300X first, not H100 fast-follow. The Repercep Implementation Plan
(§5.4 + §7.1) explicitly named non-NVIDIA silicon as the defensible
wedge for a world-model runtime — on NVIDIA, NIM and TensorRT-LLM are
free and bundled; on AMD, there was no production-grade WM serving at
all. We measured 192 GiB HBM (vs H100's 80 GB) was a structural
advantage: no offload needed, model + activations + caching all fit
resident. Records: `docs/adr/0001`.

#### 3. The optimization stack (500 words)
Three things composed:

- **CFG batching** + **inference_mode** — a one-day fix that took us
  from "OOM at 189 GiB" to "52.5 GiB peak." The autograd graph for all
  36 diffusion steps was alive without an `inference_mode` wrap. Bug
  was there since the first commit; it didn't show up until 17f / 8 steps
  was scaled to 121f / 36. (F18 in `BUILD_LOG.md`.)
- **Adaptive caching** (TeaCache-style input-similarity gate). After
  warmup, skip a denoising step when the relative-L1 distance of the
  timestep-conditioned input is below threshold. 36 steps → ~11 full
  forwards at threshold 0.30. **2.52× headline.**
- **Autotuned FP8 Triton kernel.** A 19-config autotune at the Cosmos
  production shape (B=2, H=32, D=128, S≈109k) lands at
  BLOCK_M=256 / BLOCK_N=128 / num_warps=4 / num_stages=2-3.
  1.16× over SDPA → aotriton at this shape; composes to 142 s end-to-end
  for an extra ~9 s saved. **2.68× new headline.**

The autotune story is interesting: a fixed-tile FP8 kernel that won
1.92× at B=1, H=8 (benchmark shape) was *0.99× at Cosmos's production
shape* — the 8-column program grid amortized the tile, the 64-column
grid didn't. Without autotune, FP8 was a regression. With autotune,
it's the headline. (F20, F21 in `BUILD_LOG.md`.)

#### 4. Verification — the part we got wrong, then corrected (400 words)
The original "motion stat 4.65 vs 4.66 ≈ same quality" claim survived
several sessions before we asked the right question:

> Is the cached output pixel-equivalent to the no-cache reference at
> the same prompt + seed?

We ran the no-cache reference, computed LPIPS:

| Pair | LPIPS | Interpretation |
|---|--:|---|
| no-cache vs adaptive | **0.645** | "substantially different" |
| no-cache vs adaptive + tuned-FP8 | 0.642 | "substantially different" |
| adaptive vs adaptive + tuned-FP8 | 0.117 | "perceptually very similar" |
| adaptive vs adaptive + fixed-FP8 | 0.122 | "small but visible diff" |

The cached output is **not** the no-cache output's twin. The earlier
motion-stat comparison was between *two cached outputs* — both had
motion ~30 % below the no-cache truth. We retracted the "lossless"
framing and replaced it with "trajectory-divergent valid Cosmos
output."

This is the right thing to do. A diffusion cache that skips steps
*does* change the trajectory; pixel LPIPS will be high; the question
is whether the output distribution remains close to the no-cache
distribution. The right metric is FVD against a held-out eval set,
which we're working on. v0.1 ships small-N FVD with the caveat
documented.

(F23 in `BUILD_LOG.md`; full table in `COSMOS_ON_MI300X.md`
§"Quantitative cache quality"; methodology accounting in
`METHODOLOGY.md`.)

#### 5. Wan-2.2 — the second model family (200 words)
Phase 2 of the plan said: prove the runtime is WM-native, not Cosmos-
specific. Wan-2.2-T2V-A14B (Alibaba's MoE flagship; 14 B active per
step; ~52 GiB BF16 weights, 118 GiB total snapshot) runs end-to-end
through Repercep on the same MI300X. 81 f / 40 step quality reference
at ~1700 s steady-state, 85 GiB peak. ~1.6× behind the Wan team's
single-H100 reference (1041 s with FP8 + offload). To our knowledge,
the first publicly reported Wan-2.2-T2V-A14B number on any AMD MI300X
via the diffusers path. (Reference: `docs/WAN_ON_MI300X.md` for the
full H100 / 8×H100 / A100 catalog and the small-MI355X-MLPerf
datapoint.)

#### 6. The Rust core (200 words)
The Repercep codebase is polyglot by design: Python surface (model
loading, denoise loop, attention, FastAPI), Rust core (cache
management, scheduler, request router, all PyO3-bound). v0.1 ships the
three crates + the v2 serving path that routes through them. The
Implementation Plan's hypothetical robotics-OEM design partner would
get edge-deployment-ready Rust without a 6–9 month cold-start rewrite.
(ADR-0004, ADR-0005.)

#### 7. What's open (150 words)
- FVD against a 1000+ reference set
- HIP FP8 kernel correctness (Triton ships; HIP is research-grade today)
- Continuous batching + action conditioning
- Bare-metal MI300X validation (we measure on a VF slice)

#### 8. How to reproduce (50 words)
`git clone github.com/miteshs/Repercep && cd Mirage` + the one-liner in
`README.md`. Hardware: MI300X + ROCm 7.2. Total time to first
generation: ~15 min (env setup) + 7 min (cold first run) or 142 s
(warm + cached). Reproducer in `docs/HANDOFF.md` §11.

---

## Option 4 — GitHub release notes (already drafted)

See `docs/RELEASE_NOTES_v0.1.md`. Paste it into the GitHub Releases UI
when cutting v0.1.

---

## Publication checklist (before pressing send)

- [ ] Bump `Cargo.toml` workspace version to `0.1.0` (currently `0.0.1`).
- [ ] Bump `pyproject.toml` version to `0.1.0`.
- [ ] Tag the release: `git tag -a v0.1.0 -m "Repercep Runtime v0.1.0"` +
      `git push origin v0.1.0`.
- [ ] Cut a GitHub release from the tag, paste `RELEASE_NOTES_v0.1.md`.
- [ ] Update the repo `README.md` headline number from "2.47× headline"
      to "2.68×" if not already done. (Already done as of `231ee3e`.)
- [ ] Sanity-check all docs links are GitHub-relative (start with
      `docs/` not absolute).
- [ ] One final `make lint typecheck test` and a fresh-clone smoke on a
      colleague's machine.
- [ ] Post the X thread first; let it settle for a day; then HN; then
      blog if you want long-form.
- [ ] Be ready for the "yes but H100 with the same stack…" pushback.
      The methodology doc is the right link to drop.
