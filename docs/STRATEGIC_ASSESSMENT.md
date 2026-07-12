# Repercep — Independent Strategic Assessment

- **Date:** 2026-05-28
- **Repo HEAD at assessment:** `2e9096e` (`main`, Session 23 handoff)
- **Provenance:** Independent, outside-in review produced at the maintainer's
  request. This is the *skeptical external reader's* view — the one a technical
  due-diligence pass, a Show HN crowd, or a hiring panel would take.
- **Based on:** a full audit of the Python source; a separate audit of the Rust
  crates + HIP/AMX/Triton kernels + test suite; a reconstruction of the 23-session
  build history and the F1–F47 findings ledger; and a web-verified competitive
  scan (sources in Appendix A).
- **How to read this:** it is the counterweight to `docs/POSITIONING.md`. POSITIONING
  is the first-person framing of what's defensible to claim; this doc is the
  outside view of what survives diligence. Where they disagree, treat this as the
  adversarial case. It cross-references `docs/METHODOLOGY.md`, `docs/BUILD_LOG.md`
  (F-numbers), and the per-target writeups.

---

## TL;DR — the one-paragraph verdict

Repercep is genuinely impressive **engineering** and a genuinely rare **honest
artifact**: a clean, vendor-neutral, multi-target world-model runtime that boots
Cosmos, Wan-2.2, and V-JEPA 2 on AMD, NVIDIA, and CPU, built in ~4 days and
documented with more intellectual honesty than almost anything in the inference
space. But the **moat as currently framed is thin-to-nonexistent under scrutiny**:
the headline speedup is a *public* technique (TeaCache-style caching), running on
*cheaper* hardware, against an *unoptimized published reference* — none of which is
defensible. The project built the **wedge-opener** (first/fast world-model
inference on AMD) and deferred the **moat** (world-model-native *action-conditioned,
closed-loop* serving) — and the wedge-opener is exactly the part of this space
that is commoditizing fastest. The strategic move is to stop hardening the
wedge-opener and start building the deferred half of the original vision.

---

## 1. Original vision vs. what got built

Reconstructed from `docs/adr/0001`, the referenced `~/Repercep_Implementation_Plan.pdf`,
`docs/POSITIONING.md`, and `docs/ANNOUNCEMENT.md`, the vision was:

> A **world-model-native inference engine** — purpose-built for diffusion-temporal
> serving (bidirectional attention, latent-volume denoising, **action conditioning,
> closed-loop latency**) — with an **OSS-first GTM**, a **non-NVIDIA-silicon wedge**
> (the plan named AMD/Trainium/Apple as places with "no production-grade WM serving
> today"), a **Rust+Python polyglot core** so one codebase serves datacenter *and*
> edge, eventually landing **robotics/AV design partners** who need action-conditioned,
> closed-loop world simulation.

ADR-0001 inverted the plan's H100-first sequencing: *"Leading on H100 means launching
into a market with a free, bundled, vertically-integrated incumbent. Leading on
MI300X means launching where the value proposition is 'this is the only
production-grade option that exists.'"*

**What got built (Sessions 1–23):** the *infrastructure + credibility layer* —
the vendor-neutral runtime, the backend/attention seams, the Cosmos/Wan/V-JEPA
engines, the benchmark harness, the caching optimization, three FP8 Triton kernels,
the Rust scheduler/router, a FastAPI serving skeleton, and an exemplary measurement
methodology.

**What did NOT get built** — and this is the crux — is the part the plan treated
as the actual differentiator: **action conditioning, closed-loop interactive
latency, edge deployment.** These appear in the mission statement and are then
deferred in every handoff (*"action conditioning hooks — deferred until a robotics
OEM is in the design-partner pipeline"*).

> **The central finding:** the wedge-opener (first/fast on AMD) was executed
> brilliantly; the moat (world-model-native action-conditioned serving) sits on the
> shelf — and the wedge-opener is the piece competitors are commoditizing fastest.

---

## 2. Where we actually are

### 2.1 What's genuinely real (this is not a Potemkin repo)

- **The vendor-neutral Backend Protocol (ADR-0003) is the real deal.** Audited as
  not-leaky: zero concrete-backend imports above the seam (only two, both in
  comments). CUDA and CPU backends were genuinely added as "one class + one registry
  entry," exactly as advertised. Good architecture with taste.
- **Three real, distinct FP8 Triton flash-attention kernels** (gfx942 / Hopper / Ada
  — 547 / 589 / 564 lines, not copy-paste; different FP8 formats and autotune grids),
  with a persistent on-disk autotune cache.
- **Real AMX BF16 and INT8 flash-attention C++ kernels** (716 / 868 lines of genuine
  `_tile_dpbf16ps` / `_tile_dpbssd` intrinsics + hand-written AVX-512 `exp`). The
  FP16 variant is an honest SDPA-fallback stub that labels itself.
- **Three real Rust+PyO3 crates.** Scheduler (priority buckets, tokio, GIL-release)
  and router (per-request state machine, async frame channels) are *actually wired*
  into the v2 serving path, with real `cargo test` coverage.
- **The headline numbers are real**, and the flagship H100 one is *well-backed*:
  **99.6 ± 3.9 s, N=25** (5 prompts × 5 seeds), reproduced across session boundaries
  and venv resets.
- **The intellectual honesty is a genuine, rare asset.** The F8→F10 retraction, the
  F23 "quality-preserved" walk-back, the relentless stack-vs-silicon separation in
  METHODOLOGY §3, the labeling of "contested"/"projected"/"methodology-tainted"
  numbers. This is the trust layer, and it is worth more than any single benchmark.

### 2.2 What's thinner than the framing implies

| The framing | The reality |
|---|---|
| "Adaptive caching" as a Repercep capability | An explicit, *conservative* TeaCache re-implementation — the docstring cites the paper and omits TeaCache's clever part (the polynomial rescaler), using a raw identity accumulator. Not novel; the code never claims to be. |
| "3.81× faster than NVIDIA" | = **cache 2.75× × FA-3 1.39×**. The cache is public IP; FA-3 is NVIDIA's own kernel. At the *kernel* level you're at parity-or-behind; the no-cache baseline was *slower* than NVIDIA's reference (446 vs 380 s; F47 later flips to 1.18× faster on newer torch, n=1). |
| "FP8 kernel as an optimization" | Cosmos production shape: a **wash** (0.98×) until painfully autotuned to ~1.13× (F20→F21). On Hopper: a **net loss** vs cuDNN-FA3 (F27/F29); the docs say `REPERCEP_FP8_ATTENTION=1` on Hopper is "demonstration that the kernel runs," not a perf setting. |
| "Polyglot Rust core" | Real and partly wired — but on any box without compiled wheels (the default), the v2 path runs entirely on **Python fallbacks**, not Rust. The cache crate is **orphaned** (importable, never instantiated). |
| "Quality is preserved" | Rests on **small-N FVD (N=5, self-labeled "preliminary")**. Literature standard is N≥1000 (~7 GPU-days here). POSITIONING calls this "the one thing every skeptical external reader will probe first." |
| HIP FP8 GEMM kernel | Compiles and loads but produces **numerically wrong output** (B-operand load under-samples K: reads 8 of 32 values). Zero test coverage. Open since Session 9. |
| AMX kernels | **Never numerically validated** — every dev VM masks the AMX flags. CI proves the plumbing, not the compute. |

Meta-fact: this is a **single-developer, ~4-calendar-day, single-virtualized-GPU-slice
research artifact.** No bare-metal validation; no multi-GPU (the league where the
*real* competitive Wan numbers live — 8×H100 at 49–110 s — which the project
explicitly "doesn't enter"); no load-tested serving. That's not a knock on the
effort; it's a calibration of "where we are": **a high-quality proof-of-competence,
not a product, and not yet a moat.**

---

## 3. The moat question, answered honestly

| Candidate moat | Verdict | Why |
|---|---|---|
| Adaptive caching | ❌ Not a moat | TeaCache (arXiv 2411.19108, Nov 2024; CVPR 2025) *is* "adaptive threshold caching," explicitly accelerates **Cosmos and Wan**, and already ships in vLLM-Omni and FastVideo. POSITIONING agrees. |
| "Only WM serving on AMD" | ⚠️ Eroding fast | AMD ships `rocm/pytorch-xdit` serving Wan2.1/2.2/HunyuanVideo on MI300X; SGLang-Diffusion serves WAN on MI300X; AMD put **Wan2.2 in MLPerf 6.0 (Apr 2026)**. Defensible only as "no *production-hardened, WM-native* stack — only research baselines." Shrinking window. |
| Kernel / silicon performance | ❌ Not a moat | At parity-or-behind by your own measurement. NVIDIA closes the system gap in one release by adding caching — and is *already* doing it (Cosmos-Transfer2.5, Feb 2026: TensorRT diffusion backend + action-conditioning distillation + edge-distilled low-latency model). |
| MI300X 192 GB structural memory | ✅ Real, narrow, time-boxed | The one *physics* fact: 80 GB H100 can't fit Wan-2.2-A14B both-experts-resident + FP32 VAE peak without offload/tiling; 192 GB can. Doesn't get patched in a release — but it's one workload shape and MI325X/MI355X/H200/B200 move the goalposts. |
| Multi-target runtime + WM breadth | ✅ Asset, not a deep moat | "Hard to copy in a weekend" (POSITIONING) — but a funded team copies it in a quarter, and vLLM/SGLang are going multimodal *and* run on AMD. Good architecture, not a durable barrier. |

### So what *is* the moat?

**Today: there is no durable technical one.** The most defensible assets are softer
than the project's framing admits: (1) the 192 GB memory physics (narrow, time-boxed);
(2) the architecture + breadth (replicable by a funded team); (3) — genuinely — the
**demonstrated ability to execute honest, multi-target world-model infrastructure
fast.** For hiring / credibility / design-partner trust, (3) is the strongest card.

**The moat that's actually available** is the unbuilt half of the vision:
**owning the action-conditioned, closed-loop, interactive world-model serving
regime.** Why it's structurally defensible where the others aren't:

- General-purpose engines (vLLM-Omni, SGLang-Diffusion, xDiT, FastVideo) are built
  for **request→response batch generation** and will keep absorbing batch tricks
  (caching, sequence parallelism) because that's their shape. They are *not* built
  for a **stateful interactive loop** where an agent/robot/player injects an action
  per step, the world-state latent persists and is reused across steps, and the
  metric is **closed-loop FPS under state carryover** — a different leaderboard
  nobody owns.
- It maps to where world-model demand *with money* is heading: robotics/AV
  synthetic-data-in-the-loop (Cosmos-Predict2.5 video2world + action conditioning;
  the robot-policy models NVIDIA shipped Feb 2026) and interactive neural worlds.
- It's where a *genuinely novel* caching/scheduling contribution lives: reuse across
  the **action loop**, not just across denoise steps within one generation. Recent
  "inter-request caching" work (arXiv 2604.04451, Apr 2026) shows the field is only
  now pushing caching past single-generation — room to get there first on the
  *action-conditioned* version and publish it.

> **Load-bearing caveat:** the marquee real-time interactive world models today
> (Google Genie 3, Decart Oasis) are **autoregressive**, not diffusion-temporal.
> Repercep's specialization (bidirectional attention, latent-volume denoising) targets
> the offline/short-horizon diffusion regime (Cosmos T2W, Wan, Marble-style export).
> Moving toward "interactive" forces a choice: serve the *diffusion* interactive
> regime (Cosmos video2world, real-time-diffusion research) or broaden to
> autoregressive too. **Do not pitch "real-time interactive" while serving only
> bidirectional batch diffusion** — that's the one place architecture and market are
> misaligned, and a sharp reviewer will catch it.

---

## 4. Competitive reality (web-verified 2026-05-28; see Appendix A)

- **NVIDIA is a moving, well-funded target closing your claimed differentiators.**
  There is an official **NIM microservice for Cosmos** (Triton-based; PyTorch + NeMo
  + TensorRT + TensorRT-LLM). **Cosmos-Predict2.5 / Transfer2.5** (late 2025–Feb 2026)
  added action-conditioning distillation, robot policy models, a TensorRT diffusion
  backend, and an edge-distilled low-latency model. *Targeting Cosmos-Predict-**7B**
  is last-gen* — the current center of gravity is Predict2.5-**2B**.
- **The "LLM engines are token-only" line is no longer true.** vLLM→**vLLM-Omni**
  (diffusion/video; **ships a TeaCache implementation**) and SGLang→**SGLang-Diffusion**
  (serves WAN, runs on MI300X) are absorbing video-diffusion serving. Specialized
  engines **xDiT** and **FastVideo** already cover Wan/HunyuanVideo with the same
  caching/attention tricks.
- **AMD is filling its own gap.** Beyond xDiT/SGLang above, AMD's ROCm blog serves
  Wan2.1/2.2 (self-described "research baseline, not production-grade") and
  MLPerf-6.0'd Wan2.2 on MI355X. The honest thesis is "no production-hardened
  WM-native stack on AMD," not "nothing on AMD."
- **MI300X-first is a credible-but-time-boxed GTM.** Real cost advantage (~15–40%
  below H100 SXM5) + 192 GB + relative software thinness — but AMD is closing the
  gap, the latency-sensitive segment still favors CUDA/TensorRT at low batch, and the
  demand-side WM leaders (Cosmos, Genie, Oasis) are NVIDIA-anchored (Decart explicitly
  uses NVIDIA for edge). MI300X-first may mean fishing where today's WM deployers
  largely aren't.

---

## 5. What to do next — the strategic fork

First name the game, because the right next step differs:

- **(A) Portfolio / credibility / hiring artifact.** You're *mostly done* and
  over-investing. Ship a carefully-framed writeup (§6) and stop racing benchmarks.
- **(B) A company / fundable thing.** The current roadmap (Lane A: Wan cache;
  Lane B: FVD N≥1000) is the **wrong bet** — it hardens claims that aren't
  differentiating. Pivot from "faster batch Cosmos/Wan" to "own a regime."
- **(C) A research contribution.** The novel-caching-for-action-loops idea is the
  highest-leverage move.

Assuming (B)/(C), the re-ranked roadmap:

**Tier 1 — moat-building (do these):**
1. **Build the action-conditioning + closed-loop serving surface.** Streaming latent
   world-state, per-step action injection, latent/KV reuse *across an interactive
   rollout*. Target **Cosmos-Predict2.5 video2world** (current gen) with action
   conditioning. Metric: interactive latency / FPS under state carryover — not batch
   seconds. This is what general-purpose engines structurally won't build and what
   the original vision named.
2. **Invent and *measure* one genuinely novel mechanism** — action-conditioned /
   cross-step / cross-request world-state cache reuse — **benchmarked head-to-head
   against TeaCache and DiCache** on an interactive workload. Win → a paper *and* a
   moat. Lose → learned cheaply. This is the line between "we re-implemented TeaCache"
   and "we have IP."
3. **Convert the 192 GB physics into a workload H100 *can't run at all*** — long-horizon
   rollouts with large resident world-state, or N-stream interactive serving with many
   resident world-states. "Faster" is a race; "runs what the other silicon physically
   can't" is a wedge.

**Tier 2 — credibility-hardening (minimum for trust, then stop):**
4. **FVD at N≥100, not N≥1000.** N≥100 crosses the rank-deficiency threshold and lets
   you make a *distribution-level* quality statement instead of single-pair LPIPS.
   Keep this one item from the current roadmap — it closes the "every skeptic probes
   this first" gap. Table-stakes, not a moat.
5. **One bare-metal MI300X run** to retire the "it's only a VF slice" asterisk.

**Tier 3 — explicitly *don't* (POSITIONING half-knows this):**
- Wan-shaped adaptive cache (Lane A) *as a flagship* — it widens the commoditizing
  wedge-opener. Do it only as a stepping-stone to the action-loop cache.
- Multi-GPU sequence parallelism — high effort to enter a race others lead.
- FP8-everywhere — a wash-to-loss on your shapes by your own findings.

---

## 6. Before you publish anything — credibility risks

`docs/ANNOUNCEMENT.md` is poised to lead with the misreadable number, in front of
the exact audience that will dismantle it:

- **The X hook leads with "2.68× faster than H100"; Tweet 3 retracts it** to
  "system-vs-system, MI300X is 1.24× *slower* on raw compute." Leading with the
  number you immediately walk back loses the room. **Lead with the honesty** —
  "first Cosmos on AMD + here's exactly where the speedup comes from and why it's
  not a silicon win." The honesty *is* the best content; bury the hype-number.
- **TeaCache will be the top comment.** Pre-empt it: "the cache is TeaCache-style and
  public; our contribution is the AMD path + the honest measurement." Hide the ball
  and you look like every other benchmark-inflation post.
- **Don't ship the quality-preserving implication** until FVD ≥ N=100. Today it's a
  single LPIPS pair on H100.
- **"First Cosmos on AMD" is fine** as a hook — but it's small and perishable (AMD is
  already MLPerf-ing Wan2.2). Not a thesis.

---

## 7. What this project proves regardless

The moat verdict doesn't erase three durable truths:

1. **It proves you can stand up honest, working, multi-target world-model
   infrastructure faster than almost anyone** — and document it with a discipline
   (the F-ledger, the retractions, METHODOLOGY §3) most teams never produce. That is
   itself a credibility/hiring/trust asset.
2. **The architecture is genuinely good.** The Backend Protocol is a seam that pays
   off for years; an action-conditioning surface would sit cleanly *above* it.
3. **You already privately know most of this** — POSITIONING lists "the TeaCache idea
   itself" and "kernel-level performance" under "Not moats." The gap isn't analysis;
   it's that the *active roadmap* hardens the credibility layer while the moat layer
   (action-conditioned closed-loop serving) waits behind a "need a design partner"
   gate.

> **Highest-leverage move:** flip that. Don't wait for the robotics design partner to
> justify the action-conditioning loop — build a credible version *first*, on
> Cosmos-Predict2.5 video2world + the 192 GB advantage, with one novel cache mechanism
> measured against TeaCache. That's what turns "an impressive AMD benchmark" into
> "the runtime nobody else is building" — the only path to a moat that survives
> diligence.

---

## Appendix A — competitive sources (web-verified 2026-05-28)

The competitive landscape moves fast; re-verify before citing. Key sources:

- **NIM for Cosmos:** https://docs.nvidia.com/nim/cosmos/latest/introduction.html ·
  release notes https://docs.nvidia.com/nim/cosmos/2.0.0/release-notes.html
- **Cosmos Predict2.5 / Transfer2.5:** https://huggingface.co/blog/nvidia/cosmos-predict-and-transfer2-5 ·
  https://github.com/nvidia-cosmos/cosmos-predict2.5 ·
  https://nvidianews.nvidia.com/news/nvidia-announces-major-release-of-cosmos-world-foundation-models-and-physical-ai-data-tools
- **vLLM-Omni** (TeaCache module): https://github.com/vllm-project/vllm-omni ·
  https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/cache/teacache/
- **SGLang-Diffusion:** https://github.com/sgl-project/sglang
- **xDiT / FastVideo:** https://github.com/xdit-project/xDiT · https://haoailab.com/blogs/fastvideo/ ·
  inter-request caching https://arxiv.org/pdf/2604.04451
- **AMD ROCm video/diffusion + MLPerf 6.0:**
  https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/xdit-diffusion-inference.html ·
  https://rocm.blogs.amd.com/artificial-intelligence/serving-videogen-v1/README.html ·
  https://www.amd.com/en/blogs/2026/amd-delivers-breakthrough-mlperf-inference-6-0-results.html
- **Caching lineage:** DeepCache https://arxiv.org/abs/2312.00858 (CVPR'24) ·
  Learning-to-Cache https://arxiv.org/abs/2406.01733 (NeurIPS'24) ·
  TeaCache https://arxiv.org/pdf/2411.19108 (CVPR'25), project https://liewfeng.github.io/TeaCache/ ·
  DiCache https://arxiv.org/pdf/2508.17356
- **World-model market:** Genie 3 https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/ ·
  World Labs / Marble https://techcrunch.com/2025/11/12/fei-fei-lis-world-labs-speeds-up-the-world-model-race-with-marble-its-first-commercial-product/ ·
  Decart Oasis https://decart.ai/publications/oasis-interactive-ai-video-game-model
- **MI300X adoption / ROCm-vs-CUDA gap:**
  https://blogs.oracle.com/cloud-infrastructure/announcing-ga-oci-compute-amd-mi300x-gpus ·
  https://www.spheron.network/blog/rocm-vs-cuda-gpu-cloud-2026/

## Appendix B — assessment methodology

This review was produced by an independent agent (Claude Code) at the maintainer's
request. It is based on:

1. **Python source audit** — backend/attention/runtime/models/serving/bench layers,
   assessing the ADR-0003 seam, the attention kernels, the `denoise.py` adaptive-cache
   state machine, the engines, and serving, with a real-vs-scaffold lens.
2. **Rust + kernel + test audit** — the three PyO3 crates (which are wired, which is
   orphaned), the HIP FP8 GEMM (compiles / wrong output), the AMX kernels (real /
   unvalidated), the Triton FP8 kernels, and what CI actually verifies on a no-GPU box.
3. **History reconstruction** — the 23-session arc and the F1–F47 findings ledger,
   with emphasis on retractions, negative results, and the claim-rigor gradient.
4. **Competitive scan** — web-verified as of the assessment date (Appendix A),
   distinguishing web-confirmed facts from training-era knowledge.

All performance numbers quoted here are the project's own measured figures (see
`docs/METHODOLOGY.md` and the per-target writeups); this doc does not re-run them.
The verdicts are analytical, not adversarially re-measured — the FVD-quality and
bare-metal gaps it flags are gaps in the *evidence*, which only new measurement closes.
