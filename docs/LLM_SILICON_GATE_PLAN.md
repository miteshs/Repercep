# Pre-registration: the MI300X-vs-H100 LLM throughput gate (2026-08-09)

**Binding.** Written and committed before any measurement. The point of writing
it first is that the analysis cannot be chosen after seeing the data — the same
discipline `LLM_BESTOFN_PLAN.md` imposed, which is why that result was
believable when it came back negative.

## 1. What is under test, and why it matters this much

`BUSINESS_PLAN_2026.md` §233–234 carries two kill criteria for **Lever B**
(silicon arbitrage):

| criterion | status | kill threshold |
|---|---|---|
| MI300X committed rate | $2.20/hr **[A]** | `> $2.80/hr` |
| **MI300X vs H100 token throughput** | **parity [A] — never measured** | **`< 0.75×`** |

The first is safe: provider-level MI300X on-demand is **$1.71–1.99/hr**
(TensorWave / Vultr / DigitalOcean, July 2026), comfortably under the threshold.

The second has never been measured, and **since commit `6ca6889` withdrew
`BUSINESS_PLAN` §6.3, silicon arbitrage is the entire LLM economic case.** So an
unmeasured assumption is currently load-bearing for the only surviving LLM claim
in the deck.

There is now public evidence pointing the wrong way. SemiAnalysis's AMD-vs-NVIDIA
inference benchmark concludes that **for rentals under six months NVIDIA wins
decisively across all scenarios**, and separately reports ROCm CI at "<10% parity
with NVIDIA" and "25% of tested models failing accuracy tests." AMD wins in their
data only on large dense models, memory-bandwidth-bound work, and relaxed
latency.

**This is the adversarial hypothesis, handed to us for free. Test it.**

## 2. Design

| | |
|---|---|
| Engine | **vLLM** (primary) — the most mature ROCm path and the most defensible cross-vendor comparison |
| Models | `Qwen/Qwen2.5-7B-Instruct` (~15 GB) and `Qwen/Qwen2.5-32B-Instruct` (~64 GB) — both fit **both** GPUs in bf16 |
| Precision | bf16 on both. No FP8 on either side, even where supported — a precision asymmetry would make the comparison meaningless |
| Client | **co-located on the pod**, against `127.0.0.1` |
| Repeats | 5 timed, median reported, warm-up discarded |

**Why vLLM and not SGLang**, given July's P0 deployment decision was "default the
LLM path to SGLang": SGLang's ROCm support is less proven than vLLM's, and a
cross-vendor number is only worth having if the engine is equally mature on both
sides. Recorded consequence: **if anything this understates our deployed
position**, since SGLang measured 9–17% faster on the fan-out shape. Do not
switch engines mid-experiment to chase a better number.

### 2.1 Workload shapes — reported separately, never averaged

July's standing lesson is that a lever is workload-shaped and quoting one number
without its shape is the fastest way to be wrong in public.

| tag | prompt | decode | concurrency | represents |
|---|---:|---:|---:|---|
| **S1-interactive** | 512 | 512 | 1 | chat / low-latency, where SemiAnalysis says NVIDIA wins |
| **S2-interactive-load** | 512 | 512 | 32 | the same shape under load |
| **S3-batch** | 4096 | 128 | 32 | prefill-heavy / agentic, where SemiAnalysis says AMD is competitive |

### 2.2 Metrics

Primary: **median output tokens/sec**. Secondary: TTFT, end-to-end request
latency, and completion-token totals (the equal-work check).

Derived: **cost per million output tokens** = `price_per_hr / (tok_s × 3600 / 1e6)`.

## 3. Kill criteria — decided now, not after

Let `R = median MI300X throughput / median H100 throughput`, computed **per shape
and per model**.

1. **If `R < 0.75` on every shape** → **Lever B is dead.** `BUSINESS_PLAN`'s
   silicon claim is withdrawn, not restated smaller, exactly as §6.3 was.
   Consequence, stated in advance so it cannot be softened later: with the
   best-of-N lever already withdrawn, the LLM line of business would have **no
   surviving economic differentiator**, and should be presented as breadth
   (serving many model classes) rather than as cost.
2. **If `R ≥ 0.75` on some shapes and `< 0.75` on others** → the claim survives
   **only scoped to the shapes where it holds**, and every future quote of it
   must carry the shape inline. This is the most likely outcome (§5).
3. **If `R ≥ 1.0` anywhere** → report it, but do not lead with the best shape.
   Lead with the interactive shape, because that is the one an unfriendly
   diligence question will pick.

## 4. Confounds pre-declared

- **Provider confound on cost, not on throughput.** MI300X is on AMD Developer
  Cloud at $1.99/hr; H100 is on RunPod at RunPod's rate. The **throughput ratio
  is a clean silicon comparison; the cost ratio is not** — it mixes silicon with
  provider pricing. Report them as two separate numbers and never present the
  cost ratio as a property of the silicon.
- **Feature parity between engine builds.** If vLLM's ROCm build differs
  materially from its CUDA build (chunked prefill, attention backend, CUDA
  graphs / HIP graphs), we are comparing feature sets, not chips. **Record the
  attention backend and feature flags on both sides.** A material difference is
  a caveat on the number and may itself be the real finding.
- **Cold start / autotune.** Today's Cosmos 3 run established that ROCm pays
  autotune **per tensor shape**, not just per process (7.5× on a new resolution).
  Every shape must be warmed before timing on both GPUs, or the ROCm side eats a
  penalty that has nothing to do with steady-state throughput. This trap already
  produced three wrong numbers today.
- **Payload assertion, not status codes.** RunPod's nginx owns several ports and
  returns its own 200s. Readiness must assert the model id in
  `/v1/models`, never a 2xx.
- **Equal work.** Compare equal completion-token totals across GPUs; a
  throughput win that ships fewer tokens is not a win.

## 5. Predictions on record (so being wrong is visible)

1. **`R ≥ 0.75` on S3-batch** — bandwidth-bound, MI300X's 192 GB and higher HBM
   bandwidth should hold up. Confidence: moderate.
2. **`R < 0.75` on S1-interactive at concurrency 1** — SemiAnalysis's strongest
   claim, and low concurrency exposes kernel-launch and scheduler overhead where
   ROCm has historically lagged. Confidence: moderate.
3. **The 32B model favours MI300X relatively more than the 7B** — larger dense
   models are where SemiAnalysis found AMD competitive. Confidence: low-moderate.
4. **Outcome 2 in §3 is what happens** — a shape-scoped survival, not a clean
   pass or a clean kill.

If prediction 4 holds, the honest reading is that **the AMD advantage is
workload-specific**, which points the strategy toward the lane where it holds
rather than toward "cheaper inference for everything."

## 6. Accuracy smoke test (not the gate, but reportable)

SemiAnalysis claims 25% of tested models fail accuracy checks on ROCm. Run the
same 20 prompts greedy (`temperature=0`) on both GPUs and diff the outputs.

This is a smoke test, not an eval. Pre-declared reading: **identical outputs are
expected and unremarkable; any divergence beyond the first few tokens is a
finding worth reporting even though it does not move the gate.** Bit-exactness
across vendors is not expected and is not the bar.

## 7. Session scope and sequencing

The H100 leg runs on RunPod, driven directly. **The MI300X leg needs a manual
AMD Developer Cloud instance** (browser-blocked for automation), so the
comparison completes across two sittings, H100 first.

While the H100 box is up, two Cosmos 3 items ride along for free:

- **Cosmos 3 H100 row** — `scripts/cosmos3_rocm_probe.py --full` already exists
  and runs unchanged; gives the cross-vendor comparison
  `COSMOS3_ON_MI300X.md` §7 currently has to defer.
- **`RoboLab` success-rate harness** — turns the levers ladder from a latency
  table into something quotable. It is `--runtime nvidia` Docker, so H100 is the
  only place it can run at all; if it will not run on AMD, that is itself worth
  recording.

**Boxes are terminated at the end of the session regardless of outcome.**
