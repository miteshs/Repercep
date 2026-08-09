# Cosmos 3 Nano Port Plan (2026-08-08)

- **Target:** `nvidia/Cosmos3-Nano-Policy-DROID` — the policy (world-action) variant
  of NVIDIA's Cosmos3-Nano, the fourth model on the `InteractiveWorldModel` seam.
- **Why this one, and why now:** Cosmos 3 (released 2026-05-31, announced at GTC
  Taipei 2026-06-01) is NVIDIA's flagship *open* physical-AI model and the model the
  newly-formed Cosmos Coalition (Agile Robots, Black Forest Labs, Generalist, LTX,
  Runway, Skild AI) is standardizing on. The Policy-DROID checkpoint is ranked **#1 on
  the RoboArena policy leaderboard** (their claim, HF model card). It is also the
  *same DROID embodiment* we already ported DreamZero against, which makes this the
  first port that yields a genuine same-embodiment head-to-head on our own seam
  rather than another isolated row.
- **Sources verified locally:** `NVIDIA/cosmos` cloned and read (scratchpad). Every
  number in §1–§2 below is from that checkout — `cookbooks/cosmos3/generator/action/`
  (`README.md`, `run_policy_with_diffusers.ipynb`,
  `run_policy_with_cosmos_framework.md`), `cookbooks/cosmos3/README.md`, and
  `inference_benchmarks.md` — not from the press release. Model-card facts are marked
  as such. License **OpenMDW-1.1** (Linux Foundation Open Model Development Weights),
  stated commercial-use-permitted.

---

## 0. The competitive clock, stated first — it is different this time

Every prior port (V-JEPA 2-AC, LingBot-VA, DreamZero) had the same shape: the vendor
shipped an open-slow reference, the fast path was closed or NVIDIA-only, and our claim
was "first fast serving." **That claim is not available here and the plan should not
pretend otherwise.** Cosmos 3 ships on day one with:

| Path | Status in the checkout |
|---|---|
| Diffusers `Cosmos3OmniPipeline` | full policy support, git-main only |
| vLLM-Omni | OpenAI-compatible `/v1/videos`, policy via async POST |
| SGLang Diffusion | OpenAI-compatible serving, `run_policy_with_sglang.ipynb` |
| NIM | prebuilt NGC container, FP8 |
| Cosmos Framework | native PyTorch + a real policy server (`action_policy_server_robolab`) |
| TensorRT-LLM | reasoner path |

The DreamZero port plan's §5 "competitive clock" risk — *"vLLM-Omni RFC #4127; if they
ship optimized serving before our Phase 2, the claim narrows"* — **has now happened**,
one model over. vLLM-Omni and SGLang Diffusion are shipped Cosmos 3 entrypoints.

So the honest scoping of what is still ours, and the whole reason to do this port:

1. **No AMD.** Model card and repo state NVIDIA Ampere / Hopper / Blackwell, Linux,
   CUDA 13 (or 12.8). ROCm appears nowhere. The Diffusers path is the lever — exactly
   as it was for Cosmos-Predict1, where `models/cosmos.py`'s existing docstring records
   that diffusers has no TransformerEngine/apex dependency and therefore runs on ROCm
   PyTorch with SDPA dispatching to aotriton flash kernels on MI300X. **[A]** that this
   still holds for `Cosmos3OmniPipeline`; it is the first thing Phase 1 checks.
2. **No policy latency numbers — from anyone, including NVIDIA.**
   `inference_benchmarks.md`'s intro paragraph claims the tables cover "forward and
   inverse dynamics, and policy generation," but the actual TOC and every published
   table are **t2v / i2v / t2i only**. Their own note concedes it: *"Generator results
   are published incrementally… empty cells mean that combination has not been measured
   yet."* There is no ms/chunk, no actions/sec, no resident-sessions/GPU for the policy
   path anywhere. That is the metric our `CONTROL_LOOP_BENCH.md` leaderboard exists to
   produce.
3. **Candidate batching on policy fan-out.** The measured 5.3–9.9× shared-prefix lever
   (OpenVLA, both silicons). Untested on a diffusion-policy fan-out, and worth a
   pre-registered gate rather than an assumption — see §5.

**The claim this port can support, written before the measuring starts:** *first AMD
row, first control-loop metrics, and first candidate-batching result on NVIDIA's
flagship open physical-AI policy model.* Not "first fast Cosmos 3 serving." If Phase 1
finds the diffusers path doesn't run on ROCm, item 1 dies and the port is worth
substantially less — that is the gate, and it comes first.

---

## 1. What the model is (verified against the checkout + model card)

Cosmos3-Nano is a **16B omnimodal world model** on a **Mixture-of-Transformers (MoT)**
architecture: an autoregressive transformer for discrete token generation (the
*Reasoner*) and a diffusion transformer for continuous multimodal synthesis (the
*Generator*), sharing 3D rotary position embeddings over spatial + temporal structure.
Model card gives 16B total; secondary sources describe it as 8B reasoner + 8B
generator **[A — not confirmed in the checkout, verify from `config.json` in Phase 1]**.
**BF16 only** — the card is explicit that FP4/FP8/FP16 are not officially supported,
which removes a whole rung of our usual quantization ladder before we start.

The Generator exposes three action tasks (`cookbooks/.../action/README.md`) **[V]**:

| Mode | Meaning |
|---|---|
| `forward_dynamics` (`fd`) | start image + action trajectory → future observations |
| `inverse_dynamics` (`id`) | video → ego-motion / end-effector trajectory |
| **`policy`** | start image + task instruction + state → **future observations *and* action chunk, jointly** |

`policy` is our port target and is the same regime as LingBot-VA and DreamZero: the
model is its own policy, no external CEM.

### 1.1 The DROID action space — and how it differs from our existing DROID row

Cosmos 3 treats action as a modality whose tokens are transitions between consecutive
visual states. Poses are **9D deltas** = 3D translation + 6D continuous rotation
(`rot6d`); grasp is a 1D open/close value **[V]**.

| | Cosmos3-Nano-Policy-DROID | DreamZero-DROID (already ported) |
|---|---|---|
| action width | **10D** = EE pose (9D) + gripper (1D) | 8D = joint_position (7) + gripper (1), zero-padded to 32 |
| action space | **end-effector pose deltas, meters** | **joint space** |
| rotation | 6D continuous (`rot6d`) | n/a |
| chunk | **16 actions @ 15 FPS** | 24 actions/block |
| cameras | **3 composited into one canvas** | 3 separate streams, 880 tok/frame |
| normalization | quantile (`droid_lerobot_stats.json`) | q99 quantile (`metadata.json`) |

Same robot, **different action representation** — end-effector vs joint space. Worth
recording loudly because it means the head-to-head is a *serving-cost* comparison on
one embodiment, **not** an interchangeable-policy comparison. Anyone reading a
side-by-side table will assume the latter unless told.

The normalization family is the same quantile shape LingBot-VA and DreamZero both
already reimplement, which is the third time that formula has shown up — a good sign it
should be shared rather than hand-rolled a third time (§3).

### 1.2 Policy inference geometry (from `run_policy_with_diffusers.ipynb`) **[V]**

| knob | value | source |
|---|---|---|
| `mode` | `"policy"` | `ACTION_SETS["droid_policy"]` |
| `domain_name` | `"droid_lerobot"` (*not* `"droid"`) | same |
| `chunk_size` | **16** → frames generated = `chunk_size + 1` = **17** | same; pipeline derives frame count |
| `resolution_tier` | **480** | same |
| `view_point` | `"concat_view"` | same |
| `fps` | **15** | same |
| prompt | task instruction, e.g. *"Pick up the object and place it in the target container."* | same |
| `num_inference_steps` | **30** | `FIXED_SAMPLING` |
| `guidance_scale` | **1.0** (no CFG — halves the DreamZero cost shape) | same |
| `flow_shift` | **5.0** via `UniPCMultistepScheduler.from_config(..., use_karras_sigmas=False)` | same |
| seed | 0 | same |
| `use_system_prompt` | `False` | pipeline call |
| dtype | `torch.bfloat16` | `from_pretrained` |

**Conditioning canvas** — the three DROID cameras are composited into a single
640×540 image before the pipeline ever sees them (`build_concat_frame`) **[V]**: wrist
across the full top half, `exterior_1` and `exterior_2` side by side across the bottom
half, each fitted with `ImageOps.fit` + bicubic. This is a *pre-processing* step in the
notebook, not something the pipeline does — so our engine must own it, and its exact
geometry is part of the model contract.

**Output** — `result.video` (17 frames) and `result.action[0]`, a `[16, 10]` tensor in
**model-normalized space**. Denormalization is `denormalize_action(method="quantile",
stats=DROID_ACTION_STATS)` where stats load from
`cosmos_framework/data/generator/action/normalizer_stats/droid_lerobot_stats.json`
**[V]**. Channel layout: `[:, :9]` pose deltas consumed by
`pose_rel_to_abs(rotation_format="rot6d", pose_convention="backward_framewise")`,
`[:, 9]` gripper **[V]**.

**`flow_shift` — resolved, 2026-08-08, not a discrepancy.** The cookbook README's action
defaults table gives `flow_shift = 10.0` and the policy notebook uses `5.0`, which
initially read as a conflict. Reading `run_fd_with_diffusers.ipynb`'s `FIXED_SAMPLING`
settles it **[V]**: **10.0 is the forward/inverse-dynamics value, 5.0 is the policy
value** — the README table documents the fd/id vLLM path only. Use **5.0 for policy**.
Recorded because the README table is the more discoverable of the two and would send a
reimplementation to the wrong trajectory silently.

---

## 2. The inference path (what we port)

**We port the Diffusers path, not the Cosmos Framework path.** Reasons, in order:

1. It is the ROCm lever (§0.1), and ROCm is the reason to do the port at all.
2. `run_policy_with_cosmos_framework.md` **[V]** shows the framework path is
   Docker-first with `--runtime nvidia`, a `uv sync --group=cu130-train
   --group=policy-server` dependency closure, and a `RoboLab` simulation client in a
   *second* Docker image. That is an eval harness, and the standing lesson from
   LingBot-VA (`VA_Server`) and DreamZero (`ARDroidRoboarenaPolicy`) is: **port the
   component, never the eval-harness wrapper.**
3. Diffusers gives us one Python call with no hydra/dataset-framework closure behind
   it — a much smaller vendoring surface than any prior port. This may be the first
   port where **nothing gets vendored at all**.

The call we wrap, verbatim from the notebook **[V]**:

```python
pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano-Policy-DROID",
    torch_dtype=torch.bfloat16,
    safety_checker=None,
    enable_safety_checker=GUARDRAILS,
)
pipe.to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=5.0, use_karras_sigmas=False
)

result = pipe(
    prompt=task_instruction,
    action=CosmosActionCondition(
        mode="policy", chunk_size=16, domain_name="droid_lerobot",
        resolution_tier=480, image=concat_frame, view_point="concat_view",
    ),
    fps=15, num_inference_steps=30, guidance_scale=1.0,
    use_system_prompt=False, generator=torch.Generator("cuda").manual_seed(0),
)
# result.video -> 17 frames ; result.action[0] -> [16, 10] normalized
```

**Dependency pin, and it is a sharp one:** `Cosmos3OmniPipeline` and
`CosmosActionCondition` are **not in any released diffusers**. The cookbook installs
`diffusers @ git+https://github.com/huggingface/diffusers.git` **[V,
`cookbooks/cosmos3/README.md:157`]**. Phase 1 must pin an exact commit SHA, record it,
and never float `main` — a moving pipeline definition under a benchmark is how a
leaderboard row silently stops being reproducible.

### 2.1 RESOLVED 2026-08-08: the model is **stateless per chunk**

Every prior interactive port had an explicit per-layer KV cache to reason about and
eventually reuse (ADR-0009). The open question was whether Cosmos 3's "autoregressive
chunked generation" meant a cached latent path or repeated one-shot calls. Reading
`run_fd_with_diffusers.ipynb`'s `run_rollout` settles it **[V]** — its own docstring:

> *"Chain forward-dynamics chunks, feeding each chunk's last frame into the next one.
> Every chunk regenerates its conditioning frame as frame 0, so only frames 1.. are kept."*

The loop calls `generate_chunk` — a fresh, complete `pipe(...)` — once per chunk, passing
`image=conditioning` where `conditioning = chunk_frames[-1]` from the previous call, and
`seed=chunk_index`. **Nothing is carried across chunks except a decoded RGB frame.**
"Autoregressive" here means **pixel-space frame chaining**, not cached-latent
autoregression. Four consequences, all load-bearing:

1. **ADR-0009 KV-latent reuse does not apply to this model.** There is no cross-chunk
   cache to reuse. The sliding-window eviction problem that ADR-0009 left open is
   simply absent here — this port neither needs nor tests that work.
2. **The seam's branching guarantee finally holds.** `InteractiveWorldModel.step`
   promises *"the previous state is not mutated, so a planner can branch rollouts from
   a shared prefix."* LingBot-VA and DreamZero both had to break that promise — their
   native KV caches mutate in place, so v0 of each keeps one live branch per session.
   Cosmos 3's `WorldState` is a **conditioning frame + prompt + sampling config**:
   small, serializable, trivially copyable. **This is the first port where the seam
   works as designed.** Worth saying plainly in the bench doc; it is a point in the
   seam's favour that three ports of evidence had so far failed to produce.
3. **Per-session HBM is weights-dominated, not cache-dominated.** DreamZero's headline
   constraint was 22.88 GiB of KV *per session* (H100 holds 1). Here, sessions cost
   approximately nothing beyond one image — the resident-sessions question collapses
   into "how many concurrent *denoise* calls fit," which is a batching question, not a
   memory-per-session one. **[A]**, to be measured in Phase 2.
4. **Error compounds through pixels.** Each chunk conditions on the previous chunk's
   *generated* final frame, re-encoded from RGB. Long rollouts accumulate
   generation error with no latent path to keep them honest. Not our bug to fix, but
   it bounds how long a rollout means anything, and Phase 2 should note where quality
   visibly degrades rather than benchmarking 50 chunks as if chunk 50 were valid.

---

## 3. Mapping onto the `InteractiveWorldModel` seam

| Seam | Cosmos 3 Nano realization |
|---|---|
| `reset(conditioning, params)` | build the 640×540 concat canvas from the three camera frames; stash task instruction + sampling config. `conditioning.uri` = the three camera sources. Pipeline load is engine-held per config, not per session. |
| `step(state, action)` | one `pipe(...)` policy call against the state's conditioning frame → `(17 frames, [16,10] chunk)`; denormalize; return a **new** state whose conditioning frame is the last generated frame (§2.1's chaining). `Action.space = "cosmos3_chunk_droid_ee"` (the `_ee` matters — §1.1). |
| `plan(state, goal, horizon)` | policy regime, same as LingBot-VA/DreamZero: return the model's own proposed chunk. Goal conditioning is the text instruction at reset. No CEM. |

Because of §2.1, `WorldState` here is a genuine value object — no engine-held mutable
cache keyed by `session_id`, which every prior interactive port needed. The engine
becomes noticeably thinner than `dreamzero.py`: no session-swap machinery, no cache
lifecycle, no window-reset bookkeeping.

**Shared-code opportunity, now the third instance:** quantile action denormalization
exists in `lingbot_va_pipeline.py` and `dreamzero_pipeline.py` already. The DreamZero
plan deferred a shared base as "a Phase-0 design question, not a commitment." Three
implementations of one formula is enough evidence — extract it in Phase 0, but scope it
to *just the denormalizer*, not a speculative shared VA base class (the chunk-loop
shapes still differ: Wan-family causal DiT with a live KV cache vs. a diffusers
one-shot pipeline; forcing those together would be the wrong abstraction).

---

## 4. Phases

- **Phase 0 (CPU): DONE 2026-08-09.** This doc; the autoregressive question closed
  (§2.1 — stateless per chunk); `models/cosmos3.py` — chunk loop against an injectable
  pipeline Protocol, 19 tests in `tests/test_cosmos3.py`, all green alongside the full
  suite (310 passed / 61 GPU-skipped), ruff and `mypy --strict` clean.
  `models/action_norm.py` carries the shared quantile denormalizer with an explicit
  `eps` (§3) — **the two existing callers were deliberately not migrated onto it**:
  both are GPU-verified exact-parity paths and swapping a numerically-sensitive helper
  underneath them without a pod to re-verify on would trade a real guarantee for
  tidiness. Do that migration when Phase 1's box is up.
  Concat-canvas geometry is implemented and tested (pure PIL, no GPU) because getting
  it wrong degrades every downstream number silently.
  Two scaffold decisions worth knowing before Phase 1 touches it: conditioning
  resolution is a **pipeline** Protocol method, not an engine helper (it is entirely
  model-specific, and `ConditioningInput` forbids extra fields so there is nowhere to
  smuggle pre-resolved pixels); and `plan()` **raises rather than returning normalized
  values** when quantile stats are absent, because a plausible-looking wrong trajectory
  is worse than a crash.
- **Phase 1 (GPU verify):** **H100 first, then MI300X — and the MI300X leg is the
  actual point of this port, so do not let it slip to a follow-on session the way
  DreamZero's levers ladder did.** Pin the diffusers commit SHA. Resolve the
  `flow_shift` 5.0-vs-10.0 discrepancy (§1.2). Confirm resident bf16 footprint at load
  (16B bf16 ≈ 32 GB weights **[A]**, plus activations — an H100 80 GB should hold
  several sessions, a much better story than DreamZero's 22.88 GiB/session; measure,
  don't extrapolate). Parity gate: our engine's `[16,10]` denormalized chunk vs. the
  notebook's, same seed/prompt/frame, exact match — the same gate every prior port used.
  **Guardrail access is a prerequisite and a human action:**
  `nvidia/Cosmos-1.0-Guardrail` is a **gated** HF repo requiring an access request
  **[V]**; the Generator requires it unless disabled via
  `COSMOS3_DIFFUSERS_GUARDRAILS=false`. Request access before booking the pod — we
  already handle this repo in `models/cosmos.py` for Cosmos-Predict1, so the account
  may already be approved; check first.
- **Phase 2 (bench):** `CONTROL_LOOP_BENCH.md` rows for H100 + MI300X — ms/chunk,
  actions/sec, resident sessions/GPU. This is the table nobody has published (§0.2).
  Levers ladder in a `COSMOS3_LEVERS_2026_08.md`: denoise-step reduction from the
  default 30 (the biggest single knob, and `guidance_scale=1.0` means there is no CFG
  batch to fuse — a lever DreamZero had that this model does not), attention backend,
  torch.compile, resolution tier 256 vs 480. **Then** the candidate-batching gate (§5).
- **Phase 3 (serving):** wire into `/v2/world/session` next to LingBot-VA and
  DreamZero.

---

## 5. The candidate-batching gate — pre-register it, do not assume it

The 5.3–9.9× shared-prefix lever is measured on **OpenVLA token decoding**. The
best-of-N session (2026-07-27) is the standing lesson on what happens when a lever
measured on one workload is *assumed* onto another: it transferred at 1.5×, not 3×, and
then turned out to be uncapturable. **Do not put a Cosmos 3 candidate-batching number in
any external artifact before it is measured, and pre-register the gate first.**

§2.1 sharpens the prediction, and it is not favourable. OpenVLA's lever came from
sharing a long autoregressive *prefill* across candidates. Here there is no prefill and
no KV cache: N candidate rollouts from one state are N independent pipeline calls that
share only a single image encode, while the divergent work is 30 full denoise steps
each. **[A] the prefix-sharing lever is therefore close to 1× on this model.**

What *would* show up instead is plain **batch efficiency**: diffusion at batch 1
underutilizes an H100 badly, so running N candidates as one batched call should beat N
sequential calls by a lot. That is a real effect and worth measuring — **but it is a
different mechanism, and naming it "our candidate-batching lever" would repeat exactly
the July error.** Batched generation is stock in diffusers; a customer who passes a
batch gets it from anyone. Pre-registered rule, written before the run:

- Report prefix-sharing and batch-efficiency as **two separate numbers**, with the
  baseline for each stated.
- If the only win is batch efficiency, it goes in the **deployment playbook**, not the
  moat — the same demotion `n=N` got.
- Only a measured advantage over *stock batched diffusers on the same hardware* is
  ours to claim.

## 6. Risks

- **ROCm is unproven for this pipeline.** The Cosmos-Predict1 precedent is real but is
  a different, older pipeline class. If `Cosmos3OmniPipeline` pulls TransformerEngine,
  FlashAttention-3, or a CUDA-only custom op, the AMD claim — the main reason for this
  port — is gone. **Gate this in Phase 1 before any bench work.** Cheap early check:
  import the pipeline on MI300X and inspect its module tree before downloading 32 GB
  of weights.
- **Diffusers `main` is a moving target** (§2). Pin a SHA, record it in the bench doc.
- **BF16-only** removes the FP8/FP4 rungs from the levers ladder. Trying them anyway is
  fine as an experiment; quoting them as supported is not.
- ~~The one-shot-vs-autoregressive question~~ **RESOLVED (§2.1): stateless per chunk.**
  The residual risk is presentational — do not dress a stateless call in session
  machinery, and do not let a `CONTROL_LOOP_BENCH.md` row imply this model carries
  context the way LingBot-VA and DreamZero do. It does not; it chains on decoded
  pixels, and long-rollout quality decays accordingly (§2.1 item 4).
- **Two DROID action spaces (§1.1)** will be misread as comparable policies. Any table
  that puts Cosmos 3 and DreamZero side by side must carry the EE-vs-joint-space note
  inline, not in a footnote.
- **Competitive clock, honestly:** unlike prior ports there is no first-mover window on
  the NVIDIA side — it is already served four ways. The window that exists is AMD, and
  ROCm.ai general availability was announced to begin **August 2026** (this month).
  If AMD or a neocloud publishes a Cosmos 3 ROCm row first, this port's headline claim
  narrows to control-loop metrics only. That argues for doing the MI300X leg first, or
  at least in the same session as H100.
