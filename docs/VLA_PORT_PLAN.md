# VLA Action-Token Port Plan (2026-07-25)

- **Why:** the on-thesis answer to "support LLM inference with the same stack"
  (`REVISED_STRATEGY.md`). A token-decoding VLA *is* LLM inference — but
  control-camp, so it belongs on the `InteractiveWorldModel` seam next to
  V-JEPA 2-AC and LingBot-VA, not on the co-located chat proxy
  (`docs/LLM_PROXY.md`, the *other* answer). It is the third planning regime on
  one seam (search / generate / **decode+score**) and a leaderboard row nobody
  owns: candidate-batched action-token decode on any silicon.
- **Sources:** this Phase-0 scaffold is built against the **published**
  OpenVLA / π0-FAST interfaces (papers + HF model cards), **not** a local clone
  yet — unlike the LingBot-VA plan, which cloned the repo first. Phase 1 begins
  by cloning `openvla/openvla-7b` and verifying `predict_action` against the
  code, same discipline. Everything below marked "verify" is unverified until
  then; nothing here is load-bearing on press claims.

## 1. What the model class is

A **token-decoding VLA** maps `(image(s), language instruction)` to robot
actions by **autoregressively decoding action tokens** with a KV cache — the
same machine as an LLM, pointed at a control vocabulary.

- **OpenVLA-7B** (lead, MIT, `openvla/openvla-7b`): a Prismatic VLM — Llama-2 7B
  backbone + fused DINOv2/SigLIP vision — where the 256 least-used tokenizer
  slots are overwritten as **action tokens**. A 7-DoF action (6-DoF EE delta +
  gripper) is **256-bin-per-dim discretized**, so one step decodes **7 tokens**,
  greedily, then un-discretizes via per-dataset `q01/q99` stats. ~14 GB bf16 →
  fits one L4/A100/H100/MI300X trivially. Single-step (no native chunk).
- **π0-FAST** (Apache-2.0, `openpi`): PaliGemma-3B backbone; **FAST** (DCT +
  BPE) tokenizes an action *chunk* into a variable-length token sequence →
  autoregressive decode. Chunked (`action_chunk > 1`), same seam.
- **RT-2 / others**: same shape, different backbone/tokenizer.

The serving cost center is **short-horizon autoregressive decode** (7 tokens for
OpenVLA; tens for a π0-FAST chunk) — *not* thousand-token context. That is why
paged attention / big-sequence continuous batching are unnecessary here.

## 2. The inference loop (what we port)

Per OpenVLA's `predict_action` (verify against the code in Phase 1):

1. **reset** — build the KV cache; embed the instruction + system prompt.
2. **encode observation** — processor(image) → vision tokens prepended to the
   prompt; this is the decode *prefix* (the context our seam carries).
3. **decode action chunk** — autoregressively emit `action_dim` (OpenVLA=7)
   action tokens, then **detokenize** through the per-dataset bin centers to a
   continuous action. For a **candidate set** (planning), decode `N` sequences
   that **share the prompt+vision prefix KV** — one batched forward, `N` in the
   batch dim (the lever).
4. **advance** — the executed action (and, in a world-model variant, the next
   observation) extends the context for the next decode.

Serving state = **(KV cache, prompt+vision prefix, per-dataset action stats)**
keyed by `session_id` — real session state, the object the KV-reuse lever
manipulates.

## 3. Mapping onto the `InteractiveWorldModel` seam

Landed in `models/vla.py` (Phase 0):

| Seam | VLA realization |
|---|---|
| `reset(conditioning, params)` | build KV cache + embed instruction (`config.prompt`); `conditioning.uri` = the seed image. Returns `WorldState` whose `context` is the decode prefix `(T, D)`; the KV cache lives in the pipeline keyed by `session_id`. |
| `step(state, action)` | push the executed chunk (`append_executed`), then greedily decode the next proposed chunk (`decode_action_chunk(..., 1)`), parked for a following `plan`. `Action.values` = flat `k×action_dim`, `space="vla_7d"`. |
| `plan(state, goal, horizon)` | **decode + score**: decode `plan_candidates` action-token sequences (batched, shared prefix KV — the `plan_batched` lever, the token-VLA analogue of `VJepa2ACConfig.plan_batched`), score against `goal`, return the first action of the argmin. |

**Honest caveats (recorded, not hidden):**

- **The scorer is the open modeling question, not the batching.** OpenVLA has no
  value head. `plan()`'s *throughput* win (decode N candidates in one forward)
  is real and measurable; the *selection* needs a cost source —
  `score_candidates` is the seam, but a real scorer (learned value/Q, a short
  world-model rollout, or a reward model) is Phase-1+ work. v0 scores single
  chunks and exposes the hook; it does not claim a better policy.
- **Candidate diversity needs stochastic decode.** Greedy decode makes all N
  candidates identical, so the batched lever only *matters* under sampled decode
  (temperature / top-k). Phase 1 wires sampling; the fake in `test_vla.py` fakes
  diversity to exercise the loop shape.
- **Single live branch per session** (same as LingBot-VA): the KV cache mutates
  in place; branching rollouts = forking the cache, which is the KV-reuse lever,
  not a v0 guarantee.

## 4. Phases

- **Phase 0 (this session, CPU) — DONE.** This doc + `models/vla.py` +
  `models/vla_pipeline.py` (build raises the recipe) + 14 CPU tests against a
  fake pipeline (protocol conformance, the step loop, the batched-vs-loop plan
  path, session isolation, validation). Green on `ruff`/`mypy --strict`/`pytest`.
- **Phase 1 (GPU verify).** Implement `_VLAPipeline` against real OpenVLA:
  processor, KV-cached `generate` of the action tokens, and the **action
  detokenizer** (per-dataset `q01/q99` stats carried per-config, *not*
  hardcoded — the LingBot norm-stats lesson). Clone `openvla/openvla-7b`; pin
  its transformers/timm; flash-attn optional (fall back to SDPA for the
  ROCm-portable path). Box: one H100 or MI300X (`repercep-gpu-verify-recipe` /
  `repercep-runpod-bench-platform`). Verify: `predict_action` parity through the
  seam on a BridgeData/LIBERO sample.
- **Phase 2 (bench) — the differentiator.** `scripts/bench_vla_levers.py`
  (skeleton next), mirroring `scripts/bench_cem_batched.py`: candidate-batched
  vs per-candidate action-token decode, **H100 + MI300X**. Metrics:
  planning-decisions/sec, decode latency per candidate-set, KV memory/session, N
  resident sessions/GPU. This is the row neither vLLM nor NIM optimizes.
- **Phase 3 (serving).** Wire into `/v2/world/session` — **zero serving changes
  needed**: the WebSocket is already engine-agnostic (`active_interactive`), so
  passing a `VLAEngine` as `interactive_engine` serves it. Multi-session
  resident = the 192 GB MI300X "runs what H100 can't" story.

## 5. Risks

- **Scorer, not decoder, is the hard part** (see §3). If no cheap scorer beats
  greedy `step`, `plan` is a throughput demo, not a better policy — call that
  honestly; the *serving-lever* result stands regardless.
- **transformers version pin** conflicts with the Cosmos/Wan/LingBot trees —
  Phase 1 runs in its own venv (same discipline as every other GPU port) before
  any dependency unification.
- **Action-stat provenance** — normalization stats are per-dataset; the engine
  carries them per-config (`VLAConfig`), and a wrong stat set is a silent
  wrong-numbers bug, so Phase 1 asserts them against the checkpoint card.
- **π0-FAST divergence** — variable-length FAST tokens mean `action_chunk` is
  not fixed per step; keep the pipeline's `decode_action_chunk` return shape the
  seam of record, not the token count.
