# DreamZero Port Plan (2026-07-12)

- **Why:** next-ports research 2026-07-12 ranked DreamZero #1: the third model on the
  `InteractiveWorldModel` seam, same policy regime as LingBot-VA but attached to the
  NVIDIA GEAR / GR00T 2 ecosystem (GR00T 2 is publicly built on DreamZero research).
  The reference release is the familiar open-slow/closed-fast ladder: 2-GPU torchrun
  reference at ~3 s/chunk on H100, TRT/Transformer-Engine rungs NVIDIA-only, and the
  paper's 7 Hz "DreamZero-Flash" not in the release. **Clock:** vLLM-Omni RFC #4127 is
  an open DreamZero optimization roadmap — the NVIDIA-side serving gap is being
  absorbed; the single-GPU / AMD / control-loop-metrics gap is ours to take first.
- **Sources verified locally:** `dreamzero0/dreamzero` cloned and read (scratchpad;
  key files `socket_test_optimized_AR.py` — the reference WebSocket policy server —
  `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`,
  `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`,
  `docs/WAN22_BACKBONE.md`, configs under `groot/vla/configs/`). Everything below is
  from the code, not press. Apache-2.0 (LICENSE + NVIDIA COPYRIGHT header).

## 1. What the model is (verified against the code)

DreamZero is an autoregressive **world-action model** (WAM): a **CausalWanModel** —
Wan2.1-I2V-14B-480P DiT (dim 5120, 40 layers, 40 heads, ffn 13824) extended with
**action/state registers** — over Wan2.1 VAE latents, trained with flow matching,
generating **block-by-block** with an explicit per-layer KV cache. The DiT sees one
sequence `[video_tokens | action_register]` under blockwise-causal masking (3D RoPE
for video, 1D for the register) and jointly denoises **video noise + action noise**;
the register slice is decoded by `action_decoder` into the action chunk. The model is
its own policy — one block of frames in, one action chunk out.

Checkpoints (HF, public): `GEAR-Dreams/DreamZero-DROID` (14B, inference-ready) and
`GEAR-Dreams/DreamZero-AgiBot` (~45 GB, for post-training new embodiments — ~30 min of
play data per their README). A **Wan2.2-TI2V-5B backbone** path exists in the repo
(`docs/WAN22_BACKBONE.md`, `serve_dreamzero_wan22.py`) — same `CausalWanModel` class,
config-switched — but **train-your-own, no public 5B checkpoint**; the port target is
the 14B DROID model, with 5B noted as a future cheap-serving rung if weights appear.

Key numbers from the 14B config (`wan_flow_matching_action_tf.yaml` + server):

| knob | value | note |
|---|---|---|
| `num_frame_per_block` | 1 | one latent frame per attention block |
| `frame_seqlen` | 880 | tokens per frame (3 cameras: 2 exterior + 1 wrist) |
| `num_action_per_block` / `num_state_per_block` | 32 / 1 | one chunk = 32 actions |
| context window | `max_attention_size = 21 × frame_seqlen` | ≈18.5k video tokens; cache resets when full or task/language changes |
| denoise steps | `num_inference_steps = 16`, `cfg_scale = 5.0`, `sigma_shift = 5.0` | hardcoded in the action head; yaml also carries `num_inference_timesteps: 4` ("not used during training") — **which loop uses which is a Phase-1 verify item** |
| KV cache | per layer `[2, B, L, 40, 128]` × 40 layers, **plus a full negative-prompt copy** (CFG) + 512-token cross-attn caches | bf16 |
| attention | flash-attn 3 (Hopper) / flash-attn 2 / Transformer-Engine cuDNN — **all try/except-guarded with a plain-SDPA fallback** (`wan2_1_attention.py`) | → ROCm-portable without stubbing (better than LingBot-VA, whose flash-attn import was hard) |
| dtype | bf16 end-to-end (`post_initialize` casts model/T5/CLIP/VAE) | fp32-RoPE lesson n/a |

External claims (theirs, not ours): ~3 s/chunk on H100 and ~0.6 s on GB200 after
warmup **with DiT caching on**; DreamZero-Flash ~7 Hz (paper only, not released).

## 2. The inference loop (what we port)

From `ARDroidRoboarenaPolicy` / the action head:

1. **Reset** — create per-layer KV caches + cross-attn caches (×2 for CFG), encode the
   text prompt (umT5), CLIP-encode + VAE-encode the first observation. Session state =
   (KV caches ×2, cross-attn caches ×2, `current_start_frame`, prompt embeds, frame
   buffers). Server accumulates `FRAMES_PER_CHUNK = 4` frames per call after the first.
2. **Per decision** — encode the new block's frames to latents; run the 16-step
   flow-matching loop: each step calls the DiT once per prompt context (cond + uncond)
   with `kv_cache`, `crossattn_cache`, `current_start_frame`; KV is written only on the
   designated update step. Scheduler steps the noisy action toward clean → one
   denormalized 32-action chunk.
3. **DiT cache** (`--enable-dit-cache`, off by default): dynamic skip schedule —
   cosine similarity of the last two action-noise predictions >0.95/0.93 skips the next
   4/2 DiT calls, filling in with a first-order extrapolation
   (`cache_predict_order1`, damping 0.25). This is their headline latency lever and it
   is an **approximation** — quality impact is theirs to claim, ours to measure
   (same discipline as the LingBot-VA CFG-off caveat).
4. **Advance** — `current_start_frame += num_frame_per_block`; cache reset on task
   change or window exhaustion (21 frames). No pixel decode in the loop.
5. **Parallelism:** the reference launch is `torchrun --nproc_per_node=2`, but
   `parallelize()` asserts `ip_size ∈ {1, 2}` and **ip=2 only splits CFG** — rank 0
   conditional, rank 1 unconditional, exchanged via P2P (`_exchange_predictions`).
   On ip=1 the two contexts run **sequentially** in a Python loop. So "minimum 2
   GPUs" is a CFG-split convenience, not a sharding requirement — **single-GPU is
   code-supported, and batching cond+uncond into one forward (what LingBot-VA's
   levers already do) is the immediate, obvious lever.**
6. Also present, not ported: optional TensorRT engine path (`ENABLE_TENSORRT`,
   `ar_14B`) and Transformer-Engine attention — the NVIDIA-only rungs we benchmark
   against, not through. `torch.compile` is applied to T5/CLIP/VAE-encode only; the
   code comment says Dynamo fullgraph fails on the DiT's shape variation — consistent
   with our LingBot-VA negative finding.

## 3. Mapping onto the `InteractiveWorldModel` seam

| Seam | DreamZero realization |
|---|---|
| `reset(conditioning, params)` | cache creation + prompt/T5 + first-obs CLIP/VAE encode; `conditioning.uri` = seed image(s), prompt via params. Heavyweight state (caches, `current_start_frame`) engine-held per `session_id`, as in LingBot-VA. |
| `step(state, action)` | recondition-then-predict: push executed actions + real observation frames for the new block, run the denoise loop, return the next 32-action chunk + predicted latents. `Action.space = "dreamzero_chunk_droid"`; norm stats + `embodiment_id` carried per-config from the checkpoint, not hardcoded. |
| `plan(state, goal, horizon)` | the model is its own policy (policy regime, same as LingBot-VA): `plan` returns the model's proposed chunk; goal conditioning is the text prompt at reset. No CEM. |

**Seam caveat (same as LingBot-VA v0, recorded):** the native KV cache mutates in
place per session — branching = forking the cache; v0 keeps one live branch per
session. The CEM-batched growing-window work (ADR-0009) is the eventual fix; the
engine layer should be shared with `lingbot_va` where the chunk-loop shape allows —
both are "chunked VA over a Wan-family causal DiT with named KV cache." A common base
is a Phase-0 design question, not a commitment.

## 4. Phases

- **Phase 0 (CPU):** this doc + `models/dreamzero.py` scaffold — chunk-loop layer
  against an injectable pipeline Protocol, unit-tested with fakes (the
  `vjepa2_ac.py` / `lingbot_va.py` pattern). Weight-load raises `NotImplementedError`
  with the port recipe in the docstring.
- **Phase 1 (GPU verify, H100):** vendor the minimum — `CausalWanModel` +
  `wan2_1_attention` (SDPA path) + schedulers + VAE/T5/CLIP encoders from
  `groot/vla/model/dreamzero/modules/` — **do not `pip install` their package**: the
  pyproject hard-requires `tensorrt`, `ray`, `mujoco`, `deepspeed`, and a `gear`
  package, none of which the inference loop needs (and `tensorrt` won't install on
  ROCm). Own venv: torch 2.8.0, transformers 4.51.3, diffusers 0.30.2, py3.11.
  Checkpoint `GEAR-Dreams/DreamZero-DROID`. Verify: single-GPU `ip_size=1` parity vs
  their 2-GPU server on identical DROID inputs (their `test_client_AR.py` gives the
  harness); resolve the 16-vs-4 step question; confirm bf16 weights footprint
  (est. ~42 GB: 28 DiT + ~11 umT5-xxl + CLIP + VAE — estimate, measure at load).
- **Phase 2 (bench):** RunPod H100 + MI300X rows in `bench_control_loop` /
  `CONTROL_LOOP_BENCH.md`: warm chunk latency (vs their ~3 s H100 claim),
  decisions/sec under state carryover, **KV memory per session** — back-of-envelope
  says ~800 KB/token × 18.5k tokens × 2 (CFG) ≈ **~30 GB/session at full window**,
  which would make resident-sessions/GPU the headline: H100 holds ~1, MI300X ~5.
  If that holds under measurement it is the cleanest "runs what H100 can't" story yet
  (strategy T1.3) — but it is arithmetic until measured. Levers ladder, in order:
  CFG cond+uncond batched into one forward (vs their sequential/2-GPU), DiT-cache
  on/off (quality-flagged), step count, KV window (`local_attn_size`), SDPA vs ROCm
  AITER. MI300X via the guarded SDPA fallback; no stub needed.
- **Phase 3 (serving):** wire into `/v2/world/session` next to LingBot-VA; the
  session-state object already generalized to "context + engine-held cache" in the
  LingBot port. N-session load test per the LingBot-VA levers methodology.

## 5. Risks

- **Per-session KV memory (~30 GB est.) is 5× LingBot-VA's** — could flip the
  multi-session economics story from "many sessions/GPU" to "MI300X-only
  multi-session." Measure before claiming either.
- **ip=1 is code-supported but likely under-tested upstream** (their launch docs are
  2-GPU only) — Phase-1 parity check is the gate, not an afterthought.
- **Research code hygiene:** hydra config sprawl, hardcoded inference constants
  (`num_inference_steps = 16`, `seed = 1140`), roboarena-specific observation/action
  conversion in the server — vendor components, never their server (LingBot lesson).
- **DROID embodiment semantics:** norm stats, `embodiment_id` values, and the
  3-camera layout live in checkpoint/config; the engine must carry them per-config.
  Post-train checkpoints (AgiBot/YAM) change all of them.
- **Env conflicts:** torch 2.8.0 / transformers 4.51.3 vs our tree — own venv in
  Phase 1, same discipline as V-JEPA and LingBot-VA.
- **Competitive clock:** vLLM-Omni RFC #4127. If they ship optimized DreamZero
  serving before our Phase 2, the claim narrows from "first fast DreamZero serving"
  to "first AMD / first single-GPU / first control-loop-metrics" — still ours, but
  publish early rather than polishing.
