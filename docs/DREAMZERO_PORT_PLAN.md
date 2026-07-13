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

Key numbers, **corrected 2026-07-13 against the actual released checkpoint's
`config.json` + `experiment_cfg/{conf.yaml,metadata.json}`** (fetched on an H100
RunPod pod, no auth needed — `GEAR-Dreams/DreamZero-DROID` is public, `files_metadata`
listing + `hf_hub_download` of the three small files, no weights needed for this).
The demo/socket-server comments this section originally cited turn out to describe a
*different* config than what's actually shipped — the structural DiT numbers
(`num_frame_per_block`, `num_action_per_block`) are baked into the checkpoint's RoPE
and action-register layout, not a runtime choice, so the deployed values are the ones
that matter:

| knob | value (checkpoint-verified) | note |
|---|---|---|
| `num_frame_per_block` | **2** (not 1 — demo config differs from the shipped DROID checkpoint) | two latent frames per attention block |
| `max_chunk_size` | 4 | new, previously unknown — a DiT-level cap, distinct from `num_frame_per_block` |
| `frame_seqlen` | 880 | tokens per frame (3 cameras: 2 exterior + 1 wrist) |
| `num_action_per_block` | **24** (not 32) | == `action_horizon` (top-level `VLAConfig.action_horizon=24`) |
| `num_state_per_block` | 1 | unchanged from the original read |
| `in_dim` / `out_dim` | 36 / 16 | DiT channel dims, not previously recorded |
| action_dim (model, padded) | **32** | `WANPolicyHeadConfig.action_dim` == `max_action_dim` — zero-padded across *multiple embodiments'* action spaces jointly trained (`action_loss_embodiment_ids: [26, 17]`), **not** a DROID-specific 7-DoF width as originally guessed |
| action_dim (DROID, wire/used) | **8** = `joint_position`(7) + `gripper_position`(1) | confirmed from `experiment_cfg/conf.yaml`'s DROID transform stanza: `action_concat_order: [action.joint_position, action.gripper_position]`, occupying indices `[0:8]` of the 32-wide padded tensor (zero-padded at the end — `DreamTransform`'s `np.pad(actions, ((0,0),(0, max_action_dim - n)), "constant")`, `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py:496`). **Not** cartesian-delta as originally guessed — it's joint-space + gripper, matching the reference socket server's `_convert_action` (`joint_position`(7)+`gripper_position`(1)=8), which corroborates this independently. |
| action normalization | `mode="q99"`: `(x+1)/2*(q99-q01)+q01` (denorm), symmetric quantile per channel | **Resolves the §5 denorm risk.** Identical formula to LingBot-VA's `_denormalize_actions`. Stats live in the checkpoint's `experiment_cfg/metadata.json` → `["oxe_droid"]["statistics"]["action"]["joint_position"|"gripper_position"]["q01"|"q99"]` (flat per-channel, confirmed **not** per-horizon/stateful for this checkpoint — `relative_action_per_horizon` was a red herring, that path isn't used for `oxe_droid`). Verified against `groot/vla/data/transform/state_action.py`'s `StateActionTransform` (read for reference only — **not vendored**, reimplemented directly, same discipline as LingBot-VA). |
| context window | `max_attention_size = 21 × frame_seqlen` | ≈18.5k video tokens; cache resets when full or task/language changes (§2 has the exact reset conditions now) |
| denoise steps | `num_inference_steps = 16`, `cfg_scale = 5.0`, `sigma_shift = 5.0` | confirmed the only steps actually used — `num_inference_timesteps` (checkpoint value: 4) is dead code for this path, see §2 |
| KV cache | per layer `[2, B, L, 40, 128]` × 40 layers, **plus a full negative-prompt copy** (CFG) + 512-token cross-attn caches | bf16 |
| attention | flash-attn 3 (Hopper) / flash-attn 2 / Transformer-Engine cuDNN — **all try/except-guarded with a plain-SDPA fallback** (`wan2_1_attention.py`) | → ROCm-portable without stubbing (better than LingBot-VA, whose flash-attn import was hard) |
| dtype | bf16 end-to-end (`post_initialize` casts model/T5/CLIP/VAE) | fp32-RoPE lesson n/a |
| checkpoint size | 10 safetensors shards, ~48.7 GB total | `model.safetensors.index.json` lists them; the repo also carries a `tensorrt/` subtree (~35 GB: ONNX + a pre-built `.trt` engine) — **skip it**, not needed for the PyTorch path we're porting |

External claims (theirs, not ours): ~3 s/chunk on H100 and ~0.6 s on GB200 after
warmup **with DiT caching on**; DreamZero-Flash ~7 Hz (paper only, not released).

## 2. The inference loop (what we port)

**Correction, 2026-07-13 (re-verified against source, not the press-release-level
summary this section originally had):** the real chunk loop is one level deeper than
`ARDroidRoboarenaPolicy` — that class is the roboarena eval-harness adapter (frame
accumulation, obs/action dict format conversion) and is **never ported**, same
discipline as LingBot-VA's `VA_Server`. The real per-block work is
`GrootSimPolicy.lazy_joint_forward_causal` (`groot/vla/model/n1_5/sim_policy.py:679`)
→ `VLA.lazy_joint_video_action_causal` (`groot/vla/model/dreamzero/base_vla.py:180`,
a thin `backbone(IdentityBackbone, a no-op) + action_head` dispatch) →
`WANPolicyHead.lazy_joint_video_action` (`action_head/wan_flow_matching_action_tf.py:973`)
— **this last method is what we port**; the two wrapper layers above it are trivial
(no hydra/dataset-framework dependency survives past them — `VLA.__init__` builds
`backbone`/`action_head` from a small `config.json`-carried dict via
`hydra.utils.instantiate`, not from the GR00T-N1.5 training config tree. The original
worry that this needs the full GR00T-N1.5 data-schema/transform stack was **wrong**
for the model itself; it is only true for one piece — action denormalization, §5).

One call to `lazy_joint_video_action(backbone_output, action_input, latent_video=...)`
does what our `_DreamZeroPipeline.recondition()` + `.infer_chunk()` do as two
Protocol methods — there is no separate "just push, don't predict" entrypoint in the
reference. The mapping for Phase 1: `recondition()` should stash the given
`obs_latent`/executed actions on the session (the real push only happens inside the
next call); `infer_chunk()` does the actual work, calling
`lazy_joint_video_action(..., latent_video=<stashed obs_latent, or None for
imagination>)`. Verified mechanics inside that one call:

1. **Reset condition** — `current_start_frame` resets to 0 (not lazily on our
   `reset()` alone) whenever: this is the first call (`self.language is None`), the
   text prompt changed, **the caller passes a single-frame observation** (so our
   `reset()` naturally triggers this by construction), or
   `current_start_frame >= model.local_attn_size` (window exhaustion;
   `local_attn_size` defaults to `-1` → `max_attention_size = 21 * frame_seqlen`,
   confirming the plan's "21 frames" as the *default*, not a hardcoded constant).
2. **First call priming** (`current_start_frame == 0`): CLIP+VAE-encode the seed
   frame → `clip_feas`, `ys` (the image-conditioning tensor spanning the whole
   attention window, mask-concatenated with VAE latents); create fresh KV caches
   (`_create_kv_caches`: per-layer `[2, B, L, num_heads, head_dim]`, `L` starts at 0
   and grows — confirms the plan's cache-shape claim) + cross-attn caches (`_create_
   crossattn_caches`: per-layer `[2, B, 512, num_heads, head_dim]`) — **one cache pair
   for cond, one full duplicate for uncond** (`kv_cache_neg`/`crossattn_cache_neg`),
   confirming "plus a full negative-prompt copy for CFG". Then one **clean pass**
   (`timestep=0`, `update_kv_cache=True`, no noise) seeds the cache with the observed
   frame's real K/V; `current_start_frame` becomes 1.
3. **Recondition push** (every call where `current_start_frame != 1`): the new real
   observation frames the caller passed in this call (`videos`, VAE-encoded — or
   `latent_video` verbatim if the caller supplies it directly, the imagination path)
   are pushed into the cache via another clean pass (`update_kv_cache=True`,
   `start_frame = current_start_frame - num_frame_per_block`) — grounding on
   whatever the caller fed in, real pixels or the model's own prior prediction.
4. **Joint denoise loop** — **one loop, not two separate video/action loops as this
   section previously said.** `noise_obs` and `noise_action` both start from Gaussian
   noise (same seed=1140 — a research-code determinism quirk, not to reproduce
   exactly); `FlowUniPCMultistepScheduler` (not `FlowMatchScheduler`, which is only
   used at `.set_timesteps(1000, training=True)` during training) steps both, video
   and action on separate scheduler instances but **the same 16-step index loop** —
   `self.num_inference_steps = 16` is hardcoded at `__init__` and is what's actually
   used; **`num_inference_timesteps` from `WANPolicyHeadConfig` is read into
   `self.num_inference_timesteps` but never referenced inside
   `lazy_joint_video_action` — it is dead for this path.** (Resolves the "Phase-1
   verify item" this section used to flag as open.) Each step: one `self.model(...)`
   call per CFG context (cond, uncond — **not already batched into one forward**,
   confirming batching them is a real, available lever) with
   `kv_cache=<growing>, crossattn_cache=<fixed 512>, current_start_frame=<block
   start>, update_kv_cache=False` (the searched/noisy steps are never cached — only
   the two clean passes above ever write the cache). CFG combine:
   `flow_pred = uncond + cfg_scale * (cond - uncond)`. An optional
   `decouple_inference_noise` config rescales the video (not action) noise schedule to
   stop short of full denoise (`video_inference_final_noise`) — off by default,
   another config knob to carry, not implement first.
5. **DiT cache** (`--enable-dit-cache` / `DYNAMIC_CACHE_SCHEDULE` env, off by
   default): dynamic skip schedule — cosine similarity of the last two
   *action*-noise predictions >0.95/0.93 skips the next 4/2 DiT calls, reusing the
   last prediction verbatim (not a first-order extrapolation as this section
   previously guessed — `cache_predict_order1` exists but is not what's called from
   the skip path in `should_run_model`/`lazy_joint_video_action`). Off-by-default
   also has a static variant: `dit_step_mask` (env `NUM_DIT_STEPS`) fixed-pattern
   skip. Approximation — quality impact is ours to measure, same discipline as the
   LingBot-VA CFG-off caveat.
6. **Advance** — `current_start_frame += num_frame_per_block` after the loop (no
   pixel decode in the loop; `video_pred` returned is still latents).
7. **Parallelism:** unchanged from the original read — `ip_size ∈ {1, 2}`, ip=2 only
   splits CFG across ranks via P2P (`_exchange_predictions`), ip=1 runs both
   contexts sequentially. Single-GPU is code-supported; batching cond+uncond into one
   forward (point 4 above) is the immediate lever, now confirmed against the exact
   call site (`_run_diffusion_steps`'s `for index, prompt_emb in enumerate(context)`
   loop).
8. Also present, not ported: an optional TensorRT engine path
   (`self.trt_engine`, gated in `_run_diffusion_steps`) and Transformer-Engine
   attention — the NVIDIA-only rungs we benchmark against, not through.

## 3. Mapping onto the `InteractiveWorldModel` seam

| Seam | DreamZero realization |
|---|---|
| `reset(conditioning, params)` | cache creation + prompt/T5 + first-obs CLIP/VAE encode; `conditioning.uri` = seed image(s), prompt via params. Heavyweight state (caches, `current_start_frame`) engine-held per `session_id`, as in LingBot-VA. |
| `step(state, action)` | recondition-then-predict, both inside one `lazy_joint_video_action` call (§2): push executed actions + real observation frames for the new block, run the joint denoise loop, return the next 32-action chunk + predicted latents. `Action.space = "dreamzero_chunk_droid"`; `embodiment_id` carried per-config from the checkpoint. Action *denormalization* is **not** inside the action head (§5 risk) — it happens in the eval harness we don't port, so the engine must own it against the checkpoint's `metadata.json` stats directly. |
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
- **Phase 1 (GPU verify, H100):** vendor the minimum, now precisely scoped (§2
  correction) — `base_vla.py` (`VLA`, trivial HF `PreTrainedModel` wrapper),
  `backbone/identity.py` (no-op), `action_head/wan_flow_matching_action_tf.py`
  (`WANPolicyHead` — the real orchestration: cache creation, the joint denoise loop,
  `encode_prompt`/`encode_image`/`encode_video`), `modules/wan_video_dit_action_
  casual_chunk.py` (`CausalWanModel`), `modules/wan2_1_attention.py` (SDPA path),
  `modules/flow_unipc_multistep_scheduler.py` (the scheduler actually used at
  inference — not `flow_match_scheduler.py`, which is training-only), `modules/
  wan_video_vae.py` + `wan_video_text_encoder.py` (umT5) + `wan_video_image_encoder.py`
  (CLIP), `modules/utils.py` + `wan2_1_submodule.py`. All self-contained under
  `groot/vla/model/dreamzero/` — confirmed the GR00T-N1.5 hydra/dataset framework
  does *not* leak into this closure (§2): `WANPolicyHeadConfig` also carries
  `vl_self_attention_cfg`/`load_pretrained_det_decode_layer_path` fields that
  reference `groot.vla.model.n1_5.*` and an internal NVIDIA filesystem path, but
  checked `WANPolicyHead.__init__` line-by-line (2026-07-13) — neither is ever
  `instantiate()`d or read; genuinely dead config fields for this checkpoint, not a
  hidden dependency. Denormalization (§5, resolved) needs
  `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py` read for reference only
  (channel padding order) — not vendored, hand-rolled instead. **Do not `pip install`
  their package**: the pyproject hard-requires `tensorrt`, `ray`, `mujoco`,
  `deepspeed`, and a `gear` package, none of which the inference loop needs (and
  `tensorrt` won't install on ROCm). Own venv: torch 2.8.0, transformers 4.51.3,
  diffusers 0.30.2, py3.11 (or 3.12, both satisfy the repo's `>=3.11`), plus `einops`,
  `peft` (LoRA plumbing `WANPolicyHead.__init__` imports unconditionally),
  `hydra-core`/`omegaconf` (only for `instantiate`, not the training config tree) —
  **confirmed installable clean together** on a RunPod `runpod-torch-v280` H100 image
  (torch 2.8.0+cu128 preinstalled, reused via `--system-site-packages`), 2026-07-13.
  Checkpoint `GEAR-Dreams/DreamZero-DROID`: **confirmed public, no HF auth needed**;
  10 safetensors shards, ~48.7 GB total (`config.json` + `model.safetensors` carry the
  trained deltas; the base Wan2.1-I2V-14B-480P VAE/T5/CLIP/DiT weights are pulled
  separately from `Wan-AI/Wan2.1-I2V-14B-480P`, also public, and then overwritten
  `strict=False` — two downloads, not one). **Skip the checkpoint's `tensorrt/`
  subtree** (~35 GB: ONNX export + prebuilt `.trt` engine) — irrelevant to the PyTorch
  SDPA path. Verify: single-GPU `ip_size=1` parity vs their 2-GPU server on identical
  DROID inputs (their `test_client_AR.py` gives the harness); confirm actual bf16
  resident footprint at load — the ~48.7 GB DROID checkpoint shards likely overlap
  substantially with the base Wan2.1-I2V-14B-480P download (same DiT/VAE/T5/CLIP
  architecture, DROID shards are the fine-tuned deltas loaded `strict=False` on top),
  so don't assume the two downloads sum to resident GPU memory — measure at load.
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
- **Research code hygiene:** hardcoded inference constants (`num_inference_steps =
  16` — confirmed the only value actually used, `num_inference_timesteps` from config
  is dead code, §2; `seed = 1140`), roboarena-specific observation/action conversion
  in `ARDroidRoboarenaPolicy` — vendor components (down through `WANPolicyHead`),
  never the eval-harness wrapper above it (LingBot lesson).
- ~~Action denormalization lives outside the action head~~ **RESOLVED 2026-07-13**
  (was an open risk as of the earlier revision of this doc): denormalization happens
  in `GrootSimPolicy.unapply`, which goes through the GR00T-N1.5
  `ComposedModalityTransform`/`DatasetMetadata` stack we don't want as a dependency —
  but fetching the actual `GEAR-Dreams/DreamZero-DROID` checkpoint's
  `experiment_cfg/{conf.yaml,metadata.json}` on the H100 pod (no weights needed, just
  those two small files) confirmed it's the exact same flat per-channel q01/q99
  quantile formula LingBot-VA already reimplements
  (`(x+1)/2*(q99-q01)+q01`) — **not** the stateful `relative_action_per_horizon` path
  this doc previously worried about (that's real in the codebase, referenced in
  `sim_policy.py`, but not what `oxe_droid`'s transform stanza in `conf.yaml` uses).
  §1's table has the exact stats-dict path and channel layout
  (`joint_position`(7)+`gripper_position`(1)=8, zero-padded to the model's 32-wide
  action register). Phase 1 hand-rolls this against `metadata.json`, never importing
  `groot.vla.data.transform` — confirmed low-risk, not just hoped-for.
- **DROID embodiment semantics:** `embodiment_id` values and the 3-camera layout live
  in checkpoint/config; the engine must carry them per-config. Post-train checkpoints
  (AgiBot/YAM) change all of them, including the action-transform shape above.
- **Env conflicts:** torch 2.8.0 / transformers 4.51.3 vs our tree — own venv in
  Phase 1, same discipline as V-JEPA and LingBot-VA.
- **Competitive clock:** vLLM-Omni RFC #4127. If they ship optimized DreamZero
  serving before our Phase 2, the claim narrows from "first fast DreamZero serving"
  to "first AMD / first single-GPU / first control-loop-metrics" — still ours, but
  publish early rather than polishing.
