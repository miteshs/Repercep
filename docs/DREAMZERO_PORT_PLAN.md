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
5. **DiT cache — correction, GPU-verified 2026-07-13: on by default, not off.**
   Two independent skip mechanisms share `should_run_model`: a *dynamic* schedule
   (`DYNAMIC_CACHE_SCHEDULE` env, defaults `False`) — cosine similarity of the last
   two *action*-noise predictions >0.95/0.93 skips the next 4/2 DiT calls, reusing
   the last prediction verbatim (not a first-order extrapolation as this section
   previously guessed — `cache_predict_order1` exists but isn't called from this
   path) — and a *static* fallback, `dit_step_mask`, keyed by `NUM_DIT_STEPS` (env,
   **defaults to `8`, unconditionally — not gated behind any opt-in flag**).
   `should_run_model` returns the static mask whenever the dynamic schedule is off
   (the default), so **a fresh `WANPolicyHead()` skips 8 of 16 DiT calls out of the
   box** — confirmed empirically: the first real forward on the H100 logged
   `"DIT Compute Steps 8 steps"` out of 16 scheduler iterations, with no env vars
   set. This means the reference's own default configuration is *already* running
   the approximation — "baseline vs DiT-cache-off" isn't the reference's own
   default; measuring true full-16-step baseline needs `NUM_DIT_STEPS=16`
   explicitly. Quality impact still ours to measure, same discipline as the
   LingBot-VA CFG-off caveat — but the baseline to measure *against* just changed.
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

## 2b. First real forward pass — GPU-verified 2026-07-13 (H100, RunPod)

**A synthetic-input forward through the actual 16.5B-param `DreamZero-DROID`
checkpoint ran end-to-end successfully** (`VLA` → `WANPolicyHead.lazy_joint_video_
action`, called directly — no vendoring needed, just `sys.path.insert` on the raw
research clone, exactly the LingBot-VA `import wan_va` pattern; `pip install`-free,
so none of the heavy pyproject deps triggered). This is the first hard evidence the
port plan's model-agnostic engine (`DreamZeroEngine`) sits on a real, loadable,
runnable model — not just a plausible-sounding plan. Not a correctness/parity check
(synthetic random-pixel input, no real DROID episode or reference-server comparison
— that needs `test_client_AR.py` + real eval data, still open) but a genuine
does-it-run-with-right-shapes gate, and it closed clean:

- **Loads with zero missing/unexpected keys** against `strict=False` — confirms
  `train_architecture="full"` really does mean the DROID checkpoint's shards fully
  cover the 16.48B-param model (`skip_component_loading=True` override, avoiding a
  wasted 65.6 GB Wan2.1 base-DiT download, was correct and necessary — the reference
  default `skip_component_loading=False` would download it needlessly since DROID's
  own shards overwrite it anyway).
- **Peak HBM: 42.8 GiB at load, 49.5 GiB after one chunk's forward** (bf16) — the
  plan's earlier ~42 GB estimate for load lands almost exactly on the measured
  number.
- **First-chunk latency: 3.77 s, in pure eager mode** (`TORCHDYNAMO_DISABLE=1` —
  their scheduler's `torch.compile` hits `FailOnRecompileLimitHit` under a
  single-chunk smoke test because internal history tensors change rank step to
  step, tripping Dynamo's `recompile_limit`; not a correctness issue, an
  environment knob — Phase 2's compile-on/off lever, not resolved here). This
  includes one-time first-call overhead (text/image encode, KV cache creation) —
  directly comparable in kind, not yet magnitude, to their "~3 s/chunk H100" claim,
  and notably **already close to it running in eager mode** (their number
  presumably includes `torch.compile` warmup benefit, ip=2 CFG split, and the
  default 8-of-16 DiT-cache skip — see point 5's correction above, which this same
  run empirically confirmed: `"DIT Compute Steps 8 steps"` logged with zero env
  vars set).
- **Output shapes all cross-check against the confirmed config numbers**:
  `action_pred` `(1, 24, 32)` = `(B, action_horizon, action_dim)`; `video_pred`
  `(1, 16, 3, 44, 80)` = `(B, out_dim, frames, H_latent, W_latent)` (the method's
  final `output.transpose(1, 2)` swaps frames and channels back for the caller)
  where 3 frames = 1 primed frame + `num_frame_per_block=2` newly generated, 16 =
  `out_dim` (VAE latent channels), 44×80 = 352×640 canvas ÷ 8 (VAE spatial
  downsample).
  `current_start_frame` advanced 0→1 (priming pass)→3 (+`num_frame_per_block`),
  exactly as §2 describes.

**New finding while constructing synthetic inputs — `max_state_dim=64` is
distinct from `action_dim=32`:** the state encoder
(`MultiEmbodimentActionEncoder`, keyed on `embodiment_id` exactly like the action
encoder) expects the state tensor zero-padded to width **64**
(`WANPolicyHeadConfig.max_state_dim`), not 32 or 8 — confirmed by a `torch.bmm`
shape-mismatch crash (`expected [1,8], got [1,64]`) that pinned the number exactly.
DROID's real state channels are the same 8 as its actions
(`state_concat_order: [state.joint_position, state.gripper_position]`, per
`conf.yaml`), zero-padded to 64, mirroring the 8→32 pad for actions but with a
different max width. `DreamZeroConfig` needs a `max_state_dim` field distinct from
`action_dim` (Phase-1 pipeline TODO — not yet added to the engine config).

**Other inputs resolved and verified working, not just planned:**
- **`embodiment_id = 17`** for `oxe_droid` — read directly from the checkpoint's
  `experiment_cfg/conf.yaml` `embodiment_tag_mapping` dict (a fixed training-time
  registry spanning all co-trained embodiments; `action_loss_embodiment_ids: [26,
  17]` in `config.json` = agibot(26) + oxe_droid(17), confirming this checkpoint is
  co-trained across both, consistent with the sibling `GEAR-Dreams/DreamZero-AgiBot`
  checkpoint existing). This is **not optional to get right**: `embodiment_id`
  indexes a per-embodiment weight matrix (`CategorySpecificLinear.forward`:
  `self.W[cat_ids]`) — any in-range wrong value would run without error and produce
  silently-wrong output routed through another embodiment's trained weights. This
  was the main reason Phase 1 stopped short of guessing it and instead read the
  actual registry.
- **Multi-camera layout: a 2×2 pixel grid, not the spatial-width-concat this plan
  first assumed (LingBot-VA's pattern).** `groot/vla/model/dreamzero/transform/
  dreamzero_cotrain.py`'s `_prepare_video` (DROID-specific branch, `v >= 3`):
  canvas `(2h, 2w)` where the **wrist view fills the entire top row** (nearest-
  neighbor doubled in width, `np.repeat(wrist, 2, axis=-1)`) and **left/right
  exterior views fill the bottom row** (`[h:, :w]`, `[h:, w:]`). Per-camera
  resolution 176×320 (`h=176, w=320`, `conf.yaml`'s `VideoResize`) → canvas
  352×640 → **VAE/patch-downsampled to 44×80 = 3520 spatial positions... divided by
  patch stride → 880 = `frame_seqlen`, an exact arithmetic match** that
  independently confirms this layout is right (352/16 × 640/16 = 22×40 = 880, or
  equivalently the /8 VAE downsample × /2 patch stride used above). Confirmed
  working end-to-end in the smoke test.
- **Tokenizer: `google/umt5-xxl`'s files from the `Wan-AI/Wan2.1-I2V-14B-480P`
  repo** (not the DROID repo — DROID only ships the DiT/action-head deltas). The
  directory has no model `config.json` (tokenizer-only), so `AutoTokenizer.
  from_pretrained` fails (`AutoConfig` can't resolve a `model_type`) — load via
  `PreTrainedTokenizerFast(tokenizer_file=<tokenizer.json>)` directly instead,
  bypassing `AutoConfig` entirely. `padding="max_length", max_length=512` (matches
  LingBot-VA's T5 padding convention).

**Not yet done:** the actual `repercep.models.dreamzero_pipeline` module (this was
all done in a standalone smoke-test script, not yet wired into
`_DreamZeroPipeline`); the KV-cache **session-swap mechanism** (`WANPolicyHead` has
*no* named/multi-session cache API like LingBot-VA's `wan_va` transformer — its
`kv_cache1`/`current_start_frame`/`language`/`ys`/`clip_feas` are plain mutable
instance attributes on one `nn.Module`, not keyed by session. Serving N concurrent
sessions on one resident model needs the pipeline to save/restore these attributes
per `session_id` around each call — new architectural finding, not previously
known, and a real difference from LingBot-VA's native multi-session support);
`encode_observation`'s real image loading (the smoke test used random pixels, not
a decoded image file); real-vs-reference parity (needs `test_client_AR.py` + actual
DROID eval data).

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
