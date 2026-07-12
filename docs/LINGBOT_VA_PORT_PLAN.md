# LingBot-VA 2.0 Port Plan (2026-07-11)

- **Why:** `COMPETITIVE_RESPONSE_REACTOR_2026_07.md` §4 item 3 — the second model on
  the `InteractiveWorldModel` seam. Kills the V-JEPA-AC single-model-bet risk
  (`REVISED_STRATEGY.md` §5) and positions Repercep as the self-host, any-silicon
  runtime for the video-action model class Reactor would host in the cloud.
- **Sources verified locally:** `robbyant/lingbot-va` cloned and read (scratchpad;
  key file `wan_va/wan_va_server.py` — the reference single-GPU inference server),
  configs under `wan_va/configs/`, README checkpoint table. Everything below is from
  the code, not press.

## 1. What the model is (verified against the code)

LingBot-VA 2.0 is an autoregressive **video-action** world model: a Mixture-of-
Transformers DiT (video expert + action expert, shared backbone) over **Wan2.2 VAE
latents**, trained with flow matching, generating **chunk-by-chunk** with an explicit
named **KV cache**. Per chunk it (a) denoises a video-latent block, then (b) denoises
an action block — both conditioned on the cached causal context. Apache-2.0.

Checkpoints (HF + ModelScope): `robbyant/lingbot-va-base`,
`…-posttrain-robotwin`, `…-posttrain-libero-long`. The checkpoint directory is a
Wan2.2-style bundle: `vae/`, `tokenizer/`, `text_encoder/` (T5), `transformer/`.

Key numbers from `va_demo_cfg` (single-GPU demo config):

| knob | value | note |
|---|---|---|
| resolution | 256×256, latent 16×16 per camera | 2 cameras concat on width |
| `frame_chunk_size` | 4 | latent frames per chunk |
| `action_dim` / `action_per_frame` | 30 / 8 | one chunk = **32 actions × 30 dims** |
| `attn_window` | 30 | KV-cache window, in chunks |
| denoise steps | **5 video + 10 action** per chunk | flow matching, `snr_shift` 5.0/1.0 |
| CFG | video 5.0, action 1.0 | CFG doubles the cache batch |
| dtype / attention | bf16, `attn_mode ∈ {torch, flashattn, flex}` | **`torch` (SDPA) is a first-class path → ROCm-portable** |
| VRAM | ~18–24 GB with VAE/T5 CPU-offload | fits L4/A100-40G; H100/MI300X trivially |

Action channels are quantile-normalized (`q01/q99` per-channel stats in the config);
the 30-dim layout is dual-arm EE/joints/gripper, task configs mask to
`used_action_channel_ids`.

## 2. The inference loop (what we port)

From `VA_Server` (`wan_va/wan_va_server.py`):

1. **`_reset(prompt)`** — clears/creates the named transformer KV cache
   (`create_empty_cache(name, attn_window, latent_tokens_per_chunk,
   action_tokens_per_chunk, batch=2 if CFG)`), clears the **streaming VAE** cache,
   encodes the text prompt (T5).
2. **`_encode_obs(obs)`** — per-camera frames → resize → streaming Wan-VAE encode →
   normalized latents, cameras concatenated on width.
3. **`_infer(obs, frame_st_id)`** — one chunk: sample noise `(1, 48, chunk, H_lat,
   W_lat)` + `(1, 30, chunk, 8, 1)`; run the video flow-matching loop (transformer
   called with `action_mode=False`, `update_cache=1` on the last step), then the
   action loop (`action_mode=True`); first frame is clamped to the encoded first
   observation on chunk 0. Returns denormalized actions + latents. **No pixel decode
   in the loop** — decode is a separate, optional `vae.decode` at the end.
4. **`_compute_kv_cache(obs)`** — the closed-loop recondition: after the robot
   executes, the *real* observation latents + *executed* actions are pushed into the
   KV cache (`update_cache=2`), replacing the model's imagined rollout
   (`clear_pred_cache` first). `frame_st_id` advances by the consumed frames.

So the serving state = **(named KV cache in the transformer, streaming-VAE cache,
frame_st_id, prompt embeds)** — real session state, exactly what our
multi-session/world-state serving story needs, and the object our KV-reuse lever
manipulates.

## 3. Mapping onto the `InteractiveWorldModel` seam

| Seam | LingBot-VA realization |
|---|---|
| `reset(conditioning, params)` | `_reset(prompt)` + encode initial obs; `conditioning.uri` = the seed image(s), prompt via conditioning/params. Returns `WorldState` whose `context` is the encoded obs-latent block `(T, D)`-flattened; heavyweight state (KV caches, `frame_st_id`) lives in the engine keyed by `session_id`. |
| `step(state, action)` | recondition-then-predict: push the executed action chunk + (when available) the real obs into the cache (`_compute_kv_cache`), infer the next latent chunk. `Action.values` = a flat chunk (`k×30`), `space="lingbot_va_chunk_30d"`. |
| `plan(state, goal, horizon)` | **The model is its own policy** — the action-denoise loop *is* the planner. `plan` returns the first action of the model's proposed chunk. Goal conditioning is the **text prompt** (set at reset), not a latent goal embedding; a `goal`-tensor→prompt-embed path is honest future work. No CEM needed — this is a *different planning regime* on the same seam, which is exactly the dual-regime story. |

**Seam caveat (recorded, not hidden):** `step`'s "previous state is not mutated, so a
planner can branch rollouts" does not hold for v0 — the native KV cache mutates in
place per session. Branching = forking the KV cache, which is precisely the
**KV-reuse mechanism** of the latency workstream; v0 documents the limitation and the
serving layer keeps one live branch per session.

## 4. Phases

- **Phase 0 (this session, CPU):** scoping (this doc) + `models/lingbot_va.py`
  scaffold — the model-agnostic chunk-loop layer against an injectable pipeline
  Protocol, unit-tested on CPU with fakes (the `vjepa2_ac.py` pattern). Weight-load
  raises `NotImplementedError` with the port recipe in the docstring.
- **Phase 1 (GPU verify):** implement `_LingBotVAPipeline` against the real
  components (`load_vae/load_tokenizer/load_text_encoder/load_transformer` +
  `FlowMatchScheduler` + the two denoise loops — vendored minimally, or the repo
  used as a dependency). Box: GCP A100-40G or RunPod H100 (per
  `repercep-gpu-verify-recipe` / `repercep-runpod-bench-platform`). Deps (their
  `requirements.txt`, verified): torch 2.9.0, transformers **4.55.2**, diffusers
  0.36.0, numpy 1.26.4. flash-attn is never *called* with `attn_mode="torch"` but
  `wan_va/modules/model.py` hard-imports it (`try: flash_attn_interface / except:
  flash_attn`) — stub `flash_attn.flash_attn_func` on the box instead of building
  the wheel. Demo config key: `demo_i2av` (`VA_CONFIGS`), single GPU via
  `NGPU=1 CONFIG_NAME=demo_i2av script/run_launch_va_server_sync.sh`; the
  checkpoint bundle is **24.4 GB** on HF (public, ungated). Verify: i2va demo
  parity through the Repercep seam.
- **Phase 2 (bench):** RunPod H100 + MI300X rows — chunk latency (their published
  H-series number: 142 ms/chunk, 225 Hz async), steps/sec under state carryover,
  KV-cache memory per session, N resident sessions per GPU. MI300X via
  `attn_mode="torch"`; the fp32-RoPE lesson from V-JEPA does not apply (this stack
  is bf16-native).
- **Phase 3 (serving):** wire into `/v2/world/session` next to V-JEPA-AC; the
  session state object generalizes from "context tensor" to "context + engine-held
  cache". Multi-session resident world-states = the 192 GB MI300X story.

## 5. Risks

- **transformers 5.0 / torch 2.9 vs our tree** — likely env conflicts with the
  Cosmos/Wan paths; Phase 1 runs in its own venv (same discipline as the V-JEPA
  GPU verify) before any dependency unification.
- **Their repo is research code** (in-place config mutation, asserts, async saves in
  the hot loop) — vendor the minimum, don't import the server.
- **Action-space semantics** (quantile stats are per-task) — post-train checkpoints
  ship their own norm stats; the engine must carry them per-config, not hardcode.
- **`generate()` deletes the transformer after use** (research hygiene) — do not
  reuse their driver; only their components.
