"""The model-specific LingBot-VA pipeline — Phase 1 of the port.

Implements the :class:`~mirage.models.lingbot_va._VAPipeline` Protocol against
the real components, treating the research repo (``robbyant/lingbot-va``,
Apache-2.0) as a dependency rather than vendoring 2k+ lines: it must be
importable as ``wan_va`` (clone + ``sys.path``, or ``pip install -e``). What
this module owns is the **session-keyed orchestration** the reference
``VA_Server`` doesn't have:

- **multi-session KV state**: the MoT transformer's cache protocol is keyed by
  ``cache_name`` — we key it by Mirage ``session_id``, so one resident model
  serves N independent world-model sessions (the resident-sessions metric of
  ``docs/CONTROL_LOOP_BENCH.md``);
- per-session streaming-VAE caches and prompt embeddings;
- none of the reference server's hot-loop side effects (async ``.pt`` saves,
  experiment directories, ``del``-on-exit teardown).

The denoise loops, mesh-id/patch plumbing, and quantile action normalization
follow ``wan_va/wan_va_server.py`` (read + verified 2026-07-11; the H100 run in
``docs/LINGBOT_VA_ON_H100.md`` is the reference-stack behavior this pipeline
must reproduce through the seam — that comparison is the Phase-1 GPU verify).

flash-attn note: ``wan_va.modules.model`` hard-imports ``flash_attn`` but the
``attn_mode="torch"`` (SDPA) path never calls it; :func:`_ensure_flash_attn_stub`
installs an import-time shim so the dependency stays optional (and the ROCm
door stays open).

GPU-only: nothing here runs without the checkpoint bundle and a device. The
engine's unit tests inject a fake pipeline instead (``tests/test_lingbot_va.py``).
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from mirage.backend.protocol import Backend
    from mirage.models.lingbot_va import LingBotVAConfig
    from mirage.runtime.types import ConditioningInput

_IMPORT_HELP = (
    "LingBot-VA pipeline needs the research repo importable as 'wan_va' "
    "(git clone https://github.com/robbyant/lingbot-va and add it plus its "
    "wan_va/ dir to sys.path) and the checkpoint bundle downloaded — "
    "see docs/LINGBOT_VA_PORT_PLAN.md §4 and docs/LINGBOT_VA_ON_H100.md."
)


def _ensure_flash_attn_stub() -> None:
    """Install an import shim for ``flash_attn`` if it isn't installed.

    ``attn_mode='torch'`` never calls it; the stub only satisfies the
    module-level import in ``wan_va.modules.model``.
    """
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        stub = types.ModuleType("flash_attn")

        def flash_attn_func(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("flash_attn stub: attn_mode='torch' should never reach here")

        stub.flash_attn_func = flash_attn_func  # type: ignore[attr-defined]
        sys.modules["flash_attn"] = stub


def build_pipeline(backend: Backend, config: LingBotVAConfig) -> LingBotVAPipeline:
    """Import the research package and construct the real pipeline."""
    _ensure_flash_attn_stub()
    try:
        import wan_va  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only w/o the repo
        raise RuntimeError(_IMPORT_HELP) from exc
    return LingBotVAPipeline(backend, config)


class LingBotVAPipeline:
    """Session-keyed LingBot-VA inference on one resident model.

    Loads the Wan2.2 bundle once (VAE, T5 tokenizer/encoder, MoT transformer
    with ``attn_mode`` from config) and serves any number of sessions, each
    owning a named transformer KV cache (``cache_name == session_id``), a
    streaming-VAE cache, and prompt embeddings.
    """

    def __init__(self, backend: Backend, config: LingBotVAConfig) -> None:
        import os

        import torch
        from wan_va.modules.utils import (
            load_text_encoder,
            load_tokenizer,
            load_transformer,
            load_vae,
        )
        from wan_va.utils import FlowMatchScheduler

        self._config = config
        self._device = backend.torch_device(config.device_index)
        self._dtype = getattr(torch, config.dtype)
        root = config.repo if os.path.isdir(config.repo) else _download_bundle(config.repo)

        self._vae = load_vae(os.path.join(root, "vae"), torch_dtype=self._dtype, torch_device="cpu")
        self._tokenizer = load_tokenizer(os.path.join(root, "tokenizer"))
        self._text_encoder = load_text_encoder(
            os.path.join(root, "text_encoder"), torch_dtype=self._dtype, torch_device="cpu"
        )
        self._transformer = (
            load_transformer(
                os.path.join(root, "transformer"),
                torch_dtype=self._dtype,
                torch_device=self._device,
                attn_mode=config.attn_mode,
            )
            .to(self._dtype)
            .to(self._device)
            .eval()
            .requires_grad_(False)
        )

        self._scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self._action_scheduler = FlowMatchScheduler(shift=1.0, sigma_min=0.0, extra_one_step=True)
        self._task = _task_config(config)
        # (C, h, w) of one latent frame, fixed after the first encode; used to
        # flatten/unflatten between the pipeline's 5-D latents and the seam's
        # (T, D) ``WorldState.context``.
        self._latent_geom: tuple[int, int, int] | None = None
        self._sessions: dict[str, dict[str, Any]] = {}

    # --- _VAPipeline protocol ---

    def reset(self, session_id: str, prompt: str | None) -> None:
        from wan_va.modules.utils import WanVAEStreamingWrapper

        cfg, task = self._config, self._task
        use_cfg = cfg.guidance_scale > 1 or cfg.action_guidance_scale > 1
        latent_h = task.height // 16
        latent_w = task.width // 16 * len(task.obs_cam_keys)
        patch = task.patch_size
        latent_tokens = (cfg.frame_chunk_size * latent_h * latent_w) // (
            patch[0] * patch[1] * patch[2]
        )
        action_tokens = cfg.frame_chunk_size * cfg.action_per_frame
        self._transformer.clear_cache(session_id)
        self._transformer.create_empty_cache(
            session_id,
            cfg.attn_window,
            latent_tokens,
            action_tokens,
            dtype=self._dtype,
            device=self._device,
            batch_size=2 if use_cfg else 1,
        )
        embeds = negative = None
        if prompt is not None:
            embeds, negative = _encode_prompt(
                self._tokenizer,
                self._text_encoder,
                prompt,
                use_cfg,
                self._device,
                self._dtype,
            )
        self._sessions[session_id] = {
            "vae_wrap": WanVAEStreamingWrapper(self._vae),
            "prompt_embeds": embeds,
            "negative_embeds": negative,
            "use_cfg": use_cfg,
            "latent_hw": (latent_h, latent_w),
            "init_latent": None,
        }

    def encode_observation(self, conditioning: ConditioningInput) -> torch.Tensor:
        """Streaming-VAE encode of the seed observation → flattened ``(T, D)``.

        ``conditioning.uri`` is a directory holding one ``<cam_key>.png`` per
        camera (the reference demo layout). The 5-D latent is cached on the
        *most recently reset* session (reset → encode is the seam's call
        order) and returned flattened for ``WorldState.context``.
        """
        latent5d = _encode_obs_dir(self, conditioning)
        session = self._sessions[next(reversed(self._sessions))]
        session["init_latent"] = latent5d
        c, t, h, w = latent5d.shape[1:]
        self._latent_geom = (int(c), int(h), int(w))
        return latent5d[0].permute(1, 0, 2, 3).reshape(int(t), -1)

    def infer_chunk(
        self, session_id: str, frame_st_id: int, init_latent: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One chunk: video denoise loop, then action denoise loop.

        Follows ``VA_Server._infer`` (CFG repeat, ``update_cache=1`` on each
        loop's last step, first-frame clamp on chunk 0) with the session's
        named cache. Returns the chunk's latents flattened ``(T, D)`` and the
        denormalized executed-frame actions ``(K, len(used_channels))``.
        """
        return _infer_chunk(self, session_id, frame_st_id)

    def recondition(
        self,
        session_id: str,
        actions: torch.Tensor,
        obs_latent: torch.Tensor | None,
        frame_st_id: int,
    ) -> int:
        """``VA_Server._compute_kv_cache``: drop predicted cache entries, push
        executed actions (+ real obs when given; imagination keeps the
        prediction) with ``update_cache=2``. Returns latent frames consumed."""
        return _recondition(self, session_id, actions, obs_latent, frame_st_id)

    def close(self, session_id: str) -> None:
        self._transformer.clear_cache(session_id)
        self._sessions.pop(session_id, None)


# --- internals (thin adaptations of the reference server's methods) ---


def _download_bundle(repo: str) -> str:
    from huggingface_hub import snapshot_download

    path: str = snapshot_download(repo)
    return path


def _task_config(config: LingBotVAConfig) -> Any:
    """The reference task EasyDict (norm stats, cameras, geometry) by name.

    LingBot-VA's action quantile stats and camera layout are task-specific and
    ship inside the research repo's config module; reusing them beats
    re-transcribing 30-dim quantile tables into Mirage config (drift risk).
    Mirage's ``LingBotVAConfig`` still owns the serving knobs (chunk geometry,
    steps, CFG, attn mode, dtype) — asserted to match so a silent divergence
    fails loudly.
    """
    from wan_va.configs import VA_CONFIGS

    task = VA_CONFIGS["demo"]
    if task.action_dim != config.action_dim or task.frame_chunk_size != config.frame_chunk_size:
        raise ValueError(
            "LingBotVAConfig chunk geometry diverged from the reference task "
            f"config: {config.frame_chunk_size}x{config.action_dim} vs "
            f"{task.frame_chunk_size}x{task.action_dim}"
        )
    return task


def _encode_prompt(
    tokenizer: Any,
    text_encoder: Any,
    prompt: str,
    use_cfg: bool,
    device: Any,
    dtype: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """T5 prompt (+ empty negative when CFG) — ``VA_Server.encode_prompt``."""
    import torch

    def _embed(text: str) -> torch.Tensor:
        inputs = tokenizer(
            [text],
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        enc_device = next(text_encoder.parameters()).device
        with torch.no_grad():
            hidden = text_encoder(
                inputs.input_ids.to(enc_device), inputs.attention_mask.to(enc_device)
            ).last_hidden_state
        seq_len = int(inputs.attention_mask.gt(0).sum())
        hidden = hidden[0, :seq_len]
        padded = torch.cat([hidden, hidden.new_zeros(512 - hidden.size(0), hidden.size(1))])
        return padded.unsqueeze(0).to(device=device, dtype=dtype)

    embeds = _embed(prompt)
    negative = _embed("") if use_cfg else None
    return embeds, negative


def _encode_obs_dir(pipe: LingBotVAPipeline, conditioning: ConditioningInput) -> torch.Tensor:
    """Per-camera PNGs → resized clip → streaming-VAE latents (5-D, normalized)."""
    import os

    import numpy as np
    import torch
    import torch.nn.functional as F  # noqa: N812
    from PIL import Image

    if not conditioning.uri:
        raise ValueError("LingBot-VA reset needs conditioning.uri = obs image directory")
    task = pipe._task
    session = pipe._sessions[next(reversed(pipe._sessions))]
    videos = []
    for key in task.obs_cam_keys:
        img = np.array(Image.open(os.path.join(conditioning.uri, f"{key}.png")).convert("RGB"))
        vid = torch.from_numpy(img).float().permute(2, 0, 1)[:, None]  # C,1,H,W
        vid = F.interpolate(
            vid.permute(1, 0, 2, 3), size=(task.height, task.width), mode="bilinear"
        ).permute(1, 0, 2, 3)
        videos.append(vid.unsqueeze(0))
    clip = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
    wrap = session["vae_wrap"]
    enc = wrap.encode_chunk(clip.to(next(pipe._vae.parameters()).device).to(pipe._dtype))
    mu, _logvar = torch.chunk(enc, 2, dim=1)
    mean = torch.tensor(pipe._vae.config.latents_mean, device=mu.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(pipe._vae.config.latents_std, device=mu.device).view(1, -1, 1, 1, 1)
    mu = ((mu.float() - mean) / std).to(mu.dtype)
    latent: torch.Tensor = torch.cat(torch.split(mu, 1, dim=0), dim=-1).to(pipe._device)
    return latent


def _infer_chunk(
    pipe: LingBotVAPipeline, session_id: str, frame_st_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch
    import torch.nn.functional as F  # noqa: N812
    from einops import rearrange
    from wan_va.utils import data_seq_to_patch, get_mesh_id

    cfg, task = pipe._config, pipe._task
    session = pipe._sessions[session_id]
    latent_h, latent_w = session["latent_hw"]
    use_cfg = session["use_cfg"]
    bsz = 2 if use_cfg else 1
    patch = task.patch_size

    latents = torch.randn(
        1, 48, cfg.frame_chunk_size, latent_h, latent_w, device=pipe._device, dtype=pipe._dtype
    )
    actions = torch.randn(
        1,
        cfg.action_dim,
        cfg.frame_chunk_size,
        cfg.action_per_frame,
        1,
        device=pipe._device,
        dtype=pipe._dtype,
    )
    pipe._scheduler.set_timesteps(cfg.num_inference_steps)
    pipe._action_scheduler.set_timesteps(cfg.action_num_inference_steps)
    timesteps = F.pad(pipe._scheduler.timesteps, (0, 1), value=0)
    action_timesteps = F.pad(pipe._action_scheduler.timesteps, (0, 1), value=0)
    init_latent = session["init_latent"]
    action_mask = torch.zeros(cfg.action_dim, dtype=torch.bool)
    action_mask[task.used_action_channel_ids] = True

    def _video_input(t: Any) -> dict[str, Any]:
        d = {
            "noisy_latents": latents,
            "timesteps": torch.full(
                [latents.shape[2]], float(t), dtype=torch.float32, device=pipe._device
            ),
            "grid_id": get_mesh_id(
                latents.shape[-3] // patch[0],
                latents.shape[-2] // patch[1],
                latents.shape[-1] // patch[2],
                0,
                1,
                frame_st_id,
            ).to(pipe._device),
            "text_emb": session["prompt_embeds"].to(pipe._dtype).clone(),
        }
        if frame_st_id == 0 and init_latent is not None:
            d["noisy_latents"][:, :, 0:1] = init_latent[:, :, 0:1].to(pipe._dtype)
            d["timesteps"][0:1] *= 0
        return _repeat_for_cfg(d, session, bsz)

    def _action_input(t: Any) -> dict[str, Any]:
        d = {
            "noisy_latents": actions,
            "timesteps": torch.full(
                [actions.shape[2]], float(t), dtype=torch.float32, device=pipe._device
            ),
            "grid_id": get_mesh_id(
                actions.shape[-3],
                actions.shape[-2],
                actions.shape[-1],
                1,
                1,
                frame_st_id,
                action=True,
            ).to(pipe._device),
            "text_emb": session["prompt_embeds"].to(pipe._dtype).clone(),
        }
        if frame_st_id == 0:
            d["noisy_latents"][:, :, 0:1] = 0
            d["timesteps"][0:1] *= 0
        d["noisy_latents"][:, ~action_mask] *= 0
        return _repeat_for_cfg(d, session, bsz)

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            last = i == len(timesteps) - 1
            pred = pipe._transformer(
                _video_input(t),
                update_cache=1 if last else 0,
                cache_name=session_id,
                action_mode=False,
            )
            if not last:
                pred = data_seq_to_patch(
                    patch, pred, cfg.frame_chunk_size, latent_h, latent_w, batch_size=bsz
                )
                if cfg.guidance_scale > 1:
                    pred = pred[1:] + cfg.guidance_scale * (pred[:1] - pred[1:])
                else:
                    pred = pred[:1]
                latents = pipe._scheduler.step(pred, t, latents, return_dict=False)
            if frame_st_id == 0 and init_latent is not None:
                latents[:, :, 0:1] = init_latent[:, :, 0:1].to(pipe._dtype)

        for i, t in enumerate(action_timesteps):
            last = i == len(action_timesteps) - 1
            pred = pipe._transformer(
                _action_input(t),
                update_cache=1 if last else 0,
                cache_name=session_id,
                action_mode=True,
            )
            if not last:
                pred = rearrange(pred, "b (f n) c -> b c f n 1", f=cfg.frame_chunk_size)
                if cfg.action_guidance_scale > 1:
                    pred = pred[1:] + cfg.action_guidance_scale * (pred[:1] - pred[1:])
                else:
                    pred = pred[:1]
                actions = pipe._action_scheduler.step(pred, t, actions, return_dict=False)
            if frame_st_id == 0:
                actions[:, :, 0:1] = 0

    actions[:, ~action_mask] *= 0
    flat_latents = latents[0].permute(1, 0, 2, 3).reshape(int(latents.shape[2]), -1)
    return flat_latents, _denormalize_actions(task, actions)


def _repeat_for_cfg(d: dict[str, Any], session: dict[str, Any], bsz: int) -> dict[str, Any]:
    import torch

    if bsz == 2:
        d["noisy_latents"] = d["noisy_latents"].repeat(2, 1, 1, 1, 1)
        d["text_emb"] = torch.cat(
            [d["text_emb"], session["negative_embeds"].to(d["text_emb"].dtype).clone()], dim=0
        )
        d["grid_id"] = d["grid_id"][None].repeat(2, 1, 1)
        d["timesteps"] = d["timesteps"][None].repeat(2, 1)
    else:
        d["grid_id"] = d["grid_id"][None]
        d["timesteps"] = d["timesteps"][None]
    return d


def _denormalize_actions(task: Any, actions: torch.Tensor) -> torch.Tensor:
    """``VA_Server.postprocess_action`` → ``(K, used_channels)`` executed order."""
    import torch

    q01 = torch.tensor(task.norm_stat["q01"], dtype=torch.float32).reshape(-1, 1, 1)
    q99 = torch.tensor(task.norm_stat["q99"], dtype=torch.float32).reshape(-1, 1, 1)
    a = actions[0, ..., 0].float().cpu()  # C, F, H
    a = (a + 1) / 2 * (q99 - q01 + 1e-6) + q01
    a = a[task.used_action_channel_ids]  # C_used, F, H
    return a.permute(1, 2, 0).reshape(-1, a.shape[0])  # (F*H, C_used)


def _recondition(
    pipe: LingBotVAPipeline,
    session_id: str,
    actions: torch.Tensor,
    obs_latent: torch.Tensor | None,
    frame_st_id: int,
) -> int:
    import torch

    cfg, task = pipe._config, pipe._task
    session = pipe._sessions[session_id]

    latent5d: torch.Tensor | None
    if obs_latent is not None and pipe._latent_geom is not None:
        c, h, w = pipe._latent_geom
        latent5d = obs_latent.reshape(-1, c, h, w).permute(1, 0, 2, 3).unsqueeze(0)
    elif frame_st_id == 0:
        latent5d = session["init_latent"]
    else:
        latent5d = None
    if latent5d is None:
        # Imagination mode (reference ``generate()``): the predicted chunk
        # already entered the cache via ``update_cache=1`` in infer_chunk;
        # keep it (no clear_pred_cache!) and just advance the frame clock.
        return int(cfg.frame_chunk_size)
    # Real observation: drop the predicted entries, push grounded reality.
    pipe._transformer.clear_pred_cache(session_id)

    # Normalize executed actions back to model space (inverse of postprocess).
    q01 = torch.tensor(task.norm_stat["q01"], dtype=torch.float32).reshape(1, 1, -1)
    q99 = torch.tensor(task.norm_stat["q99"], dtype=torch.float32).reshape(1, 1, -1)
    k = actions.shape[0] // cfg.action_per_frame
    a = actions.float().cpu().reshape(k, cfg.action_per_frame, -1)
    a_norm = (a - q01[..., task.used_action_channel_ids]) / (
        q99[..., task.used_action_channel_ids] - q01[..., task.used_action_channel_ids] + 1e-6
    ) * 2.0 - 1.0
    full = torch.zeros(k, cfg.action_per_frame, cfg.action_dim)
    full[..., task.used_action_channel_ids] = a_norm
    action5d = full.permute(2, 0, 1)[None, ..., None].to(pipe._device, pipe._dtype)

    # Two update_cache=2 passes, mirroring _compute_kv_cache (video then action).
    from wan_va.utils import get_mesh_id

    patch = task.patch_size
    latent5d = latent5d.to(pipe._device, pipe._dtype)
    bsz = 2 if session["use_cfg"] else 1
    video_d = {
        "noisy_latents": latent5d,
        "timesteps": torch.zeros([latent5d.shape[2]], dtype=torch.float32, device=pipe._device),
        "grid_id": get_mesh_id(
            latent5d.shape[-3] // patch[0],
            latent5d.shape[-2] // patch[1],
            latent5d.shape[-1] // patch[2],
            0,
            1,
            frame_st_id,
        ).to(pipe._device),
        "text_emb": session["prompt_embeds"].to(pipe._dtype).clone(),
    }
    action_d = {
        "noisy_latents": action5d,
        "timesteps": torch.zeros([action5d.shape[2]], dtype=torch.float32, device=pipe._device),
        "grid_id": get_mesh_id(
            action5d.shape[-3],
            action5d.shape[-2],
            action5d.shape[-1],
            1,
            1,
            frame_st_id,
            action=True,
        ).to(pipe._device),
        "text_emb": session["prompt_embeds"].to(pipe._dtype).clone(),
    }
    with torch.no_grad():
        pipe._transformer(
            _repeat_for_cfg(video_d, session, bsz),
            update_cache=2,
            cache_name=session_id,
            action_mode=False,
        )
        pipe._transformer(
            _repeat_for_cfg(action_d, session, bsz),
            update_cache=2,
            cache_name=session_id,
            action_mode=True,
        )
    return int(latent5d.shape[2])
