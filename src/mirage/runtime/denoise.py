"""Mirage-native Cosmos denoising loop.

Replaces the diffusers ``CosmosTextToWorldPipeline.__call__`` with a loop that
uses the same components (T5 encoder, DiT transformer, VAE, EDM-Euler
scheduler) but folds the classifier-free-guidance forward passes into a single
batched call. The diffusers loop runs two sequential transformer forwards per
step (BUILD_LOG F9); this collapses them into one batch-2 forward.

The data path mirrors the reference loop step-for-step — encoder pre-batching,
``scheduler.scale_model_input``, the scheduler's two-call pattern around CFG,
and the VAE postprocess — so behaviour stays bit-equivalent except for the
batched-vs-sequential difference in the transformer call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def denoise_cosmos_video(
    pipe: Any,
    *,
    prompt: str,
    negative_prompt: str | None = None,
    height: int = 704,
    width: int = 1280,
    num_frames: int = 121,
    num_inference_steps: int = 36,
    guidance_scale: float = 7.0,
    fps: int = 30,
    seed: int | None = None,
    output_type: str = "pt",
    cfg_batched: bool = True,
    cache_skip_every: int = 0,
    cache_warmup_steps: int = 4,
) -> Any:
    """Run the Cosmos denoising loop end-to-end and return a video tensor.

    Args:
        pipe: a loaded ``CosmosTextToWorldPipeline``.
        cfg_batched: if ``True`` (default), the per-step conditional and
            unconditional transformer forwards are folded into one batch-2
            call. If ``False``, the function uses the diffusers reference
            behaviour (two sequential forwards) as a correctness reference.
        cache_skip_every: if ``>= 2``, after the warmup window, run a full DiT
            forward only on every Nth step and reuse the cached output on the
            remaining N-1 steps. ``0`` (default) disables caching — every step
            runs the full forward. Aggressive values trade quality for speed.
        cache_warmup_steps: number of leading steps that always run a full
            forward, before caching kicks in. Defaults to 4.

    Returns:
        With ``output_type='pt'`` (default): a tensor shaped ``(B, T, C, H, W)``
        in ``[0, 1]`` — what ``pipe(...).frames`` returns for the same inputs.
    """
    import torch

    device = pipe._execution_device
    transformer_dtype = pipe.transformer.dtype

    # 1. Encode prompts (cond + uncond).
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        device=device,
        dtype=transformer_dtype,
    )

    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    # 2. Scheduler timesteps.
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # 3. Initial latents.
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.transformer.config.in_channels,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=torch.float32,
        device=device,
        generator=generator,
        latents=None,
    )
    padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)

    # 4. Pre-batch encoder hidden states for CFG.  Order matches the diffusers
    # reference: position 0 = uncond, position 1 = cond.  The padding mask is
    # NOT pre-batched: the transformer repeats it internally by `batch_size`,
    # so a `(2, 1, H, W)` mask would double again to `(4, ...)`.
    encoder_pair: Any = None
    if cfg_batched:
        encoder_pair = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # 5. Denoising loop, optionally with step-skip caching after a warmup window.
    cached_noise_pred: Any = None
    for step_idx, t in enumerate(timesteps):
        latent_model_input = pipe.scheduler.scale_model_input(latents, t).to(transformer_dtype)
        timestep = t.expand(latents.shape[0]).to(transformer_dtype)

        should_skip = (
            cache_skip_every >= 2
            and step_idx >= cache_warmup_steps
            and cached_noise_pred is not None
            and (step_idx - cache_warmup_steps) % cache_skip_every != 0
        )

        if should_skip:
            noise_pred = cached_noise_pred
        elif cfg_batched:
            batched_input = latent_model_input.repeat(2, 1, 1, 1, 1)
            batched_timestep = timestep.repeat(2)
            noise_pred = pipe.transformer(
                hidden_states=batched_input,
                timestep=batched_timestep,
                encoder_hidden_states=encoder_pair,
                fps=fps,
                padding_mask=padding_mask,
                return_dict=False,
            )[0]
            cached_noise_pred = noise_pred
        else:
            noise_pred_cond = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                fps=fps,
                padding_mask=padding_mask,
                return_dict=False,
            )[0]
            noise_pred_uncond = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                fps=fps,
                padding_mask=padding_mask,
                return_dict=False,
            )[0]
            noise_pred = torch.cat([noise_pred_uncond, noise_pred_cond], dim=0)
            cached_noise_pred = noise_pred

        sample = torch.cat([latents, latents], dim=0)

        # First scheduler call: returns pred_original_sample (x0).
        noise_pred = pipe.scheduler.step(noise_pred, t, sample, return_dict=False)[1]
        pipe.scheduler._step_index -= 1

        # Apply CFG on x0.
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_cond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Second scheduler call: actually advance latents.
        latents = pipe.scheduler.step(
            noise_pred,
            t,
            latents,
            return_dict=False,
            pred_original_sample=noise_pred,
        )[0]

    # 6. Latent un-normalization + VAE decode + postprocess (reference path).
    if pipe.vae.config.latents_mean is not None:
        latents_mean = pipe.vae.config.latents_mean
        latents_std = pipe.vae.config.latents_std
        latents_mean = (
            torch.tensor(latents_mean)
            .view(1, pipe.vae.config.latent_channels, -1, 1, 1)[:, :, : latents.size(2)]
            .to(latents)
        )
        latents_std = (
            torch.tensor(latents_std)
            .view(1, pipe.vae.config.latent_channels, -1, 1, 1)[:, :, : latents.size(2)]
            .to(latents)
        )
        latents = latents * latents_std / pipe.scheduler.config.sigma_data + latents_mean
    else:
        latents = latents / pipe.scheduler.config.sigma_data

    video = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.video_processor.postprocess_video(video, output_type=output_type)
