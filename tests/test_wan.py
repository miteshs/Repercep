"""Tests for the Wan-2.2 engine.

Construction, identity, defaults, and ``WorldModelEngine`` Protocol conformance
are checked here. Actual generation needs ~52 GiB of downloaded weights and a
GPU, and is exercised by ``scripts/run_wan.py`` rather than the unit suite.
"""

from __future__ import annotations

from typing import Any

from mirage.backend.rocm import ROCmBackend
from mirage.models import (
    WAN_DEFAULT_REPO,
    WAN_NATIVE_FPS,
    WAN_SMALL_REPO,
    WanConfig,
    WanEngine,
)
from mirage.runtime.engine import WorldModelEngine


def test_wan_engine_satisfies_engine_protocol() -> None:
    assert isinstance(WanEngine(backend=ROCmBackend()), WorldModelEngine)


def test_wan_engine_info_before_load() -> None:
    engine = WanEngine(backend=ROCmBackend())
    info = engine.info()
    assert info.model_name == "wan-2.2-t2v-a14b"
    assert info.backend == "rocm"
    assert info.dtype == "bfloat16"
    assert info.ready is False  # lazy: no weights touched yet


def test_wan_engine_is_not_loaded_on_construction() -> None:
    assert WanEngine(backend=ROCmBackend()).is_loaded is False


def test_wan_engine_pipeline_raises_before_load() -> None:
    import pytest

    engine = WanEngine(backend=ROCmBackend())
    with pytest.raises(RuntimeError, match="not loaded"):
        _ = engine.pipeline


def test_wan_default_repo_is_t2v_a14b_diffusers() -> None:
    assert WAN_DEFAULT_REPO == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert WanConfig().repo_id == WAN_DEFAULT_REPO


def test_wan_small_repo_is_ti2v_5b() -> None:
    assert WAN_SMALL_REPO == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def test_wan_native_fps_is_sixteen() -> None:
    # Wan-2.2 was trained at 16 FPS; the downstream writer relies on this.
    assert WAN_NATIVE_FPS == 16


def test_wan_config_defaults() -> None:
    config = WanConfig()
    # BF16 is native on CDNA3; FP32 VAE matches the Wan reference path.
    assert config.dtype == "bfloat16"
    assert config.vae_dtype == "float32"
    # Wan-2.2-T2V-A14B model-card default for second-stage guidance.
    assert config.guidance_scale_2 == 3.0
    # Forward-compat hooks (Phase-2 native-loop follow-up).
    assert config.use_native_loop is False
    assert config.cache_skip_every == 0


def test_wan_engine_accepts_custom_config() -> None:
    config = WanConfig(device_index=1, dtype="float16", guidance_scale_2=2.5)
    engine = WanEngine(backend=ROCmBackend(), config=config)
    info = engine.info()
    assert info.dtype == "float16"
    assert info.device == "rocm:1"


def test_wan_profile_exports_at_bench_top_level() -> None:
    # The Wan-specific profiler mirrors profile_cosmos in shape and lives in
    # the same package. Both should re-export from mirage.bench so external
    # callers can pick one without touching the submodule path.
    import mirage.bench as bench

    assert hasattr(bench, "WanProfile")
    assert hasattr(bench, "profile_wan")


def test_wan_profile_schema_includes_moe_split() -> None:
    # The MoE second-expert handoff is non-trivial for Wan-2.2 (Session 10
    # noted ~270 s of the 326 s smoke run was non-DiT work). The profiler must
    # report transformer and transformer_2 timings as separate fields so the
    # handoff is visible in the breakdown — not collapsed into a single
    # dit_loop_s.
    from mirage.bench.profile import WanProfile

    fields = set(WanProfile.model_fields)
    assert "dit_high_noise_s" in fields
    assert "dit_low_noise_s" in fields
    assert "dit_high_noise_calls" in fields
    assert "dit_low_noise_calls" in fields
    # And dit_loop_s = high + low, same units as the Cosmos counterpart so
    # the two reports are directly comparable.
    assert "dit_loop_s" in fields


def test_wan_profile_dit_share_zero_when_total_zero() -> None:
    from mirage.bench.profile import WanProfile

    prof = WanProfile(
        total_s=0.0,
        text_encode_s=0.0,
        dit_loop_s=0.0,
        dit_high_noise_s=0.0,
        dit_low_noise_s=0.0,
        vae_decode_s=0.0,
        other_s=0.0,
        dit_calls=0,
        dit_high_noise_calls=0,
        dit_low_noise_calls=0,
        text_encode_calls=0,
        vae_decode_calls=0,
        compiled=False,
    )
    assert prof.dit_share == 0.0


def test_wan_config_vae_tiling_default_off() -> None:
    # Default-off keeps the MI300X (192 GiB) baseline path unchanged; the flag
    # is opt-in for the 80 GiB H100 envelope.
    assert WanConfig().vae_tiling is False


def test_wan_engine_load_calls_enable_tiling_when_vae_tiling_true() -> None:
    """vae_tiling=True must invoke pipe.vae.enable_tiling() during load()."""
    from unittest.mock import MagicMock, patch

    fake_vae = MagicMock(name="vae")
    fake_pipe = MagicMock(name="pipe")
    fake_pipe.vae = fake_vae

    with (
        patch("diffusers.AutoencoderKLWan.from_pretrained", return_value=fake_vae),
        patch("diffusers.WanPipeline.from_pretrained", return_value=fake_pipe),
    ):
        engine = WanEngine(backend=ROCmBackend(), config=WanConfig(vae_tiling=True))
        engine.load()

    fake_vae.enable_tiling.assert_called_once()


def test_wan_engine_load_does_not_call_enable_tiling_when_vae_tiling_false() -> None:
    """vae_tiling=False (default) must NOT touch enable_tiling."""
    from unittest.mock import MagicMock, patch

    fake_vae = MagicMock(name="vae")
    fake_pipe = MagicMock(name="pipe")
    fake_pipe.vae = fake_vae

    with (
        patch("diffusers.AutoencoderKLWan.from_pretrained", return_value=fake_vae),
        patch("diffusers.WanPipeline.from_pretrained", return_value=fake_pipe),
    ):
        engine = WanEngine(backend=ROCmBackend(), config=WanConfig(vae_tiling=False))
        engine.load()

    fake_vae.enable_tiling.assert_not_called()


def _make_engine_with_fake_pipe(boundary_ratio: float | None) -> tuple[WanEngine, Any]:
    """Construct a WanEngine whose loaded pipe is a MagicMock with a chosen
    boundary_ratio. Returns (engine, fake_pipe) so the test can inspect the
    kwargs passed to fake_pipe.__call__."""
    from unittest.mock import MagicMock

    fake_pipe = MagicMock(name="pipe")
    fake_pipe.config.boundary_ratio = boundary_ratio
    # Mock the __call__ return: output.frames[0] must be a tensor convertible
    # via _as_frame_tensor. Easiest: return uint8 frames-tensor of shape
    # (T, C, H, W) in [0, 1].
    import torch

    fake_frames = torch.zeros((2, 3, 64, 64), dtype=torch.float32)
    fake_pipe.return_value.frames = [fake_frames]

    engine = WanEngine(backend=ROCmBackend())
    engine._pipe = fake_pipe  # bypass load()
    return engine, fake_pipe


def test_wan_engine_passes_guidance_scale_2_for_moe_variant() -> None:
    """A14B (boundary_ratio != None) must receive guidance_scale_2."""
    from mirage.runtime.types import GenerationParams, GenerationRequest

    engine, fake_pipe = _make_engine_with_fake_pipe(boundary_ratio=0.875)
    request = GenerationRequest(
        prompt="x", negative_prompt="y",
        params=GenerationParams(num_frames=2, num_inference_steps=2, height=64, width=64,
                                guidance_scale=4.0, fps=16, seed=0),
    )
    _ = list(engine.generate(request))
    _, kwargs = fake_pipe.call_args
    assert "guidance_scale_2" in kwargs
    assert kwargs["guidance_scale_2"] == 3.0  # WanConfig default


def test_wan_engine_omits_guidance_scale_2_for_non_moe_variant() -> None:
    """TI2V-5B (boundary_ratio == None) must NOT receive guidance_scale_2.

    Diffusers raises ``ValueError: guidance_scale_2 is only supported when
    the pipeline's boundary_ratio is not None`` if the kwarg is passed to a
    non-MoE Wan variant. See BUILD_LOG F39.
    """
    from mirage.runtime.types import GenerationParams, GenerationRequest

    engine, fake_pipe = _make_engine_with_fake_pipe(boundary_ratio=None)
    request = GenerationRequest(
        prompt="x", negative_prompt="y",
        params=GenerationParams(num_frames=2, num_inference_steps=2, height=64, width=64,
                                guidance_scale=4.0, fps=16, seed=0),
    )
    _ = list(engine.generate(request))
    _, kwargs = fake_pipe.call_args
    assert "guidance_scale_2" not in kwargs


def test_wan_profile_dit_share_reports_loop_fraction() -> None:
    from mirage.bench.profile import WanProfile

    # 80% of total in the DiT loop (a typical Wan-shaped breakdown — far more
    # DiT-bound than the 17f/8-step smoke, where VAE+postprocess dominate).
    prof = WanProfile(
        total_s=100.0,
        text_encode_s=1.0,
        dit_loop_s=80.0,
        dit_high_noise_s=40.0,
        dit_low_noise_s=40.0,
        vae_decode_s=10.0,
        other_s=9.0,
        dit_calls=80,
        dit_high_noise_calls=40,
        dit_low_noise_calls=40,
        text_encode_calls=2,
        vae_decode_calls=1,
        compiled=False,
    )
    assert prof.dit_share == 0.8
