"""Tests for the Wan-2.2 engine.

Construction, identity, defaults, and ``WorldModelEngine`` Protocol conformance
are checked here. Actual generation needs ~52 GiB of downloaded weights and a
GPU, and is exercised by ``scripts/run_wan.py`` rather than the unit suite.
"""

from __future__ import annotations

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
