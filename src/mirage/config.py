"""Runtime configuration.

All settings are overridable via ``MIRAGE_*`` environment variables, e.g.
``MIRAGE_DTYPE=fp16`` or ``MIRAGE_DEVICE_INDEX=0``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mirage.hardware import DType


def _default_weights_dir() -> Path:
    return Path.home() / ".cache" / "mirage" / "weights"


class RuntimeConfig(BaseSettings):
    """Top-level runtime configuration.

    Reads ``MIRAGE_*`` environment variables; unknown keys are rejected so a
    typo in an env var fails loud rather than being silently ignored.
    """

    model_config = SettingsConfigDict(env_prefix="MIRAGE_", extra="forbid")

    device_index: int = Field(0, ge=0, description="Which accelerator to bind to.")
    dtype: DType = Field(DType.BF16, description="Compute dtype for the model.")
    attention_backend: str = Field(
        "auto",
        description="'auto' lets the backend pick; or pin 'rocm-ck-flash' / 'naive-sdpa'.",
    )
    max_batch_size: int = Field(1, ge=1, description="Upper bound on concurrent requests.")
    latent_cache_gib: float = Field(8.0, gt=0, description="HBM budget for the paged latent cache.")
    weights_dir: Path = Field(
        default_factory=_default_weights_dir,
        description="Local directory for downloaded model weights.",
    )
    hf_token: str | None = Field(
        None,
        description="HuggingFace token for gated weights (Cosmos). Prefer MIRAGE_HF_TOKEN.",
    )
