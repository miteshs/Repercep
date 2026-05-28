"""World-model engines for Mirage.

Lead model: Cosmos-Predict-7B. Second model family: Wan-2.2 (T2V-A14B), which
proves the runtime is world-model-native rather than Cosmos-specific.
"""

from __future__ import annotations

from mirage.models.cosmos import (
    DEFAULT_REPO,
    CosmosConfig,
    CosmosEngine,
    GuardrailError,
)
from mirage.models.vjepa2_ac import (
    DEFAULT_ENCODER_REPO,
    VJepa2ACConfig,
    VJepa2ACEngine,
)
from mirage.models.wan import DEFAULT_REPO as WAN_DEFAULT_REPO
from mirage.models.wan import NATIVE_FPS as WAN_NATIVE_FPS
from mirage.models.wan import SMALL_REPO as WAN_SMALL_REPO
from mirage.models.wan import WanConfig, WanEngine

__all__ = [
    "DEFAULT_ENCODER_REPO",
    "DEFAULT_REPO",
    "WAN_DEFAULT_REPO",
    "WAN_NATIVE_FPS",
    "WAN_SMALL_REPO",
    "CosmosConfig",
    "CosmosEngine",
    "GuardrailError",
    "VJepa2ACConfig",
    "VJepa2ACEngine",
    "WanConfig",
    "WanEngine",
]
