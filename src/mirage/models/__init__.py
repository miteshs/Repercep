"""World-model engines for Mirage.

Generation camp: Cosmos-Predict-7B and Wan-2.2 (T2V-A14B). Control camp on the
interactive seam: V-JEPA 2-AC (energy-MPC planning) and LingBot-VA 2.0
(video-action policy) — two planning regimes, one Protocol.
"""

from __future__ import annotations

from mirage.models.cosmos import (
    DEFAULT_REPO,
    CosmosConfig,
    CosmosEngine,
    GuardrailError,
)
from mirage.models.lingbot_va import (
    DEFAULT_REPO as LINGBOT_VA_DEFAULT_REPO,
)
from mirage.models.lingbot_va import (
    LingBotVAConfig,
    LingBotVAEngine,
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
    "LINGBOT_VA_DEFAULT_REPO",
    "WAN_DEFAULT_REPO",
    "WAN_NATIVE_FPS",
    "WAN_SMALL_REPO",
    "CosmosConfig",
    "CosmosEngine",
    "GuardrailError",
    "LingBotVAConfig",
    "LingBotVAEngine",
    "VJepa2ACConfig",
    "VJepa2ACEngine",
    "WanConfig",
    "WanEngine",
]
