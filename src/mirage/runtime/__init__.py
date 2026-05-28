"""The Mirage Runtime: request types, the engine seam, and the latent cache."""

from __future__ import annotations

from mirage.runtime.denoise import denoise_cosmos_video
from mirage.runtime.engine import EngineInfo, WorldModelEngine
from mirage.runtime.interactive import InteractiveWorldModel
from mirage.runtime.latent_cache import CacheStats, PagedLatentCache
from mirage.runtime.stub_engine import StubEngine
from mirage.runtime.types import (
    Action,
    ConditioningInput,
    ConditioningKind,
    Frame,
    FrameChunk,
    GenerationParams,
    GenerationRequest,
    GenerationResult,
    LatentStep,
    ResetRequest,
    RolloutParams,
    WorldState,
)

__all__ = [
    "Action",
    "CacheStats",
    "ConditioningInput",
    "ConditioningKind",
    "EngineInfo",
    "Frame",
    "FrameChunk",
    "GenerationParams",
    "GenerationRequest",
    "GenerationResult",
    "InteractiveWorldModel",
    "LatentStep",
    "PagedLatentCache",
    "ResetRequest",
    "RolloutParams",
    "StubEngine",
    "WorldModelEngine",
    "WorldState",
    "denoise_cosmos_video",
]
