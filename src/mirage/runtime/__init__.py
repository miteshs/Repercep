"""The Mirage Runtime: request types, the engine seam, and the latent cache."""

from __future__ import annotations

from mirage.runtime.denoise import denoise_cosmos_video
from mirage.runtime.engine import EngineInfo, WorldModelEngine
from mirage.runtime.latent_cache import CacheStats, PagedLatentCache
from mirage.runtime.stub_engine import StubEngine
from mirage.runtime.types import (
    ConditioningInput,
    ConditioningKind,
    Frame,
    FrameChunk,
    GenerationParams,
    GenerationRequest,
    GenerationResult,
)

__all__ = [
    "CacheStats",
    "ConditioningInput",
    "ConditioningKind",
    "EngineInfo",
    "Frame",
    "FrameChunk",
    "GenerationParams",
    "GenerationRequest",
    "GenerationResult",
    "PagedLatentCache",
    "StubEngine",
    "WorldModelEngine",
    "denoise_cosmos_video",
]
