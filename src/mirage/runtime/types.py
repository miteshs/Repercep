"""Request, response, and frame types for the Mirage Runtime.

These types are the wire contract. Every externally-visible type is a Pydantic
model, so the HTTP and gRPC layers serialize it directly and a malformed
request fails validation at the edge rather than deep in the diffusion loop.
``Frame`` is the one internal exception — it carries a live tensor and never
crosses the wire.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import torch

_STRICT = ConfigDict(extra="forbid")


class ConditioningKind(enum.StrEnum):
    """How a generation is conditioned on visual input."""

    NONE = "none"  # pure Text2World
    IMAGE = "image"  # Video2World rolled out from a single frame
    VIDEO = "video"  # Video2World continuing an existing clip


class ConditioningInput(BaseModel):
    """Optional visual conditioning for Video2World generation.

    Raw pixels never travel in JSON. ``uri`` is a reference the serving layer
    resolves: a data URI, an object-store key, or a server-local path.
    """

    model_config = _STRICT

    kind: ConditioningKind = ConditioningKind.NONE
    uri: str | None = None


class GenerationParams(BaseModel):
    """Diffusion and sampling knobs.

    Defaults track the Cosmos-Predict-7B reference configuration (121 frames at
    704x1280, 35 denoising steps). Bounds are deliberately conservative for
    v0.1 and will widen as the runtime is hardened.
    """

    model_config = _STRICT

    num_frames: int = Field(121, ge=1, le=256)
    height: int = Field(704, ge=64, le=2048)
    width: int = Field(1280, ge=64, le=2048)
    num_inference_steps: int = Field(35, ge=1, le=200)
    guidance_scale: float = Field(7.0, ge=0.0, le=30.0)
    fps: int = Field(24, ge=1, le=120)
    seed: int | None = Field(None, description="None yields a nondeterministic run.")


class GenerationRequest(BaseModel):
    """A single world-model generation request."""

    model_config = _STRICT

    prompt: str = Field(..., min_length=1)
    negative_prompt: str | None = None
    params: GenerationParams = Field(default_factory=GenerationParams)
    conditioning: ConditioningInput = Field(default_factory=ConditioningInput)


@dataclass(slots=True)
class Frame:
    """One generated frame, in-process.

    Internal type: ``pixels`` is a live tensor, so ``Frame`` never crosses the
    wire — the serving layer converts it to a ``FrameChunk`` plus an
    out-of-band pixel payload.
    """

    index: int
    total: int
    pixels: torch.Tensor  # (height, width, 3)


class FrameChunk(BaseModel):
    """One frame as it streams to a client — the metadata envelope.

    The pixel payload is carried out of band (a follow-up binary message, or an
    object-store URI in ``uri``); keeping bytes out of the JSON keeps the
    stream cheap to parse and log.
    """

    model_config = _STRICT

    frame_index: int
    total_frames: int
    height: int
    width: int
    latency_ms: float
    uri: str | None = None


class GenerationResult(BaseModel):
    """Terminal summary of a completed generation."""

    model_config = _STRICT

    num_frames: int
    total_latency_ms: float
    frames_per_second: float
    backend: str
    device: str
    attention_op: str
