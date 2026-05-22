"""Frame-streaming HTTP API for the Mirage Runtime.

API-first: this module *is* the HTTP contract. The gRPC contract is the sibling
``proto/mirage.proto``, kept in lockstep with ``mirage.runtime.types``.
Endpoints are versioned under ``/v1``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from mirage import __version__
from mirage.runtime.engine import EngineInfo
from mirage.runtime.stub_engine import StubEngine

# EngineInfo and GenerationRequest must stay runtime imports: FastAPI resolves
# route annotations at startup via get_type_hints (see per-file ruff ignore).
from mirage.runtime.types import FrameChunk, GenerationRequest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mirage.runtime.engine import WorldModelEngine


def create_app(engine: WorldModelEngine | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        engine: the world-model engine to serve. Defaults to ``StubEngine`` so
            the API is runnable and testable before the Cosmos-Predict-7B
            engine lands (Task #7).
    """
    active_engine: WorldModelEngine = engine if engine is not None else StubEngine()
    app = FastAPI(title="Mirage Runtime", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/info")
    def info() -> EngineInfo:
        return active_engine.info()

    @app.post("/v1/generate/stream")
    def generate_stream(request: GenerationRequest) -> StreamingResponse:
        """Stream frames as newline-delimited JSON ``FrameChunk`` records.

        The response begins as soon as the first frame is ready; the runtime
        does not buffer the whole clip. This is the frame-level streaming the
        implementation plan calls for.
        """

        def frames() -> Iterator[str]:
            started = time.perf_counter()
            for frame in active_engine.generate(request):
                chunk = FrameChunk(
                    frame_index=frame.index,
                    total_frames=frame.total,
                    height=int(frame.pixels.shape[0]),
                    width=int(frame.pixels.shape[1]),
                    latency_ms=(time.perf_counter() - started) * 1e3,
                )
                yield chunk.model_dump_json() + "\n"

        return StreamingResponse(frames(), media_type="application/x-ndjson")

    return app
