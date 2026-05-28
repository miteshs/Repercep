"""Frame-streaming HTTP API for the Mirage Runtime.

API-first: this module *is* the HTTP contract. The gRPC contract is the sibling
``proto/mirage.proto``, kept in lockstep with ``mirage.runtime.types``.

Two API surfaces live here:

* ``/v1/...`` — original synchronous path. The handler runs the engine
  inline and yields :class:`mirage.runtime.types.FrameChunk` over NDJSON.
  Untouched for backward compatibility.

* ``/v2/...`` — Stage-4 path that routes through the Rust core. The handler
  mints a request id, stages the body, calls ``Router.accept`` (which submits
  to the scheduler), and streams the router's frame channel back to the
  client. The engine itself runs in a single background driver thread that
  drains the scheduler — see :mod:`mirage.serving.driver`.

The driver thread, scheduler, and router are owned by the FastAPI app's
``lifespan`` context: created at startup, torn down on shutdown.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mirage import __version__
from mirage.runtime.engine import EngineInfo
from mirage.runtime.router import Router, RouterError
from mirage.runtime.scheduler import Scheduler
from mirage.runtime.stub_engine import StubEngine

# These must stay runtime imports: FastAPI resolves route annotations at startup
# via get_type_hints, and the WebSocket session validates Action/ResetRequest
# and emits LatentStep at runtime (see per-file ruff ignore).
from mirage.runtime.types import (
    Action,
    FrameChunk,
    GenerationRequest,
    LatentStep,
    ResetRequest,
)
from mirage.serving.driver import EngineDriver, _SchedulerAdapter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from mirage.runtime.engine import WorldModelEngine
    from mirage.runtime.interactive import InteractiveWorldModel
    from mirage.runtime.types import WorldState


# Default scheduler capacity for the v2 path. Sized so a single-engine driver
# can absorb a small burst without rejecting; backpressure surfaces as a 503
# at the v2 endpoint when this fills.
_DEFAULT_SCHEDULER_CAPACITY = 64

# Per-request frame channel depth on the router. The driver pushes one frame
# at a time and the streaming handler drains continuously, so a small bound
# is enough; backpressure here would only fire if a client TCP-stalls.
_DEFAULT_FRAME_QUEUE_DEPTH = 64


# ---------------------------------------------------------------------------
# v2 request body
# ---------------------------------------------------------------------------


_PRIORITY = Literal["low", "normal", "high"]


class V2GenerationRequest(BaseModel):
    """v2 generation request — :class:`GenerationRequest` plus a priority hint.

    Composition over inheritance: keeping :class:`GenerationRequest` itself
    untouched means the v1 path's wire contract is bit-stable. The v2 body
    is the v1 body inside ``request`` plus an optional ``priority`` field.
    """

    model_config = ConfigDict(extra="forbid")

    request: GenerationRequest
    priority: _PRIORITY = "normal"


# ---------------------------------------------------------------------------
# Lifespan-owned state
# ---------------------------------------------------------------------------


class _V2State:
    """Holds the v2 path's owned objects so the lifespan can tear them down."""

    def __init__(
        self,
        *,
        engine: WorldModelEngine,
        scheduler: Scheduler,
        adapter: _SchedulerAdapter,
        router: Router,
        driver: EngineDriver,
    ) -> None:
        self.engine = engine
        self.scheduler = scheduler
        self.adapter = adapter
        self.router = router
        self.driver = driver


def _build_v2_state(
    engine: WorldModelEngine,
    *,
    scheduler_capacity: int,
    frame_queue_depth: int,
    loop: asyncio.AbstractEventLoop,
) -> _V2State:
    scheduler = Scheduler(capacity=scheduler_capacity)
    adapter = _SchedulerAdapter(scheduler)
    router = Router(adapter, per_request_queue_depth=frame_queue_depth)
    driver = EngineDriver(
        engine=engine, scheduler=scheduler, router=router, loop=loop
    )
    return _V2State(
        engine=engine,
        scheduler=scheduler,
        adapter=adapter,
        router=router,
        driver=driver,
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    engine: WorldModelEngine | None = None,
    *,
    interactive_engine: InteractiveWorldModel | None = None,
    scheduler_capacity: int = _DEFAULT_SCHEDULER_CAPACITY,
    frame_queue_depth: int = _DEFAULT_FRAME_QUEUE_DEPTH,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        engine: the world-model engine to serve. Defaults to ``StubEngine`` so
            the API is runnable and testable before the Cosmos-Predict-7B
            engine lands.
        interactive_engine: optional action-conditioned world model served over
            the ``/v2/world/session`` WebSocket (ADR-0008). ``None`` disables it.
        scheduler_capacity: max in-flight requests on the v2 path before
            ``submit`` raises ``QueueFull`` (surfaces as HTTP 503).
        frame_queue_depth: per-request frame channel depth. Backpressure
            within the router fires when a client TCP-stalls beyond this.
    """
    active_engine: WorldModelEngine = engine if engine is not None else StubEngine()
    active_interactive = interactive_engine

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # The driver thread needs a handle to *this* loop so it can bounce
        # router.push_frame back via run_coroutine_threadsafe.
        loop = asyncio.get_running_loop()
        state = _build_v2_state(
            active_engine,
            scheduler_capacity=scheduler_capacity,
            frame_queue_depth=frame_queue_depth,
            loop=loop,
        )
        _app.state.v2 = state
        state.driver.start()
        try:
            yield
        finally:
            # Order matters: stop the driver (which calls scheduler.shutdown()
            # internally, draining the queue and waking the thread), then
            # mark the router shutting down so any in-flight subscribers see
            # end-of-stream cleanly.
            state.driver.stop(timeout=5.0)
            with suppress(RouterError):
                state.router.shutdown()

    app = FastAPI(title="Mirage Runtime", version=__version__, lifespan=lifespan)

    # -----------------------------------------------------------------------
    # v1 (untouched)
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # v2 (Rust-router path)
    # -----------------------------------------------------------------------

    @app.post("/v2/generate/stream")
    async def generate_stream_v2(
        body: Annotated[V2GenerationRequest, Field()],
    ) -> StreamingResponse:
        """Submit through the router/scheduler; stream NDJSON frames back.

        Each line is one frame as encoded by
        :func:`mirage.serving.driver.encode_frame_line` — the v1
        ``FrameChunk`` fields plus ``is_final`` and a base64-encoded pixel
        payload. Mirror the v1 format exactly so the migration is a URL swap.
        """
        state: _V2State = app.state.v2
        request_id = uuid.uuid4().hex
        # Stage the payload so the adapter has it when the router calls
        # submit. Wrap in a single-key envelope so the driver can future-
        # extend without re-flowing the wire shape.
        payload = {"request": body.request.model_dump()}
        state.adapter.stage(request_id, payload)
        try:
            state.router.accept(request_id, body.priority, payload)
        except RouterError as exc:
            # Roll back the staged payload — accept may have failed before
            # the adapter consumed it.
            state.adapter.discard(request_id)
            msg = str(exc).lower()
            # Map scheduler-side capacity errors to 503; everything else 400.
            if "queue is full" in msg or "queuefull" in msg:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        stream = state.router.subscribe(request_id)

        async def body_iter() -> AsyncIterator[bytes]:
            try:
                async for frame in stream:
                    # frame is the dict returned by the FrameStream:
                    # {request_id, frame_index, payload (bytes), is_final}.
                    yield bytes(frame["payload"]) + b"\n"
            except Exception:
                # Best-effort cancel on stream interruption (client disconnect
                # or driver failure). Avoid raising — the response is already
                # being streamed.
                with suppress(RouterError):
                    state.router.cancel(request_id)

        return StreamingResponse(body_iter(), media_type="application/x-ndjson")

    @app.post("/v2/generate/{request_id}/cancel")
    async def cancel_v2(request_id: str) -> dict[str, str]:
        """Cancel an in-flight v2 request. 404 if the id is unknown."""
        state: _V2State = app.state.v2
        try:
            state.router.cancel(request_id)
        except RouterError as exc:
            msg = str(exc).lower()
            if "unknown request" in msg:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "cancelled", "request_id": request_id}

    @app.get("/v2/generate/{request_id}/state")
    def state_v2(request_id: str) -> dict[str, str | None]:
        """Look up the router's current state for a request. ``None`` if unknown."""
        state: _V2State = app.state.v2
        return {"request_id": request_id, "state": state.router.state(request_id)}

    # -----------------------------------------------------------------------
    # v2 interactive world-model session (action-conditioned, closed-loop)
    # -----------------------------------------------------------------------

    @app.websocket("/v2/world/session")
    async def world_session(ws: WebSocket) -> None:
        """Bidirectional interactive world-model session.

        Protocol: the client sends a ``ResetRequest`` JSON to open the session,
        then one ``Action`` JSON per step; the server replies with a
        ``LatentStep`` JSON per step (``step_index`` 0 acknowledges the reset).
        State persists for the life of the connection. Engine calls run in a
        threadpool so the event loop stays free — the same rationale as the v2
        driver thread. See ADR-0008.
        """
        await ws.accept()
        engine = active_interactive
        if engine is None:
            await ws.send_json({"error": "no interactive engine configured"})
            await ws.close(code=1008)
            return
        try:
            reset = ResetRequest.model_validate_json(await ws.receive_text())
        except ValidationError:
            await ws.send_json({"error": "first message must be a ResetRequest"})
            await ws.close(code=1008)
            return
        except WebSocketDisconnect:
            return
        state: WorldState = await run_in_threadpool(
            engine.reset, reset.conditioning, reset.params
        )
        await ws.send_text(LatentStep(step_index=0).model_dump_json())
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    action = Action.model_validate_json(raw)
                except ValidationError:
                    await ws.send_json({"error": "invalid action"})
                    continue
                state, latent_step = await run_in_threadpool(engine.step, state, action)
                await ws.send_text(latent_step.model_dump_json())
        except WebSocketDisconnect:
            return

    return app
