"""OpenAI-compatible reverse-proxy to a co-located vLLM/SGLang server.

The runtime's *world-model* path (router → scheduler → driver) and this LLM
path are deliberately separate. An LLM upstream (vLLM, SGLang) already does
continuous batching + paged attention *internally*; routing its traffic
through Repercep's single-driver v2 scheduler (``serving/driver.py``) would
**serialize exactly what the upstream parallelizes** — strictly worse than
talking to it directly. So this module is a thin, stateless async passthrough
that reuses only the gateway's auth and deployment surface. See
``docs/LLM_PROXY.md`` for the design rationale and co-location recipe.

Security note: the client's own ``Authorization`` header (the *gateway* token,
meaningless upstream) is never forwarded. A configured ``api_key`` is injected
instead, so a secured upstream (``vllm serve --api-key``) works without leaking
the gateway credential.
"""

from __future__ import annotations

import json
import logging

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from repercep.serving.llm_fusing import CandidateFuser, is_fusable

_LOG = logging.getLogger(__name__)

# A connection-level failure to the upstream (down, wrong port, DNS) maps to
# 502 Bad Gateway — the gateway is up, the thing behind it isn't.
_UPSTREAM_UNREACHABLE = 502


class LlmProxy:
    """Thin OpenAI-compatible reverse-proxy to one upstream LLM server.

    Construct once per app, mount :attr:`router` behind the gateway's auth
    dependency, and :meth:`aclose` on shutdown. Stateless beyond the pooled
    :class:`httpx.AsyncClient`, so it holds none of the world-model path's
    per-session state.
    """

    def __init__(
        self,
        *,
        upstream_url: str,
        timeout_s: float = 600.0,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        fusing_enabled: bool = False,
        fusing_window_ms: float = 8.0,
        fusing_max_batch: int = 32,
    ) -> None:
        self._upstream_url = upstream_url.rstrip("/")
        self._api_key = api_key
        # An injected client (tests use ``httpx.MockTransport``) carries its own
        # base_url; otherwise build one pinned to the upstream. Constructing an
        # AsyncClient outside a running loop is fine — it lazily opens the pool.
        self._client = client or httpx.AsyncClient(base_url=self._upstream_url, timeout=timeout_s)
        # Off by default. When enabled, only /v1/completions is eligible — chat
        # completions carry message structure this has not been measured on.
        self._fuser = (
            CandidateFuser(
                self._client,
                window_ms=fusing_window_ms,
                max_fuse=fusing_max_batch,
                api_key=api_key,
            )
            if fusing_enabled
            else None
        )
        self.router = self._build_router()

    async def aclose(self) -> None:
        """Close the pooled upstream client. Idempotent per httpx semantics."""
        await self._client.aclose()

    # -- upstream request helpers -------------------------------------------

    def _upstream_headers(self, *, content_type: str, accept: str | None) -> dict[str, str]:
        headers = {"content-type": content_type}
        if accept is not None:
            headers["accept"] = accept
        if self._api_key is not None:
            headers["authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _proxy_post_stream(self, path: str, request: Request) -> Response:
        """Forward a POST body upstream and stream the response back verbatim."""
        body = await request.body()
        headers = self._upstream_headers(
            content_type=request.headers.get("content-type", "application/json"),
            accept=request.headers.get("accept"),
        )
        upstream_req = self._client.build_request("POST", path, content=body, headers=headers)
        try:
            upstream = await self._client.send(upstream_req, stream=True)
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=_UPSTREAM_UNREACHABLE, detail=f"llm upstream unreachable: {exc}"
            ) from exc
        if upstream.status_code >= 400:
            # Surface the upstream's own error body/status rather than masking
            # it — a 400 from vLLM (bad params) should read as a 400 here.
            detail = (await upstream.aread()).decode(errors="replace")
            await upstream.aclose()
            raise HTTPException(status_code=upstream.status_code, detail=detail)
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            background=BackgroundTask(upstream.aclose),
        )

    async def _proxy_get(self, path: str) -> Response:
        """Forward a GET upstream and return its (non-streamed) JSON body."""
        headers = (
            {"authorization": f"Bearer {self._api_key}"} if self._api_key is not None else None
        )
        try:
            upstream = await self._client.get(path, headers=headers)
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=_UPSTREAM_UNREACHABLE, detail=f"llm upstream unreachable: {exc}"
            ) from exc
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    async def _maybe_fused_completion(self, request: Request) -> Response | None:
        """Serve a completion through the fuser, or ``None`` to fall through.

        Returning ``None`` for anything unrecognised means an enabled fuser can
        never make a request *fail* that the plain passthrough would have
        served — the worst case is that it does not get coalesced.
        """
        assert self._fuser is not None
        raw = await request.body()
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None  # let the upstream author the parse error
        if not isinstance(body, dict) or not is_fusable(body):
            return None
        try:
            payload = await self._fuser.complete(body)
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=_UPSTREAM_UNREACHABLE, detail=f"llm upstream unreachable: {exc}"
            ) from exc
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if isinstance(status, int):
                raise HTTPException(
                    status_code=status, detail=getattr(exc, "detail", str(exc))
                ) from exc
            raise HTTPException(status_code=_UPSTREAM_UNREACHABLE, detail=str(exc)) from exc
        return JSONResponse(payload)

    async def _readiness(self) -> Response:
        """Best-effort upstream reachability probe (readiness, not liveness).

        ``/v1/models`` is implemented by both vLLM and SGLang and needs the
        weights loaded to answer, so a 200 here is a genuine "ready to serve"
        signal — unlike the gateway's own always-200 ``/health`` liveness.
        """
        try:
            probe = await self._client.get("/v1/models")
            ok = probe.status_code == 200
        except httpx.RequestError:
            ok = False
        return JSONResponse(
            {"upstream": "ok" if ok else "unreachable", "url": self._upstream_url},
            status_code=200 if ok else 503,
        )

    def _build_router(self) -> APIRouter:
        router = APIRouter(tags=["llm"])

        @router.post("/v1/chat/completions")
        async def chat_completions(request: Request) -> Response:
            return await self._proxy_post_stream("/v1/chat/completions", request)

        @router.post("/v1/completions")
        async def completions(request: Request) -> Response:
            if self._fuser is not None:
                fused = await self._maybe_fused_completion(request)
                if fused is not None:
                    return fused
            return await self._proxy_post_stream("/v1/completions", request)

        @router.get("/v1/models")
        async def models() -> Response:
            return await self._proxy_get("/v1/models")

        @router.get("/v1/llm/health")
        async def llm_health() -> Response:
            return await self._readiness()

        return router


__all__ = ["LlmProxy"]
