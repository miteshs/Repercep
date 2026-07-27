"""Coalesce concurrent prefix-sharing completions into one upstream ``n=N`` call.

This is the serving lever measured in ``docs/LLM_BESTOFN_RESULT.md``, turned
into a gateway feature. The measurement found that the win does **not** come
from prefill arithmetic — it comes from a *cache-timing race*:

    N prefix-sharing requests issued simultaneously all miss the prefix cache
    together, because none of them has finished prefilling to populate it.
    Automatic prefix caching is a temporal optimization, and a simultaneous
    fan-out defeats it. One ``n=N`` request shares the prefill structurally,
    inside a single request, and cannot lose that race.

So the product is not "we beat the upstream engine" — it is "we deliver the
upstream's own best path to callers who emit N separate requests, without them
rewriting anything." Agent frameworks emit exactly that shape.

Measured on H100 + vLLM 0.26.0 with prefix caching **on**: 1.5x at a 2k shared
prefix with 32-token decodes at moderate concurrency, 1.04x-2.5x across the
shape grid. Modest, real, and workload-dependent — see the result doc before
quoting a number.

**Off by default**, like the proxy it plugs into. Fusing changes the shape of an
upstream call, so it is opt-in.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

_LOG = logging.getLogger(__name__)

# Fields that must match for two requests' responses to be interchangeable.
# Anything outside this set makes a request unfusable rather than silently
# fused under a assumption we did not check.
_FUSION_KEY_FIELDS = (
    "model",
    "prompt",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "logit_bias",
    "seed",
)

# Request fields we understand well enough to reason about. A body carrying
# anything else is passed through untouched — the safe default is to not fuse.
_KNOWN_FIELDS = frozenset(
    (
        *_FUSION_KEY_FIELDS,
        "n",
        "stream",
        "logprobs",
        "echo",
        "user",
        "best_of",
        "stream_options",
    )
)


def is_fusable(body: dict[str, Any]) -> bool:
    """Whether a ``/v1/completions`` body may be coalesced with its peers.

    Deliberately conservative — every ``False`` here costs at most the lever,
    while a wrong ``True`` corrupts someone's response.
    """
    # Streaming callers each want their own token stream; a fused call has one.
    if body.get("stream"):
        return False
    # Only fuse callers asking for a single completion. A caller already using
    # n>1 has done the fusing themselves and needs nothing from us.
    if int(body.get("n") or 1) != 1:
        return False
    if int(body.get("best_of") or 1) != 1:
        return False
    # **Greedy must never be fused.** vLLM rejects n>1 under greedy sampling
    # outright ("n must be 1 when using greedy sampling"), so a fused greedy
    # batch would 400 for callers whose individual requests were perfectly
    # valid. Learned from the measurement, not from the docs.
    if float(body.get("temperature", 1.0)) <= 0.0:
        return False
    # A prompt must exist and be a single string: list prompts are already a
    # batch with their own index semantics, which we would have to re-derive.
    if not isinstance(body.get("prompt"), str) or not body["prompt"]:
        return False
    # Unknown fields could change response shape or sampling in ways this
    # module has not reasoned about.
    return all(k in _KNOWN_FIELDS for k in body)


def fusion_key(body: dict[str, Any]) -> str:
    """Canonical key identifying requests whose completions are interchangeable.

    Two bodies share a key only if every sampling-relevant field matches, so a
    caller can be handed any choice from the fused response.
    """
    keyed = {k: body[k] for k in _FUSION_KEY_FIELDS if k in body}
    return json.dumps(keyed, sort_keys=True, separators=(",", ":"))


class _Batch:
    """One in-flight coalescing window for a single fusion key."""

    __slots__ = ("bodies", "futures", "key", "timer")

    def __init__(self, key: str) -> None:
        self.key = key
        self.futures: list[asyncio.Future[dict[str, Any]]] = []
        self.bodies: list[dict[str, Any]] = []
        self.timer: asyncio.Task[None] | None = None

    def __len__(self) -> int:
        return len(self.futures)


class CandidateFuser:
    """Coalesces concurrent equivalent completions into one ``n=N`` upstream call.

    A request arriving with a fusion key that has no open batch opens one and
    starts a short timer. Peers arriving inside that window join it. The batch
    fires when the window closes or ``max_fuse`` is reached, whichever is first.

    Failure policy: a fused call that errors resolves **every** waiter with the
    same error, which is what each would have seen individually — with one
    exception. If the upstream rejects the *fusion itself* (a 4xx caused by
    ``n>1``), waiters are retried individually, because that failure is ours,
    not theirs.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        window_ms: float = 8.0,
        max_fuse: int = 32,
        api_key: str | None = None,
    ) -> None:
        self._client = client
        self._window_s = max(0.0, window_ms / 1000.0)
        self._max_fuse = max(1, max_fuse)
        self._api_key = api_key
        self._batches: dict[str, _Batch] = {}
        self._lock = asyncio.Lock()
        self.stats = {"fused_calls": 0, "requests_fused": 0, "passthrough": 0, "fallbacks": 0}

    async def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        """Return one completion payload, fusing with concurrent peers if possible.

        The returned payload is shaped exactly like an unfused ``n=1`` response,
        so callers cannot tell whether fusion happened — except via the
        additive ``repercep_fusion`` block, which never replaces a standard
        field.
        """
        if self._max_fuse == 1 or not is_fusable(body):
            self.stats["passthrough"] += 1
            return await self._post(body)

        key = fusion_key(body)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()

        async with self._lock:
            batch = self._batches.get(key)
            if batch is None:
                batch = _Batch(key)
                self._batches[key] = batch
                batch.timer = asyncio.create_task(self._fire_after_window(key))
            batch.futures.append(future)
            batch.bodies.append(body)
            full = len(batch) >= self._max_fuse

        if full:
            # Reached the cap before the window closed — fire immediately rather
            # than making the earliest waiter pay the remaining window.
            await self._fire(key)
        return await future

    async def _fire_after_window(self, key: str) -> None:
        try:
            await asyncio.sleep(self._window_s)
        except asyncio.CancelledError:
            return
        await self._fire(key)

    async def _fire(self, key: str) -> None:
        async with self._lock:
            batch = self._batches.pop(key, None)
            if batch is None or not batch.futures:
                return
            # Cancel the window timer only when someone else is firing (the
            # max_fuse path). When the timer itself is the caller, cancelling
            # here would cancel the *running* task, raising CancelledError at
            # the next await inside this method and leaving every waiter's
            # future unresolved — a permanent hang, not a slow request.
            timer = batch.timer
            if timer is not None and timer is not asyncio.current_task() and not timer.done():
                timer.cancel()

        n = len(batch)
        if n == 1:
            # Nobody joined; send it as the plain n=1 request it already is,
            # so a quiet gateway never pays a fusion penalty.
            self.stats["passthrough"] += 1
            await self._resolve_with(batch, self._post(batch.bodies[0]), fused=False)
            return

        fused_body = dict(batch.bodies[0])
        fused_body["n"] = n
        try:
            payload = await self._post(fused_body)
        except _UpstreamRejectedFusionError:
            # The upstream refused the n>1 shape. Our doing, so make each
            # waiter whole by sending its own original request.
            self.stats["fallbacks"] += 1
            _LOG.warning("llm fusion rejected upstream (n=%d); falling back to %d calls", n, n)
            await asyncio.gather(
                *(self._resolve_one(f, b) for f, b in zip(batch.futures, batch.bodies, strict=True))
            )
            return
        except Exception as exc:
            for f in batch.futures:
                if not f.done():
                    f.set_exception(exc)
            return

        self.stats["fused_calls"] += 1
        self.stats["requests_fused"] += n
        self._distribute(batch, payload, n)

    def _distribute(self, batch: _Batch, payload: dict[str, Any], n: int) -> None:
        """Hand each waiter its own choice, reshaped as a standalone response."""
        choices = payload.get("choices") or []
        if len(choices) < n:
            # Fewer completions than callers — resolve what we can and fail the
            # rest explicitly rather than handing anyone another's text.
            _LOG.error("llm fusion got %d choices for %d waiters", len(choices), n)
        usage = payload.get("usage") or {}
        total_completion = int(usage.get("completion_tokens") or 0)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)

        for i, future in enumerate(batch.futures):
            if future.done():
                continue
            if i >= len(choices):
                future.set_exception(
                    RuntimeError(f"fused upstream returned {len(choices)} choices for {n} callers")
                )
                continue
            choice = dict(choices[i])
            choice["index"] = 0
            single = {
                k: v for k, v in payload.items() if k not in ("choices", "usage", "repercep_fusion")
            }
            single["choices"] = [choice]
            # Per-caller usage is reported as if unfused: this is what the caller
            # would have been billed alone, and it is what their client expects.
            # The true shared cost is disclosed separately rather than hidden.
            single["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total_completion // n if n else total_completion,
                "total_tokens": prompt_tokens + (total_completion // n if n else total_completion),
            }
            single["repercep_fusion"] = {
                "fused": True,
                "batch_size": n,
                "shared_prompt_tokens": prompt_tokens,
                "batch_completion_tokens": total_completion,
            }
            future.set_result(single)

    async def _resolve_one(
        self, future: asyncio.Future[dict[str, Any]], body: dict[str, Any]
    ) -> None:
        if future.done():
            return
        try:
            future.set_result(await self._post(body))
        except Exception as exc:
            future.set_exception(exc)

    async def _resolve_with(self, batch: _Batch, coro: Any, *, fused: bool) -> None:
        try:
            payload = await coro
        except Exception as exc:
            for f in batch.futures:
                if not f.done():
                    f.set_exception(exc)
            return
        for f in batch.futures:
            if not f.done():
                f.set_result(payload)

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"content-type": "application/json"}
        if self._api_key is not None:
            headers["authorization"] = f"Bearer {self._api_key}"
        resp = await self._client.post("/v1/completions", json=body, headers=headers)
        if resp.status_code >= 400:
            text = resp.text
            if int(body.get("n") or 1) > 1 and 400 <= resp.status_code < 500:
                raise _UpstreamRejectedFusionError(text)
            raise _UpstreamError(resp.status_code, text)
        payload: dict[str, Any] = resp.json()
        return payload


class _UpstreamError(RuntimeError):
    """Upstream returned a non-2xx for a request we did not reshape."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"upstream {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class _UpstreamRejectedFusionError(RuntimeError):
    """Upstream 4xx'd a request *because* we fused it — recoverable by us."""


__all__ = ["CandidateFuser", "fusion_key", "is_fusable"]
