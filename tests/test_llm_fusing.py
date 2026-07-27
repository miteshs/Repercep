"""Tests for candidate fusing (``serving/llm_fusing.py``).

The upstream is faked with ``httpx.MockTransport`` — no LLM server, no GPU —
following ``test_llm_proxy.py``. What is asserted is the contract that makes
fusing safe to turn on: concurrent equivalent requests become ONE upstream call,
each caller gets its OWN distinct completion shaped exactly like an unfused
response, and everything we are not certain about is passed through instead.

The correctness cases matter more than the happy path here: this code hands one
caller a response produced by a request another caller made.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from repercep.serving.llm_fusing import CandidateFuser, fusion_key, is_fusable

if TYPE_CHECKING:
    from collections.abc import Callable

_BASE = "http://llm-upstream.test"


def _body(**over: Any) -> dict[str, Any]:
    b: dict[str, Any] = {
        "model": "m",
        "prompt": "shared prefix",
        "max_tokens": 8,
        "temperature": 0.8,
    }
    b.update(over)
    return b


def _fuser(
    handler: Callable[[httpx.Request], httpx.Response], **kw: Any
) -> tuple[CandidateFuser, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def _record(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(_record), base_url=_BASE)
    return CandidateFuser(client, **kw), seen


def _ok(request: httpx.Request) -> httpx.Response:
    """Upstream that returns n distinct completions and honest usage."""
    body = json.loads(request.content)
    n = int(body.get("n") or 1)
    return httpx.Response(
        200,
        json={
            "id": "cmpl-1",
            "object": "text_completion",
            "model": body["model"],
            "choices": [{"text": f" candidate-{i}", "index": i} for i in range(n)],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 8 * n,
                "total_tokens": 100 + 8 * n,
            },
        },
    )


# -- eligibility ------------------------------------------------------------


@pytest.mark.parametrize(
    ("over", "why"),
    [
        ({"stream": True}, "streaming callers each need their own stream"),
        ({"n": 4}, "caller already fused it themselves"),
        ({"best_of": 2}, "best_of changes selection semantics"),
        ({"temperature": 0.0}, "greedy: upstream rejects n>1 outright"),
        ({"prompt": ["a", "b"]}, "list prompt is already a batch"),
        ({"prompt": ""}, "empty prompt"),
        ({"suffix": "x"}, "unknown field we have not reasoned about"),
    ],
)
def test_ineligible_requests_are_not_fused(over: dict[str, Any], why: str) -> None:
    assert is_fusable(_body(**over)) is False, why


def test_plain_single_completion_is_fusable() -> None:
    assert is_fusable(_body()) is True


def test_greedy_is_never_fusable() -> None:
    """The rule learned from the measurement, not the docs.

    vLLM rejects ``n>1`` under greedy sampling ("n must be 1 when using greedy
    sampling"). Fusing a greedy batch would 400 for callers whose individual
    requests were entirely valid.
    """
    assert is_fusable(_body(temperature=0.0)) is False
    assert is_fusable(_body(temperature=0.01)) is True


def test_fusion_key_separates_incompatible_sampling() -> None:
    assert fusion_key(_body()) == fusion_key(_body())
    assert fusion_key(_body()) != fusion_key(_body(temperature=0.9))
    assert fusion_key(_body()) != fusion_key(_body(prompt="other"))
    assert fusion_key(_body()) != fusion_key(_body(max_tokens=9))
    # Fields that do not affect interchangeability must not split the batch.
    assert fusion_key(_body()) == fusion_key(_body(user="alice"))


# -- fusing behaviour -------------------------------------------------------


def test_concurrent_equivalent_requests_become_one_upstream_call() -> None:
    fuser, seen = _fuser(_ok, window_ms=25, max_fuse=32)

    async def run() -> list[dict[str, Any]]:
        return await asyncio.gather(*(fuser.complete(_body()) for _ in range(6)))

    results = asyncio.run(run())

    assert len(seen) == 1, f"expected 1 fused upstream call, saw {len(seen)}"
    assert seen[0]["n"] == 6
    assert fuser.stats["fused_calls"] == 1
    assert fuser.stats["requests_fused"] == 6
    # Every caller got a DISTINCT completion — nobody was handed a duplicate.
    texts = [r["choices"][0]["text"] for r in results]
    assert len(set(texts)) == 6, texts


def test_each_caller_gets_an_unfused_response_shape() -> None:
    fuser, _ = _fuser(_ok, window_ms=25)

    async def run() -> list[dict[str, Any]]:
        return await asyncio.gather(*(fuser.complete(_body()) for _ in range(4)))

    for r in asyncio.run(run()):
        assert len(r["choices"]) == 1, "a fused caller must not see its peers"
        assert r["choices"][0]["index"] == 0, "index must be re-based to a lone response"
        assert r["usage"]["prompt_tokens"] == 100
        assert r["usage"]["completion_tokens"] == 8
        assert r["object"] == "text_completion"
        # Disclosure is additive and never replaces a standard field.
        assert r["repercep_fusion"]["fused"] is True
        assert r["repercep_fusion"]["batch_size"] == 4


def test_incompatible_requests_do_not_share_a_batch() -> None:
    fuser, seen = _fuser(_ok, window_ms=25)

    async def run() -> tuple[dict[str, Any], ...]:
        return await asyncio.gather(
            fuser.complete(_body()),
            fuser.complete(_body()),
            fuser.complete(_body(prompt="different")),
            fuser.complete(_body(temperature=0.2)),
        )

    asyncio.run(run())
    assert len(seen) == 3, "three distinct fusion keys => three upstream calls"
    assert sorted(int(s.get("n") or 1) for s in seen) == [1, 1, 2]


def test_max_fuse_caps_the_batch() -> None:
    fuser, seen = _fuser(_ok, window_ms=200, max_fuse=3)

    async def run() -> list[dict[str, Any]]:
        return await asyncio.gather(*(fuser.complete(_body()) for _ in range(6)))

    results = asyncio.run(run())
    assert all(int(s["n"]) <= 3 for s in seen), [s.get("n") for s in seen]
    assert len(results) == 6
    assert all(len(r["choices"]) == 1 for r in results)


def test_lone_request_is_sent_unfused() -> None:
    """A quiet gateway must never pay a fusion penalty."""
    fuser, seen = _fuser(_ok, window_ms=5)
    result = asyncio.run(fuser.complete(_body()))
    assert len(seen) == 1
    assert int(seen[0].get("n") or 1) == 1
    assert fuser.stats["fused_calls"] == 0
    assert "repercep_fusion" not in result


def test_ineligible_request_bypasses_fusing_entirely() -> None:
    fuser, seen = _fuser(_ok, window_ms=200)
    asyncio.run(fuser.complete(_body(temperature=0.0)))
    assert len(seen) == 1
    assert int(seen[0].get("n") or 1) == 1
    assert fuser.stats["passthrough"] == 1


# -- failure policy ---------------------------------------------------------


def test_upstream_rejecting_the_fusion_falls_back_to_individual_calls() -> None:
    """A 4xx caused by ``n>1`` is our fault, so every caller is made whole."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls["n"] += 1
        if int(body.get("n") or 1) > 1:
            return httpx.Response(
                400, json={"error": {"message": "n must be 1 when using greedy sampling"}}
            )
        return _ok(request)

    fuser, seen = _fuser(handler, window_ms=25)

    async def run() -> list[dict[str, Any]]:
        return await asyncio.gather(*(fuser.complete(_body()) for _ in range(3)))

    results = asyncio.run(run())
    assert len(results) == 3
    assert all(len(r["choices"]) == 1 for r in results), "every caller still served"
    assert fuser.stats["fallbacks"] == 1
    # One rejected fused attempt, then one call per caller.
    assert [int(s.get("n") or 1) for s in seen] == [3, 1, 1, 1]


def test_upstream_server_error_propagates_to_every_waiter() -> None:
    """A 5xx would have hit each caller individually too — nobody hangs."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "upstream down"})

    fuser, _ = _fuser(handler, window_ms=25)

    async def run() -> list[BaseException | dict[str, Any]]:
        return await asyncio.gather(
            *(fuser.complete(_body()) for _ in range(3)), return_exceptions=True
        )

    results = asyncio.run(run())
    assert len(results) == 3
    assert all(isinstance(r, Exception) for r in results), "no waiter left unresolved"


def test_short_upstream_choice_list_fails_rather_than_duplicating() -> None:
    """If the upstream returns fewer completions than callers, the unserved
    callers get an error — never another caller's text.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        n = int(body.get("n") or 1)
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "text_completion",
                "model": body["model"],
                # Deliberately one short.
                "choices": [{"text": f" c{i}", "index": i} for i in range(max(0, n - 1))],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8 * n},
            },
        )

    fuser, _ = _fuser(handler, window_ms=25)

    async def run() -> list[BaseException | dict[str, Any]]:
        return await asyncio.gather(
            *(fuser.complete(_body()) for _ in range(4)), return_exceptions=True
        )

    results = asyncio.run(run())
    served = [r for r in results if isinstance(r, dict)]
    failed = [r for r in results if isinstance(r, Exception)]
    assert len(served) == 3
    assert len(failed) == 1
    texts = [r["choices"][0]["text"] for r in served]
    assert len(set(texts)) == len(texts), "no caller received a duplicate completion"


def test_max_fuse_one_disables_fusing() -> None:
    fuser, seen = _fuser(_ok, window_ms=200, max_fuse=1)

    async def run() -> list[dict[str, Any]]:
        return await asyncio.gather(*(fuser.complete(_body()) for _ in range(3)))

    asyncio.run(run())
    assert len(seen) == 3
    assert all(int(s.get("n") or 1) == 1 for s in seen)
    assert fuser.stats["fused_calls"] == 0
