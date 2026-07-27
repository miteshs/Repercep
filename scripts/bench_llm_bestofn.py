#!/usr/bin/env python3
"""Measure the best-of-N serving lever against a stock OpenAI-compatible server.

This is the gate experiment for ``docs/BUSINESS_PLAN_2026.md`` §6.3 — the table
whose 3x multiplier is transferred from a VLA measurement and has never been run
on LLM decode. The design, the sweeps and the **binding** decision rule are
pre-registered in ``docs/LLM_BESTOFN_PLAN.md``; read that first. This script
only measures.

The rung ladder (plan §2), all against one already-loaded server:

    R0  N separate requests, prefix caching OFF   -> strawman floor, never a baseline
    R1  N separate requests, prefix caching ON    -> the realistic baseline
    R2  ONE request with n=N                      -> the ceiling vLLM already offers

The claimable lever is ``L = R1 / R2`` (plan §5): not "we beat vLLM", but "we
deliver vLLM's own best path to workloads emitted as N separate calls, without
the caller rewriting anything". R3 (Repercep fusing R1-shaped traffic into an R2
call) is the *same upstream call* as R2, so it is deliberately not measured here
— only its overhead is, later, and only if the lever survives.

Because APC is a *server* flag, R0 vs R1 needs either two server launches or the
per-rung prefix randomization this script does by default (``--fresh-prefix``),
which denies the cache a hit without touching server config. Which one was used
is recorded in the result line; do not mix them silently.

    # against a vLLM server on :8001
    python scripts/bench_llm_bestofn.py --base-url http://127.0.0.1:8001/v1 \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --candidates 4,8,16,32 --prefix-tokens 2048 --decode-tokens 32 \\
        --concurrency 1,4,16 --repeats 10

    # harness self-check, no server, no GPU
    python scripts/bench_llm_bestofn.py --self-test

Emits one ``[bestofn] RESULT {...}`` JSON line, same convention as
``scripts/bench_vla_levers.py`` so the rows sit together without an accounting
asterisk.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import httpx

# The agentic operating point the decision rule in plan §5 is evaluated at.
_GATE_POINT = {"candidates": 16, "prefix_tokens": 2048, "decode_tokens": 32, "concurrency": 4}
_GATE_KEEP_AS_IS = 3.0
_GATE_FLOOR = 1.5


@dataclass
class RungResult:
    """Timing for one rung at one sweep point."""

    rung: str
    wall_s: list[float] = field(default_factory=list)
    completion_tokens: int = 0
    n_texts: int = 0
    sample_digest: str = ""

    @property
    def median_s(self) -> float:
        return statistics.median(self.wall_s) if self.wall_s else 0.0

    @property
    def stdev_pct(self) -> float:
        if len(self.wall_s) < 2:
            return 0.0
        m = statistics.mean(self.wall_s)
        return 100.0 * statistics.stdev(self.wall_s) / m if m else 0.0


def _synthetic_prefix(n_tokens: int, rng: random.Random) -> str:
    """A prefix of roughly ``n_tokens`` tokens.

    Uses common English words so the tokenizer yields ~1 token/word rather than
    the 3-4 a random-character string would produce; the point is to control the
    *shared prefix length*, which is the mechanism under test (plan §3.1), so a
    wrong-by-4x prefix would silently move the independent variable.
    """
    vocab = [
        "the", "system", "shall", "observe", "each", "state", "and", "report",
        "a", "concise", "summary", "of", "what", "changed", "since", "previous",
        "step", "including", "any", "anomaly", "worth", "noting",
    ]  # fmt: skip
    return " ".join(rng.choice(vocab) for _ in range(max(1, n_tokens)))


def _digest(texts: Sequence[str]) -> str:
    """Order-insensitive digest of a candidate set, for the parity gate."""
    import hashlib

    h = hashlib.sha256()
    for t in sorted(texts):
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class _Client:
    """Thin OpenAI-compatible completions client.

    Deliberately not the ``openai`` SDK: this measures wall-clock serving time
    and must not carry an SDK's retry/backoff behaviour into the measurement.
    """

    def __init__(self, base_url: str, model: str, api_key: str | None, timeout_s: float):
        self._model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def complete(
        self, prompt: str, *, n: int, max_tokens: int, seed: int, temperature: float
    ) -> dict[str, Any]:
        """One /completions call. ``n>1`` is the R2 path; ``n==1`` the R0/R1 path."""
        body = {
            "model": self._model,
            "prompt": prompt,
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        r = await self._http.post("/completions", json=body)
        r.raise_for_status()
        payload: dict[str, Any] = r.json()
        return payload


def _texts_and_tokens(payload: dict[str, Any]) -> tuple[list[str], int]:
    texts = [c.get("text", "") for c in payload.get("choices", [])]
    used = int((payload.get("usage") or {}).get("completion_tokens", 0))
    return texts, used


async def _run_r2(
    client: _Client, prompt: str, n: int, decode: int, seed: int, temperature: float
) -> tuple[list[str], int]:
    """One request, n=N — the ceiling vLLM already offers."""
    payload = await client.complete(
        prompt, n=n, max_tokens=decode, seed=seed, temperature=temperature
    )
    return _texts_and_tokens(payload)


async def _run_loop(
    client: _Client, prompt: str, n: int, decode: int, seed: int, temperature: float
) -> tuple[list[str], int]:
    """N separate requests, issued concurrently — the R0/R1 shape.

    Concurrent, not serial: a real agent framework fans these out. A serial loop
    would inflate the lever by measuring request latency N times over, which is
    the strawman this experiment exists to avoid.

    Each request gets ``seed + i``. With a shared seed and temperature > 0 every
    request would independently draw the *same* continuation, so R1 would return
    N copies of one candidate while R2's ``n=N`` returns N distinct ones — equal
    token counts, but not the same work semantically, and not best-of-N. Under
    the greedy parity pass the seed is irrelevant, so this does not weaken the
    correctness gate.
    """
    results = await asyncio.gather(
        *(
            client.complete(prompt, n=1, max_tokens=decode, seed=seed + i, temperature=temperature)
            for i in range(n)
        )
    )
    texts: list[str] = []
    used = 0
    for payload in results:
        t, u = _texts_and_tokens(payload)
        texts.extend(t)
        used += u
    return texts, used


async def _time_rung(
    client: _Client,
    rung: str,
    *,
    n: int,
    prefix_tokens: int,
    decode: int,
    concurrency: int,
    repeats: int,
    fresh_prefix: bool,
    rng: random.Random,
    seed: int,
    temperature: float,
) -> RungResult:
    """Median wall-time to obtain all N candidates for one decision.

    ``concurrency`` simulates C independent decisions in flight — the axis plan
    §4 names as the most likely killer, since continuous batching may already
    saturate the GPU before our lever gets a chance to.

    ``temperature`` must be > 0 for a *timed* run: best-of-N only exists because
    the candidates differ, and a greedy n=N is both unrepresentative and open to
    being special-cased by the server. Greedy is used only by the separate
    parity pass in ``_parity_check``.
    """
    runner = _run_r2 if rung == "R2" else _run_loop
    out = RungResult(rung=rung)

    # Seeded off ``seed`` alone, so R1 and R2 see the *same* shared prefix and
    # APC can hit for both. R0 overrides this per call, below.
    stable_prefix = _synthetic_prefix(prefix_tokens, random.Random(seed))

    def _prompt() -> str:
        # R0 denies the prefix cache a hit by making the prefix unique per call;
        # R1/R2 reuse one prefix so APC can do its job. This is the in-harness
        # alternative to relaunching the server with --no-enable-prefix-caching.
        if fresh_prefix and rung == "R0":
            salt = "".join(rng.choice(string.ascii_lowercase) for _ in range(24))
            return f"{salt} {stable_prefix}"
        return stable_prefix

    # Warm up. Every rung gets its own warm-up, so R1 and R2 are both measured
    # against a warm cache — the alternative (R2 inheriting R1's warm APC) would
    # flatter R2 and understate the lever.
    await asyncio.gather(
        *(runner(client, _prompt(), n, decode, seed, temperature) for _ in range(concurrency))
    )

    for _ in range(repeats):
        t0 = time.perf_counter()
        batches = await asyncio.gather(
            *(runner(client, _prompt(), n, decode, seed, temperature) for _ in range(concurrency))
        )
        out.wall_s.append(time.perf_counter() - t0)
        texts, used = batches[0]
        out.completion_tokens = used
        out.n_texts = len(texts)
        out.sample_digest = _digest(texts)
    return out


async def _equivalence_check(
    client: _Client, *, prefix_tokens: int, decode: int, seed: int
) -> tuple[bool, str, str]:
    """Correctness gate — run untimed, alongside the timed sweep.

    **Exact candidate-set parity between R1 and R2 is impossible, by
    construction, and this is a property of the server rather than a gap in the
    harness.** Two independent reasons:

    1. vLLM rejects greedy ``n>1`` outright — *"n must be 1 when using greedy
       sampling"* — so the obvious gate (compare greedy candidate sets) cannot
       be executed at all.
    2. Under sampling, R2's ``n`` candidates are drawn from one shared RNG
       stream while R1's N requests each seed their own, so the two sets are
       different draws from the same distribution and will never be equal.

    What *is* checkable, and is the thing a throughput comparison actually
    needs, is done instead:

    - **this function** — the two request shapes reduce to the same thing at
      ``n=1, temperature=0``, proving the harness builds equivalent requests
      rather than accidentally asking for different work;
    - **token accounting in ``_sweep``** — both rungs must report the same
      completion-token total, which is what proves they decoded the same amount.

    Text equality was the wrong gate for a throughput claim anyway; equal decode
    work is the right one.
    """
    prompt = _synthetic_prefix(prefix_tokens, random.Random(seed))
    r1_texts, _ = await _run_loop(client, prompt, 1, decode, seed, 0.0)
    r2_texts, _ = await _run_r2(client, prompt, 1, decode, seed, 0.0)
    d1, d2 = _digest(r1_texts), _digest(r2_texts)
    return d1 == d2, d1, d2


async def _sweep(client: _Client, args: argparse.Namespace, rng: random.Random) -> dict[str, Any]:
    results: dict[str, Any] = {}
    parity: list[str] = []
    for n in args.candidates:
        for prefix_tokens in args.prefix_tokens:
            for decode in args.decode_tokens:
                for conc in args.concurrency:
                    key = f"N{n}_p{prefix_tokens}_d{decode}_c{conc}"
                    point: dict[str, Any] = {}
                    for rung in ("R0", "R1", "R2"):
                        res = await _time_rung(
                            client,
                            rung,
                            n=n,
                            prefix_tokens=prefix_tokens,
                            decode=decode,
                            concurrency=conc,
                            repeats=args.repeats,
                            fresh_prefix=args.fresh_prefix,
                            rng=rng,
                            seed=args.seed,
                            temperature=args.temperature,
                        )
                        point[rung] = {
                            "median_s": round(res.median_s, 4),
                            "stdev_pct": round(res.stdev_pct, 2),
                            "candidates_returned": res.n_texts,
                            "completion_tokens": res.completion_tokens,
                        }
                        if res.n_texts != n:
                            point[rung]["WARN"] = f"expected {n} candidates, got {res.n_texts}"

                    r1, r2 = point["R1"]["median_s"], point["R2"]["median_s"]
                    point["lever_R1_over_R2"] = round(r1 / r2, 2) if r2 else 0.0
                    point["strawman_R0_over_R2"] = (
                        round(point["R0"]["median_s"] / r2, 2) if r2 else 0.0
                    )
                    # Correctness gate, untimed (plan §6). Two parts, because
                    # exact candidate-set parity is impossible here — see
                    # _equivalence_check.
                    ok, d1, d2 = await _equivalence_check(
                        client, prefix_tokens=prefix_tokens, decode=decode, seed=args.seed
                    )
                    point["greedy_shape_ok"] = ok
                    if not ok:
                        point["greedy_digests"] = {"R1": d1, "R2": d2}

                    # The load-bearing check: equal decode work across rungs. If
                    # R1 and R2 did not decode the same number of tokens, the
                    # wall-clock ratio is not a serving lever, it is an
                    # accounting error.
                    t1 = point["R1"]["completion_tokens"]
                    t2 = point["R2"]["completion_tokens"]
                    tokens_ok = t1 > 0 and t2 > 0 and abs(t1 - t2) / max(t1, t2) <= 0.02
                    point["tokens_ok"] = tokens_ok
                    point["completion_tokens"] = {"R1": t1, "R2": t2}

                    point["parity_ok"] = ok and tokens_ok
                    if not point["parity_ok"]:
                        parity.append(key)
                    results[key] = point
                    print(
                        f"[bestofn] {key}: R0 {point['R0']['median_s'] * 1e3:.0f}ms  "
                        f"R1 {r1 * 1e3:.0f}ms  R2 {r2 * 1e3:.0f}ms  "
                        f"lever(R1/R2)={point['lever_R1_over_R2']}x"
                        f"{'' if point['parity_ok'] else '  [PARITY MISMATCH]'}",
                        flush=True,
                    )
    return {"points": results, "parity_mismatches": parity}


def _verdict(points: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered decision rule (plan §5) to the gate point."""
    key = (
        f"N{_GATE_POINT['candidates']}_p{_GATE_POINT['prefix_tokens']}"
        f"_d{_GATE_POINT['decode_tokens']}_c{_GATE_POINT['concurrency']}"
    )
    point = points.get(key)
    if point is None:
        return {
            "gate_point": key,
            "measured": False,
            "note": "gate point not in sweep; decision rule cannot be applied",
        }
    lever = point["lever_R1_over_R2"]
    if lever >= _GATE_KEEP_AS_IS:
        action = "BUSINESS_PLAN §6.3 stands as written; publish the ladder"
    elif lever >= _GATE_FLOOR:
        action = f"REBUILD BUSINESS_PLAN §6.3 at the measured lever ({lever}x), not 3x"
    else:
        action = (
            "DELETE BUSINESS_PLAN §6.3. Drop the LLM efficiency claim from deck, "
            "plan, site and report. Compete on breadth + silicon only (PIVOT §9)"
        )
    return {"gate_point": key, "measured": True, "lever": lever, "action": action}


def _self_test() -> int:
    """Exercise the rung logic against a mock upstream — no server, no GPU.

    Proves request construction, the n=1-loop vs n=N shapes, parity digesting
    and the result-line format before a GPU session pays for those bugs.
    """
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        n = int(body.get("n", 1))
        # Deterministic body so greedy parity holds across rungs. The flat
        # per-call cost means R2 (one call) beats an R1 fan-out (N calls) here —
        # that is the mock exercising the code path, NOT a performance result.
        time.sleep(0.002)
        return httpx.Response(
            200,
            json={
                "choices": [{"text": " candidate"} for _ in range(n)],
                "usage": {"completion_tokens": 7 * n},
            },
        )

    async def run() -> dict[str, Any]:
        client = _Client("http://mock.test/v1", "mock-model", None, 30.0)
        client._http = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://mock.test/v1"
        )
        args = argparse.Namespace(
            candidates=[4],
            prefix_tokens=[64],
            decode_tokens=[8],
            concurrency=[1],
            repeats=3,
            fresh_prefix=True,
            seed=0,
            temperature=0.8,
        )
        out = await _sweep(client, args, random.Random(0))
        await client.aclose()
        return out

    out = asyncio.run(run())
    point = out["points"]["N4_p64_d8_c1"]

    failures = []
    if point["R1"]["candidates_returned"] != 4:
        failures.append(f"R1 returned {point['R1']['candidates_returned']} candidates, want 4")
    if point["R2"]["candidates_returned"] != 4:
        failures.append(f"R2 returned {point['R2']['candidates_returned']} candidates, want 4")
    if not point["parity_ok"]:
        failures.append("greedy parity gate failed on a deterministic mock")
    n_ones = sum(1 for c in calls if int(c.get("n", 1)) == 1)
    n_many = sum(1 for c in calls if int(c.get("n", 1)) > 1)
    if n_many == 0 or n_ones == 0:
        failures.append(f"expected both n=1 and n>1 call shapes, saw {n_ones}/{n_many}")
    # The timed sweep must sample (temperature > 0); the parity gate must be
    # greedy. Both shapes have to appear, or one of the two is not running.
    temps = {c["temperature"] for c in calls}
    if 0.0 not in temps:
        failures.append("no greedy calls seen — the parity gate did not run")
    if not any(t > 0.0 for t in temps):
        failures.append("no sampling calls seen — the timed sweep ran greedy")
    # R0 must not reuse the R1/R2 prefix, or the cache-denial is a no-op.
    prompts = {c["prompt"][:24] for c in calls}
    if len(prompts) < 2:
        failures.append("R0 prefix randomization did not produce distinct prompts")

    for f in failures:
        print(f"[bestofn] SELF-TEST FAIL: {f}", flush=True)
    if failures:
        return 1
    print(
        f"[bestofn] self-test OK — {len(calls)} mock calls, "
        f"{n_ones} at n=1, {n_many} at n>1, parity gate held",
        flush=True,
    )
    return 0


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default=None, help="Model id as the server names it.")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--candidates", type=_int_list, default=[4, 8, 16, 32])
    ap.add_argument("--prefix-tokens", type=_int_list, default=[256, 2048, 8192])
    ap.add_argument("--decode-tokens", type=_int_list, default=[8, 32, 200])
    ap.add_argument("--concurrency", type=_int_list, default=[1, 4, 16])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for the TIMED sweep. Must be > 0 — best-of-N "
        "only exists because the candidates differ, and a greedy n=N is both "
        "unrepresentative and open to server-side special-casing. The greedy "
        "parity gate runs separately and is never timed.",
    )
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument(
        "--fresh-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="R0 randomizes its prefix to deny APC a hit (the alternative is "
        "relaunching the server with prefix caching disabled).",
    )
    ap.add_argument("--out", type=Path, default=None, help="Also write the result JSON here.")
    ap.add_argument("--self-test", action="store_true", help="Mock-upstream check; no server.")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if not args.model:
        print("[bestofn] --model is required (or use --self-test)", flush=True)
        return 2
    if args.temperature <= 0.0:
        print(
            "[bestofn] --temperature must be > 0 for a timed run: a greedy n=N is "
            "not a best-of-N workload (plan §6). The parity gate handles greedy.",
            flush=True,
        )
        return 2

    rng = random.Random(args.seed)

    async def _go() -> dict[str, Any]:
        # Sweep and close in ONE event loop. A second asyncio.run() cannot close
        # a client whose connections are bound to the first loop — it raises
        # "Event loop is closed" and discards a completed run's results.
        client = _Client(args.base_url, args.model, args.api_key, args.timeout_s)
        try:
            return await _sweep(client, args, rng)
        finally:
            await client.aclose()

    swept = asyncio.run(_go())

    out = {
        "experiment": "llm_bestofn",
        "plan": "docs/LLM_BESTOFN_PLAN.md",
        "model": args.model,
        "base_url": args.base_url,
        "repeats": args.repeats,
        "seed": args.seed,
        "temperature_timed": args.temperature,
        "temperature_parity": 0.0,
        "r0_cache_denial": "fresh-prefix" if args.fresh_prefix else "server-flag",
        "points": swept["points"],
        "parity_mismatches": swept["parity_mismatches"],
        "verdict": _verdict(swept["points"]),
    }
    line = "[bestofn] RESULT " + json.dumps(out)
    print(line, flush=True)
    if args.out:
        args.out.write_text(line + "\n")
    if swept["parity_mismatches"]:
        print(
            "[bestofn] PARITY MISMATCH at "
            f"{len(swept['parity_mismatches'])} point(s) — treat every number in this "
            "run as suspect until explained (plan §6).",
            flush=True,
        )
        return 1
    print(f"[bestofn] VERDICT: {out['verdict'].get('action', 'n/a')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
