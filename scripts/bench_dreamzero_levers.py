#!/usr/bin/env python3
"""Serving-latency ladder for the REAL DreamZero pipeline.

Follows the ``bench_lingbot_va_levers.py`` house style (timed stages, warm-up
then measure, one verbatim ``RESULT`` JSON line) but ladders through the
DreamZero-specific levers scoped in ``docs/DREAMZERO_PORT_PLAN.md`` §4:

- **rung 0 (true baseline)**: ``num_dit_steps=16`` — every DiT call in the
  16-step joint denoise loop actually runs. This is deliberately NOT the
  reference's own out-of-the-box behavior (see rung 0b) — it is the
  full-compute number everything else in this ladder is measured against.
- **rung 0b (reference default)**: ``num_dit_steps=8`` — the static
  ``NUM_DIT_STEPS`` skip the reference silently runs with no opt-in flag
  (port plan §2 point 5). Their published "~3s/chunk H100" claim is already
  this number, not rung 0's — reported side by side so the two are never
  conflated.
- **rung 1 (batched CFG)**: ``cfg_batched=True`` — the single-GPU
  differentiator vs. the reference's 2-GPU ``ip=2`` split (which only
  distributes the same two sequential cond/uncond forwards across ranks,
  same total compute). ``DreamZeroPipeline`` does not yet implement the
  batched forward (needs ``WANPolicyHead._run_diffusion_steps`` read from the
  research clone on a GPU pod) — it warns and falls back to sequential CFG.
  This script detects that fallback and reports the rung as not measurable
  rather than silently recording a sequential-CFG number under a "batched"
  label.
- **rung 2 (cfg_scale=1)**: sets the CFG combine weight to 1 (mathematically
  ``flow_pred = cond``). Whether the reference's ``_run_diffusion_steps``
  actually *skips* the uncond forward at this value (a real latency win, as
  confirmed for LingBot-VA's ``guidance_scale=1.0``) or still runs both
  passes and only changes the combine math (no latency win) is **not
  confirmed** for DreamZero — this rung reports the number with that
  uncertainty disclosed, not asserted as a win.
- **rung 3 (DiT-step scan + dynamic cache)**: ``num_dit_steps`` in
  ``{12, 8, 4}``, plus the independent dynamic cosine-similarity skip
  schedule (``enable_dit_cache=True``) at the rung-0 step count — two
  separate skip mechanisms (port plan §2 point 5), reported separately.
- **rung 4 (torch.compile)**: new ``DreamZeroConfig.compile`` flag. Cold
  (first chunk, compile tax included) and warm reported separately. Phase 1
  hit ``FailOnRecompileLimitHit`` running eager-forced
  (``TORCHDYNAMO_DISABLE=1``) — if compile still breaks here, this reports
  "not measurable" rather than a fabricated number (LingBot-VA's rung 3 was
  itself a negative result; this may be too).
- **rung 5 (KV window)**: ``local_attn_size`` reduced from the checkpoint
  default (21 frames) — primarily a KV-memory lever (pair with
  ``bench_control_loop.py --engine dreamzero --attn-window N --sessions N``
  for the marginal-HBM side of this number); latency is reported here as a
  secondary signal.

Quality guardrail (disclosed, not oversold): same as LingBot-VA — no offline
task-success metric exists for DreamZero in this repo, so the only quality
signal is per-channel |action| magnitude-distribution consistency (p50, p95)
against rung 0's, at a fixed seed. Consistency is evidence a lever didn't
silently change *what* the model outputs; it is not a correctness or
task-success proof.

This is a GPU-pod deliverable: it needs the research repo importable as
``groot`` plus the ``GEAR-Dreams/DreamZero-DROID`` checkpoint
(``DreamZeroEngine.load()`` raises a ``RuntimeError`` with the setup recipe
otherwise — this script does not catch or swallow that error for the
baseline load, so running it without a GPU fails immediately and
informatively).

    python scripts/bench_dreamzero_levers.py \
        --obs-dir /workspace/dreamzero/example/droid \
        --prompt "pick up the mug"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any


def _setup() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))


_setup()

import torch  # noqa: E402

from repercep.backend.registry import select_backend  # noqa: E402
from repercep.models.dreamzero import DreamZeroConfig, DreamZeroEngine  # noqa: E402
from repercep.runtime.types import (  # noqa: E402
    Action,
    ConditioningInput,
    ConditioningKind,
    RolloutParams,
)

if TYPE_CHECKING:
    from repercep.backend.protocol import Backend
    from repercep.runtime.types import WorldState


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _magnitude_stats(actions: torch.Tensor) -> dict[str, list[float]]:
    """Per-channel ``|action|`` p50/p95 — see the module docstring's guardrail note."""
    if actions.numel() == 0:
        return {"p50": [], "p95": []}
    mag = actions.abs().float()
    p50 = torch.quantile(mag, 0.5, dim=0)
    p95 = torch.quantile(mag, 0.95, dim=0)
    return {"p50": [round(v, 4) for v in p50.tolist()], "p95": [round(v, 4) for v in p95.tolist()]}


def _max_rel_diff(a: list[float], b: list[float]) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    diffs = [abs(x - y) / max(abs(y), 1e-6) for x, y in zip(a, b, strict=True)]
    return round(max(diffs), 4)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _load_engine(backend: Backend, **cfg_kwargs: Any) -> tuple[DreamZeroEngine, float]:
    """Construct + load a fresh engine. NOT wrapped in try/except here — the
    ``groot`` "not importable"/checkpoint-missing ``RuntimeError`` from
    ``DreamZeroEngine.load()`` must propagate uncaught so this script fails
    loudly (with the setup recipe) rather than silently, exactly as running
    the engine directly would.
    """
    engine = DreamZeroEngine(backend, DreamZeroConfig(**cfg_kwargs))
    t = time.perf_counter()
    engine.load()
    _sync()
    return engine, time.perf_counter() - t


def _open_session(engine: DreamZeroEngine, obs_dir: str, prompt: str) -> WorldState:
    return engine.reset(
        ConditioningInput(kind=ConditioningKind.IMAGE, uri=obs_dir), RolloutParams()
    )


def _run_chunks(
    engine: DreamZeroEngine,
    state: WorldState,
    goal: torch.Tensor,
    *,
    warmup: int,
    measure: int,
) -> tuple[WorldState, list[float], torch.Tensor]:
    """Drive plan→step warmup then measured chunks. Returns the trailing
    state, per-chunk ms (measured chunks only), and the proposed actions
    concatenated across measured chunks (for the magnitude guardrail).
    """
    horizon = engine._config.num_action_per_block

    for _ in range(warmup):
        engine.plan(state, goal, horizon=horizon)
        proposed = engine._sessions[state.session_id].pending_actions
        assert proposed is not None
        state, _ = engine.step(state, Action(values=proposed.flatten().tolist()))

    chunk_ms: list[float] = []
    action_rows: list[torch.Tensor] = []
    for _ in range(measure):
        engine.plan(state, goal, horizon=horizon)
        proposed = engine._sessions[state.session_id].pending_actions
        assert proposed is not None
        _sync()
        t = time.perf_counter()
        state, _ = engine.step(state, Action(values=proposed.flatten().tolist()))
        _sync()
        chunk_ms.append((time.perf_counter() - t) * 1000)
        action_rows.append(proposed.detach().float().cpu())
    actions_cat = (
        torch.cat(action_rows, dim=0) if action_rows else torch.zeros(0, 1, dtype=torch.float32)
    )
    return state, chunk_ms, actions_cat


def _run_rung_detecting_fallback(
    engine: DreamZeroEngine,
    state: WorldState,
    goal: torch.Tensor,
    *,
    warmup: int,
    measure: int,
) -> tuple[WorldState, list[float], torch.Tensor, list[str]]:
    """Like :func:`_run_chunks`, but also captures any
    ``DreamZeroPipeline._apply_levers`` fallback warnings (e.g.
    ``cfg_batched`` not yet implemented) so the caller can mark the rung
    "not measurable" instead of recording a misleadingly-labeled number.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        state, chunk_ms, actions = _run_chunks(engine, state, goal, warmup=warmup, measure=measure)
        messages = [str(w.message) for w in caught]
    return state, chunk_ms, actions, messages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="GEAR-Dreams/DreamZero-DROID", help="HF id or local bundle dir")
    ap.add_argument("--obs-dir", required=True, help="dir with seed camera images")
    ap.add_argument("--prompt", default="pick up the mug")
    ap.add_argument("--backend", choices=["auto", "cuda", "rocm"], default="auto")
    ap.add_argument("--warmup", type=int, default=2, help="untimed chunks before measuring")
    ap.add_argument("--measure", type=int, default=6, help="timed chunks per rung")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    backend = select_backend(prefer=None if args.backend == "auto" else args.backend)
    dev = backend.devices()[0]
    print(f"[levers] backend={backend.name} device={dev.name}", flush=True)

    rungs: dict[str, Any] = {}
    rung0_mag: dict[str, list[float]] | None = None

    def _record(
        name: str,
        chunk_ms: list[float],
        actions_cat: torch.Tensor,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        nonlocal rung0_mag
        mag: dict[str, Any] = _magnitude_stats(actions_cat)
        if rung0_mag is None:
            rung0_mag = {"p50": mag["p50"], "p95": mag["p95"]}
        else:
            mag["p50_max_rel_diff_vs_rung0"] = _max_rel_diff(mag["p50"], rung0_mag["p50"])
            mag["p95_max_rel_diff_vs_rung0"] = _max_rel_diff(mag["p95"], rung0_mag["p95"])
        entry = {
            "chunk_ms": [round(x, 1) for x in chunk_ms],
            "chunk_ms_warm_mean": round(_mean(chunk_ms), 1),
            "action_magnitude": mag,
        }
        if extra:
            entry.update(extra)
        rungs[name] = entry
        print(f"[levers] {name}: warm_mean={entry['chunk_ms_warm_mean']}ms", flush=True)

    goal = torch.zeros(1)  # policy-regime: the prompt is the goal; tensor unused

    # --- rungs 0 - 3 (except dynamic-cache/step-scan, see below): one load,
    # reused. num_dit_steps/enable_dit_cache/cfg_scale are read fresh every
    # _infer call (DreamZeroPipeline._apply_levers), so mutating
    # engine._config between rungs is enough -- no cache-shape dependency the
    # way LingBot-VA's guidance_scale has, so no fresh reset() is strictly
    # required, but one is taken per rung anyway for a clean per-rung
    # first-block prime, matching house style. ---
    engine, load_s = _load_engine(backend, repo=args.repo, prompt=args.prompt)
    print(f"[levers] load (rungs 0-3, shared): {load_s:.1f}s", flush=True)

    torch.manual_seed(args.seed)
    state = _open_session(engine, args.obs_dir, args.prompt)
    state, chunk_ms, actions = _run_chunks(engine, state, goal, warmup=args.warmup, measure=args.measure)
    _record("rung0_baseline_16steps", chunk_ms, actions, extra={"num_dit_steps": 16})

    engine._config.num_dit_steps = 8
    torch.manual_seed(args.seed)
    state = _open_session(engine, args.obs_dir, args.prompt)
    state, chunk_ms, actions = _run_chunks(engine, state, goal, warmup=args.warmup, measure=args.measure)
    _record(
        "rung0b_reference_default_8steps",
        chunk_ms,
        actions,
        extra={
            "num_dit_steps": 8,
            "note": "the reference's own undisclosed default (NUM_DIT_STEPS=8) "
            "-- their published ~3s/chunk H100 claim is this number, not rung 0's",
        },
    )
    engine._config.num_dit_steps = 16

    engine._config.cfg_batched = True
    torch.manual_seed(args.seed)
    state = _open_session(engine, args.obs_dir, args.prompt)
    state, chunk_ms, actions, fallback_msgs = _run_rung_detecting_fallback(
        engine, state, goal, warmup=args.warmup, measure=args.measure
    )
    if fallback_msgs:
        rungs["rung1_cfg_batched"] = {
            "measurable": False,
            "reason": "DreamZeroPipeline fell back to sequential CFG "
            "(cfg_batched not yet implemented -- needs the research clone's "
            "_run_diffusion_steps read on a GPU pod, port plan §4)",
            "warning": fallback_msgs[0],
        }
        print("[levers] rung1_cfg_batched: not measurable -- pipeline fell back", flush=True)
    else:
        _record("rung1_cfg_batched", chunk_ms, actions, extra={"cfg_batched": True})
    engine._config.cfg_batched = False

    engine._config.cfg_scale = 1.0
    torch.manual_seed(args.seed)
    state = _open_session(engine, args.obs_dir, args.prompt)
    state, chunk_ms, actions = _run_chunks(engine, state, goal, warmup=args.warmup, measure=args.measure)
    _record(
        "rung2_cfg_scale_1",
        chunk_ms,
        actions,
        extra={
            "cfg_scale": 1.0,
            "note": "whether this actually skips the reference's uncond forward "
            "(a real latency lever, confirmed for LingBot-VA's guidance_scale=1.0) "
            "or only changes the CFG combine math (no latency win) is NOT "
            "confirmed for DreamZero -- read _run_diffusion_steps to resolve",
        },
    )
    engine._config.cfg_scale = 5.0

    for steps in (12, 8, 4):
        engine._config.num_dit_steps = steps
        torch.manual_seed(args.seed)
        state = _open_session(engine, args.obs_dir, args.prompt)
        state, chunk_ms, actions = _run_chunks(
            engine, state, goal, warmup=args.warmup, measure=args.measure
        )
        _record(f"rung3_dit_steps_{steps}", chunk_ms, actions, extra={"num_dit_steps": steps})
    engine._config.num_dit_steps = 16

    engine._config.enable_dit_cache = True
    torch.manual_seed(args.seed)
    state = _open_session(engine, args.obs_dir, args.prompt)
    state, chunk_ms, actions = _run_chunks(engine, state, goal, warmup=args.warmup, measure=args.measure)
    _record(
        "rung3_dynamic_dit_cache",
        chunk_ms,
        actions,
        extra={"num_dit_steps": 16, "enable_dit_cache": True},
    )
    engine._config.enable_dit_cache = False

    # --- rung 4: torch.compile — needs a fresh transformer construction. ---
    try:
        compiled_engine, compiled_load_s = _load_engine(
            backend, repo=args.repo, prompt=args.prompt, compile=True
        )
        torch.manual_seed(args.seed)
        state = _open_session(compiled_engine, args.obs_dir, args.prompt)
        # Cold: first measured chunk, compile tax included.
        state, cold_ms, _cold_actions = _run_chunks(
            compiled_engine, state, goal, warmup=0, measure=1
        )
        # Warm: compile already paid for, isolated from rungs 0-3's mean.
        state, warm_ms, warm_actions = _run_chunks(
            compiled_engine, state, goal, warmup=args.warmup, measure=args.measure
        )
        _record(
            "rung4_torch_compile",
            warm_ms,
            warm_actions,
            extra={
                "load_seconds": round(compiled_load_s, 1),
                "cold_first_chunk_ms": round(cold_ms[0], 1) if cold_ms else None,
            },
        )
    except Exception as exc:  # the graph-break/compile failure mode itself
        rungs["rung4_torch_compile"] = {
            "measurable": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"[levers] rung4 torch.compile: not measurable — {exc}", flush=True)

    # --- rung 5: KV window (local_attn_size). Primarily a memory lever --
    # pair with bench_control_loop.py --attn-window N --sessions N for the
    # marginal-HBM side; latency reported here as a secondary signal. ---
    for window in (12, 6):
        engine._config.local_attn_size = window
        torch.manual_seed(args.seed)
        state = _open_session(engine, args.obs_dir, args.prompt)
        state, chunk_ms, actions = _run_chunks(
            engine, state, goal, warmup=args.warmup, measure=args.measure
        )
        _record(
            f"rung5_local_attn_size_{window}",
            chunk_ms,
            actions,
            extra={
                "local_attn_size": window,
                "note": "latency only -- pair with bench_control_loop.py "
                "--attn-window for the marginal-HBM/session-density side",
            },
        )
    engine._config.local_attn_size = None

    result = {
        "bench": "dreamzero_levers",
        "device": dev.name,
        "seed": args.seed,
        "warmup_chunks": args.warmup,
        "measured_chunks": args.measure,
        "quality_guardrail_note": (
            "action-magnitude (|action| p50/p95 per channel) vs rung0, at a "
            "fixed seed, is the only quality signal available for DreamZero "
            "in this repo — there is no offline task-success metric. "
            "Consistency shows a lever didn't silently change the output "
            "distribution; it does not prove task correctness."
        ),
        "rungs": rungs,
    }
    print("RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
