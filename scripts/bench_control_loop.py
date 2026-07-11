#!/usr/bin/env python3
"""Control-Loop Serving Benchmark v0 — the leaderboard harness.

Drives ANY :class:`mirage.runtime.interactive.InteractiveWorldModel` through
the closed-loop protocol (``reset -> step* -> plan``) and measures the four
metrics `docs/CONTROL_LOOP_BENCH.md` defines as the control-regime serving
leaderboard nobody else publishes:

1. **Closed-loop step latency under state carryover** — warm ms per ``step``
   on one persistent session (the state carries; no re-encode per step).
2. **Planning-decisions/sec** — full ``plan()`` calls per second.
3. **Energy-evals/sec** — rollout-energy evaluations per second inside a plan
   (V-JEPA-class energy planners; ``null`` for policy-mode engines).
4. **Resident sessions per GPU** — measured single-session marginal HBM
   extrapolated against the device's total (weights counted once).

Emits one verbatim ``RESULT`` JSON line (repo provenance convention).

    python scripts/bench_control_loop.py --engine vjepa2-ac          # real weights, GPU
    python scripts/bench_control_loop.py --engine vjepa2-ac --fake   # CPU toy weights (CI)
    python scripts/bench_control_loop.py --engine lingbot-va \
        --obs-dir /workspace/lingbot-va/example/demo                # real weights, GPU
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _setup() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))


_setup()

import torch  # noqa: E402

from mirage.backend.registry import select_backend  # noqa: E402
from mirage.models.lingbot_va import LingBotVAConfig, LingBotVAEngine  # noqa: E402
from mirage.models.vjepa2_ac import VJepa2ACConfig, VJepa2ACEngine  # noqa: E402
from mirage.runtime.types import (  # noqa: E402
    Action,
    ConditioningInput,
    ConditioningKind,
    RolloutParams,
)

_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _ToyEncoder:
    """Deterministic (1, T, D) features — the CPU/CI stand-in for real weights."""

    def __init__(self, frames: int, dim: int) -> None:
        self._ctx = torch.arange(frames * dim, dtype=torch.float32).reshape(frames, dim) / dim

    def get_vision_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        return self._ctx.unsqueeze(0)


class _ToyPredictor:
    """Linear toy dynamics: next frame = last context frame + action."""

    def __call__(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return context[-1] + action


def build_engine(args: argparse.Namespace) -> Any:
    """Construct the engine under test from CLI args."""
    backend = select_backend(prefer=None if args.backend == "auto" else args.backend)
    if args.engine == "vjepa2-ac":
        cfg = VJepa2ACConfig(
            action_dim=args.action_dim,
            context_frames=8,
            dtype=_DTYPES[args.dtype],
            plan_samples=args.plan_samples,
            # Elites must not exceed samples (topk) — clamp for small CI budgets.
            plan_elites=min(8, args.plan_samples),
            plan_iters=args.plan_cem_iters,
        )
        if args.fake:
            return VJepa2ACEngine(
                backend,
                cfg,
                encoder=_ToyEncoder(frames=cfg.context_frames, dim=args.action_dim),
                predictor=_ToyPredictor(),
            )
        return VJepa2ACEngine(backend, cfg)
    # LingBot-VA: policy-regime engine (Phase-1 pipeline, docs/LINGBOT_VA_PORT_PLAN.md
    # §4). Needs a real prompt + seed-observation directory (--prompt/--obs-dir);
    # the fake path is unsupported (its rollout is a real chunked denoise loop,
    # not a toy-weights CPU stand-in).
    if args.fake:
        raise NotImplementedError("--fake is not supported for --engine lingbot-va")
    return LingBotVAEngine(backend, LingBotVAConfig(prompt=args.prompt))


def bench_control_loop(
    engine: Any,
    *,
    warmup: int = 3,
    step_iters: int = 20,
    plan_calls: int = 1,
    horizon: int = 4,
    action_dim: int = 7,
    obs_dir: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure the four leaderboard metrics on one persistent session.

    ``action_dim``/random-action generation is V-JEPA-shaped (one control
    vector per ``step()``, advancing one latent frame). LingBot-VA's ``step()``
    is chunk-shaped instead (a full executed chunk — frame_chunk_size x
    action_per_frame rows of its wire action width — advances one denoise
    chunk); detected via ``_wire_action_dim`` (see ``mirage.models.lingbot_va``)
    so one harness drives both regimes without an engine-kind flag threaded
    through every call site.
    """
    torch.manual_seed(seed)
    on_gpu = torch.cuda.is_available()
    is_chunked = hasattr(engine, "_wire_action_dim")

    t = time.perf_counter()
    loader = getattr(engine, "load", None)
    if callable(loader) and not getattr(engine, "is_loaded", False):
        loader()
    _sync()
    load_s = time.perf_counter() - t
    baseline_gib = torch.cuda.memory_allocated() / 1024**3 if on_gpu else 0.0

    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    conditioning = (
        ConditioningInput(kind=ConditioningKind.IMAGE, uri=obs_dir)
        if is_chunked
        else ConditioningInput()
    )
    state = engine.reset(conditioning, RolloutParams(horizon=horizon))
    _sync()
    reset_s = time.perf_counter() - t

    def rand_action() -> Action:
        if is_chunked:
            n = engine._config.frame_chunk_size * engine._config.action_per_frame
            return Action(values=torch.randn(n * engine._wire_action_dim()).tolist())
        return Action(values=torch.randn(action_dim).tolist())

    # 1. closed-loop step latency under state carryover (warm).
    for _ in range(warmup):
        state, _ = engine.step(state, rand_action())
    _sync()
    t = time.perf_counter()
    for _ in range(step_iters):
        state, _ = engine.step(state, rand_action())
    _sync()
    step_s = (time.perf_counter() - t) / step_iters

    # 2./3. planning-decisions/sec and energy-evals/sec. The goal is a reached
    # state's terminal frame block (P patch-token rows; 1 on toy paths), so the
    # plan target is guaranteed feasible.
    tokens_per_frame = int(getattr(engine, "_tokens_per_frame", 1))
    goal_state, _ = engine.step(state, rand_action())
    goal = goal_state.context[-tokens_per_frame:]
    if is_chunked:
        # LingBot-VA's plan() reuses a chunk step() already parked
        # (session.pending_actions) instead of recomputing — correct for real
        # closed-loop use, but the step() above just parked one, which would
        # make the timed plan() below a free cache hit. Clear it so each timed
        # call does the real chunk denoise it's measuring.
        engine._sessions[state.session_id].pending_actions = None
    _sync()
    t = time.perf_counter()
    for _ in range(plan_calls):
        engine.plan(state, goal, horizon)
        if is_chunked:
            engine._sessions[state.session_id].pending_actions = None
    _sync()
    plan_s = (time.perf_counter() - t) / plan_calls

    cfg = getattr(engine, "_config", None)
    samples = getattr(cfg, "plan_samples", None)
    iters = getattr(cfg, "plan_iters", None)
    energy_evals = samples * iters if samples and iters else None

    # 4. resident sessions per GPU: marginal session HBM vs device total,
    # counting the (shared) weights once.
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3 if on_gpu else 0.0
    session_gib = max(peak_gib - baseline_gib, 0.0)
    if on_gpu and session_gib > 0:
        hbm_total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        resident_sessions = int((hbm_total_gib - baseline_gib) // session_gib)
    else:
        hbm_total_gib = 0.0
        resident_sessions = None

    info = engine.info()
    return {
        "mode": "control_loop_bench_v0",
        "model": info.model_name,
        "device": info.device,
        "dtype": info.dtype,
        "load_seconds": round(load_s, 1),
        "reset_seconds": round(reset_s, 3),
        "step_ms_warm": round(step_s * 1000, 2),
        "steps_per_sec": round(1 / step_s, 1) if step_s > 0 else None,
        "plan_seconds": round(plan_s, 3),
        "planning_decisions_per_sec": round(1 / plan_s, 4) if plan_s > 0 else None,
        "plan_horizon": horizon,
        "cem": {"samples": samples, "iters": iters} if samples else None,
        "energy_evals_per_plan": energy_evals,
        "energy_evals_per_sec": (
            round(energy_evals / plan_s, 1) if energy_evals and plan_s > 0 else None
        ),
        "weights_gib": round(baseline_gib, 2),
        "session_marginal_gib": round(session_gib, 2),
        "hbm_total_gib": round(hbm_total_gib, 1),
        "resident_sessions_per_gpu": resident_sessions,
        "step_iters": step_iters,
        "plan_calls": plan_calls,
        "state_carryover": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=["vjepa2-ac", "lingbot-va"], default="vjepa2-ac")
    ap.add_argument("--backend", choices=["auto", "cuda", "rocm", "cpu"], default="auto")
    ap.add_argument("--dtype", choices=sorted(_DTYPES), default="bf16")
    ap.add_argument(
        "--fake", action="store_true", help="toy encoder/predictor (CPU/CI, no weights)"
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--step-iters", type=int, default=20)
    ap.add_argument("--plan-calls", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--action-dim", type=int, default=7)
    ap.add_argument("--plan-samples", type=int, default=64)
    ap.add_argument("--plan-cem-iters", type=int, default=3)
    ap.add_argument(
        "--obs-dir", default=None, help="lingbot-va: dir with <cam_key>.png seed images"
    )
    ap.add_argument(
        "--prompt",
        default="Pick the green cube and place it inside the blue box",
        help="lingbot-va: the goal instruction (text-conditioned policy)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.engine == "lingbot-va" and not args.obs_dir:
        ap.error("--engine lingbot-va requires --obs-dir")

    engine = build_engine(args)
    print(f"[clb] engine={args.engine} fake={args.fake}", flush=True)
    result = bench_control_loop(
        engine,
        warmup=args.warmup,
        step_iters=args.step_iters,
        plan_calls=args.plan_calls,
        horizon=args.horizon,
        action_dim=args.action_dim,
        obs_dir=args.obs_dir,
        seed=args.seed,
    )
    print("[clb] RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
