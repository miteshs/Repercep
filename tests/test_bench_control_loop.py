"""Tests for ``scripts/bench_control_loop.py`` — the control-loop leaderboard harness.

Drives the measurement core against the V-JEPA engine with toy weights (the
``--fake`` path), so the four leaderboard metrics are exercised end-to-end on
CPU without downloads. Mirrors ``test_fvd.py``'s load-by-path convention for
script modules.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "scripts" / "bench_control_loop.py"
_spec = importlib.util.spec_from_file_location("bench_control_loop_mod", _SRC)
assert _spec is not None and _spec.loader is not None
_clb = importlib.util.module_from_spec(_spec)
sys.modules["bench_control_loop_mod"] = _clb
_spec.loader.exec_module(_clb)


def _fake_args(**overrides: object) -> object:
    import argparse

    ns = argparse.Namespace(
        engine="vjepa2-ac",
        backend="cpu",
        dtype="fp32",
        fake=True,
        action_dim=4,
        plan_samples=6,
        plan_cem_iters=2,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_bench_control_loop_fake_engine_metrics() -> None:
    pytest.importorskip("torch")
    engine = _clb.build_engine(_fake_args())
    result = _clb.bench_control_loop(
        engine, warmup=1, step_iters=5, plan_calls=1, horizon=2, action_dim=4
    )
    assert result["mode"] == "control_loop_bench_v0"
    assert result["model"] == "vjepa2-ac-300m"
    assert result["step_ms_warm"] > 0
    assert result["planning_decisions_per_sec"] > 0
    # V-JEPA-class engine: energy metrics populated from the CEM config.
    assert result["energy_evals_per_plan"] == 6 * 2
    assert result["energy_evals_per_sec"] > 0
    # CPU run: no HBM story to tell.
    assert result["resident_sessions_per_gpu"] is None
    assert result["state_carryover"] is True


def test_bench_control_loop_result_is_json_serializable() -> None:
    pytest.importorskip("torch")
    import json

    engine = _clb.build_engine(_fake_args())
    result = _clb.bench_control_loop(
        engine, warmup=0, step_iters=2, plan_calls=1, horizon=2, action_dim=4
    )
    assert json.loads(json.dumps(result))["plan_horizon"] == 2


def test_bench_control_loop_chunked_engine_forces_real_plan_call() -> None:
    """LingBot-VA-shaped (chunked) engines: plan() must do real work, not a
    cache hit off the pending_actions a preceding step() parked (the bug this
    guards: the harness's goal-computation step() would otherwise make every
    timed plan() call a free no-op lookup instead of a real chunk denoise).
    """
    pytest.importorskip("torch")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from repercep.models.lingbot_va import LingBotVAConfig, LingBotVAEngine
    from test_lingbot_va import _FakePipeline, _NamedBackend  # type: ignore[import-not-found]

    class _CountingPipeline(_FakePipeline):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            self.infer_chunk_calls = 0

        def infer_chunk(self, *args: object, **kwargs: object) -> object:
            self.infer_chunk_calls += 1
            return super().infer_chunk(*args, **kwargs)  # type: ignore[misc]

    pipeline = _CountingPipeline(chunk=2, action_dim=3, dim=4)
    engine = LingBotVAEngine(
        _NamedBackend("fake"),  # type: ignore[arg-type]
        LingBotVAConfig(prompt="test", frame_chunk_size=2, action_per_frame=1, used_action_dim=3),
        pipeline=pipeline,
    )

    result = _clb.bench_control_loop(
        engine, warmup=1, step_iters=1, plan_calls=2, horizon=2, obs_dir="unused"
    )
    assert result["state_carryover"] is True
    # warmup step + timed step + goal step = 3 step()-driven infer_chunk calls,
    # then one REAL infer_chunk per timed plan() call (pending_actions cleared
    # each time) — 3 + 2 = 5. Without the fix this would be 3 + 1 (or fewer):
    # the goal step's cached proposal would serve every plan() call for free.
    assert pipeline.infer_chunk_calls == 3 + 2
