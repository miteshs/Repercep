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
