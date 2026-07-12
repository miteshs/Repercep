"""Tests for the ``/v2/world/session`` interactive WebSocket surface.

Drives the bidirectional session end-to-end through FastAPI's TestClient with a
stub action-conditioned engine. Gated on torch (the stub holds a latent tensor)
and fastapi; skips cleanly where either is absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from repercep.runtime.engine import EngineInfo
from repercep.runtime.types import (
    Action,
    ConditioningInput,
    LatentStep,
    ResetRequest,
    RolloutParams,
    WorldState,
)

if TYPE_CHECKING:
    import torch


class _StubInteractive:
    """Counts steps; echoes ``len(action.values)`` as the step energy."""

    def info(self) -> EngineInfo:
        return EngineInfo(
            model_name="stub-ac", backend="cpu", device="cpu:0", dtype="float32", ready=True
        )

    def reset(self, conditioning: ConditioningInput, params: RolloutParams) -> WorldState:
        import torch

        return WorldState(context=torch.zeros(1, 2), step_index=0, session_id="sess")

    def step(self, state: WorldState, action: Action) -> tuple[WorldState, LatentStep]:
        nxt = WorldState(
            context=state.context, step_index=state.step_index + 1, session_id=state.session_id
        )
        return nxt, LatentStep(step_index=nxt.step_index, energy=float(len(action.values)))

    def plan(self, state: WorldState, goal: torch.Tensor, horizon: int) -> Action:
        return Action(values=[0.0])


def test_interactive_session_streams_steps() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from repercep.serving.app import create_app

    with (
        TestClient(create_app(interactive_engine=_StubInteractive())) as client,
        client.websocket_connect("/v2/world/session") as ws,
    ):
        ws.send_text(ResetRequest().model_dump_json())
        ack = ws.receive_json()
        assert ack["step_index"] == 0  # reset acknowledged

        for expected in (1, 2, 3):
            ws.send_text(Action(values=[0.5, -0.5]).model_dump_json())
            msg = ws.receive_json()
            assert msg["step_index"] == expected
            assert msg["energy"] == 2.0  # len(action.values)


def test_interactive_session_requires_engine() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from repercep.serving.app import create_app

    with (
        TestClient(create_app()) as client,
        client.websocket_connect("/v2/world/session") as ws,
    ):
        assert "error" in ws.receive_json()
