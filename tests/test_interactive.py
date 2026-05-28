"""Tests for the interactive (action-conditioned) world-model seam.

Exercises the wire contract and Protocol conformance. The parts that build
latent tensors are gated on torch and skip cleanly on a box without it, matching
the rest of the suite (accelerator-specific paths skip rather than fail).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from pydantic import ValidationError

from mirage.models.vjepa2_ac import VJepa2ACConfig, VJepa2ACEngine
from mirage.runtime.engine import EngineInfo
from mirage.runtime.interactive import InteractiveWorldModel
from mirage.runtime.types import (
    Action,
    ConditioningInput,
    LatentStep,
    ResetRequest,
    RolloutParams,
    WorldState,
)

if TYPE_CHECKING:
    import torch

    from mirage.backend.protocol import Backend


class _NamedBackend:
    """Minimal stand-in carrying just the attribute the engine reads (``name``)."""

    def __init__(self, name: str) -> None:
        self.name = name


class _StubInteractive:
    """A torch-backed stub world model that satisfies ``InteractiveWorldModel``."""

    def info(self) -> EngineInfo:
        return EngineInfo(
            model_name="stub", backend="cpu", device="cpu:0", dtype="float32", ready=True
        )

    def reset(self, conditioning: ConditioningInput, params: RolloutParams) -> WorldState:
        import torch

        return WorldState(context=torch.zeros(1, 4), step_index=0, session_id="s0")

    def step(self, state: WorldState, action: Action) -> tuple[WorldState, LatentStep]:
        nxt = WorldState(
            context=state.context,
            step_index=state.step_index + 1,
            session_id=state.session_id,
        )
        return nxt, LatentStep(step_index=nxt.step_index)

    def plan(self, state: WorldState, goal: torch.Tensor, horizon: int) -> Action:
        return Action(values=[0.0])


# --- wire types (no torch) ---


def test_action_requires_at_least_one_value() -> None:
    with pytest.raises(ValidationError):
        Action(values=[])
    assert Action(values=[0.1, -0.2], space="ee_delta").space == "ee_delta"


def test_action_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        Action.model_validate({"values": [0.0], "bogus": 1})


def test_rollout_params_bounds() -> None:
    assert RolloutParams().horizon == 16
    with pytest.raises(ValidationError):
        RolloutParams(horizon=0)
    with pytest.raises(ValidationError):
        RolloutParams(horizon=10_000)


def test_reset_request_defaults() -> None:
    req = ResetRequest()
    assert req.conditioning.kind is ConditioningInput().kind
    assert req.params.decode_pixels is False


def test_latent_step_envelope_carries_no_tensor() -> None:
    step = LatentStep(step_index=3, energy=1.5)
    assert '"step_index":3' in step.model_dump_json()
    assert LatentStep(step_index=0).frame is None


# --- Protocol conformance (no torch) ---


def test_engine_satisfies_interactive_protocol() -> None:
    # Method-only Protocol → issubclass works with no instance and no torch.
    assert issubclass(VJepa2ACEngine, InteractiveWorldModel)


def test_stub_is_interactive_instance() -> None:
    assert isinstance(_StubInteractive(), InteractiveWorldModel)


def test_engine_info_and_scaffold_state() -> None:
    engine = VJepa2ACEngine(cast("Backend", _NamedBackend("fake-cpu")), VJepa2ACConfig())
    info = engine.info()
    assert info.model_name == "vjepa2-ac-300m"
    assert info.ready is False
    # The model-dependent path is scaffolded until the AC predictor is ported.
    with pytest.raises(NotImplementedError):
        engine.load()


# --- seam loop contract (needs torch for the latent tensors) ---


def test_step_advances_and_streams_in_order() -> None:
    pytest.importorskip("torch")
    engine = _StubInteractive()
    state = engine.reset(ConditioningInput(), RolloutParams())
    seen: list[int] = []
    for _ in range(3):
        state, step = engine.step(state, Action(values=[0.0]))
        seen.append(step.step_index)
    assert seen == [1, 2, 3]
    assert state.step_index == 3
