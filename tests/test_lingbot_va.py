"""Tests for the LingBot-VA engine's model-agnostic chunk-loop layer.

Exercises Protocol conformance, session bookkeeping, and — with an injected
fake pipeline — the recondition-then-predict step loop and policy-mode plan.
Mirrors ``test_interactive.py``: torch-dependent parts skip cleanly without
torch; the model-specific pipeline remains a Phase-1 GPU port and is asserted
to raise (see ``docs/LINGBOT_VA_PORT_PLAN.md``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from repercep.models.lingbot_va import LingBotVAConfig, LingBotVAEngine
from repercep.runtime.interactive import InteractiveWorldModel
from repercep.runtime.types import Action, ConditioningInput, RolloutParams, WorldState

if TYPE_CHECKING:
    import torch

    from repercep.backend.protocol import Backend


class _NamedBackend:
    """Minimal stand-in carrying just the attribute the engine reads (``name``)."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakePipeline:
    """Deterministic fake of the reference server's session surface.

    ``infer_chunk`` returns latents filled with ``frame_st_id`` (so tests can
    see the cache clock advance) and an action chunk whose first row is
    ``[frame_st_id, 0, ...]``; ``recondition`` consumes ``chunk`` frames per
    executed action chunk and records every call for assertions.
    """

    def __init__(self, chunk: int = 4, action_dim: int = 30, dim: int = 8) -> None:
        self.chunk = chunk
        self.action_dim = action_dim
        self.dim = dim
        self.reset_calls: list[tuple[str, str | None]] = []
        self.recondition_calls: list[tuple[str, tuple[int, ...], int]] = []
        self.close_calls: list[str] = []

    def reset(self, session_id: str, prompt: str | None) -> None:
        self.reset_calls.append((session_id, prompt))

    def close(self, session_id: str) -> None:
        self.close_calls.append(session_id)

    def encode_observation(self, conditioning: ConditioningInput) -> torch.Tensor:
        import torch

        return torch.zeros(self.chunk, self.dim)

    def infer_chunk(
        self, session_id: str, frame_st_id: int, init_latent: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        latents = torch.full((self.chunk, self.dim), float(frame_st_id))
        actions = torch.zeros(self.chunk * 2, self.action_dim)
        actions[0, 0] = float(frame_st_id)
        return latents, actions

    def recondition(
        self,
        session_id: str,
        actions: torch.Tensor,
        obs_latent: torch.Tensor | None,
        frame_st_id: int,
    ) -> int:
        self.recondition_calls.append((session_id, tuple(actions.shape), frame_st_id))
        return self.chunk


def _toy_engine(pipeline: _FakePipeline | None = None) -> LingBotVAEngine:
    return LingBotVAEngine(
        cast("Backend", _NamedBackend("fake")),
        LingBotVAConfig(prompt="pick the green cube", action_dim=30),
        pipeline=pipeline,
    )


# --- Protocol conformance / readiness (no torch) ---


def test_engine_satisfies_interactive_protocol() -> None:
    assert issubclass(LingBotVAEngine, InteractiveWorldModel)


def test_engine_not_ready_without_pipeline() -> None:
    engine = _toy_engine(pipeline=None)
    assert engine.info().ready is False


def test_load_requires_research_repo() -> None:
    # Without the wan_va research package the real-pipeline build fails with
    # the setup recipe (the GPU box installs it per the port plan).
    with pytest.raises(RuntimeError, match="LINGBOT_VA_PORT_PLAN"):
        _toy_engine(pipeline=None).load()


# --- the chunk loop (needs torch for the latent tensors) ---


def test_reset_opens_session_with_prompt() -> None:
    pytest.importorskip("torch")
    pipeline = _FakePipeline()
    engine = _toy_engine(pipeline)
    state = engine.reset(ConditioningInput(), RolloutParams())
    assert state.step_index == 0
    assert tuple(state.context.shape) == (4, 8)
    assert pipeline.reset_calls == [(state.session_id, "pick the green cube")]


def test_release_closes_session_on_pipeline_and_drops_engine_bookkeeping() -> None:
    """``release()`` frees both layers of per-session state (the leak fix).

    Two layers hold session state: this engine's own ``_sessions`` (frame
    clock, parked action proposal) and the pipeline's session-keyed named
    KV cache. Without dropping both, a churn of short-lived sessions leaks
    GPU tensors forever.
    """
    pytest.importorskip("torch")
    pipeline = _FakePipeline()
    engine = _toy_engine(pipeline)
    state = engine.reset(ConditioningInput(), RolloutParams())
    assert state.session_id in engine._sessions

    engine.release(state)

    assert state.session_id not in engine._sessions
    assert pipeline.close_calls == [state.session_id]

    # A session release before the engine ever loaded a pipeline (e.g. a
    # WebSocket that disconnects before its first reset()) must not raise.
    _toy_engine(pipeline=None).release(state)


def test_step_reconditions_then_predicts() -> None:
    torch = pytest.importorskip("torch")
    pipeline = _FakePipeline()
    engine = _toy_engine(pipeline)
    state = engine.reset(ConditioningInput(), RolloutParams())

    executed = Action(values=[0.0] * (2 * 30), space="lingbot_va_30d")
    nxt, step = engine.step(state, executed)

    # Executed chunk shaped (k, action_dim) and pushed before predicting.
    assert pipeline.recondition_calls == [(state.session_id, (2, 30), 0)]
    # The cache clock advanced by one chunk; the new context is the new chunk.
    assert step.step_index == 1 and nxt.step_index == 1
    assert torch.all(nxt.context == 4.0)
    # The previous WorldState object is untouched (single live branch caveat
    # is about the cache, not the envelope).
    assert state.step_index == 0 and torch.all(state.context == 0.0)


def test_plan_returns_models_proposed_action() -> None:
    torch = pytest.importorskip("torch")
    pipeline = _FakePipeline()
    engine = _toy_engine(pipeline)
    state = engine.reset(ConditioningInput(), RolloutParams())

    # Fresh session: plan predicts a chunk itself (frame_st_id == 0).
    action = engine.plan(state, torch.zeros(8), horizon=4)
    assert action.space == "lingbot_va_30d"
    assert len(action.values) == 30
    assert action.values[0] == 0.0

    # After a step, plan reuses the chunk the step already predicted (parked),
    # whose first row encodes the advanced cache clock.
    nxt, _ = engine.step(state, Action(values=[0.0] * 30))
    action2 = engine.plan(nxt, torch.zeros(8), horizon=4)
    assert action2.values[0] == 4.0


def test_step_rejects_misshapen_action_chunk() -> None:
    pytest.importorskip("torch")
    engine = _toy_engine(_FakePipeline())
    state = engine.reset(ConditioningInput(), RolloutParams())
    with pytest.raises(ValueError, match="multiple of action_dim"):
        engine.step(state, Action(values=[0.0] * 31))


def test_step_rejects_foreign_world_state() -> None:
    torch = pytest.importorskip("torch")
    engine = _toy_engine(_FakePipeline())
    engine.reset(ConditioningInput(), RolloutParams())
    foreign = WorldState(context=torch.zeros(1, 8), step_index=0, session_id="nope")
    with pytest.raises(KeyError, match="unknown session"):
        engine.step(foreign, Action(values=[0.0] * 30))
