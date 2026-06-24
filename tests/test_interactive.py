"""Tests for the interactive (action-conditioned) world-model seam.

Exercises the wire contract, Protocol conformance, and — with an injected fake
encoder + predictor — the real rollout and CEM/energy planner. The parts that
build latent tensors are gated on torch and skip cleanly on a box without it,
matching the rest of the suite (accelerator-specific paths skip rather than
fail). The model-specific weight load remains a port and is asserted to raise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import ValidationError

from mirage.models.vjepa2_ac import (
    VJepa2ACConfig,
    VJepa2ACEngine,
    _AcPredictorAdapter,
    _infer_tokens_per_frame,
)
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


class _FakeEncoder:
    """Returns a fixed ``(1, T, D)`` feature tensor regardless of input."""

    def __init__(self, context: torch.Tensor) -> None:
        self._context = context

    def get_vision_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        return self._context.unsqueeze(0)


class _FakePredictor:
    """Linear toy dynamics: next state = last context frame + action."""

    def __call__(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return context[-1] + action


def _toy_engine(ctx0: torch.Tensor, *, context_frames: int) -> VJepa2ACEngine:
    return VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")),
        VJepa2ACConfig(
            action_dim=int(ctx0.shape[-1]),
            context_frames=context_frames,
            seed_frames=2,
            seed_resolution=8,
        ),
        encoder=_FakeEncoder(ctx0),
        predictor=_FakePredictor(),
    )


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


def test_engine_not_ready_without_predictor() -> None:
    # Inject only the encoder; the predictor still needs its (GPU/network) load,
    # so the engine reports not-ready until both halves are present.
    engine = VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")), VJepa2ACConfig(), encoder=object()
    )
    assert engine.info().ready is False


def test_infer_tokens_per_frame() -> None:
    # Real encoder: (crop/patch)**2 spatial patch tokens per frame; stub: 1.
    class _Cfg:
        crop_size = 256
        patch_size = 16

    class _Enc:
        config = _Cfg()

    assert _infer_tokens_per_frame(_Enc()) == 256
    assert _infer_tokens_per_frame(object()) == 1


def test_ac_predictor_adapter_returns_next_frame_block() -> None:
    torch = pytest.importorskip("torch")

    class _RawPredictor:
        """Mimics the research predictor: (x, actions, states) -> (B, N, D).

        Asserts the per-frame action/state shape the real predictor requires.
        """

        def __call__(self, x: Any, actions: Any, states: Any) -> Any:
            assert x.shape[0] == 1 and actions.shape == states.shape
            assert actions.shape == (1, 3, 7)  # (B, T frames, action_dim)
            return x + actions.sum()

    adapter = _AcPredictorAdapter(_RawPredictor(), tokens_per_frame=4, action_dim=7)
    context = torch.zeros(12, 8)  # 3 frames x P=4 patch tokens, D=8
    nxt = adapter(context, torch.ones(7))
    assert tuple(nxt.shape) == (4, 8)  # the trailing P rows = predicted next frame


def test_step_appends_patch_token_frame_block() -> None:
    torch = pytest.importorskip("torch")
    # The real path emits P>1 patch tokens per frame; step must append the block
    # and cap the window by frames (not rows).
    p, d = 3, 4
    ctx0 = torch.zeros(2 * p, d)  # 2 frames

    class _BlockPredictor:
        def __call__(self, context: Any, action: Any) -> Any:
            return torch.ones(p, d) * action.sum()

    engine = VJepa2ACEngine(
        cast("Backend", _NamedBackend("fake")),
        VJepa2ACConfig(action_dim=d, context_frames=2),
        encoder=_FakeEncoder(ctx0),
        predictor=_BlockPredictor(),
    )
    engine._tokens_per_frame = p  # real path sets this from the encoder config
    state = engine.reset(ConditioningInput(), RolloutParams())
    assert tuple(state.context.shape) == (2 * p, d)  # 2 frames kept
    nxt, _ = engine.step(state, Action(values=[1.0, 1.0, 1.0, 1.0]))
    assert tuple(nxt.context.shape) == (2 * p, d)  # appended P, capped to 2 frames


# --- seam loop contract (needs torch for the latent tensors) ---


def test_stub_step_streams_in_order() -> None:
    pytest.importorskip("torch")
    engine = _StubInteractive()
    state = engine.reset(ConditioningInput(), RolloutParams())
    seen: list[int] = []
    for _ in range(3):
        state, step = engine.step(state, Action(values=[0.0]))
        seen.append(step.step_index)
    assert seen == [1, 2, 3]
    assert state.step_index == 3


def test_reset_and_step_advance_context() -> None:
    torch = pytest.importorskip("torch")
    ctx0 = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    engine = _toy_engine(ctx0, context_frames=4)
    assert engine.info().ready is True

    state = engine.reset(ConditioningInput(), RolloutParams())
    assert state.step_index == 0
    assert tuple(state.context.shape) == (2, 4)

    nxt, step = engine.step(state, Action(values=[1.0, 1.0, 1.0, 1.0]))
    assert step.step_index == 1
    assert tuple(nxt.context.shape) == (3, 4)
    # Toy dynamics: new last frame == old last frame + action.
    assert torch.allclose(nxt.context[-1], state.context[-1] + torch.ones(4))


def test_context_window_is_capped() -> None:
    torch = pytest.importorskip("torch")
    ctx0 = torch.zeros(1, 3)
    engine = _toy_engine(ctx0, context_frames=2)
    state = engine.reset(ConditioningInput(), RolloutParams())
    for _ in range(5):
        state, _ = engine.step(state, Action(values=[0.0, 0.0, 0.0]))
    assert tuple(state.context.shape) == (2, 3)  # capped at context_frames


def test_plan_reduces_energy_toward_goal() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    engine = _toy_engine(torch.zeros(2, 4), context_frames=8)
    state = engine.reset(ConditioningInput(), RolloutParams())
    goal = state.context[-1] + torch.tensor([2.0, 0.0, -1.0, 0.5])

    e_zero = engine._rollout_energy(state, torch.zeros(3, 4), goal)
    sequence = engine._plan_sequence(state, goal, horizon=3)
    e_planned = engine._rollout_energy(state, sequence, goal)
    # CEM minimizes the terminal latent energy → planned beats the zero action.
    assert float(e_planned) < float(e_zero)

    action = engine.plan(state, goal, horizon=3)
    assert len(action.values) == 4
    assert action.space == "ee_delta"
