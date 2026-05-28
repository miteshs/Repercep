"""V-JEPA 2-AC engine — the lead interactive, energy-based world model.

V-JEPA 2 (Meta, 2025) is an encoder-predictor video world model from Yann
LeCun's group: it predicts in *representation space*, not pixels, and is the
flagship of the energy-based / JEPA family. The action-conditioned variant
(V-JEPA 2-AC, ~300M params, block-causal) autoregressively predicts the next
state embedding conditioned on an action, and *plans* by minimizing a
latent-space energy (embedding distance to a goal) over candidate action
sequences — model-predictive control as energy minimization.

This module wraps it as an
:class:`~mirage.runtime.interactive.InteractiveWorldModel` on a Mirage
``Backend``. What is **implemented and tested** here is the model-agnostic
algorithmic layer: the latent rollout (:meth:`VJepa2ACEngine.step`), the
energy function (:meth:`VJepa2ACEngine._rollout_energy`), and the CEM/MPC
planner (:meth:`VJepa2ACEngine._plan_sequence` / :meth:`VJepa2ACEngine.plan`).
They run against an injected ``encoder`` + ``predictor`` (see ``__init__``), so
the loop and the planner are exercised on CPU without any model weights.

What remains a **port** (raises ``NotImplementedError`` with the intended body
in its docstring) is the model-specific weight loading: the encoder is the same
HuggingFace checkpoint ``scripts/run_vjepa2.py`` already runs
(``facebook/vjepa2-vitg-fpc64-256``); the AC predictor head lives in
``facebookresearch/vjepa2`` and is not an HF ``AutoModel`` today. Decoding a
real image/video observation URI is a serving-IO concern (Phase 2). See
``docs/adr/0008-interactive-world-model-seam.md``.

This is NOT a diffusion model: there is no denoise loop, so the adaptive cache
does not apply. What carries over from the Cosmos/Wan path is the ``Backend``
seam (ADR-0003), the attention abstraction, the config + ``MIRAGE_*``
conventions, the ``EngineInfo`` contract, and ``torch.inference_mode()``
discipline (the F18 lesson).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from mirage.runtime.engine import EngineInfo
from mirage.runtime.types import Action, ConditioningKind, LatentStep, WorldState

if TYPE_CHECKING:
    import torch

    from mirage.backend.protocol import Backend
    from mirage.runtime.types import ConditioningInput, RolloutParams

#: The V-JEPA 2 encoder checkpoint (the one ``scripts/run_vjepa2.py`` benchmarks).
DEFAULT_ENCODER_REPO = "facebook/vjepa2-vitg-fpc64-256"

#: The action-conditioned predictor head. NOTE: this lives in the
#: ``facebookresearch/vjepa2`` research repo today, not as an HF ``AutoModel`` —
#: the exact checkpoint + loader is a Phase-1 port item (see ADR-0008).
DEFAULT_PREDICTOR_REPO = "facebookresearch/vjepa2"


class _Predictor(Protocol):
    """The action-conditioned next-state predictor, as a callable.

    Maps the current latent context window and an action to the next state
    embedding. Keeping this an injectable callable is what isolates the
    model-specific port from the model-agnostic rollout/planning logic.
    """

    def __call__(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor: ...


@dataclass(slots=True)
class VJepa2ACConfig:
    """Load-time + planning configuration for :class:`VJepa2ACEngine`."""

    encoder_repo: str = DEFAULT_ENCODER_REPO
    predictor_repo: str = DEFAULT_PREDICTOR_REPO
    device_index: int = 0
    dtype: str = "bfloat16"
    # Block-causal window the predictor attends over (state embeddings retained).
    context_frames: int = 8
    # Synthetic-seed clip shape for an unconditioned (NONE) reset. Real
    # image/video URI decode is Phase 2 (serving IO).
    seed_frames: int = 64
    seed_resolution: int = 256
    # Energy-based planning (CEM / MPC) knobs, used by ``plan()``.
    plan_samples: int = 64  # candidate action sequences sampled per planning call
    plan_elites: int = 8  # top-k by lowest energy, refit each iteration
    plan_iters: int = 3  # CEM refit iterations
    action_dim: int = 7  # control dimensionality (e.g. 7-DoF end-effector delta)


class VJepa2ACEngine:
    """V-JEPA 2-AC served on a Mirage backend (the interactive seam).

    The rollout (:meth:`step`), the energy (:meth:`_rollout_energy`), and the
    CEM planner (:meth:`plan`) are implemented model-agnostically against an
    ``encoder`` + ``predictor``. In production those are loaded lazily on the
    first :meth:`load`; in tests they are injected so the loop and planner run
    on CPU without weights. The weight loaders are the remaining port (see the
    module docstring and ADR-0008).
    """

    model_name = "vjepa2-ac-300m"

    def __init__(
        self,
        backend: Backend,
        config: VJepa2ACConfig | None = None,
        *,
        encoder: Any | None = None,
        predictor: _Predictor | None = None,
    ) -> None:
        self._backend = backend
        self._config = config if config is not None else VJepa2ACConfig()
        # Injected for testing / advanced use; otherwise loaded lazily.
        self._encoder: Any | None = encoder
        self._predictor: _Predictor | None = predictor

    @property
    def is_loaded(self) -> bool:
        return self._encoder is not None and self._predictor is not None

    def info(self) -> EngineInfo:
        return EngineInfo(
            model_name=self.model_name,
            backend=self._backend.name,
            device=f"{self._backend.name}:{self._config.device_index}",
            dtype=self._config.dtype,
            ready=self.is_loaded,
        )

    # --- loading (the model-specific port) ---

    def _ensure_encoder(self) -> None:
        """Load the V-JEPA 2 encoder if not already present/injected.

        Intended body (the proven path from ``scripts/run_vjepa2.py``)::

            import torch
            from transformers import AutoModel

            device = self._backend.torch_device(self._config.device_index)
            dtype = getattr(torch, self._config.dtype)
            self._encoder = (
                AutoModel.from_pretrained(self._config.encoder_repo, dtype=dtype)
                .to(device)
                .eval()
            )
        """
        if self._encoder is not None:
            return
        raise NotImplementedError(
            "V-JEPA 2 encoder load pending — see ADR-0008 (Phase 1); the body is "
            "the AutoModel path from scripts/run_vjepa2.py (works in the [models] env)."
        )

    def _ensure_predictor(self) -> None:
        """Load the action-conditioned predictor head if not already present."""
        if self._predictor is not None:
            return
        self._predictor = _load_ac_predictor(self._config)

    def load(self) -> None:
        """Load the encoder and the AC predictor. Idempotent."""
        self._ensure_encoder()
        self._ensure_predictor()

    # --- the interactive seam (implemented) ---

    def reset(self, conditioning: ConditioningInput, params: RolloutParams) -> WorldState:
        """Encode the conditioning observation into the initial world state."""
        import torch

        self._ensure_encoder()
        assert self._encoder is not None
        frames = self._resolve_frames(conditioning)
        with torch.inference_mode():
            features = self._encoder.get_vision_features(pixel_values_videos=frames)
        context = _as_context(features)[-self._config.context_frames :]
        return WorldState(context=context, step_index=0, session_id=_new_session_id())

    def step(self, state: WorldState, action: Action) -> tuple[WorldState, LatentStep]:
        """Advance one latent step under ``action`` (no pixel decode)."""
        import torch

        self._ensure_predictor()
        assert self._predictor is not None
        with torch.inference_mode():
            vec = torch.tensor(
                action.values, dtype=state.context.dtype, device=state.context.device
            )
            nxt = self._predictor(state.context, vec)
            context = torch.cat([state.context, nxt.unsqueeze(0)], dim=0)
            context = context[-self._config.context_frames :]
        new_state = WorldState(
            context=context, step_index=state.step_index + 1, session_id=state.session_id
        )
        return new_state, LatentStep(step_index=new_state.step_index)

    def plan(self, state: WorldState, goal: torch.Tensor, horizon: int) -> Action:
        """Energy-minimizing MPC (CEM): the next action toward ``goal``.

        Samples action sequences, rolls each out via :meth:`step`, scores them
        by the terminal latent energy, refits to the elite set, and returns the
        first action of the energy-minimizing sequence.
        """
        sequence = self._plan_sequence(state, goal, horizon)
        return Action(values=sequence[0].tolist(), space="ee_delta")

    # --- planning internals (energy-based) ---

    def _rollout_energy(
        self, state: WorldState, action_seq: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        """Energy of a rollout: distance of its terminal state to ``goal``.

        This is the scalar the planner minimizes — low energy means the action
        sequence drives the world model toward the goal embedding.
        """
        import torch

        rolled = state
        for t in range(int(action_seq.shape[0])):
            rolled, _ = self.step(rolled, Action(values=action_seq[t].tolist()))
        energy: torch.Tensor = torch.linalg.vector_norm(rolled.context[-1] - goal)
        return energy

    def _plan_sequence(self, state: WorldState, goal: torch.Tensor, horizon: int) -> torch.Tensor:
        """Cross-entropy-method search for the energy-minimizing action sequence."""
        import torch

        a_dim = self._config.action_dim
        mean = torch.zeros(horizon, a_dim, dtype=goal.dtype, device=goal.device)
        std = torch.ones_like(mean)
        for _ in range(self._config.plan_iters):
            noise = torch.randn(
                self._config.plan_samples, horizon, a_dim, dtype=goal.dtype, device=goal.device
            )
            seqs = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            energies = torch.stack(
                [self._rollout_energy(state, seqs[i], goal) for i in range(int(seqs.shape[0]))]
            )
            elite_idx = torch.topk(energies, self._config.plan_elites, largest=False).indices
            elite = seqs[elite_idx]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0).clamp_min(1e-6)
        return mean

    def _resolve_frames(self, conditioning: ConditioningInput) -> torch.Tensor:
        """Resolve a conditioning observation to a pixel clip for the encoder.

        ``NONE`` yields a synthetic seed clip (the encoder maps it to the prior).
        Decoding a real image/video URI is a serving-IO concern (Phase 2).
        """
        import torch

        if conditioning.kind is ConditioningKind.NONE:
            n, r = self._config.seed_frames, self._config.seed_resolution
            return torch.zeros(1, n, 3, r, r)
        raise NotImplementedError(
            "observation URI decode (image/video) is Phase 2 serving IO — see ADR-0008"
        )


def _load_ac_predictor(config: VJepa2ACConfig) -> _Predictor:
    """Load the V-JEPA 2-AC predictor head as a ``_Predictor`` callable.

    The remaining model-specific port: wire the action-conditioned head from
    ``facebookresearch/vjepa2`` (it is not an HF ``AutoModel`` today) into a
    callable ``(context, action) -> next_state_embedding``.
    """
    raise NotImplementedError(
        "V-JEPA 2-AC predictor port pending — see ADR-0008 (Phase 1): wire the "
        f"action-conditioned head from {config.predictor_repo!r}."
    )


def _as_context(features: Any) -> torch.Tensor:
    """Normalize an encoder output to a ``(T, D)`` state-embedding context.

    ``get_vision_features`` returns either a tensor or an object carrying
    ``last_hidden_state`` (``scripts/run_vjepa2.py`` handles both); a leading
    batch axis is squeezed.
    """
    tensor = features if hasattr(features, "shape") else features.last_hidden_state
    context: torch.Tensor = tensor[0] if tensor.ndim == 3 else tensor
    return context


def _new_session_id() -> str:
    return uuid.uuid4().hex
