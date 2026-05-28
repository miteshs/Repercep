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
``Backend``. The encoder is the same HuggingFace checkpoint
``scripts/run_vjepa2.py`` already runs (``facebook/vjepa2-vitg-fpc64-256``); the
AC predictor head + planner currently live in ``facebookresearch/vjepa2`` and
are a small port, not a one-line ``from_pretrained`` — the model-dependent
methods below are scaffolded with ``NotImplementedError`` and the intended body
specified in their docstrings. See
``docs/adr/0008-interactive-world-model-seam.md`` for the phased plan.

This is NOT a diffusion model: there is no denoise loop, so the adaptive cache
does not apply. What carries over from the Cosmos/Wan path is the ``Backend``
seam (ADR-0003), the attention abstraction, the config + ``MIRAGE_*``
conventions, the ``EngineInfo`` contract, and ``torch.inference_mode()``
discipline (the F18 lesson).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mirage.runtime.engine import EngineInfo

if TYPE_CHECKING:
    import torch

    from mirage.backend.protocol import Backend
    from mirage.runtime.types import (
        Action,
        ConditioningInput,
        LatentStep,
        RolloutParams,
        WorldState,
    )

#: The V-JEPA 2 encoder checkpoint (the one ``scripts/run_vjepa2.py`` benchmarks).
DEFAULT_ENCODER_REPO = "facebook/vjepa2-vitg-fpc64-256"

#: The action-conditioned predictor head. NOTE: this lives in the
#: ``facebookresearch/vjepa2`` research repo today, not as an HF ``AutoModel`` —
#: the exact checkpoint + loader is a Phase-1 port item (see ADR-0008).
DEFAULT_PREDICTOR_REPO = "facebookresearch/vjepa2"


@dataclass(slots=True)
class VJepa2ACConfig:
    """Load-time + planning configuration for :class:`VJepa2ACEngine`."""

    encoder_repo: str = DEFAULT_ENCODER_REPO
    predictor_repo: str = DEFAULT_PREDICTOR_REPO
    device_index: int = 0
    dtype: str = "bfloat16"
    # Block-causal window the predictor attends over (state embeddings retained).
    context_frames: int = 8
    # Energy-based planning (CEM / MPC) knobs, used by ``plan()``.
    plan_samples: int = 64  # candidate action sequences sampled per planning call
    plan_elites: int = 8  # top-k by lowest energy, refit each iteration
    plan_iters: int = 3  # CEM refit iterations
    action_dim: int = 7  # control dimensionality (e.g. 7-DoF end-effector delta)


class VJepa2ACEngine:
    """V-JEPA 2-AC served on a Mirage backend (the interactive seam).

    Construction is cheap; weights load lazily on the first :meth:`load`. The
    model-dependent methods are scaffolded (``NotImplementedError``) with the
    intended implementation in their docstrings — see the module docstring and
    ADR-0008 for the phased plan.
    """

    model_name = "vjepa2-ac-300m"

    def __init__(self, backend: Backend, config: VJepa2ACConfig | None = None) -> None:
        self._backend = backend
        self._config = config if config is not None else VJepa2ACConfig()
        self._encoder: Any | None = None
        self._predictor: Any | None = None

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

    def load(self) -> None:
        """Load the encoder and the AC predictor. Idempotent.

        Intended body (Phase 1)::

            import torch
            from transformers import AutoModel

            device = self._backend.torch_device(self._config.device_index)
            dtype = getattr(torch, self._config.dtype)
            self._encoder = (
                AutoModel.from_pretrained(self._config.encoder_repo, dtype=dtype)
                .to(device)
                .eval()
            )
            self._predictor = _load_ac_predictor(self._config.predictor_repo, device, dtype)

        The encoder path is proven in ``scripts/run_vjepa2.py``; the predictor
        loader is the port (the AC head is not an HF ``AutoModel`` today).
        """
        raise NotImplementedError(
            "VJepa2ACEngine.load: AC predictor port pending — see ADR-0008 (Phase 1)"
        )

    def reset(self, conditioning: ConditioningInput, params: RolloutParams) -> WorldState:
        """Encode the conditioning observation into the initial world state.

        Intended body::

            import torch

            from mirage.runtime.types import WorldState

            self.load()
            frames = _resolve_frames(conditioning)  # serving resolves the URI
            with torch.inference_mode():
                ctx = self._encoder.get_vision_features(pixel_values_videos=frames)
            window = ctx[:, -self._config.context_frames :]
            return WorldState(context=window, step_index=0, session_id=_new_session_id())
        """
        raise NotImplementedError(
            "VJepa2ACEngine.reset: encoder-rollout port pending — see ADR-0008 (Phase 1)"
        )

    def step(self, state: WorldState, action: Action) -> tuple[WorldState, LatentStep]:
        """Advance one latent step under ``action`` (no pixel decode).

        Intended body::

            import torch

            from mirage.runtime.types import LatentStep, WorldState

            with torch.inference_mode():
                a = torch.tensor(
                    action.values, device=state.context.device, dtype=state.context.dtype
                )
                nxt = self._predictor(state.context, a)  # block-causal next state
                ctx = torch.cat([state.context, nxt[None]], dim=0)[-self._config.context_frames :]
            new = WorldState(
                context=ctx, step_index=state.step_index + 1, session_id=state.session_id
            )
            return new, LatentStep(step_index=new.step_index)
        """
        raise NotImplementedError(
            "VJepa2ACEngine.step: predictor port pending — see ADR-0008 (Phase 1)"
        )

    def plan(self, state: WorldState, goal: torch.Tensor, horizon: int) -> Action:
        """Energy-minimizing MPC (CEM): the next action toward ``goal``.

        Sample action sequences, roll out :meth:`step`, score each by the
        terminal latent energy ``||s_T - goal||``, refit to the elite set, and
        return the first action of the best sequence. Intended body::

            import torch

            from mirage.runtime.types import Action

            mean = torch.zeros(horizon, self._config.action_dim, device=goal.device)
            std = torch.ones_like(mean)
            for _ in range(self._config.plan_iters):
                seqs = mean + std * torch.randn(
                    self._config.plan_samples, horizon, self._config.action_dim, device=goal.device
                )
                energies = torch.stack([self._rollout_energy(state, seq, goal) for seq in seqs])
                elite = seqs[energies.topk(self._config.plan_elites, largest=False).indices]
                mean, std = elite.mean(0), elite.std(0).clamp_min(1e-6)
            return Action(values=mean[0].tolist(), space="ee_delta")
        """
        raise NotImplementedError(
            "VJepa2ACEngine.plan: energy-MPC port pending — see ADR-0008 (Phase 1)"
        )
