"""LingBot-VA 2.0 engine — the video-action world model on the interactive seam.

LingBot-VA (Ant Group / Robbyant, 2026) is an autoregressive video-**action**
world model: a Mixture-of-Transformers DiT (video expert + action expert) over
Wan2.2 VAE latents, trained with flow matching, generating chunk-by-chunk with a
named KV cache. Per chunk it denoises a video-latent block and an action block;
the action block *is* the policy output — unlike V-JEPA 2-AC there is no
external CEM/energy search, the model proposes its own actions and the goal is
conditioned as a text prompt. Weights are Apache-2.0 on HF
(``robbyant/lingbot-va-base`` + RoboTwin/LIBERO post-trains). See
``docs/LINGBOT_VA_PORT_PLAN.md`` for the full scoping (interfaces verified
against ``robbyant/lingbot-va`` ``wan_va/wan_va_server.py``).

This module wraps it as an
:class:`~mirage.runtime.interactive.InteractiveWorldModel`. What is
**implemented and tested** here is the model-agnostic chunk-loop layer:
session bookkeeping, the recondition-then-predict :meth:`LingBotVAEngine.step`,
and the policy-mode :meth:`LingBotVAEngine.plan`. They run against an injected
``pipeline`` (see :class:`_VAPipeline`), so the loop is exercised on CPU
without weights — the ``vjepa2_ac.py`` pattern. The model-specific pipeline
(Wan-VAE streaming encode, T5 prompt embeds, the MoT transformer's
``create_empty_cache``/``update_cache`` KV protocol, and the two flow-matching
denoise loops) is the Phase-1 GPU port: :meth:`LingBotVAEngine.load` raises
``NotImplementedError`` with the recipe until it lands.

Regime note (the dual-regime story on one seam): V-JEPA 2-AC plans by
*searching* actions against a latent energy; LingBot-VA plans by *generating*
actions from the denoiser. Both satisfy the same Protocol; ``plan`` differs in
mechanism, not in contract.

State note: the native serving state is a per-session mutable KV cache inside
the transformer (plus a streaming-VAE cache and ``frame_st_id``), so the seam's
"previous state is not mutated" branching guarantee does **not** hold in v0 —
the engine keeps one live branch per session and documents it. Forking the KV
cache across branches is the KV-reuse mechanism of the latency workstream
(``docs/COMPETITIVE_RESPONSE_REACTOR_2026_07.md`` §4), not a v0 requirement.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from mirage.runtime.engine import EngineInfo
from mirage.runtime.types import Action, LatentStep, WorldState

if TYPE_CHECKING:
    import torch

    from mirage.backend.protocol import Backend
    from mirage.runtime.types import ConditioningInput, RolloutParams

#: The pretrained bundle (Wan2.2-style directory: vae/, tokenizer/,
#: text_encoder/, transformer/). Post-train variants carry their own
#: action-normalization stats: ``…-posttrain-robotwin``, ``…-posttrain-libero-long``.
DEFAULT_REPO = "robbyant/lingbot-va-base"


class _VAPipeline(Protocol):
    """The model-specific half of the port, as an injectable session pipeline.

    Mirrors the reference server's surface (``VA_Server`` in
    ``robbyant/lingbot-va``): reset builds the named KV cache + prompt embeds,
    ``encode_observation`` is the streaming Wan-VAE path, ``infer_chunk`` runs
    the video-then-action flow-matching loops (caching the prediction), and
    ``recondition`` is ``_compute_kv_cache`` — it drops the predicted cache
    entries and pushes executed reality (obs latents + actions) instead.
    Keeping this a Protocol isolates the chunk-loop logic from the port, so the
    loop is testable on CPU with fakes.
    """

    def reset(self, session_id: str, prompt: str | None) -> None: ...

    def encode_observation(self, conditioning: ConditioningInput) -> torch.Tensor: ...

    def infer_chunk(
        self, session_id: str, frame_st_id: int, init_latent: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one chunk → ``(latents (T, D), actions (K, action_dim))``."""
        ...

    def recondition(
        self,
        session_id: str,
        actions: torch.Tensor,
        obs_latent: torch.Tensor | None,
        frame_st_id: int,
    ) -> int:
        """Push executed actions (+ real obs when available) into the cache.

        Returns the number of latent frames consumed (advances ``frame_st_id``).
        ``obs_latent=None`` is imagination mode: the model's own predicted
        latents (already cached by :meth:`infer_chunk`) stand in for reality.
        """
        ...


@dataclass(slots=True)
class LingBotVAConfig:
    """Load-time + rollout configuration for :class:`LingBotVAEngine`.

    Defaults are the reference single-GPU demo config (``va_demo_cfg`` in the
    research repo), verified 2026-07-11: 256x256 two-camera obs, 4-latent-frame
    chunks, 30-dim actions x 8 per frame, 30-chunk KV window, 5/10 video/action
    flow-matching steps, CFG 5.0/1.0, bf16, SDPA attention (``torch`` mode —
    the ROCm-portable path; flash-attn is optional, not required).
    """

    repo: str = DEFAULT_REPO
    device_index: int = 0
    dtype: str = "bfloat16"
    # The task instruction — LingBot-VA's goal conditioning is textual (T5).
    prompt: str | None = None
    # Chunked generation geometry (va_demo_cfg names kept for greppability).
    frame_chunk_size: int = 4
    action_dim: int = 30
    action_per_frame: int = 8
    attn_window: int = 30
    # Flow-matching denoise budgets per chunk.
    num_inference_steps: int = 5
    action_num_inference_steps: int = 10
    guidance_scale: float = 5.0
    action_guidance_scale: float = 1.0
    attn_mode: str = "torch"


@dataclass(slots=True)
class _Session:
    """Engine-held per-session state the wire-safe ``WorldState`` cannot carry."""

    frame_st_id: int = 0
    pending_actions: torch.Tensor | None = field(default=None)


class LingBotVAEngine:
    """LingBot-VA 2.0 served on a Mirage backend (the interactive seam).

    The chunk loop (recondition → predict) and session bookkeeping are
    implemented model-agnostically against an injected :class:`_VAPipeline`; in
    production the pipeline is built from the HF bundle on the first
    :meth:`load` (the Phase-1 port), in tests it is a fake. ``WorldState.context``
    carries the latest latent chunk ``(T, D)``; the KV cache lives in the
    pipeline keyed by ``session_id`` (see the module docstring's state note).
    """

    model_name = "lingbot-va-2"

    def __init__(
        self,
        backend: Backend,
        config: LingBotVAConfig | None = None,
        *,
        pipeline: _VAPipeline | None = None,
    ) -> None:
        self._backend = backend
        self._config = config if config is not None else LingBotVAConfig()
        # Injected for testing / advanced use; otherwise loaded by ``load()``.
        self._pipeline: _VAPipeline | None = pipeline
        self._sessions: dict[str, _Session] = {}

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def info(self) -> EngineInfo:
        return EngineInfo(
            model_name=self.model_name,
            backend=self._backend.name,
            device=f"{self._backend.name}:{self._config.device_index}",
            dtype=self._config.dtype,
            ready=self.is_loaded,
        )

    # --- loading (the model-specific port, Phase 1) ---

    def load(self) -> None:
        """Build the real :class:`_VAPipeline` from the HF bundle.

        NOT YET PORTED (Phase 1, GPU): load ``vae/`` (streaming wrapper),
        ``tokenizer/`` + ``text_encoder/`` (T5, CPU-offloadable), and
        ``transformer/`` with ``attn_mode=self._config.attn_mode``; drive the
        named-KV-cache protocol (``create_empty_cache`` at reset,
        ``update_cache=1`` on the last denoise step, ``update_cache=2`` +
        ``clear_pred_cache`` on recondition) and the two ``FlowMatchScheduler``
        loops. Recipe + dependency pins (torch 2.9 / transformers 5.0 /
        diffusers 0.36, own venv) in ``docs/LINGBOT_VA_PORT_PLAN.md`` §4.
        """
        raise NotImplementedError(
            "LingBot-VA weight load is the Phase-1 GPU port — see docs/LINGBOT_VA_PORT_PLAN.md §4"
        )

    # --- the interactive seam ---

    def reset(self, conditioning: ConditioningInput, params: RolloutParams) -> WorldState:
        """Open a session: build caches, encode the seed observation."""
        self._require_pipeline()
        assert self._pipeline is not None
        session_id = uuid.uuid4().hex
        self._pipeline.reset(session_id, self._config.prompt)
        init_latent = self._pipeline.encode_observation(conditioning)
        self._sessions[session_id] = _Session()
        return WorldState(context=init_latent, step_index=0, session_id=session_id)

    def step(self, state: WorldState, action: Action) -> tuple[WorldState, LatentStep]:
        """Advance one chunk under the *executed* action chunk.

        Native order (the reference client loop): the previous prediction's
        cache entries are dropped and executed reality is pushed
        (``recondition``), then the next chunk is predicted. ``action.values``
        is a flat executed chunk — ``k x action_dim`` floats. The model's
        *proposed* actions for the new chunk are parked on the session for a
        following :meth:`plan` call.
        """
        self._require_pipeline()
        assert self._pipeline is not None
        session = self._session_for(state)
        actions = self._parse_action_chunk(action, like=state.context)
        consumed = self._pipeline.recondition(state.session_id, actions, None, session.frame_st_id)
        session.frame_st_id += consumed
        latents, proposed = self._pipeline.infer_chunk(state.session_id, session.frame_st_id, None)
        session.pending_actions = proposed
        new_state = WorldState(
            context=latents, step_index=state.step_index + 1, session_id=state.session_id
        )
        return new_state, LatentStep(step_index=new_state.step_index)

    def plan(self, state: WorldState, goal: torch.Tensor, horizon: int) -> Action:
        """Policy-mode planning: the model's own proposed next action.

        LingBot-VA *generates* actions (the action-denoise loop is the
        planner); the goal is the text prompt fixed at :meth:`reset`, so
        ``goal``/``horizon`` are accepted for seam compatibility and unused in
        v0 (a goal-embedding→prompt-embed path is future work, recorded in the
        port plan). Returns the first action of the chunk proposed for the
        current state — predicting a fresh chunk if :meth:`step` has not
        already parked one.
        """
        self._require_pipeline()
        assert self._pipeline is not None
        session = self._session_for(state)
        proposed = session.pending_actions
        if proposed is None:
            _, proposed = self._pipeline.infer_chunk(
                state.session_id, session.frame_st_id, state.context
            )
            session.pending_actions = proposed
        return Action(values=proposed[0].tolist(), space="lingbot_va_30d")

    # --- internals ---

    def _require_pipeline(self) -> None:
        if self._pipeline is None:
            self.load()

    def _session_for(self, state: WorldState) -> _Session:
        session = self._sessions.get(state.session_id)
        if session is None:
            raise KeyError(
                f"unknown session {state.session_id!r} — WorldState must come "
                "from this engine's reset()"
            )
        return session

    def _parse_action_chunk(self, action: Action, *, like: torch.Tensor) -> torch.Tensor:
        """Validate + shape a flat executed chunk to ``(k, action_dim)``."""
        import torch

        a_dim = self._config.action_dim
        if len(action.values) % a_dim != 0:
            raise ValueError(
                f"action chunk length {len(action.values)} is not a multiple "
                f"of action_dim={a_dim} (space={action.space!r})"
            )
        vec = torch.tensor(action.values, dtype=torch.float32, device=like.device)
        return vec.reshape(-1, a_dim)
