"""DreamZero engine — the third model on the interactive seam (Phase 0 scaffold).

DreamZero (NVIDIA GEAR, 2026) is an autoregressive **world-action model**: a
``CausalWanModel`` — Wan2.1-I2V-14B-480P DiT extended with action/state
registers — over Wan2.1 VAE latents, trained with flow matching, generating
**block-by-block** with a named per-layer KV cache. Per block it jointly
denoises video-latent noise and action-register noise; the action-register
slice decodes into an executed action chunk. Same policy regime as
LingBot-VA (the model proposes its own actions; no external CEM), but a
different backbone/cache shape and attached to the GR00T 2 ecosystem. See
``docs/DREAMZERO_PORT_PLAN.md`` for the full scoping (interfaces verified
against ``dreamzero0/dreamzero`` ``socket_test_optimized_AR.py`` +
``groot/vla/model/dreamzero/``).

This module wraps it as an
:class:`~repercep.runtime.interactive.InteractiveWorldModel`. What is
**implemented and tested** here is the model-agnostic chunk-loop layer:
session bookkeeping, the recondition-then-predict :meth:`DreamZeroEngine.step`,
and the policy-mode :meth:`DreamZeroEngine.plan`. They run against an
injected ``pipeline`` (see :class:`_DreamZeroPipeline`), so the loop is
exercised on CPU without weights — the ``vjepa2_ac.py`` / ``lingbot_va.py``
pattern. Unlike LingBot-VA, there is no pipeline module yet: the model-
specific half (vendoring ``CausalWanModel``, the SDPA attention path,
schedulers, and the VAE/T5/CLIP encoders) is Phase 1 of the port plan, so
:meth:`DreamZeroEngine.load` raises ``NotImplementedError`` unconditionally
with the recipe until it lands.

Chunk-loop geometry note (the DreamZero-specific numbers, vs LingBot-VA's
4-frame chunks): one block is a *single* latent frame
(``num_frame_per_block=1``); the executed unit per :meth:`step`/:meth:`plan`
is one **action** chunk of ``num_action_per_block=32`` actions. The KV cache
carries a full negative-prompt copy for CFG plus a 512-token cross-attn
cache, and resets when the attention window (``attn_window_frames=21``
frames) fills or the task/language changes — bookkeeping the pipeline owns,
not this layer.

State note (same caveat as LingBot-VA): the native serving state is a
per-session mutable KV cache inside the transformer, so the seam's "previous
state is not mutated" branching guarantee does not hold in v0 — the engine
keeps one live branch per session. A base shared with ``lingbot_va`` ("chunked
VA over a Wan-family causal DiT with named KV cache") is a Phase-0 design
question per the port plan, not a commitment made here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from repercep.runtime.engine import EngineInfo
from repercep.runtime.types import Action, LatentStep, WorldState

if TYPE_CHECKING:
    import torch

    from repercep.backend.protocol import Backend
    from repercep.runtime.types import ConditioningInput, RolloutParams

#: The inference-ready DROID checkpoint (HF, Apache-2.0). ``GEAR-Dreams/DreamZero-AgiBot``
#: is the post-training bundle (new embodiments), not the serving default.
DEFAULT_REPO = "GEAR-Dreams/DreamZero-DROID"


class _DreamZeroPipeline(Protocol):
    """The model-specific half of the port, as an injectable session pipeline.

    Mirrors the reference server's surface (``ARDroidRoboarenaPolicy`` in
    ``dreamzero0/dreamzero``): ``reset`` builds the per-layer KV cache (+ CFG
    negative-prompt copy + cross-attn caches) and encodes the text prompt,
    ``encode_observation`` is the first-frame CLIP/VAE encode,
    ``infer_chunk`` runs the 16-step flow-matching loop over one block
    (caching the prediction), and ``recondition`` pushes executed reality
    (real observation frames + executed actions) into the cache — the
    "push-then-predict" order the reference client loop uses. Keeping this a
    Protocol isolates the chunk-loop logic from the port, so the loop is
    testable on CPU with fakes.
    """

    def reset(self, session_id: str, prompt: str | None) -> None: ...

    def encode_observation(
        self, session_id: str, conditioning: ConditioningInput
    ) -> torch.Tensor: ...

    def infer_chunk(
        self, session_id: str, current_start_frame: int, init_latent: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one block → ``(latents (num_frame_per_block, D), actions (K, action_dim))``."""
        ...

    def recondition(
        self,
        session_id: str,
        actions: torch.Tensor,
        obs_latent: torch.Tensor | None,
        current_start_frame: int,
    ) -> int:
        """Push executed actions (+ real obs when available) into the cache.

        Returns the number of latent frames consumed (advances
        ``current_start_frame``). ``obs_latent=None`` is imagination mode:
        the model's own predicted latents (already cached by
        :meth:`infer_chunk`) stand in for reality.
        """
        ...

    def close(self, session_id: str) -> None:
        """Drop the session's named KV + cross-attn caches. The other half of ``reset``."""
        ...


@dataclass(slots=True)
class DreamZeroConfig:
    """Load-time + rollout configuration for :class:`DreamZeroEngine`.

    Defaults are the reference 14B DROID config verified 2026-07-12
    (``wan_flow_matching_action_tf.yaml`` + the WebSocket server): one latent
    frame per block, 3-camera 880-token frames, 32-action blocks, a 21-frame
    (~18.5k token) attention window, 16-step flow matching, CFG 5.0, sigma
    shift 5.0, bf16, SDPA attention (the ROCm-portable path — flash-attn is
    optional and try/except-guarded in the reference code, no stub needed
    unlike LingBot-VA).
    """

    repo: str = DEFAULT_REPO
    device_index: int = 0
    dtype: str = "bfloat16"
    # The task instruction — DreamZero's goal conditioning is textual (umT5).
    prompt: str | None = None
    # Chunked generation geometry (reference yaml names kept for greppability).
    num_frame_per_block: int = 1
    frame_seqlen: int = 880  # tokens/frame across 2 exterior + 1 wrist camera
    num_action_per_block: int = 32
    num_state_per_block: int = 1
    # DROID's common 7-DoF end-effector-delta + gripper convention
    # (dx, dy, dz, droll, dpitch, dyaw, gripper). Unconfirmed against the
    # checkpoint's actual action-head config — a Phase-1 verify item, same
    # caveat as LingBotVAConfig.used_action_dim.
    action_dim: int = 7
    used_action_dim: int | None = None
    # attn window in frames; max_attention_size = attn_window_frames * frame_seqlen.
    attn_window_frames: int = 21
    # Flow-matching denoise budget per block. The yaml also carries
    # num_inference_timesteps=4 ("not used during training") — which loop the
    # server actually runs is a Phase-1 verify item (port plan §1); this
    # config tracks the server-hardcoded value.
    num_inference_steps: int = 16
    cfg_scale: float = 5.0
    sigma_shift: float = 5.0
    # Their headline latency lever (dynamic DiT-call skip via cosine
    # similarity of consecutive action-noise predictions) — off by default,
    # quality impact is ours to measure, not the pipeline's default stance.
    enable_dit_cache: bool = False
    attn_mode: str = "sdpa"
    # ip=2 only splits CFG (rank 0 conditional / rank 1 unconditional,
    # exchanged via P2P) — a convenience, not a sharding requirement. ip=1
    # runs both contexts sequentially in one process; batching them into one
    # forward is the obvious single-GPU lever (port plan §2.5), a pipeline
    # concern this config doesn't encode.
    ip_size: int = 1


@dataclass(slots=True)
class _Session:
    """Engine-held per-session state the wire-safe ``WorldState`` cannot carry."""

    current_start_frame: int = 0
    pending_actions: torch.Tensor | None = field(default=None)


class DreamZeroEngine:
    """DreamZero served on a Repercep backend (the interactive seam).

    The chunk loop (recondition → predict) and session bookkeeping are
    implemented model-agnostically against an injected
    :class:`_DreamZeroPipeline`; in production the pipeline would be built
    from the HF bundle on the first :meth:`load` (the Phase-1 port, not
    landed — see :meth:`load`), in tests it is a fake.
    ``WorldState.context`` carries the latest predicted latent block
    ``(num_frame_per_block, D)``; the KV cache lives in the pipeline keyed by
    ``session_id`` (see the module docstring's state note).
    """

    model_name = "dreamzero-droid"

    def __init__(
        self,
        backend: Backend,
        config: DreamZeroConfig | None = None,
        *,
        pipeline: _DreamZeroPipeline | None = None,
    ) -> None:
        self._backend = backend
        self._config = config if config is not None else DreamZeroConfig()
        # Injected for testing / advanced use; otherwise built by ``load()``
        # once Phase 1 lands.
        self._pipeline: _DreamZeroPipeline | None = pipeline
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

    # --- loading (the model-specific port, Phase 1 — not landed) ---

    def load(self) -> None:
        """Build the real pipeline. Not yet — Phase 1 of the port plan.

        Unlike :meth:`LingBotVAEngine.load` (which fails only when the
        research package isn't importable), there is no
        ``repercep.models.dreamzero_pipeline`` module yet: vendoring
        ``CausalWanModel`` + the SDPA attention path + schedulers +
        VAE/T5/CLIP encoders from ``groot/vla/model/dreamzero/modules/`` is
        unstarted work. Raises unconditionally (idempotent only in the sense
        that an already-injected test pipeline short-circuits it) until that
        lands. Do **not** ``pip install`` the ``dreamzero0/dreamzero``
        package as a shortcut: its pyproject hard-requires ``tensorrt``,
        ``ray``, ``mujoco``, ``deepspeed``, and a ``gear`` package, none of
        which the inference loop needs — and ``tensorrt`` won't install on
        ROCm. See ``docs/DREAMZERO_PORT_PLAN.md`` Phase 1.
        """
        if self._pipeline is not None:
            return
        raise NotImplementedError(
            "DreamZero pipeline not ported yet (Phase 1, docs/DREAMZERO_PORT_PLAN.md): "
            "vendor CausalWanModel + wan2_1_attention (SDPA path) + schedulers + "
            "VAE/T5/CLIP encoders from groot/vla/model/dreamzero/modules/ into a new "
            "repercep.models.dreamzero_pipeline module against the GEAR-Dreams/DreamZero-DROID "
            "checkpoint. Do not pip install dreamzero0/dreamzero directly (pulls in "
            "tensorrt/ray/mujoco/deepspeed/gear; tensorrt won't build on ROCm)."
        )

    # --- the interactive seam ---

    def reset(self, conditioning: ConditioningInput, params: RolloutParams) -> WorldState:
        """Open a session: build caches, encode the seed observation."""
        self._require_pipeline()
        assert self._pipeline is not None
        session_id = uuid.uuid4().hex
        self._pipeline.reset(session_id, self._config.prompt)
        init_latent = self._pipeline.encode_observation(session_id, conditioning)
        self._sessions[session_id] = _Session()
        return WorldState(context=init_latent, step_index=0, session_id=session_id)

    def step(self, state: WorldState, action: Action) -> tuple[WorldState, LatentStep]:
        """Advance one block under the *executed* action chunk.

        Native order (the reference client loop): the previous prediction's
        cache entries are dropped and executed reality is pushed
        (``recondition``), then the next block is predicted. ``action.values``
        is a flat executed chunk — ``k x action_dim`` floats,
        ``space="dreamzero_chunk_droid"``. The model's *proposed* actions for
        the new block are parked on the session for a following
        :meth:`plan` call.
        """
        self._require_pipeline()
        assert self._pipeline is not None
        session = self._session_for(state)
        actions = self._parse_action_chunk(action, like=state.context)
        consumed = self._pipeline.recondition(
            state.session_id, actions, None, session.current_start_frame
        )
        session.current_start_frame += consumed
        latents, proposed = self._pipeline.infer_chunk(
            state.session_id, session.current_start_frame, None
        )
        session.pending_actions = proposed
        new_state = WorldState(
            context=latents, step_index=state.step_index + 1, session_id=state.session_id
        )
        return new_state, LatentStep(step_index=new_state.step_index)

    def plan(self, state: WorldState, goal: torch.Tensor, horizon: int) -> Action:
        """Policy-mode planning: the model's own proposed next action chunk.

        DreamZero *generates* actions (the action-register denoise loop is
        the planner); the goal is the text prompt fixed at :meth:`reset`, so
        ``goal``/``horizon`` are accepted for seam compatibility and unused
        in v0. Returns the first action of the block proposed for the
        current state — predicting a fresh block if :meth:`step` has not
        already parked one.
        """
        self._require_pipeline()
        assert self._pipeline is not None
        session = self._session_for(state)
        proposed = session.pending_actions
        if proposed is None:
            _, proposed = self._pipeline.infer_chunk(
                state.session_id, session.current_start_frame, state.context
            )
            session.pending_actions = proposed
        return Action(values=proposed[0].tolist(), space="dreamzero_chunk_droid")

    def release(self, state: WorldState) -> None:
        """Drop this session's server-side state (named KV cache + bookkeeping).

        Not part of :class:`InteractiveWorldModel` — the serving layer calls
        it duck-typed (``getattr(engine, "release", None)``) when a client
        session ends. Two layers hold state per session: this engine's own
        ``_sessions`` (frame clock, parked action proposal) and the
        pipeline's session-keyed KV/cross-attn caches (the actual GPU
        tensors) — both need dropping, or the cache leaks one entry per
        session forever under churn (the LingBot-VA lesson).
        """
        self._sessions.pop(state.session_id, None)
        if self._pipeline is not None:
            self._pipeline.close(state.session_id)

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

    def _wire_action_dim(self) -> int:
        """The executed-action-chunk width (see ``DreamZeroConfig.used_action_dim``)."""
        return self._config.used_action_dim or self._config.action_dim

    def _parse_action_chunk(self, action: Action, *, like: torch.Tensor) -> torch.Tensor:
        """Validate + shape a flat executed chunk to ``(k, wire_action_dim)``."""
        import torch

        a_dim = self._wire_action_dim()
        if len(action.values) % a_dim != 0:
            raise ValueError(
                f"action chunk length {len(action.values)} is not a multiple "
                f"of action_dim={a_dim} (space={action.space!r})"
            )
        vec = torch.tensor(action.values, dtype=torch.float32, device=like.device)
        return vec.reshape(-1, a_dim)
