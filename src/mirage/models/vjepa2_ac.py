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

The model-specific weight loading is now wired (ADR-0008 Phase 1): the encoder is
the same HuggingFace checkpoint ``scripts/run_vjepa2.py`` runs
(``facebook/vjepa2-vitg-fpc64-256``, loaded in :meth:`VJepa2ACEngine._ensure_encoder`);
the AC predictor head — which is *not* an HF ``AutoModel`` — is pulled from
``facebookresearch/vjepa2`` via its Torch Hub entrypoint and wrapped by
:class:`_AcPredictorAdapter` (:func:`_load_ac_predictor`). Because the real
predictor works over **patch tokens** and predicts a whole next frame (not the
single per-frame embedding the stub uses), the engine is patch-aware via
``_tokens_per_frame`` (1 on the stub path, ``(crop/patch)**2`` on the real one).
These real paths need a GPU + the checkpoints to run; the spots that have no
offline ground truth are marked ``VERIFY ON GPU`` in :class:`_AcPredictorAdapter`.
Real image/video URI decode is :func:`_decode_observation` (serving IO, Phase 2).
See ``docs/adr/0008-interactive-world-model-seam.md``.

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
        # Spatial patch tokens per temporal frame in the encoder's output. The
        # real V-JEPA 2 encoder emits ``(crop/patch)**2`` patch tokens per frame;
        # the model-agnostic stub path (injected encoder, no ``config``) keeps the
        # default of 1, so ``WorldState.context`` rows == frames for the tests.
        # Set from the encoder config in :meth:`_ensure_encoder` on the real path.
        self._tokens_per_frame = 1
        # The real V-JEPA 2-AC predictor is trained on layer-normed reps
        # (``normalize_reps=True`` in the reference wrapper). Enabled on the real
        # path in :meth:`_ensure_encoder`; left off for the model-agnostic stub.
        self._normalize_reps = False

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

        The proven path from ``scripts/run_vjepa2.py``: the HF ``AutoModel`` whose
        ``get_vision_features`` :meth:`reset` calls. The AC predictor's encoder is
        frozen during action-conditioned post-training, so this same checkpoint is
        the one the predictor expects embeddings from.
        """
        if self._encoder is not None:
            return
        import torch
        from transformers import AutoModel

        device = self._backend.torch_device(self._config.device_index)
        dtype = getattr(torch, self._config.dtype)
        self._encoder = (
            AutoModel.from_pretrained(self._config.encoder_repo, dtype=dtype).to(device).eval()
        )
        self._tokens_per_frame = _infer_tokens_per_frame(self._encoder)
        self._normalize_reps = True

    def _ensure_predictor(self) -> None:
        """Load the action-conditioned predictor head if not already present."""
        if self._predictor is not None:
            return
        # The encoder sets ``_tokens_per_frame`` (the spatial grid the predictor's
        # next-frame block spans), so load it first.
        self._ensure_encoder()
        self._predictor = _load_ac_predictor(self._config, self._tokens_per_frame)

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
        # Keep the last ``context_frames`` *frames*; each frame is
        # ``_tokens_per_frame`` patch-token rows (1 on the stub path).
        keep = self._config.context_frames * self._tokens_per_frame
        context = self._maybe_norm(_as_context(features)[-keep:])
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
            # The predictor returns the next frame: a single embedding ``(D,)`` on
            # the stub path, or a ``(P, D)`` block of patch tokens on the real
            # path. Shape to a 2-D block, (real-path) layer-norm it like the
            # reference wrapper, append, then cap to the window.
            block = self._maybe_norm(nxt if nxt.ndim == 2 else nxt.unsqueeze(0))
            context = torch.cat([state.context, block], dim=0)
            keep = self._config.context_frames * self._tokens_per_frame
            context = context[-keep:]
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
        # Terminal state = the last frame (its ``_tokens_per_frame`` rows; the last
        # single row on the stub path). ``vector_norm`` flattens, so the stub case
        # ``[-1:]`` is numerically identical to the previous ``[-1]``.
        terminal = rolled.context[-self._tokens_per_frame :]
        energy: torch.Tensor = torch.linalg.vector_norm(terminal - goal)
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
        """Resolve a conditioning observation to a ``(1, T, C, H, W)`` pixel clip.

        ``NONE`` yields a synthetic seed clip (the encoder maps it to the prior);
        ``IMAGE``/``VIDEO`` decode the ``uri`` into a clip. The result is placed on
        the backend device + dtype so it feeds the (real) encoder directly.

        NOTE: the exact pixel normalization the V-JEPA 2 encoder expects is the HF
        ``VJEPA2VideoProcessor`` recipe (resize/center-crop + ImageNet mean/std).
        ``_decode_observation`` applies that recipe; parity with the processor is a
        thing to confirm against real inputs on the GPU box (serving IO, Phase 2).
        """
        n, r = self._config.seed_frames, self._config.seed_resolution
        if conditioning.kind is ConditioningKind.NONE:
            import torch

            clip = torch.zeros(1, n, 3, r, r)
        else:
            clip = _decode_observation(conditioning, frames=n, resolution=r)
        return self._to_backend(clip)

    def _to_backend(self, clip: torch.Tensor) -> torch.Tensor:
        """Move a pixel clip onto the backend device + configured dtype.

        Guarded so the model-agnostic stub backend (no ``torch_device``) leaves the
        clip on CPU/float32 — the injected fake encoder ignores it anyway.
        """
        import torch

        to_device = getattr(self._backend, "torch_device", None)
        if to_device is None:
            return clip
        device = to_device(self._config.device_index)
        dtype = getattr(torch, self._config.dtype)
        return clip.to(device=device, dtype=dtype)

    def _maybe_norm(self, reps: torch.Tensor) -> torch.Tensor:
        """Layer-norm reps over the embedding dim on the real path (no-op on stub).

        Matches the reference ``WorldModel(normalize_reps=True)``: the AC predictor
        is trained on layer-normed reps, so both the seed context and each predicted
        frame are normalized.
        """
        if not self._normalize_reps:
            return reps
        import torch.nn.functional as F  # noqa: N812

        return F.layer_norm(reps, (reps.shape[-1],))


class _AcPredictorAdapter:
    """Bridge the real V-JEPA 2-AC predictor to the ``_Predictor`` contract.

    The research predictor's forward (verified against
    ``facebookresearch/vjepa2`` ``src/models/ac_predictor.py`` and the reference
    ``notebooks/utils/world_model_wrapper.py``) is
    ``forward(x, actions, states, extrinsics=None) -> tokens`` over **patch
    tokens** ``x: (B, N_ctxt, D)`` (``N_ctxt`` = ``T`` frames x ``P`` spatial
    patches). It interleaves one action + one state token *per frame*, so
    ``actions`` and ``states`` are ``(B, T, action_dim)``; it predicts
    frame-causally and the **next frame** is the trailing ``P`` rows of the output
    (``predictor(...)[:, -tokens_per_frame:]`` in the reference wrapper).

    This adapter exposes Mirage's ``(context, action) -> next_frame`` contract: it
    batches the context, drives the *next* frame with ``action`` at the last frame
    position (earlier positions are no-op zeros), supplies a zero proprioceptive
    state, calls the predictor, and returns the predicted next frame's ``P`` rows
    for :meth:`VJepa2ACEngine.step` to append.

    REFINE: the reference integrates a real 7-DoF robot pose across steps
    (``compute_new_pose``); zeros here exercise the dynamics as a function of the
    action, which is what the load/rollout/planning smoke test needs. A real
    closed-loop robot deployment would thread the pose through ``WorldState``.
    """

    def __init__(
        self,
        predictor: Any,
        tokens_per_frame: int,
        action_dim: int,
        compute_dtype: Any = None,
    ) -> None:
        self._predictor = predictor
        self._p = tokens_per_frame
        self._adim = action_dim
        # The predictor runs in this dtype; context is cast in and the result cast
        # back. The upstream RoPE attention upcasts q/k to float32, so running the
        # predictor in float32 keeps q/k/v dtypes consistent (bf16 hits a SDPA
        # dtype-mismatch). ``None`` (the stub/unit-test path) does no casting.
        self._cdtype = compute_dtype

    def __call__(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        import torch

        in_dtype = context.dtype
        x = context.unsqueeze(0)  # (1, N_ctxt, D)
        if self._cdtype is not None:
            x = x.to(self._cdtype)
        n_frames = int(x.shape[1]) // self._p
        # Per-frame action/state tokens (B, T, action_dim); drive the next frame
        # with `action` at the last position, no-op (zero) for past frames.
        actions = x.new_zeros(1, n_frames, self._adim)
        actions[:, -1] = action.to(actions.dtype)
        states = torch.zeros_like(actions)
        out = self._predictor(x, actions, states)
        tokens = out if isinstance(out, torch.Tensor) else out.last_hidden_state
        next_frame: torch.Tensor = tokens[0, -self._p :]  # predicted next frame
        return next_frame.to(in_dtype)


#: The published V-JEPA 2-AC checkpoint (encoder + predictor weights). NOTE: the
#: repo's Torch Hub entrypoint hard-codes a placeholder ``localhost`` URL, so we
#: build the model with ``pretrained=False`` and load these weights ourselves.
_AC_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"


def _load_ac_predictor(config: VJepa2ACConfig, tokens_per_frame: int) -> _Predictor:
    """Load the V-JEPA 2-AC predictor head as a ``_Predictor`` callable.

    Wires the action-conditioned head from ``facebookresearch/vjepa2`` (it is not
    an HF ``AutoModel`` today) via its Torch Hub entrypoint. The entrypoint's
    ``pretrained=True`` path points at a placeholder ``localhost`` checkpoint URL,
    so we build the architecture with ``pretrained=False`` and load the published
    checkpoint's ``predictor`` sub-state-dict ourselves (the same key-cleaning the
    repo applies). The encoder is reused from the HF load (frozen during AC
    post-training, so they match). The predictor is wrapped in
    :class:`_AcPredictorAdapter` to satisfy the ``(context, action) -> next_frame``
    contract the rollout/planner expect.
    """
    import torch

    device = config.device_index
    _, predictor = torch.hub.load(  # type: ignore[no-untyped-call]
        config.predictor_repo, "vjepa2_ac_vit_giant", pretrained=False
    )
    ckpt = torch.hub.load_state_dict_from_url(_AC_CHECKPOINT_URL, map_location="cpu")
    state_dict = {
        k.replace("module.", "").replace("backbone.", ""): v
        for k, v in ckpt["predictor"].items()
    }
    predictor.load_state_dict(state_dict)
    # Run the predictor in float32: its RoPE attention upcasts q/k to float32, so
    # bf16 weights hit an SDPA dtype mismatch. It's only ~300M params; the adapter
    # casts the (bf16) context in and the next frame back out.
    predictor = predictor.to(f"cuda:{device}", dtype=torch.float32).eval()
    return _AcPredictorAdapter(predictor, tokens_per_frame, config.action_dim, torch.float32)


def _infer_tokens_per_frame(encoder: Any) -> int:
    """Spatial patch tokens per temporal frame: ``(crop_size / patch_size) ** 2``.

    Read from the encoder's HF ``config``; falls back to 1 when there is no config
    (the model-agnostic stub path), keeping ``WorldState.context`` rows == frames.
    """
    cfg = getattr(encoder, "config", None)
    crop = getattr(cfg, "crop_size", None)
    patch = getattr(cfg, "patch_size", None)
    if not crop or not patch:
        return 1
    return int((crop // patch) ** 2)


def _decode_observation(
    conditioning: ConditioningInput, *, frames: int, resolution: int
) -> torch.Tensor:
    """Decode an image/video ``uri`` into a ``(1, T, C, H, W)`` float clip.

    A single image is tiled to ``frames``; a video is decoded and sampled to
    ``frames``. Pixels are resized to ``resolution`` and ImageNet-normalized — the
    V-JEPA 2 recipe. Parity with the HF ``VJEPA2VideoProcessor`` is to be confirmed
    on real inputs (Phase 2 serving IO); raises if no ``uri`` is given.
    """
    import torch
    from torchvision.io import read_image, read_video
    from torchvision.transforms import functional as tvf

    uri = conditioning.uri
    if not uri:
        raise ValueError(f"{conditioning.kind} conditioning requires a uri")

    if conditioning.kind is ConditioningKind.IMAGE:
        img = read_image(uri).float() / 255.0  # (C, H, W)
        clip = img.unsqueeze(0).expand(frames, -1, -1, -1)  # (T, C, H, W)
    else:  # VIDEO
        video, _audio, _info = read_video(uri, output_format="TCHW")  # (T, C, H, W)
        video = video.float() / 255.0
        idx = torch.linspace(0, max(video.shape[0] - 1, 0), frames).long()
        clip = video[idx]

    clip = tvf.resize(clip, [resolution, resolution], antialias=True)
    clip = tvf.normalize(clip, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    result: torch.Tensor = clip.unsqueeze(0)  # (1, T, C, H, W)
    return result


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
