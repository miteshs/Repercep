# ADR-0009 — KV/latent reuse across rollout steps (the structural latency lever)

- **Status:** Design accepted for both reuse dimensions (§"the two reuse
  opportunities"). The `step()` (persistent-session, temporal) path's
  engine-side seam is **implemented + CPU-tested** this session. The
  CEM-batched-candidate path's engine-side seam is **design-only, not yet
  implemented** — reconciling per-candidate cache divergence with the
  existing batched-SDPA call needs a paged/per-candidate-block cache design
  (§"scope note" below), a bigger lift than `step()`'s single-session case.
  Wrapping the real torch-hub predictor (either path) is a GPU-verify
  follow-up.
- **Date:** 2026-07-11
- **Relates to:** ADR-0008 (interactive world-model seam),
  `docs/LEVERS_2026_07_H100.md` (batching + bf16, GPU-verified same day),
  `docs/mirage-cem-batching-finding` memory (this lever ranked #1, June 26).

## Context

`VJepa2ACEngine.step()` and `_rollout_energy_batched()` feed the **entire**
context window (`context_frames` frames, `context_frames * tokens_per_frame`
tokens — 2048 tokens at the default 8 frames) through the AC predictor's full
transformer stack on **every** call, even though only one new frame's worth of
tokens is actually new. Today's batching (1.6–2.1×, then 7.3× combined with
bf16 — `docs/LEVERS_2026_07_H100.md`) cuts the *number* of forwards and their
*dtype* cost; it does not cut the *quadratic-in-window* cost of each forward.
This ADR designs the lever that does: standard incremental (KV-cache)
attention, adapted to this model's specific RoPE + sliding-window shape.

### The real predictor's architecture (verified against the source, 2026-07-11)

Fetched from `facebookresearch/vjepa2` (`src/models/ac_predictor.py`,
`src/models/utils/modules.py`) — not paraphrased from memory:

- **Token layout, per frame:** `[action_token, state_token, visual_patches...]`
  (`cond_tokens=2`, no extrinsics in our config), frames concatenated
  sequentially: `x = torch.cat([a, s, x], dim=2).flatten(1, 2)`.
- **Attention:** plain `F.scaled_dot_product_attention(q, k, v, attn_mask=...)`
  — **no native KV-cache, no `past_key_values`, no `use_cache` kwarg anywhere**
  in `VisionTransformerPredictorAC.forward(x, actions, states, extrinsics=None)`.
- **Block-causal mask** (`build_action_block_causal_attention_mask`, verbatim):
  ```python
  def build_action_block_causal_attention_mask(T, H, W, add_tokens=1):
      N_T = add_tokens + (H * W)
      N = T * N_T
      mask = torch.zeros(N, N).bool()
      mask_block = torch.ones(N_T, N_T).bool()
      for t1 in range(T):
          for t2 in range(max(0, t1 - T + 1), t1 + 1):
              mask[t1*N_T:(t1+1)*N_T, t2*N_T:(t2+1)*N_T] = mask_block
      return mask
  ```
  Frame `t`'s tokens (including its own action/state) attend to frames `<= t`
  only (`t2` ranges up to and including `t1`). **A frame's K/V never depend on
  any later frame.** This is the fact that makes any caching valid at all.
- **Positions are window-relative, recomputed fresh every call:**
  `T = N_ctxt // (grid_h * grid_w)` — derived from the *current* input's own
  token count, not a caller-supplied offset. There is no `start_pos`/`offset`
  parameter anywhere in the signature.
- **RoPE is rotate-half, standard and exactly invertible/composable**
  (verbatim, per axial band — the same function runs independently on the
  d/h/w position components our `_AcPredictorAdapter` doesn't split further
  since context rows are already flattened patch tokens):
  ```python
  def rotate_queries_or_keys(x, pos):
      omega = torch.arange(D // 2, dtype=x.dtype, device=x.device) / (D / 2.0)
      omega = 1.0 / 10000 ** omega
      freq = torch.einsum("..., f -> ... f", pos, omega)
      emb_sin, emb_cos = freq.sin().repeat(...,2), freq.cos().repeat(...,2)
      y1, y2 = x.unflatten(-1, (-1, 2)).unbind(-1)
      y = torch.stack((-y2, y1), dim=-1).flatten(-2)
      return x * emb_cos + y * emb_sin
  ```
  This is the standard complex-rotation form: `RoPE(x, pos) = R(pos) x` where
  `R(pos)` is a block-diagonal rotation by angle `pos * ω_i` per frequency
  band `i`. Rotations compose: `R(pos - 1) = R(-1) · R(pos)`. **A rotated key
  computed at position `pos` can be re-expressed at position `pos - 1` by
  applying the fixed, position-independent `R(-1)` rotation — without
  recomputing the pre-RoPE linear projection.** This is the mechanism that
  makes a *sliding* window cache-compatible, not just a *growing* one.

## The two reuse opportunities (both real, both grounded in the mask above)

1. **Temporal (within/across `step()` calls):** while the window is *growing*
   (session hasn't hit `context_frames` yet), each already-processed frame's
   K/V is exactly reusable — nothing shifted. Once the window is *full* and
   FIFO-evicts the oldest frame, every retained frame's relative position
   decreases by one — but per the RoPE composition fact above, this is a
   cheap **shift** (`R(-1)` applied to every cached K), not a recompute.
   Net cost per step: **one forward over the new frame's tokens** (query
   against the full cached-and-shifted window) instead of a full window
   forward — O(window) instead of O(window²) in the attention term.
2. **Across CEM candidates (within one `plan()`'s batched rollout):** because
   frame `t`'s K/V cannot depend on frame `t+1`'s action token (causal mask,
   confirmed above), the **pre-rollout context window's K/V is identical
   across all `S` candidates** — it can be computed **once**, not once per
   candidate (today's `_rollout_energy_batched` computes it redundantly
   inside the batched SDPA call, S times). Candidates only diverge from the
   first *new* frame onward.

## Decision

Extend the `_Predictor` seam with an **optional** incremental-call contract,
additive to the existing full-context callable (nothing about the batching/
bf16 work changes; a predictor that doesn't support it just keeps using the
full-recompute path — same pattern as `supports_batch`):

```python
class _CachedPredictor(Protocol):
    supports_kv_cache: bool  # capability flag, mirrors supports_batch

    def init_cache(self, context: torch.Tensor) -> Any:
        """Build a cache (opaque to the engine — predictor-owned per-layer
        K/V, however it wants to represent them) from a full context window,
        once, at session start / whenever the engine has no cache yet."""

    def step_cached(self, cache: Any, action: torch.Tensor) -> tuple[torch.Tensor, Any]:
        """Attend one new frame's tokens (derived from `action`, predictor's
        own concern) against `cache`, return the predicted next frame and the
        cache with that frame appended."""

    def evict(self, cache: Any) -> Any:
        """Drop the oldest frame, applying the `R(-1)`-equivalent shift to
        every remaining cached key (the rotation-composition fact above) so
        their baked-in RoPE positions stay correct after the window slides."""
```

`branch(cache) -> Any` (a cheap independent fork, for the CEM-batched case) is
part of the design but not implemented — see the scope note below.

**What is implemented and CPU-tested now** (this session, `step()` path
only): `VJepa2ACEngine` detects `supports_kv_cache` and, when present, drives
`step()` through `init_cache`/`step_cached`/`evict` instead of the
full-window `_predictor(context, action)` call, keeping one cache per
session (mirrors the `_plan_mean` per-session dict `plan_warm_start` already
added). A **fake** cached predictor (toy linear dynamics where the output
depends on the *sum of all live cached frames* — sensitive enough that a
dropped, duplicated, or stale frame changes the result) proves the
bookkeeping (growth vs. eviction) produces results **identical** to the
existing full-recompute path across many steps, including past the
`context_frames` cap where eviction kicks in — the parity test that matters,
same discipline as the June `_rollout_energy_batched` vs. `_rollout_energy`
parity test.

**Scope note — why `_rollout_energy_batched` (CEM) is design-only for now:**
that path calls the predictor once per rollout timestep with **all S
candidates stacked into one batched tensor** (the batching lever). True
per-candidate KV-caching there means each of the S candidates needs its own
cache once they diverge (from the second rollout step on), which the current
one-big-tensor batched call can't express — it needs either S independent
`step_cached` calls (defeating the batching win) or a paged-attention-style
design where S candidates' caches live in one addressable block and a single
batched kernel call still attends each candidate against only its own cache.
The latter is the right answer long-term but is substantially more engine
machinery than the persistent-session case; deferred as a named follow-up
rather than built partially. **The `step()` win (this ADR's implemented part)
matters more for real serving anyway** — it's what metric #1 of
`docs/CONTROL_LOOP_BENCH.md` (closed-loop step latency under state carryover)
directly measures, and is the online robot-control-loop cost, not just the
offline-planning cost.

**What is explicitly NOT done here (GPU-verify follow-up, tracked
separately):** wrapping the *real* torch-hub predictor's actual
`ACRoPEAttention` blocks to implement `step_cached` — this needs the real
model's internals patched or vendored (its attention/RoPE code is not part of
a stable public API we can import against blind), and correctness there can
only be established by comparing cached-rollout energies against the existing
full-recompute rollout on the real weights, live. Marked `VERIFY ON GPU` at
the seam, matching the file's existing convention.

## Consequences

- The dominant remaining serving cost — O(window²) attention recomputed every
  step — drops to O(window) per step plus a cheap per-eviction shift, which is
  the only lever left that changes the *asymptotic* shape of the cost curve
  (batching and bf16 are constant-factor wins on top of the existing shape).
- Session `WorldState.context` (the wire-safe embedding window) is unchanged;
  the cache is engine-held state keyed by `session_id`, mirroring the pattern
  `LingBotVAPipeline` already established for its own named KV cache — the two
  ports converge on the same session-state shape, which is a good sign for a
  future shared serving-layer abstraction (multi-session resident state,
  `docs/CONTROL_LOOP_BENCH.md` metric 4).
- Risk: if the real predictor's RoPE band-splitting (d/h/w/r axes) doesn't
  cleanly separate for a flattened `(context_frames * tokens_per_frame, D)`
  context the way assumed here, the shift trick may need per-axis handling
  the adapter doesn't currently expose (`_tokens_per_frame` already tracks the
  patch-grid size needed for this). Flagged for the GPU-verify pass.

## Revisit if

The real predictor wrapping proves the shift math doesn't hold exactly (e.g.
if positions are NOT purely window-relative-from-zero in some edge case, or a
band split doesn't compose as assumed) — in which case the fallback is
"cache the growing-window phase only, full-recompute after first eviction",
which still captures most of a session's early-step savings and is a much
smaller, safer change.
