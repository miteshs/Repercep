# Mirage walkthroughs — V-JEPA 2-AC, the interactive / energy-based path

A guided tour of the *other* kind of world model in Mirage: not a one-shot
diffusion video generator (that's the [Cosmos walkthrough](../walkthroughs/)),
but a **stateful, action-conditioned, closed-loop** world model that predicts in
**latent space** and *plans* by **energy minimization** — Yann LeCun's JEPA /
energy-based program, served. This is the direction
`docs/STRATEGIC_ASSESSMENT.md` argues is the actual moat, and it landed this
session as [ADR-0008](../adr/0008-interactive-world-model-seam.md).

**Who this is for / how it's pitched.** Same as the Cosmos series: *every detail*
on **both axes** — the **ML / model level** (what JEPA is, the energy view of
prediction and planning, why it isn't generative/diffusion) **and the systems /
infrastructure level** (the interactive seam, the rollout, the CEM planner, the
WebSocket serving loop). Fundamentals built up where needed; grounded in real
`file:line`; runnable snippets marked.

## The path

| # | Part | What you'll understand |
|---|------|------------------------|
| 0 | [Orientation](00-orientation.md) | What V-JEPA 2-AC *is*, the `reset → step → plan` call path, and how it differs from the Cosmos diffusion path |
| 1 | JEPA & energy-based models *(coming)* | The ML foundations: predict-in-embedding-space, the energy/compatibility view, why no pixels / no partition function, vs diffusion |
| 2 | The model + weights *(coming)* | The V-JEPA 2 encoder (HuggingFace ViT) + the AC predictor (the `facebookresearch/vjepa2` port); what's real vs scaffold in `models/vjepa2_ac.py` |
| 3 | The interactive seam + latent rollout *(coming)* | `InteractiveWorldModel` (`runtime/interactive.py`) + `VJepa2ACEngine.reset/step` line by line — the block-causal latent rollout |
| 4 | Energy-based planning *(coming)* | `_rollout_energy` / `_plan_sequence` / `plan` — CEM/MPC as energy minimization, with the verified `--stub` demo |
| 5 | Serving the closed loop + what's next *(coming)* | the `/v2/world/session` WebSocket, the bidirectional protocol; the remaining weight port; the AVID pixel sibling |

Read in order. If you haven't done the [Cosmos walkthrough](../walkthroughs/), you
don't need to — but Part 0 contrasts the two, so a glance at that series' Part 0
helps.

## Honest scope (important)

Unlike the Cosmos path (which fully runs given weights), the V-JEPA 2-AC engine is
**part real, part scaffold** — and the tour is explicit about which is which:

- **Real and CPU-tested:** the `InteractiveWorldModel` seam, the latent rollout
  (`step`), and the **energy-MPC planner** (`plan` / `_rollout_energy` /
  `_plan_sequence`) — verified end-to-end on CPU via `scripts/run_vjepa2_ac.py
  --stub` (the planner provably reduces a latent-space energy toward a goal).
- **The remaining port (raises `NotImplementedError`, intended body in the
  docstrings):** loading the real V-JEPA 2 **encoder** (HuggingFace) and the
  **AC predictor head** (`facebookresearch/vjepa2`), plus real video-URI decode.
  These need the actual checkpoints + a GPU.

So this series teaches the *architecture and the algorithm* at full fidelity (you
can run the loop + planner today), and is precise about what's a contract awaiting
weights vs working code.

> Status: Part 0 written; Parts 1–5 in progress. See `docs/SESSION_24_HANDOFF.md`
> for session context and `docs/adr/0008-...` for the design decision.
