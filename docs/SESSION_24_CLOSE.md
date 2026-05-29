# Session 24 close — assessment, the interactive seam, and a full documentation pass

**Date:** 2026-05-28 → 05-29 · **HEAD:** `ffaf369` on `main`, in sync with
`origin/main` · **Working tree:** clean, nothing uncommitted.

This is the close-out. The mid-session forward plan is in
[`docs/SESSION_24_HANDOFF.md`](SESSION_24_HANDOFF.md) and **still holds** — this
doc records *everything that landed* (including the documentation pass that came
after the handoff was written) and where to resume.

## What this session was

Run from an analysis box (a fresh `/tmp` clone, **no GPU** — so docs + CPU-verified
code, no GPU runs). It did not execute Session 23's lanes (Wan adaptive cache;
FVD N≥1000 — **both still unstarted**). Instead: an honest strategic assessment,
then the architectural lane that assessment argued is the moat (interactive /
energy-based world-model serving, ADR-0008), then a full line-by-line
documentation pass over the runtime.

## What landed (17 commits, all on `main`)

**1. Strategic assessment** — `docs/STRATEGIC_ASSESSMENT.md` (PR #1, `c89b373` →
merged `1ff8419`). The honest moat verdict: the headline speedup is the public
TeaCache cache, "only-on-AMD" is eroding, kernels are at parity-or-behind; the
defensible regime is action-conditioned closed-loop world models. The adversarial
counterweight to `docs/POSITIONING.md`.

**2. The interactive / energy-based seam (ADR-0008)** — `cfb3f5b`, `c16e0af`,
`6b6a6aa`, `ee62f39`:
- `runtime/interactive.py` — `InteractiveWorldModel` Protocol (`reset → step →
  plan`), distinct from the one-shot `WorldModelEngine`.
- `models/vjepa2_ac.py` — `VJepa2ACEngine`: real latent rollout + CEM/energy-MPC
  planner (encoder/predictor injectable). **CPU-verified** (`run_vjepa2_ac.py
  --stub`: energy 1.763 → 0.401). The encoder + AC-predictor *weight load* is the
  remaining GPU port (`NotImplementedError`, intended body in docstrings).
- `serving/app.py` — `/v2/world/session` bidirectional WebSocket.
- `scripts/run_vjepa2_ac.py` — the runner (`--stub` runs the loop + planner on CPU).
- New wire types in `runtime/types.py`; ADR-0008.
- **Verified:** ruff clean · `mypy --strict` clean · pytest green
  (`tests/test_interactive.py`, `tests/test_serving_interactive.py`).

**3. README refreshed** (`ee6107f`) to the real current status (3 model families ×
3 targets, headline numbers + methodology caveat, the Rust core/kernels, the
interactive seam) and pointers to the new docs.

**4. Three line-by-line walkthrough series** (cross-linked; both ML + systems
depth; grounded in real source + verified CPU runs):
- `docs/walkthroughs/` — **Cosmos** inference, Parts 0–6 (code tour: API → DiT →
  denoise loop → dispatch → FP8 kernel → output). Grounded in the real diffusers
  0.38.0 source + a verified tiny-DiT CPU run.
- `docs/walkthroughs-vjepa2-ac/` — **V-JEPA 2-AC**, Parts 0–5 (code tour: JEPA/EBM
  foundations → the model → seam/rollout → energy planning → serving). Grounded in
  the real transformers 5.9.0 `VJEPA2Model` + a verified tiny-encoder CPU run +
  Mirage's own code.
- `docs/walkthroughs-avid/` — **AVID**, Parts 0–3 (**design** walkthrough, not a
  code tour: the pixel world model on the same seam; ADR-0008 Phase 3). Real reuses
  cited by `file:line`; the unbuilt adapter clearly marked.

## State at close

- Clean working tree, `HEAD == origin/main == ffaf369`. Nothing to push.
- Nothing precious lives only on this pod (verified earlier in the session) — safe
  to tear down. The **off-repo strategy docs** (`~/Mirage_Implementation_Plan.pdf`,
  `~/Mirage_Cowork_Handoff.md`) remain not-in-git by design; back them up off-pod
  if they live only on a pod.

## Where to resume (forward plan — see SESSION_24_HANDOFF for detail)

1. **The GPU weight port** (highest-leverage for the interactive lane): fill the
   four `NotImplementedError` bodies in `models/vjepa2_ac.py` (docstrings are the
   spec) on a GPU box; the V-JEPA 2-AC walkthrough Part 2 maps the exact encoder +
   AC-predictor wiring. Then `run_vjepa2_ac.py` without `--stub` runs for real.
2. **The fork** (the strategic call): commit to the interactive lane vs. resume
   Session 23's *still-unstarted* lanes (Wan adaptive cache; FVD N≥1000). The
   assessment argues the former (moat) over the latter (credibility).
3. **AVID Phase 3** (only if pursuing the pixel path): the build order is in
   `docs/walkthroughs-avid/03-...`; step 2 (train the action adapter) is the
   irreducible ML.
4. **Optional**: a Wan-2.2 walkthrough — the one major real-code path without one.

## Caveats carried forward

- `tests/test_serving_v2.py` **hangs in a minimal venv** (driver/TestClient
  teardown) — pre-existing and environmental, **not** a regression (the same
  `create_app`/lifespan/driver is green in `test_serving_interactive.py`). Should
  pass in the full `[dev]` env.
- The interactive engine is **part real (seam/rollout/planner/serving, CPU-tested),
  part port (weights)**; AVID is **design-only**. Both are labelled as such
  throughout the docs — no shipped-vs-designed ambiguity.

## CPU verification recipe (reusable, no GPU)

Throwaway venv with `ruff mypy pydantic pydantic-settings numpy pytest
pytest-asyncio fastapi httpx` + CPU torch (`pip install torch --index-url
https://download.pytorch.org/whl/cpu`); for reading model internals, `diffusers`
+ `transformers`. Then `ruff check`, `mypy --config-file pyproject.toml
--follow-imports=silent <files>`, `pytest tests/test_interactive.py
tests/test_serving_interactive.py -q`. This is how the interactive code was
verified and how the walkthrough CPU runs were captured.

— Session 24 complete.
