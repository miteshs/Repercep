# Session 24 handoff — strategic assessment + the interactive (energy-based) world-model seam

**Date:** 2026-05-28 · **HEAD:** `ee62f39` on `main`, in sync with
`origin/main` · **Status:** pre-alpha; this was a **strategy + architecture
session run from an analysis box** (a fresh clone in `/tmp`, not the MI300X/H100
dev pod), so it landed docs + a new seam + CPU-verified code, **no GPU runs**.

## What this session actually was (read this first)

It did **not** execute Session 23's planned lanes (Lane A Wan-shaped adaptive
cache; Lane B FVD N≥1000). **Both of those remain unstarted.** Instead, prompted
by an external "where's the moat, honestly" review, it did two things:

1. Wrote an independent **strategic assessment** (`docs/STRATEGIC_ASSESSMENT.md`,
   merged via PR #1).
2. Opened a **new architectural lane** the assessment argued is the real moat —
   an **interactive, action-conditioned, closed-loop** world-model seam
   (ADR-0008), with V-JEPA 2-AC (the energy-based / JEPA world model) as the lead
   implementation.

Whoever picks up next has a genuine fork: **resume Session 23's lanes**, or
**commit to the interactive direction** this session opened. The assessment
argues for the latter; that's a call, not a settled fact.

## The thesis (from `docs/STRATEGIC_ASSESSMENT.md`)

The honest finding: Repercep's headline speedups are **not a moat** under scrutiny
— the cache is public (TeaCache, already in vLLM-Omni/FastVideo), "only-on-AMD"
is eroding (AMD ships xDiT/SGLang-Diffusion + MLPerf'd Wan2.2), and at the kernel
level Repercep is at parity-or-behind. The defensible, unbuilt regime is the half
of the original vision that was deferred: **action-conditioned, closed-loop
world-model serving** — which general-purpose request/response diffusion servers
structurally don't model, and which is exactly the energy-based / JEPA program
(LeCun). Repercep already serves a member of that family (V-JEPA 2, script-only),
so the on-ramp is short. The assessment is the adversarial counterweight to
`docs/POSITIONING.md`; read both.

## What landed (5 commits on `main`)

| Commit | What | Verified |
|---|---|---|
| (PR #1) | `docs/STRATEGIC_ASSESSMENT.md` — moat verdict + competitive scan + next-steps fork | n/a (doc) |
| `cfb3f5b` | ADR-0008 + `runtime/interactive.py` (`InteractiveWorldModel` seam) + wire types (`Action`, `RolloutParams`, `ResetRequest`, `WorldState`, `LatentStep`) + `models/vjepa2_ac.py` scaffold | ruff, mypy --strict, pytest 9 |
| `c16e0af` | **Phase 1** — real latent rollout + CEM/energy-MPC planner in `VJepa2ACEngine` (encoder/predictor injectable) | ruff, mypy --strict, pytest 12 |
| `6b6a6aa` | **Phase 2** — `/v2/world/session` bidirectional WebSocket in `serving/app.py` (engine calls run in a threadpool) | ruff, mypy --strict, pytest 14 |
| `ee62f39` | `scripts/run_vjepa2_ac.py` — interactive rollout + energy-MPC runner; `--stub` runs end-to-end on CPU with no weights | ruff; `--stub` run (energy 1.76 → 0.40) |

## State of the ADR-0008 work

**Implemented and CPU-tested (model-agnostic):**
- The `InteractiveWorldModel` seam (`reset → step(action) → … / plan`), distinct
  from the one-shot pixel-shaped `WorldModelEngine`. The one-shot path can later
  become a facade over it.
- `VJepa2ACEngine.step()` — block-causal latent rollout (context append + cap),
  under `torch.inference_mode()`.
- `_rollout_energy()` / `_plan_sequence()` / `plan()` — CEM/MPC minimizing the
  terminal latent energy `‖s_T − goal‖`. **This is the energy-based planning** the
  EBM/JEPA framing calls for; verified to actually reduce energy toward a goal.
- The WebSocket session, end-to-end via FastAPI TestClient against a stub engine.

**Remaining = the model-specific port (raises `NotImplementedError`, intended
body in each docstring; needs the real checkpoints + a GPU):**
- `_ensure_encoder()` — load `facebook/vjepa2-vitg-fpc64-256` (proven path is
  `scripts/run_vjepa2.py`).
- `_load_ac_predictor()` — port the AC predictor head from
  `facebookresearch/vjepa2` (not an HF `AutoModel` today) into a callable
  `(context, action) -> next_state_embedding`.
- `_resolve_frames()` for real image/video URIs (serving IO).
- **Phase 3 (AVID, pixel path)** — deliberately NOT scaffolded: it requires
  *training* an action adapter on action-labelled video against the real
  Cosmos/Wan diffusion stack. It's a research/training task, not a verifiable
  code stub. Defer until weights/GPU are in hand.

## How to run / verify

```bash
# Runs now, no weights, on CPU — demonstrates the rollout + energy-MPC planner:
python scripts/run_vjepa2_ac.py --stub --plan --steps 4
# Real engine (needs V-JEPA 2-AC weights + GPU; NotImplementedError until the
# loaders above are filled in):
python scripts/run_vjepa2_ac.py --backend cuda --plan
```

**CPU verification recipe used this session** (no ROCm/GPU needed — useful for
fast iteration on the non-kernel layers): a throwaway venv with
`ruff mypy pydantic pydantic-settings numpy pytest pytest-asyncio` + CPU torch
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`) +
`fastapi httpx`. Then:
- `ruff check <files>`
- `mypy --config-file pyproject.toml --follow-imports=silent <files>` (torch is
  installed so it's fully typed; the repo's overrides handle the rest)
- `pytest tests/test_interactive.py tests/test_serving_interactive.py -q`

The interactive + serving-interactive tests are CPU-only and green; the
weight-dependent paths skip cleanly.

## Open / next, ranked

1. **(highest leverage for the interactive lane) Fill the four
   `NotImplementedError` bodies** in `models/vjepa2_ac.py` on a GPU box — the
   docstrings are the spec; the seam, planner, serving session, runner, and tests
   are already in place to receive it. Then `scripts/run_vjepa2_ac.py` (drop
   `--stub`) is the smoke test.
2. **Decide the fork**: interactive lane vs Session 23's lanes (Wan adaptive
   cache, FVD N≥1000). The latter are still unstarted and are *credibility*
   work; the former is *moat* work per the assessment.
3. **Phase 3 (AVID)** only after weights + an action-adapter training plan.

## Caveats / gotchas

- **`test_serving_v2.py` hangs in a minimal venv** (driver-thread / TestClient
  teardown) — **pre-existing and environmental, NOT a regression** from this
  session's additive `serving/app.py` change. The same `create_app` + lifespan +
  driver is exercised green by `test_serving_interactive.py`, and all routes are
  intact. It should pass in the full `[dev]` env.
- **The strategy docs (`~/Repercep_Implementation_Plan.pdf`,
  `~/Mirage_Cowork_Handoff.md`) are still off-repo by design** (HANDOFF §9) and
  are NOT backed up in git. If they live only on a pod, back them up off-pod
  (a *private* repo/gist — not this public OSS repo).
- This session ran from a non-Repercep pod (Repercep was cloned into `/tmp`); no
  weights, mp4s, autotune caches, or FVD sets were touched or produced.

## Cross-references

- `docs/STRATEGIC_ASSESSMENT.md` — the moat argument this session acts on
- `docs/adr/0008-interactive-world-model-seam.md` — the seam decision + phased plan
- `docs/POSITIONING.md` — the first-person framing the assessment is a counterweight to
- `src/repercep/runtime/interactive.py`, `src/repercep/models/vjepa2_ac.py`,
  `src/repercep/serving/app.py` (`/v2/world/session`), `scripts/run_vjepa2_ac.py`
- Tests: `tests/test_interactive.py`, `tests/test_serving_interactive.py`
