# ADR-0004 — Polyglot build tooling scaffolded; Rust-core fork deferred

- **Status:** Accepted (scaffold only — no Rust code yet)
- **Date:** 2026-05-23
- **Relates to:** the Implementation Plan's "Language stack" section in
  `~/Mirage_Cowork_Handoff.md`; ADR-0003 (vendor-neutral backend Protocol)

## Context

The handoff doc commits to a five-language stack with non-overlapping jobs:

- **Python** — user-facing API, model loading, HF/Diffusers/PyTorch integration
- **Rust** — Runtime core (scheduler, request router, paged latent-cache manager)
- **Triton + CUDA / HIP C++** — kernels
- **C++ (MLIR)** — kernel synthesizer's dialect and lowering passes (Phase 4+)
- **TypeScript + React** — Studio dashboard (Phase 3+)

with Cargo + CMake + uv as the build trio (resist Bazel until a real polyglot
monorepo coordination problem appears).

The handoff also names a specific fork to resolve before writing the Rust
core: **if any of the first 3 design partners is a robotics OEM (closed-loop,
sub-100ms edge), Rust-core from day one; if all 3 are datacenter, Python-core
is defensible and the Rust hot-path rewrite is budgeted into Series A.**

Today the Runtime core is Python (vLLM-style), the MI300X path runs at 2.47×
the H100 reference at 121f / 36 steps, and no design partner has signed yet.

## Decision

Scaffold the polyglot tooling now, with **no Rust code yet**:

1. A virtual Cargo workspace at the repo root (`Cargo.toml`, `members = []`).
2. `crates/` for future Rust workspace members.
3. `kernels/` for future Triton / CUDA / HIP — deliberately at the repo root,
   not under `src/`, so it carries a different review standard.
4. `rust-toolchain.toml` pinning stable.
5. `make rust-*` targets that operate on the (empty-for-now) workspace.

The first crate is **not** added by this ADR. Adding the first crate is what
resolves the fork — and that decision is gated on design-partner status, not
on developer convenience.

## Rationale

- **Decouple "can we write Rust" from "should we write Rust *now*."** With
  the scaffold in place, the moment the fork resolves we can populate
  `crates/mirage-cache` (or whichever component goes first) without spending
  a sprint on `pyproject.toml` ↔ `Cargo.toml` build wiring under deadline.
- **Cheap insurance against the robotics scenario.** Per the handoff:
  Python-core means a 6–9 month Rust rewrite if a robotics OEM lands as
  design partner #1 or #2. The scaffold doesn't pre-pay any of that, but it
  removes the "build system bring-up" tax from the rewrite path entirely.
- **Forcing function for `kernels/` as a separate codebase.** The handoff is
  explicit: "Mixing clean-architecture and perf code produces clean-but-slow
  or fast-but-unmaintainable. Treat the kernel layer as a separate codebase
  with different review standards." Putting `kernels/` at the repo root,
  with no expectation that mypy or ruff covers it, locks that culturally.
- **The Cargo + CMake + uv triad, not Bazel.** Handoff explicitly defers Bazel
  until there's a "genuine polyglot monorepo coordination problem." We are
  far from that.

## Consequences

- **CI will need a Rust step soon.** None of `rust-fmt-check`, `rust-clippy`,
  `rust-test` does anything meaningful on an empty workspace, but the targets
  exist so the moment we land a crate, CI just needs to call them — no new
  pipeline design under pressure.
- **`maturin` is the planned Python↔Rust binding tool** (not pinned yet).
  Added to `pyproject.toml` `[project.optional-dependencies] dev` when the
  first PyO3 crate lands, not before.
- **`Cargo.lock` will be committed** once the first crate exists (workspace
  produces a Python extension module via PyO3 — reproducible builds are the
  right default). Until then there is no lock to commit.
- **No change to the running MI300X path.** This ADR is repo-shape only; the
  Cosmos-Predict-7B path, the 2.47× headline, the publish-ready writeup, the
  35-file Python codebase, and the 36-test suite are all untouched.

## Revisit when

The first design partner conversation converges and we know whether the
robotics-OEM branch is live. At that point either:

- **Robotics-OEM in:** populate `crates/mirage-cache` immediately (paged
  latent-cache manager has the fewest callers, cleanest PyO3 boundary).
  Then `crates/mirage-scheduler`, then `crates/mirage-router`.
- **All datacenter:** stay Python-core. Re-evaluate at Series A planning
  (Month 14–18) with the operational scars we've actually accumulated.

The trigger is *design partner mix*, not engineering preference.
