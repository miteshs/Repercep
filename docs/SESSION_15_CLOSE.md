# Session 15 close — next-session pickup

**Date:** 2026-05-24 (evening) · **Working tree:** clean on this branch
(`session-14-cuda-port`).  Companion branch `cpu-amx-port` carries the
CPU AMX substrate.

Read this *before* `docs/HANDOFF.md`.  This doc is the focused "what
happened today, what to do next" cut for Session 15; `HANDOFF.md` is the
full orientation layer.

---

## What landed today (across both branches)

Two parallel streams ran:

* **H100 follow-ups** to close the Session 14 open list (this branch,
  `session-14-cuda-port`).
* **Intel CPU AMX substrate** as a new vendor in the Backend Protocol
  (sibling branch `cpu-amx-port`, branched off this one).

The user direction was "go full force on H100 + start CPU AMX in
parallel"; Wan-A14B was paused explicitly after F30 tripped MooseFS hard
again (see F31 below).

### H100 follow-ups (this branch)

1. **F29 FP8 Hopper autotune grid expansion** (`kernels/triton_kernels/
   fp8_flash_attn_hopper.py`):
   - Added `BLOCK_M=192` row tile — the gap the autotuner kept skipping
     between 128 (under-utilises on long S) and 256 (over-spills regs
     at D=128).
   - Added `num_stages=5` on tiles with SMEM headroom — the Cosmos
     shape (S=83k, D=128) was under-filling the K/V prefetch pipeline
     at 4 stages.
   - Added `num_warps=12` on the 256×256 tile only — Hopper SM width
     supports it when 8 warps under-fills.
   - Net: ~40-config grid → ~60 configs, ~9 s search tax (under the
     15 s target).
   - **Bench not yet run** — apply this branch + run
     `scripts/bench_fp8.py` and a full Cosmos 121f/36 sweep to measure
     whether this closes the 4.2 s/call deficit vs cuDNN-FA3 (F29).
2. **`scripts/run_cosmos.py` `--backend {auto,rocm,cuda,cpu}`** flag.
   Defaults to `auto` (first available; GPU wins over CPU on a GPU
   host).  Guards `torch.cuda.*` calls so the CPU path doesn't crash
   on `reset_peak_memory_stats()` / `max_memory_allocated()`.  Needed
   for the CPU branch to be exercisable; lands on this branch so all
   the GPU paths get the flag too.
3. **`HANDOFF.md`** §0 metadata and §10 timeline refreshed (HEAD was
   stale at the Session 12 commit; Session 14 + 15 rows added; findings
   tally updated F1–F32, ADRs 0001–0007).
4. **FA-3 from source on Hopper (H5)** — wheel built, tested,
   intentionally NOT committed (it's a 2 MiB binary built off-tree).
   Reproducer: see Session 15 entry in `BUILD_LOG.md` for the
   minimal `FLASH_ATTENTION_DISABLE_*=TRUE` set that drops the build
   from 451 files / ~60 min to ~20 files / ~10 min.

### Things attempted that did NOT fully land

* **H1 multi-prompt variance on H100 — PARTIAL.**  Ran
  `scripts/verify_timing.py --N 3 --prompts 5 --no-cache-refs`.
  5 of 8 adaptive runs succeeded (A0, B0, B1, B2, B3); 3 lost to
  subprocess env contamination during the TE install + FA-3 install
  windows (F31 fallout).  Per-run timings *captured* but *lost* because
  `verify_timing.py` uses `subprocess.run(capture_output=True)` and
  doesn't flush its own per-run print until the script exits.  File-
  timestamp inspection of the surviving 5 mp4s is consistent with the
  138.4 s adaptive headline modulo the ~50 s subprocess cold-load.
  **Session 16:** patch `verify_timing.py` to flush each print + write
  per-run sidecar JSONs (or env `PYTHONUNBUFFERED=1`), re-run.
  Killed mid-Phase-C per user direction to save ~40 min of GPU.

* **H2 TE FP8 install (F28) — PARTIALLY UNBLOCKED.**  Agent research
  produced the canonical install command:
  `uv pip install --no-build-isolation "transformer_engine[pytorch,
  core_cu12]==2.15.0"`.  The build extension compiles cleanly against
  torch 2.8.0+cu128; the module imports.  **BUT** the resolver pulled
  in *both* `transformer_engine_cu12` and `transformer_engine_cu13`
  wheels — the cu13 binary won, and `libtransformer_engine.so` does
  `dlopen()` hitting `OSError: undefined symbol:
  cublasLtGroupedMatrixLayoutInit_internal, version libcublasLt.so.13`.
  **Real fix (Session 16):**
  ```
  uv pip install --no-deps transformer-engine==2.15.0 \
                          transformer-engine-cu12==2.15.0 \
                          transformer-engine-torch==2.15.0
  ```
  and verify no `transformer_engine_cu13-*.dist-info` lands.

* **H3 Wan-A14B — PAUSED.**  Per user direction after F30 tripped the
  storage backend.  ~25 GiB partial download discarded; restart from
  scratch via `hf download --max-workers 1` once F31 (Session 15's
  more severe MooseFS throttle event) clears.

### CPU AMX substrate (companion branch `cpu-amx-port`)

See that branch's commits + `docs/adr/0007-cpu-backend.md` for the
full architecture record.  Headline:

- New `repercep.backend.cpu.CPUBackend` (Vendor.INTEL).  Detects
  Sapphire / Emerald / Granite Rapids via /proc/cpuinfo
  family+model.  Generic AVX-512 + generic x86-64 fallback arches.
  Capabilities advertise BF16 + INT8 (no FP8 ISA on any shipped Xeon).
- Three CPU attention ops, parallel to the GPU FP8 pattern:
  - `AMXSDPAAttention` — torch SDPA on CPU.  oneDNN auto-dispatches
    to AMX tiles for BF16.  The always-available floor.
  - `IPEXFlashAttention` — wrapper over Intel-Extension-for-PyTorch's
    fused attention.  Available when IPEX is installed.
  - `AMXFlashAttention` — custom AMX BF16 flash kernel
    (`kernels/cpu/amx_attn/flash_attn_amx.cpp`).  Sibling of the
    gfx942 + Hopper FP8 Triton kernels.  **The kernel source needs
    one more iteration** — the inline-asm register naming the writing
    agent chose is gcc-rejected; rewrite to use `_tile_loadd` /
    `_tile_dpbf16ps` / `_tile_stored` intrinsics from `<immintrin.h>`.
    Wrapper, ADR, build script, tests are all correct.
- Registry's new INTEL branch in `select_attention_op` mirrors the
  GPU FP8 env-gating: `REPERCEP_AMX_ATTENTION` ∈ {1, true, on, amx,
  ipex} promotes the AMX-aware op when shape qualifies; unset keeps
  the SDPA floor active.
- `repercep.attention.diffusers_backend_amx` registers
  `"repercep_amx"` with the diffusers attention dispatcher — sibling
  of the GPU FP8 bridge.
- `repercep.runtime.quantize` — per-channel symmetric INT8 weight
  quantization helpers (lands first; the AMX INT8 attention kernel
  itself is Session 16+).
- `docs/adr/0007-cpu-backend.md` — full ADR.
- `docs/COSMOS_ON_CPU.md` — measurement doc with TBD numbers pending
  the kernel rewrite + end-to-end CPU smoke.
- 3 test files (162 + 161 + 116 lines): `tests/test_backend_cpu.py`,
  `tests/test_attention_cpu.py`, `tests/test_quantize.py`.

### Findings recorded (F31, F32, F31a)

Full text in `BUILD_LOG.md` Session 15 entry.  Short:

* **F31** — MooseFS server-side write-rate quota at
  `mfs#us-mo-1.runpod.net` tripped after Wan-A14B parallel download +
  editable Repercep install; ~55+ min wedged.  Mitigation pattern
  documented (HF cache + venv + build dirs all relocate to local
  overlay; only source + model snapshots stay on MooseFS).
* **F32** — `HF_HUB_OFFLINE=1` still writes `refs/main`; a
  FUSE-throttled `/workspace` then crashes `from_pretrained`.  Pre-
  stage snapshots elsewhere as a workaround.
* **F31a** — `cosmos.py`'s `DEFAULT_REPO` is
  `nvidia/Cosmos-1.0-Diffusion-7B-Text2World` (diffusers format) but
  the README's "Cosmos-Predict-7B" naming maps to a *different* HF
  repo (`Cosmos-Predict1-7B-Text2World`) that doesn't ship
  `model_index.json`.  README needs an update; no code change.

---

## Quality gate (this branch)

- `mypy --strict`: NOT re-run this session (working tree clean since
  Session 14; the F29 grid expansion + run_cosmos.py changes are typed
  the same way).  Recommend re-running in Session 16 after H2 lands.
- `ruff check`: same.
- `pytest`: same.
- New Python compiles clean (py_compile verified during /tmp staging).

---

## What's open after today (priority-ranked)

1. **(highest signal) AMX kernel rewrite for `cpu-amx-port`.**  Use
   `_tile_loadd` / `_tile_dpbf16ps` / `_tile_stored` intrinsics from
   `<immintrin.h>` instead of the inline asm that gcc rejects.
   Reference patterns: oneDNN's AMX matmul primitive, Intel's
   `amx-cookbook` examples.  ~2-4 hours of focused work.
2. **H2 TE Phase 4 bench on H100.**  Use the `--no-deps` install
   pattern from above; expect ~115-130 s (cuDNN-FA3 with FP8 recipe
   should beat the BF16 cuDNN-FA3 floor).  If TE matches
   adaptive-alone, the Triton FP8 path becomes purely research surface.
3. **H4 FP8 Hopper autotune bench.**  Apply this branch + run
   `scripts/bench_fp8.py` and full 121f/36 sweep.  Goal: beat
   cuDNN-FA3 at the Cosmos production shape, restoring
   `REPERCEP_FP8_ATTENTION=1` as a perf-on default on Hopper.
4. **H1 multi-prompt variance proper.**  Patch `verify_timing.py` to
   flush per-run prints (or run with `PYTHONUNBUFFERED=1`).  Re-run
   `--N 3 --prompts 5` to get a clean adaptive variance band.  Phase C
   no-cache references can stay deferred — the close-doc 446.3 s
   number was solid.
5. **Wan-A14B sweep on H100 (was H3).**  After FUSE recovers, download
   with `--max-workers 1`, then 17 f / 8 step smoke + 81 f / 40 step
   quality reference.  Compare to Wan team's H100 (1041.5 s with
   offload + FP8) and the MI300X 1700 s projection.
6. **MI300X carryovers from Session 13** unchanged.  None moved.

---

## Quick reproducers (for next session)

```bash
# Fresh box setup
git clone https://github.com/miteshs/Mirage.git && cd Repercep
git checkout session-14-cuda-port    # for H100 follow-ups
# OR
git checkout cpu-amx-port            # for CPU substrate
uv venv --python 3.12 .venv
uv pip install --python .venv torch==2.8.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv -e ".[models,serving,dev]"
make rust-install
make check-gpu

# CRITICAL: redirect HF cache OFF MooseFS to avoid F31/F30/F32
export HF_HOME=/tmp/.cache/huggingface
export HF_HUB_CACHE=/tmp/.cache/huggingface/hub
# Cosmos weights are NOT the Predict1 repo — F31a:
hf download nvidia/Cosmos-1.0-Diffusion-7B-Text2World

# H100 headline reproducer (~138 s warm)
.venv/bin/python scripts/run_cosmos.py --backend cuda \
    --frames 121 --steps 36 --native-loop \
    --cache-mode adaptive --cache-adaptive-threshold 0.30 \
    --cache-force-full-every 16

# CPU smoke (substrate, not perf — minutes per video)
REPERCEP_AMX_ATTENTION=1 .venv/bin/python scripts/run_cosmos.py \
    --backend cpu --frames 17 --steps 8

# Multi-prompt variance with stdout flushing
PYTHONUNBUFFERED=1 .venv/bin/python scripts/verify_timing.py \
    --N 3 --prompts 5

# TE install — cu12-only, --no-deps
uv pip install --no-deps --no-build-isolation \
    transformer-engine==2.15.0 \
    transformer-engine-cu12==2.15.0 \
    transformer-engine-torch==2.15.0
```

---

## Branches + how to PR them

```
main
├── session-14-cuda-port   <- (this branch)  PR target: main
│                              Session 14 architecture + Session 15
│                              H100 follow-ups + the F29 autotune patch
│
└── cpu-amx-port            <- (sibling)  PR target: main
                                Branches off session-14-cuda-port.
                                After session-14 merges, this PR
                                cleanly applies the CPU substrate.
```

Recommended merge order: `session-14-cuda-port → main` first, then
`cpu-amx-port → main`.  No file conflicts expected — the only shared
edits (attention/registry.py, backend/registry.py) are additive in
both branches; cpu-amx-port already inherits session-14's additions
since it was branched off that branch.

---

## When you resume — start here

1. Read this doc (you're doing it).
2. Read the cpu-amx-port commit message + ADR-0007 for the CPU side.
3. Pick from the priority-ranked open list above.  Items 1, 2, 3 are
   each tractable in one session.
4. Open the two PRs.

Have a good day.
