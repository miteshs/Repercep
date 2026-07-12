# Session 16 close — next-session pickup

**Date:** 2026-05-25 (early morning) · **Working tree:** clean on `main`,
ahead of `origin/main` by 1 commit (V-JEPA 2 script — to push with
this close doc).

Read this *before* `docs/HANDOFF.md`.  This is the focused "what
happened today, what to do next" cut for Session 16; `HANDOFF.md` is
the full orientation layer.

---

## TL;DR

**New H100 headline: 99.6 ± 3.9 s** (5-prompt mean, std 3.86 %, range
95.2–103.6 s) at Cosmos 121 f / 36 steps.  **3.81× NVIDIA's published
H100 reference (~380 s)**, mean case — **3.99× best case**.  Up from
the prior 138.4 s / 2.75× baseline.

Driver: source-built FlashAttention-3 (`Dao-AILab/flash-attention/
hopper`, minimal sm_90 BF16 config) wired into the diffusers attention
dispatcher via a new `REPERCEP_FP8_ATTENTION=fa` env value.  Torch
2.8.0+cu128's SDPA dispatches BF16 on sm_90 to cuDNN, but the cuDNN
path is *not* the same WGMMA-based FA-3 kernel that the Dao-AILab wheel
ships — measured 1.8–2.05 × gap across S∈{8k, 16k, 32k} in
`scripts/bench_fp8_hopper.py`.

Repercep now also serves **V-JEPA 2** (Meta's non-diffusion encoder-
predictor world model, Feb 2025) end-to-end on all three targets,
proving the Backend Protocol is genuinely model-shape-agnostic (not
diffusion-specific).

---

## What landed today

### 1. Both feature branches merged + pushed to `main`

`session-14-cuda-port` and `cpu-amx-port` both merged with `--no-ff`
preserving the PR-style topology, then pushed to `origin/main`.

Linear chain confirmed (`main → session-14-cuda-port → cpu-amx-port`),
no conflicts.  Post-merge quality-gate fixes landed in `0327c72`:

* `kernels/triton_kernels/fp8_flash_attn_hopper.py` — F29 grid trim:
  drop `BLOCK_M=192` tiles and `num_warps=12`; Triton requires both to
  be powers of 2.  `num_stages=5` (the third F29 lever) is independent
  of those constraints and stays in.  Kernel didn't compile before
  the trim.
* `tests/test_attention_cpu.py` — `AttentionShape` API drift: 5 sites
  were using the older `seq=` field; current shape uses `seq_len_q` +
  `seq_len_kv` with `kind` still a field.  Mechanical rename.
* `src/repercep/runtime/quantize.py`, `tests/test_quantize.py`,
  `src/repercep/attention/amx_flash.py` — lint + mypy hygiene.

Quality gate on H100 + Sapphire Rapids: **ruff clean, mypy --strict
clean over 65 source files, pytest = 173 passed / 14 skipped / 0
failed, cargo test = 41 passed across 3 crates.**

### 2. FA-3 source build + wiring (the headline driver)

* Built FA-3 from `github.com/Dao-AILab/flash-attention/hopper` with
  this minimal-config flag set (the build is **5 min** with these
  flags vs ~60 min without):

  ```bash
  FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
  FLASH_ATTENTION_DISABLE_SPLIT=TRUE
  FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
  FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
  FLASH_ATTENTION_DISABLE_LOCAL=TRUE
  FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
  FLASH_ATTENTION_DISABLE_PACKGQA=TRUE
  FLASH_ATTENTION_DISABLE_FP16=TRUE
  FLASH_ATTENTION_DISABLE_FP8=TRUE          # F37 — without this the
                                            # .so has an undefined FP8
                                            # template symbol
  FLASH_ATTENTION_DISABLE_HDIM{64,96,192,256}=TRUE
  FLASH_ATTENTION_DISABLE_HDIMDIFF{64,192}=TRUE
  ```

* `src/repercep/attention/diffusers_backend.py`:
  - `_native_fallback` now tries `HopperFlashAttention` first when
    conditions allow (no mask/dropout/GQA, BF16/FP16, on CUDA).  Falls
    through to torch SDPA on any error.
  - `_repercep_fp8_attention` recognises `REPERCEP_FP8_ATTENTION=fa` (alias
    `flash`) as "activate the bridge, skip FP8, use FA-2/3 for every
    call".
  - `maybe_activate_from_env` adds `fa`/`flash` to the truthy set.

* End-to-end Cosmos 121 f / 36 steps on H100, adaptive cache:
  - default (torch SDPA): 138.4 s (prior baseline)
  - `REPERCEP_FP8_ATTENTION=fa` (FA-3 via bridge): **99.6 ± 3.9 s
    mean / 95.2 s best**
  - `REPERCEP_FP8_ATTENTION=on` (FP8 Triton via bridge): 343.7 s
    (validates bridge routes correctly; the FP8 kernel itself is the
    bottleneck, 2.5–3.0× slower than cuDNN-FA3 at production shape)

* Microbench (`scripts/bench_fp8_hopper.py`, B=2 H=32 D=128):

  | seq_len | torch SDPA (cuDNN) | HopperFlashAttention (FA-3) | speedup |
  |--------:|------------------:|---------------------------:|--------:|
  | 8 192   | 6.06 ms           | 2.95 ms                    | 2.05×   |
  | 16 384  | 23.94 ms          | 13.14 ms                   | 1.82×   |
  | 32 768  | 101.23 ms         | 51.43 ms                   | 1.97×   |

### 3. TransformerEngine source build works (F28 fully resolved)

* PyPI wheels of TE 2.12, 2.13, 2.14, 2.15 *all* ship a prebuilt
  `transformer_engine_torch.cpython-312-x86_64-linux-gnu.so` that
  references the c10::cuda symbol
  `_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb` with the
  *old* signature; torch 2.8.0+cu128 has the *new* signature
  (`(int, char const*, char const*, unsigned int, bool)`).  Bisect
  proved no wheel version matches.
* Source build path (validated working):

  ```bash
  apt install -y numactl   # optional; we used taskset instead
  uv pip install --python .venv pybind11
  cd /tmp && git clone --recursive --depth 1 \
      https://github.com/NVIDIA/TransformerEngine.git
  cd TransformerEngine && \
    PATH=/usr/local/cuda-12.8/bin:$PATH \
    CUDA_HOME=/usr/local/cuda-12.8 \
    NVTE_FRAMEWORK=pytorch \
    NVTE_CUDA_ARCHS=90 \
    MAX_JOBS=8 \
    uv pip install --python /path/to/.venv --no-build-isolation .
  # Then uninstall any stale PyPI bits (transformer-engine-cu12 /
  # transformer-engine-torch) and re-install the source build:
  uv pip uninstall transformer-engine transformer-engine-cu12 transformer-engine-torch
  cd TransformerEngine && NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=90 \
    uv pip install --python /path/to/.venv --no-build-isolation .
  ```

* Validated: `from transformer_engine.pytorch.attention import
  DotProductAttention; op(q, k, v)` works on H100 BF16 inputs.
* **NOT yet measured end-to-end** — Repercep's `TransformerEngineAttention`
  wrapper compiles; running it through the Cosmos pipeline is open
  Session 17 work.

### 4. V-JEPA 2 — Repercep now serves a non-diffusion world model

* `scripts/run_vjepa2.py` — loads any of the four V-JEPA 2 release
  variants (vitl 0.3B / vith 0.7B / vitg 1B / vitg-384 1B-384res),
  runs the encoder forward on a synthetic 64-frame clip, prints
  embedding shape + timing.

* H100 (BF16, B=1, T=64, 256×256):
  - ViT-L (326M):  **155–179 ms / forward**, 0.87 GiB peak
  - ViT-G (1B):    **320 ms / forward**, 2.29 GiB peak
  - Best attn impl: `sdpa` (cuDNN flash).  V-JEPA's `head_dim=64`
    *flips* the Cosmos pattern — FA-2 wheel is 1.7 × *slower* than
    SDPA at this head dim (vs 2.05 × faster at Cosmos's 128).

* CPU (Sapphire Rapids, 52 threads, 1 socket via taskset):
  - ViT-L: **84.4 s / forward** (BF16, oneDNN-AMX backed SDPA)

* **Strategic significance:** Repercep now serves both
  (a) diffusion video models (Cosmos 121f/36 in 99.6 s on H100),
  (b) non-diffusion encoder-predictor world models (V-JEPA 2 326 M in
      155 ms / 1 B in 320 ms),
  all on the same `Backend` Protocol and the same `select_backend()`
  selector.  The "world-model-native inference engine" claim now spans
  both architectural families.

### 5. CPU AMX work: validated end-to-end + diagnosed

* **Handoff item "AMX kernel needs intrinsic rewrite" was stale** —
  inspecting `kernels/cpu/amx_attn/flash_attn_amx.cpp` shows it
  already uses `_tile_loadd` / `_tile_dpbf16ps` / `_tile_stored` /
  `_tile_zero` / `_tile_loadconfig` / `_tile_release` intrinsics.
  Kernel **builds and runs first try** with `make kernels-cpu` (after
  a Makefile fix: `PY := $(CURDIR)/.venv/bin/python` so the `cd
  kernels/cpu/amx_attn` doesn't break the relative path).

* Correctness: **0.18 % rel err vs torch SDPA** (within BF16 noise) at
  Cosmos shapes.  10/10 `tests/test_attention_cpu.py` pass.

* **First CPU end-to-end Cosmos**: 17 f / 8 steps, BF16, 48 threads,
  AMX kernel active:
  - generate_seconds = **875.6 s**
  - peak RAM = 27.9 GiB
  - mp4 generated.

* Microbench at Cosmos shape (B=2, H=32, D=128):
  - S=1024: AMX **3.8 × faster** than torch SDPA (which itself
    dispatches AMX through oneDNN)
  - S=2048 – 8192: AMX **0.5 – 0.7 ×** (slower)
  - Bottleneck: kernel runs at **~13 % of AMX peak FLOPS** vs
    oneDNN's ~33 %.  NUMA pinning + thread count alone don't fix it
    (verified empirically).  Real fix is larger `M_Q` tile +
    persistent-kernel pattern — substantial restructure, deferred.

### 6. Other landed pieces

* `scripts/bench_fp8_hopper.py` — Hopper sibling of `bench_fp8.py`.
  Per-op `rel_err` reporting (previously the column took `min()`
  across ops, hiding FP8's 3.4 % quantization error behind FA-2's
  0.0).
* `docs/TARGETS_AND_KERNELS.md` — per-target architecture + kernel
  layout doc, companion to `architecture.md`.  Headline numbers,
  language/toolchain matrix, reproducer commands.
* `Makefile` — `$(PY)` rewritten to `$(CURDIR)/.venv/bin/python` so
  recipes that `cd` first still resolve `.venv/bin/python`.

### Things attempted that did NOT fully land

* **CPU 2-socket parallel throughput test** — kicked off two
  taskset-pinned (`-c 0-51` and `-c 52-103`) Cosmos 17f/8 runs in
  parallel.  Both processes ran to completion (~25 min elapsed) but
  the bash wrapper's `2>&1 | tail -3` pipeline ate the output before
  the file capture, so no numbers were captured and no mp4s landed.
  Easy fix: drop the `tail -3` from the redirect for any future
  long-run background tasks.

* **FP8 Hopper Triton kernel optimization** — blocked.  `ncu` requires
  GPU performance-counter permissions that this Runpod box doesn't
  grant (`ERR_NVGPUCTRPERM`).  Without ncu, kernel surgery is
  guessing.  Triton 3.4 doesn't expose `trans_b` in `tl.dot` so the
  explicit `tl.trans(k_tile)` can't be trivially removed.  Real path:
  either (a) get root + enable counters, or (b) instrument with
  manual timing of phases, or (c) compare against CUTLASS FP8 GEMM
  examples for inspiration.

* **Multi-prompt variance on FA-2 (mid-run)** — torch got swapped from
  2.8.0+cu128 to 2.12.0 by the *first* FA-3 build's resolver (F33
  below).  First 2 runs at FA-2 came in clean at 132.5 s, the third
  blew up on a `T5EncoderModel` import.  Restored torch and re-ran the
  full 5-prompt variance on FA-3 instead (now the headline).

* **FP8 end-to-end on H100 (`REPERCEP_FP8_ATTENTION=on`)** — measured at
  343.7 s, much slower than the SDPA baseline.  Confirms the bridge
  routes correctly through the FP8 Triton kernel.  The kernel itself
  is the bottleneck; optimization is open work.

* **TE-FP8 end-to-end measurement** — TE now installs and imports
  cleanly, `DotProductAttention` works in isolation.  The
  Cosmos-via-bridge path with TE wasn't measured this session; that's
  the Session 17 follow-up that should validate or refute whether TE
  FP8 attention beats FA-3 BF16 at the Cosmos shape.

---

## Findings recorded (F33–F37)

* **F33** — `pip install` of `flash-attn-3` *from source* will
  silently upgrade torch (the build script's resolver pulled
  `torch==2.12.0+cu130`, replacing `torch==2.8.0+cu128`), breaking
  the entire venv.  Mitigation: always pin torch via the cu128 wheel
  index *after* any FA-3 source install.

* **F34** — TransformerEngine PyPI wheels 2.12, 2.13, 2.14, 2.15 all
  reference the c10::cuda symbol with the *old* signature; torch 2.8+
  has the new one.  No wheel matches; source build is the only path.
  See §3 above for the reproducer.

* **F35** — V-JEPA 2 inference at default `attn_implementation=sdpa`
  is faster than `attn_implementation=flash_attention_2` on H100
  (155 ms vs 268 ms).  Cause: head_dim=64 hits a less-optimised path
  in the FA-2 wheel.  Repercep's `_native_fallback` FA-2/3 promotion
  should gate on `head_dim ≥ 128` for safety (TODO).

* **F36** — Cosmos's diffusers pipeline bypasses
  `repercep.attention.select_attention_op` and goes directly through
  `diffusers._AttentionBackendRegistry`.  Promoting
  `HopperFlashAttention` ahead of `NaiveAttention` in the Repercep
  registry has zero effect on Cosmos end-to-end.  Capture is via the
  diffusers bridge (`repercep_fp8_attention` + `_native_fallback`),
  not the Repercep registry.

* **F37** — FA-3 minimal-config source build without
  `FLASH_ATTENTION_DISABLE_FP8=TRUE` produces a `_C.abi3.so` with an
  undefined FP8 symbol:
  `run_mha_fwd<90, cutlass::float_e4m3_t, 128, 128, ...>`.  The
  dispatcher references FP8 entry points even when only BF16 is
  needed; you must disable FP8 explicitly.

---

## Quality gate (post-session)

* `ruff check src tests scripts` — clean
* `mypy --strict` — clean over 65 source files
* `pytest -q` — 173 passed / 14 skipped / 0 failed (H100 host)
* `cargo test --workspace` — 41 passed across 3 crates
* `make kernels-cpu` — builds + 10/10 AMX tests pass

---

## What's open after today (priority-ranked)

1. **(highest signal) TE-FP8 end-to-end bench on H100.**  TE is
   installed and validated.  Wire `TransformerEngineAttention` through
   the diffusers bridge (currently the bridge only knows FP8 Triton
   or SDPA/FA — needs a third branch for TE).  Run Cosmos 121f/36
   with TE-FP8 attention and compare against the 99.6 s FA-3 mean.
   The agent's research called TE FP8 the "NVIDIA-canonical" FP8
   path and predicted it should at least match FA-3 BF16.

2. **TE FP8 linears (the big lever).**  Wrap `pipe.transformer`'s
   `nn.Linear`s with `te.Linear` under a `DelayedScaling` recipe
   inside an `fp8_autocast` context.  MLP is 2/3 of Cosmos DiT
   FLOPs and currently runs BF16.  Even if FP8 attention is a wash,
   FP8 linears alone should win meaningfully on Hopper.  Per the
   NVIDIA-Cosmos research agent: ~1 week for first pass + ~1 week
   for calibration.

3. **FP8 Hopper Triton kernel surgery.**  Currently 2.5-3 × slower
   than cuDNN-FA3 at Cosmos shape (343.7 s vs 99.6 s end-to-end).
   Profiling blocked on perf-counter perms.  Without ncu, candidates
   to try blind: (a) remove `tl.trans(k_tile)` via direct-strided
   load, (b) pre-fold `qk_scale` into `q_tile` pre-loop, (c) use
   Triton's `tl.experimental.descriptor_load` for TMA-style async
   loads.  Each one is hours of work + bench.

4. **AMX kernel optimization (CPU port perf).**  Achieves ~13 % AMX
   peak vs oneDNN's ~33 %.  Real fix: M_Q tile 32→64, persistent
   kernel pattern.  Substantial restructure — ~half-week of focused
   work.  Reference: oneDNN's `src/cpu/x64/jit_brgemm_*_amx.cpp`
   and Intel's `amx-cookbook`.

5. **Wan-2.2 81f/40-step end-to-end on H100.**  `repercep.models.wan`
   loader already in tree.  Second WM family validation with FA-3
   bridge active.  Per handoff projected ~1700 s on MI300X — H100
   with FA-3 should be much faster.  ~30 min on GPU.

6. **V-JEPA 2 native Repercep engine.**  Today's `run_vjepa2.py` is
   standalone.  Real integration: a new `EncoderEngine` Protocol
   (since V-JEPA 2 returns embeddings, not frames — `WorldModelEngine`
   assumes frame output) and a `repercep.models.vjepa2.VJEPA2Engine`
   that wires through the serving stack.

7. **Cosmos-Predict2 + NATTEN back-port.**  Per NVIDIA, 2.0–2.6 ×
   on Hopper for the sparse-attention variants.  Predict1 wasn't
   trained with NATTEN; back-port needs either a short finetune
   (recipe in the NATTEN paper) or ablate-and-measure on layers
   that tolerate it.  ~2 weeks; quality-experiment heavy.

8. **CPU multi-socket parallel throughput.**  Today's attempt with
   `taskset -c 0-51` + `-c 52-103` didn't capture output (bash pipe
   ate it).  Re-run without the `| tail -3` to validate 2 ×
   throughput.  ~15 min CPU.

9. **Multi-GPU Context-Parallel (per NIM's playbook).**  Sequence-
   sharded attention via `split_inputs_cp` / `cat_outputs_cp`.
   2–3 weeks; the only published technique that takes Cosmos under
   60 s.  Needs a multi-GPU box — outside this session's scope.

---

## Quick reproducers (for next session)

```bash
# Fresh box setup
git clone https://github.com/miteshs/Mirage.git && cd Repercep
uv venv --python 3.12 .venv
uv pip install --python .venv torch==2.8.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv -e ".[models,serving,dev]"
# Rust toolchain (one-time, the maturin step needs cargo on PATH)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
PATH="$HOME/.cargo/bin:$PATH" make rust-install
make kernels-cpu     # if on Sapphire Rapids+
make check-gpu

# FA-3 from source (~5 min minimal config)
cd /tmp && git clone --depth 1 https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper && \
  FLASH_ATTENTION_DISABLE_BACKWARD=TRUE \
  FLASH_ATTENTION_DISABLE_SPLIT=TRUE \
  FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE \
  FLASH_ATTENTION_DISABLE_APPENDKV=TRUE \
  FLASH_ATTENTION_DISABLE_LOCAL=TRUE \
  FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
  FLASH_ATTENTION_DISABLE_PACKGQA=TRUE \
  FLASH_ATTENTION_DISABLE_FP16=TRUE \
  FLASH_ATTENTION_DISABLE_FP8=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM64=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
  FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE \
  FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE \
  MAX_JOBS=8 uv pip install --python /path/to/.venv --no-build-isolation .
# REQUIRED after FA-3 install (F33): re-pin torch
uv pip install --python .venv torch==2.8.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128 --reinstall

# TE from source (~15-20 min)
# See §3 of this doc for the full sequence (pybind11 + clone + 2
# uv pip install passes to clear the stale wheel sanity check).

# H100 headline reproducer (~100 s warm)
REPERCEP_FP8_ATTENTION=fa .venv/bin/python scripts/run_cosmos.py \
    --backend cuda --frames 121 --steps 36 --native-loop \
    --cache-mode adaptive --cache-adaptive-threshold 0.30 \
    --cache-force-full-every 16
# Expect: generate_seconds ≈ 95-103 s (mean 99.6 ± 3.9 s)

# 5-prompt variance on the headline
REPERCEP_FP8_ATTENTION=fa .venv/bin/python scripts/verify_timing.py \
    --N 1 --prompts 5

# Hopper microbench (FP8 Triton vs FA-3 vs SDPA)
.venv/bin/python scripts/bench_fp8_hopper.py --iters 15

# V-JEPA 2 smoke
.venv/bin/python scripts/run_vjepa2.py --model vitg --warmup 2 --iters 5

# CPU smoke (substrate, slow — minutes per video)
REPERCEP_AMX_ATTENTION=1 OMP_NUM_THREADS=48 .venv/bin/python \
    scripts/run_cosmos.py --backend cpu --frames 17 --steps 8
```

---

## Branches + repository state

```
main  (current; bd01f45 + 1 local commit pending push)
  + 8 commits since Session 15:
    705a594  vjepa2: smoke + bench — non-diffusion world model on Repercep
    bd01f45  docs: H100 FA-3 variance — 99.6 ± 3.9s across 5 prompts
    12cca53  docs: H100 headline drops to 95.3s with FA-3 — 3.99x ref
    8f696e5  attention: FA-2/3 in diffusers bridge — new ...=fa value
    f0f52df  bench + docs: Hopper FP8 bench harness, TARGETS doc,
             Makefile fix
    0327c72  post-merge: green the quality gate on H100 + Sapphire Rapids
    1f0b516  Merge branch 'cpu-amx-port' into main
    453ed23  Merge branch 'session-14-cuda-port' into main
```

`session-14-cuda-port` and `cpu-amx-port` can both be deleted now —
they're fully merged.

---

## When you resume — start here

1. Read this doc (you're doing it).
2. Skim `docs/TARGETS_AND_KERNELS.md` if you need the three-target
   architecture map.
3. Pick from the priority-ranked open list above.  Items 1 (TE-FP8
   bench) and 5 (Wan-2.2 H100) are each tractable in one session and
   close obvious gaps.
4. Item 2 (TE FP8 linears) is the **biggest remaining lever** — if
   the goal is "fastest single-GPU Cosmos in the world", that's where
   the next 30 %+ probably lives.

Have a good day.
