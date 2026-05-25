# Session 19 H100 handoff

Date: 2026-05-25

Goal for the next pod: validate the Wan attention seam on a real H100 stack,
then decide whether to move into serving-v2 hardening or Wan-native loop work.

## What landed

- `AttentionOp.available` is now part of the protocol; `NaiveAttention`
  advertises `available=True`.
- `make kernels-cpu` now skips cleanly when AMX ISA flags are absent instead
  of falling through to a missing `.venv/bin/python` build command.
- Wan now has a diffusers attention installer:
  `mirage.attention.wan_processor.maybe_install_mirage_wan_attention`.
  When `MIRAGE_FP8_ATTENTION` is set and diffusers exposes
  `WanAttnProcessor`, `WanEngine.load()` installs a Wan processor pointed at
  Mirage's registered `mirage_fp8` backend.
- Runtime control-plane modules keep native PyO3 as the production path, but
  fall back to Python implementations when stale/missing native wheels are
  present. This keeps local tests useful before `make rust-install`.
- Serving tests now skip cleanly if `fastapi.testclient` is absent.

## Verification on the Ada dev box

Passed:

```bash
PYTHONPATH=$(pwd)/src python3 -m ruff check src tests scripts
PYTHONPATH=$(pwd)/src python3 -m pytest -q
PYTHONPATH=$(pwd)/src python3 -m pytest tests/test_fp8_attention_ada.py tests/test_quantize_gpu.py tests/test_eval_quality_parity_gpu.py -q
PYTHONPATH=$(pwd)/src python3 -m mypy src/mirage/attention/protocol.py src/mirage/attention/naive.py src/mirage/attention/wan_processor.py src/mirage/models/wan.py src/mirage/runtime/latent_cache.py src/mirage/runtime/router.py src/mirage/runtime/scheduler.py tests/test_attention.py tests/test_wan.py tests/test_runtime.py tests/test_router.py tests/test_scheduler.py
make kernels-cpu
```

Full pytest result: `210 passed, 24 skipped`.

Known limitation: full-repo `mypy` still has unrelated pre-existing failures in
quantize dynamic-class typing, optional serving/config imports, Cosmos stale
`type: ignore` comments, and eval-quality annotations/stubs.

## H100 next steps

1. Build the real env:

   ```bash
   uv venv .venv --python python3
   uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu128
   uv pip install --python .venv -e ".[models,serving,nvidia,dev]"
   make rust-install
   ```

   Install `flash-attn` separately only if the pod has the matching CUDA build
   toolchain available; otherwise start with the SDPA fallback path.

2. Confirm native control-plane imports beat the Python fallbacks:

   ```bash
   PYTHONPATH=$(pwd)/src .venv/bin/python - <<'PY'
   from mirage.runtime.latent_cache import PagedLatentCache
   from mirage.runtime.router import Router
   from mirage.runtime.scheduler import Scheduler
   print(PagedLatentCache.__module__)
   print(Router.__module__)
   print(Scheduler.__module__)
   PY
   ```

   Expected modules: `mirage_cache._native`, `mirage_router._native`,
   `mirage_scheduler._native`.

3. Run the Wan F40 attention proof before larger model work:

   ```bash
   MIRAGE_FP8_ATTENTION=fa PYTHONPATH=$(pwd)/src .venv/bin/python scripts/trace_wan_attention.py
   ```

   First target: prove Wan calls route through the diffusers dispatch seam or
   establish that the installed diffusers version is on the old direct-SDPA
   processor path.

4. Smoke the small Wan variant on H100:

   ```bash
   MIRAGE_FP8_ATTENTION=fa PYTHONPATH=$(pwd)/src .venv/bin/python scripts/run_wan.py \
     --repo-id Wan-AI/Wan2.2-TI2V-5B-Diffusers \
     --frames 17 --steps 8 --height 480 --width 832 --seed 0
   ```

5. Only after the dispatch proof is clear, choose the next workstream:

   - serving-v2 hardening: cancellation during long generation, request
     lifecycle observability, native-only production packaging;
   - or Wan-native loop/adaptive cache: bigger perf upside, but should be
     measured against the confirmed diffusers-Wan baseline.
