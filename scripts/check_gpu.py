#!/usr/bin/env python3
"""Standalone MI300X / ROCm smoke test.

Run with the project venv:  .venv/bin/python scripts/check_gpu.py

Intentionally has no Mirage imports so it works before the package is
installed — it is the first thing to run on a fresh box.
"""

from __future__ import annotations

import sys
import time


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: torch is not installed in this environment.", file=sys.stderr)
        return 1

    print(f"torch   : {torch.__version__}")
    print(f"hip     : {torch.version.hip}")

    if not torch.version.hip:
        print("FAIL: this torch build is not a ROCm build.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print(
            "FAIL: no GPU visible. Check that /dev/kfd and /dev/dri/renderD* are "
            "readable by this user (group 'render').",
            file=sys.stderr,
        )
        return 1

    count = torch.cuda.device_count()
    print(f"devices : {count}")
    for i in range(count):
        p = torch.cuda.get_device_properties(i)
        print(
            f"  [{i}] {p.name}  {p.gcnArchName}  "
            f"{p.total_memory / 1024**3:.0f} GiB  {p.multi_processor_count} CUs"
        )

    dev = torch.device("cuda", 0)
    n = 4096
    for dtype in (torch.float16, torch.bfloat16):
        a = torch.randn(n, n, device=dev, dtype=dtype)
        b = torch.randn(n, n, device=dev, dtype=dtype)
        for _ in range(3):  # warmup
            _ = a @ b
        torch.cuda.synchronize()
        iters = 50
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = a @ b
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
        tflops = 2 * n**3 / (ms * 1e-3) / 1e12
        print(f"  matmul {n}x{n} {dtype!s:>16}: {ms:7.3f} ms  ({tflops:6.1f} TFLOP/s)")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
