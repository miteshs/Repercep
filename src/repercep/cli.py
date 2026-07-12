"""The ``repercep`` command-line entry point."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_USAGE = "usage: repercep [info]"


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns a process exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "info"

    if command in ("info", "-h", "--help", "help"):
        if command in ("-h", "--help", "help"):
            print(_USAGE)
            return 0
        return _info()

    print(f"repercep: unknown command {command!r}", file=sys.stderr)
    print(_USAGE, file=sys.stderr)
    return 2


def _info() -> int:
    """Print the detected backend and devices."""
    from repercep import __version__
    from repercep.backend.registry import select_backend

    print(f"Repercep Runtime v{__version__}")
    try:
        backend = select_backend()
    except RuntimeError as exc:
        print("  backend : NONE")
        print(f"  reason  : {exc}")
        return 1

    caps = backend.capabilities()
    print(f"  backend : {backend.name} ({backend.vendor.value})")
    for dev in backend.devices():
        print(
            f"  device  : [{dev.index}] {dev.name}  {dev.arch}  "
            f"{dev.total_memory_gib:.0f} GiB  {dev.multi_processor_count} CUs"
        )
    print(f"  dtype   : {backend.default_dtype().value}")
    print(f"  attn    : {', '.join(caps.attention_ops)}")
    print(f"  fp8     : {caps.supports_fp8}   flash-attn: {caps.supports_flash_attention}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
