"""Backend discovery and selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mirage.backend.rocm import ROCmBackend

if TYPE_CHECKING:
    from mirage.backend.protocol import Backend

# Every backend Mirage knows how to construct.  An NVIDIA backend, when it
# exists, is appended here and nothing else changes (ADR-0001).
_ALL_BACKENDS: tuple[Backend, ...] = (ROCmBackend(),)


def available_backends() -> tuple[Backend, ...]:
    """Backends whose hardware is actually present on this host."""
    return tuple(b for b in _ALL_BACKENDS if b.is_available())


def select_backend(prefer: str | None = None) -> Backend:
    """Return a usable backend.

    Args:
        prefer: pin a specific backend by name (e.g. ``"rocm"``).  If ``None``,
            the first available backend is returned.

    Raises:
        ValueError: ``prefer`` names a backend Mirage does not know.
        RuntimeError: the requested (or any) backend is not available.
    """
    if prefer is not None:
        for backend in _ALL_BACKENDS:
            if backend.name == prefer:
                if not backend.is_available():
                    raise RuntimeError(f"backend {prefer!r} is not available on this host")
                return backend
        known = ", ".join(b.name for b in _ALL_BACKENDS)
        raise ValueError(f"unknown backend {prefer!r}; known backends: {known}")

    for backend in _ALL_BACKENDS:
        if backend.is_available():
            return backend
    raise RuntimeError(
        "no Mirage backend is available — is a GPU visible and torch installed? "
        "Run `python scripts/check_gpu.py` to diagnose."
    )
