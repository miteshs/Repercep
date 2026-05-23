"""Per-request router — Python wrapper over the Rust mirage_router._native impl.

Greenfield as of ADR-0005 / Stage 3. Owns request state and frame ordering;
talks to a scheduler-like object via duck-typed submit()/cancel().
"""
from __future__ import annotations

from mirage_router._native import FrameStream, Router, RouterError

__all__ = ["FrameStream", "Router", "RouterError"]
