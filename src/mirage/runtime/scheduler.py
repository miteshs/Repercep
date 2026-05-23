"""Request scheduler — Python wrapper over the Rust ``mirage_scheduler._native`` impl.

Greenfield as of ADR-0005 / Stage 3. The Rust crate owns priority, FIFO
within priority, and cancellation; this module is a thin re-export.

The async surface is deliberately ``next_blocking(timeout_ms)`` in v0 — a
true Python coroutine via ``pyo3-async-runtimes`` is a follow-up once the
router actually needs to multiplex with FastAPI's event loop. See the
top-of-file comment in ``crates/mirage-scheduler/src/lib.rs`` for the
rationale.
"""

from __future__ import annotations

from mirage_scheduler._native import Scheduler, SchedulerError

__all__ = ["Scheduler", "SchedulerError"]
