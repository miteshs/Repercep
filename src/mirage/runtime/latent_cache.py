"""Paged latent cache — Python wrapper over the Rust mirage_cache._native impl.

The pre-Stage-3 pure-Python implementation lived here. The Rust port
preserves semantics verbatim; see crates/mirage-cache/ and ADR-0005.
"""

from __future__ import annotations

from mirage_cache._native import (
    CacheStats,
    LatentCacheError,
    LatentPage,
    PagedLatentCache,
)

__all__ = ["CacheStats", "LatentCacheError", "LatentPage", "PagedLatentCache"]
