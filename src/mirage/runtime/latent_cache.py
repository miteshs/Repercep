"""Paged latent cache.

vLLM's PagedAttention pages a *KV* cache for autoregressive token decode. World
models need a different object: a cache of *latent video tiles* reused across
diffusion steps and across temporal frames. A page here is a fixed-size latent
tile; eviction is frame-aware — a tile far behind the current temporal window
is a better victim than a recently-touched one, regardless of raw LRU age.

This module is the typed skeleton. The page table, allocation accounting, and
the eviction-policy seam are real and tested; tensor-backed page storage is
wired in with the inference path (Task #7), where a page's byte size depends
on the Cosmos tokenizer's latent shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

PageId = int
RequestId = str


class LatentCacheError(RuntimeError):
    """Raised when the cache cannot satisfy an allocation."""


@dataclass(slots=True)
class LatentPage:
    """One fixed-size latent tile's worth of cache bookkeeping."""

    page_id: PageId
    request_id: RequestId
    frame_index: int
    last_used_step: int
    in_use: bool = True


@dataclass(slots=True)
class CacheStats:
    """A snapshot of cache occupancy."""

    capacity: int
    allocated: int
    free: int
    pinned: int


@runtime_checkable
class EvictionPolicy(Protocol):
    """Chooses which evictable page to reclaim when the cache is full."""

    def victim(self, pages: list[LatentPage], current_frame: int) -> PageId:
        """Return the page id to evict, or raise ``LatentCacheError``."""
        ...


class FrameAwareEviction:
    """Evict the page whose frame is furthest behind the temporal window.

    Ties are broken by least-recently-used diffusion step. This is the
    WM-specific reason the cache is not a plain LRU map: frames the rollout has
    moved past will not be read again, so they are always the right victim even
    if they were touched more recently than an in-window frame.
    """

    def victim(self, pages: list[LatentPage], current_frame: int) -> PageId:
        evictable = [p for p in pages if not p.in_use]
        if not evictable:
            raise LatentCacheError("no evictable pages: cache is fully pinned")
        # Most negative (frame_index - current_frame) == furthest behind;
        # tie-break on the oldest diffusion step.
        victim = min(
            evictable,
            key=lambda p: (p.frame_index - current_frame, p.last_used_step),
        )
        return victim.page_id


class PagedLatentCache:
    """A fixed pool of latent pages with a frame-aware eviction policy."""

    def __init__(self, num_pages: int, policy: EvictionPolicy | None = None) -> None:
        if num_pages < 1:
            raise ValueError("num_pages must be >= 1")
        self._capacity = num_pages
        self._policy: EvictionPolicy = policy or FrameAwareEviction()
        self._pages: dict[PageId, LatentPage] = {}
        self._next_id: PageId = 0

    def allocate(
        self, request_id: RequestId, frame_index: int, step: int, current_frame: int
    ) -> PageId:
        """Reserve a page, evicting a victim first if the pool is full.

        Raises:
            LatentCacheError: the pool is full and every page is pinned.
        """
        if len(self._pages) >= self._capacity:
            victim = self._policy.victim(list(self._pages.values()), current_frame)
            del self._pages[victim]
        page_id = self._next_id
        self._next_id += 1
        self._pages[page_id] = LatentPage(
            page_id=page_id,
            request_id=request_id,
            frame_index=frame_index,
            last_used_step=step,
        )
        return page_id

    def touch(self, page_id: PageId, step: int) -> None:
        """Record that ``page_id`` was read at diffusion ``step``."""
        page = self._pages.get(page_id)
        if page is not None:
            page.last_used_step = step

    def unpin(self, page_id: PageId) -> None:
        """Mark a page evictable without freeing it."""
        page = self._pages.get(page_id)
        if page is not None:
            page.in_use = False

    def release_request(self, request_id: RequestId) -> int:
        """Free every page held by a request. Returns the number freed."""
        victims = [pid for pid, p in self._pages.items() if p.request_id == request_id]
        for pid in victims:
            del self._pages[pid]
        return len(victims)

    def stats(self) -> CacheStats:
        pinned = sum(1 for p in self._pages.values() if p.in_use)
        return CacheStats(
            capacity=self._capacity,
            allocated=len(self._pages),
            free=self._capacity - len(self._pages),
            pinned=pinned,
        )
