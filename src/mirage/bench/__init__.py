"""Benchmark harness for the Mirage Runtime.

Measurement before optimization: this package times any ``WorldModelEngine``
and reports latency, throughput, and peak HBM, and profiles the Cosmos pipeline
stage by stage — so every later speedup is measured against a fixed baseline.
"""

from __future__ import annotations

from mirage.bench.harness import BenchmarkResult, benchmark_engine, speedup
from mirage.bench.profile import CosmosProfile, profile_cosmos

__all__ = [
    "BenchmarkResult",
    "CosmosProfile",
    "benchmark_engine",
    "profile_cosmos",
    "speedup",
]
