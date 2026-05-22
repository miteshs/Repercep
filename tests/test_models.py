"""Tests for the Cosmos engine.

Construction, identity, and Protocol conformance are checked here. Actual
generation needs ~38 GB of downloaded weights and a GPU, and is exercised by
the run/benchmark scripts rather than the unit suite.
"""

from __future__ import annotations

from mirage.backend.rocm import ROCmBackend
from mirage.models.cosmos import DEFAULT_REPO, CosmosConfig, CosmosEngine
from mirage.runtime.engine import WorldModelEngine


def test_cosmos_engine_satisfies_engine_protocol() -> None:
    assert isinstance(CosmosEngine(backend=ROCmBackend()), WorldModelEngine)


def test_cosmos_engine_info_before_load() -> None:
    engine = CosmosEngine(backend=ROCmBackend())
    info = engine.info()
    assert info.model_name == "cosmos-predict1-7b-text2world"
    assert info.backend == "rocm"
    assert info.ready is False  # lazy: no weights touched yet


def test_cosmos_engine_is_not_loaded_on_construction() -> None:
    assert CosmosEngine(backend=ROCmBackend()).is_loaded is False


def test_default_repo_is_the_diffusers_format() -> None:
    assert DEFAULT_REPO == "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
    assert CosmosConfig().repo_id == DEFAULT_REPO
