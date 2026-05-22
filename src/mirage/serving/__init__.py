"""HTTP and gRPC serving for the Mirage Runtime."""

from __future__ import annotations

from mirage.serving.app import create_app

__all__ = ["create_app"]
