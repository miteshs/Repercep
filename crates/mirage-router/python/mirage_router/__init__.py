"""Native extension package for the Mirage per-request router.

The Python-facing API lives in ``mirage.runtime.router``; this package only
exposes the compiled `_native` submodule produced by maturin from the
``mirage-router`` crate. See crates/README.md and ADR-0005.
"""
