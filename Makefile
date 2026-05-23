PY := .venv/bin/python
UV := ~/.local/bin/uv
CARGO := cargo

.PHONY: help install lint format typecheck test check-gpu info \
        rust-build rust-check rust-fmt rust-fmt-check rust-clippy rust-test \
        rust-install lint-all check-all

help:
	@echo "Mirage Runtime — make targets:"
	@echo ""
	@echo "  Python:"
	@echo "    install         install the package + dev/model/serving extras into .venv"
	@echo "    lint            ruff lint"
	@echo "    format          ruff format"
	@echo "    typecheck       mypy --strict"
	@echo "    test            pytest"
	@echo "    check-gpu       standalone MI300X/ROCm smoke test"
	@echo "    info            print detected backend + devices"
	@echo ""
	@echo "  Rust workspace (no members yet — see Cargo.toml and ADR-0004):"
	@echo "    rust-build      cargo build --workspace"
	@echo "    rust-check      cargo check --workspace"
	@echo "    rust-fmt        cargo fmt --all"
	@echo "    rust-fmt-check  cargo fmt --all -- --check"
	@echo "    rust-clippy     cargo clippy --workspace --all-targets -- -D warnings"
	@echo "    rust-test       cargo test --workspace"
	@echo "    rust-install    maturin develop --release for every crate (into .venv)"
	@echo ""
	@echo "  Combined:"
	@echo "    lint-all        ruff + rust-fmt-check + rust-clippy"
	@echo "    check-all       lint-all + typecheck + test + rust-test"

install:
	$(UV) pip install --python .venv -e ".[models,serving,dev]"

lint:
	$(PY) -m ruff check src tests scripts

format:
	$(PY) -m ruff format src tests scripts

typecheck:
	$(PY) -m mypy

test:
	$(PY) -m pytest -q

check-gpu:
	$(PY) scripts/check_gpu.py

info:
	$(PY) -m mirage.cli info

# Rust targets. Operate on the virtual workspace at the repo root. No-op cleanly
# while crates/ is empty; ready for the first crate when the fork is resolved
# (see docs/adr/0004-polyglot-build-tooling.md).
#
# Guard: cargo build/check/fmt/clippy/test all error on a zero-member workspace,
# so each target skips with a friendly note until any crates/*/Cargo.toml exists.
HAVE_CRATES := $(shell find crates -mindepth 2 -maxdepth 2 -name Cargo.toml -print -quit 2>/dev/null)

define rust-skip-msg
	@echo "$(1): workspace has no members yet (see docs/adr/0004) — skipping"
endef

rust-build:
ifeq ($(HAVE_CRATES),)
	$(call rust-skip-msg,rust-build)
else
	$(CARGO) build --workspace
endif

rust-check:
ifeq ($(HAVE_CRATES),)
	$(call rust-skip-msg,rust-check)
else
	$(CARGO) check --workspace
endif

rust-fmt:
ifeq ($(HAVE_CRATES),)
	$(call rust-skip-msg,rust-fmt)
else
	$(CARGO) fmt --all
endif

rust-fmt-check:
ifeq ($(HAVE_CRATES),)
	$(call rust-skip-msg,rust-fmt-check)
else
	$(CARGO) fmt --all -- --check
endif

rust-clippy:
ifeq ($(HAVE_CRATES),)
	$(call rust-skip-msg,rust-clippy)
else
	$(CARGO) clippy --workspace --all-targets -- -D warnings
endif

rust-test:
ifeq ($(HAVE_CRATES),)
	$(call rust-skip-msg,rust-test)
else
	$(CARGO) test --workspace
endif

# Build each crate's PyO3 extension module and install it into .venv. Iterates
# explicitly because `maturin develop` is per-package, not workspace-wide.
rust-install:
ifeq ($(HAVE_CRATES),)
	$(call rust-skip-msg,rust-install)
else
	@for crate in $$(find crates -mindepth 2 -maxdepth 2 -name Cargo.toml -printf '%h\n'); do \
	    echo "==> maturin develop --release in $$crate"; \
	    $(PY) -m maturin develop --release --manifest-path $$crate/Cargo.toml || exit 1; \
	done
endif

lint-all: lint rust-fmt-check rust-clippy

check-all: lint-all typecheck test rust-test
