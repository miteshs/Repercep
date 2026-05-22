PY := .venv/bin/python
UV := ~/.local/bin/uv

.PHONY: help install lint format typecheck test check-gpu info

help:
	@echo "Mirage Runtime — make targets:"
	@echo "  install    install the package + dev/model/serving extras into .venv"
	@echo "  lint       ruff lint"
	@echo "  format     ruff format"
	@echo "  typecheck  mypy --strict"
	@echo "  test       pytest"
	@echo "  check-gpu  standalone MI300X/ROCm smoke test"
	@echo "  info       print detected backend + devices"

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
