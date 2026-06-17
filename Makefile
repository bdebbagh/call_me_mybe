# ─────────────────────────────────────────────
#  call_me_mybe — Makefile
# ─────────────────────────────────────────────

PYTHON   = python3
PIP      = pip
SRC      = src
MAIN     = $(SRC)/main.py
VENV     = .venv

.PHONY: install run debug clean lint lint-strict

install:
	uv sync
# ── run ───────────────────────────────────────
run:
	uv run python -m src \
		--functions_definition data/input/functions_definition.json \
		--input               data/input/function_calling_tests.json \
		--output              data/output/function_calls.json

# ── debug ─────────────────────────────────────
debug:
	uv run python -u -m src \
		--functions_definition data/input/functions_definition.json \
		--input               data/input/function_calling_tests.json \
		--output              data/output/function_calls.json

# ── lint ──────────────────────────────────────
lint:
	uv run flake8 . && \
	uv run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs
lint-strict:
	flake8 . --max-line-length=79 --statistics && \
	mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs \
		--disallow-any-generics \
		--disallow-incomplete-defs \
		--no-implicit-optional \
		--strict-equality


clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf data/output/*
	@echo "[clean] Done."