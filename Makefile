# Makefile for Potato-ACHIS
# Using uv as package manager

.PHONY: help install install-dev sync lock lint format test clean train-baseline train-mdfan

# Default target
help:
	@echo "🥔 Potato-ACHIS - Available commands:"
	@echo ""
	@echo "  make install        Install dependencies"
	@echo "  make install-dev    Install with dev dependencies"
	@echo "  make sync           Sync environment with lockfile"
	@echo "  make lock           Update lockfile"
	@echo ""
	@echo "  make lint           Run linter (ruff)"
	@echo "  make format         Format code (ruff)"
	@echo "  make test           Run tests"
	@echo "  make typecheck      Run type checker"
	@echo ""
	@echo "  make train-baseline Train baseline model"
	@echo "  make train-mdfan    Train MDFAN model"
	@echo ""
	@echo "  make clean          Clean cache files"

# ============ Installation ============

install:
	uv sync

install-dev:
	uv sync --all-extras

sync:
	uv sync

lock:
	uv lock

# ============ Code Quality ============

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

typecheck:
	uv run mypy src/

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ --cov=src --cov-report=html

# ============ Training ============

train-baseline:
	uv run python src/train.py --model baseline --backbone mobilenet_v3_small

train-mdfan:
	uv run python src/train.py --model mdfan --backbone resnet50

# ============ Cleanup ============

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage
