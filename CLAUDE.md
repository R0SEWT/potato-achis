# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Potato-ACHIS implements a **Multi-source Domain Feature Adaptation Network (MDFAN)** for classifying potato diseases in Andean field conditions. It addresses domain shift between public datasets (PlantVillage) and real-world highland imagery using domain adaptation techniques (GRL, MMD loss, adversarial training).

Two model types: **baseline** (backbone + classifier head, no adaptation) and **MDFAN** (adds domain discriminator, feature alignment, MMD loss).

## Common Commands

```bash
# Install dependencies
uv sync                     # production deps
uv sync --all-extras        # all optional deps (dev, viz, tracking, notebooks)

# Training
uv run python src/train.py --model baseline --backbone mobilenet_v3_small --data_dir ./data/raw/plantvillage
uv run python src/train.py --model mdfan --backbone resnet50 --source_dirs ./data/raw/source1 ./data/raw/source2 --target_dir ./data/raw/target

# Evaluation
uv run python src/eval.py --checkpoint ./outputs/best_model.pt --test_dir ./data/raw/andean_field/test

# Tests
uv run pytest               # all tests
uv run pytest tests/ -v     # verbose
uv run pytest -m "not slow" # skip slow tests
uv run pytest -m "not gpu"  # skip GPU tests

# Linting & formatting
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/

# Makefile shortcuts
make test
make lint
make format
make train-baseline
make train-mdfan
```

## Architecture

- **Entry points**: `src/train.py` (argparse CLI) and `src/eval.py`. Also registered as `potato-train` / `potato-eval` console scripts.
- **Model factory**: `src/models/model.py` — `create_model()` dispatches to `BaselineModel` or `MDFAN` via `ModelFactory`.
- **Backbones**: `src/models/backbones/backbone_factory.py` wraps `timm` to create MobileNetV3 or ResNet50 feature extractors.
- **MDFAN components** (`src/models/mdfan/`): `FeatureExtractor` (bottleneck), `DomainDiscriminator` (with GRL), `ClassifierAlignment`, assembled by `MDAFNModel`.
- **Data pipeline**: `src/data/datamodule.py` (`PotatoDataModule`) handles single-source and multi-source setups. `MultiSourceIterator` synchronizes loaders during MDFAN training.
- **Losses**: `src/losses/mmd_loss.py` (MMD with RBF kernel), `src/losses/domain_adversarial_loss.py` (classification + multi-source domain loss).
- **Andean augmentations**: `src/data/transforms/andean_transforms.py` — specialized transforms simulating highland lighting/conditions.
- **Config**: Hydra YAML configs in `configs/` with defaults in `configs/config.yaml`. Model configs: `configs/model/{baseline,mdfan}.yaml`, data config: `configs/data/potato.yaml`.

## Key Conventions

- Package manager is **uv** — always prefix commands with `uv run`.
- Linter/formatter is **ruff** (replaces black, isort, flake8). Config in `pyproject.toml`.
- Line length: 88, target Python 3.10+, double quotes, space indentation.
- Pre-commit hooks: ruff lint+format, mypy, trailing whitespace, large file check (max 1000KB). Install with `uv run pre-commit install`.
- PyTorch is sourced from the CUDA 12.1 index (configured in `pyproject.toml` under `[tool.uv.sources]`).
- 5 disease classes (early_blight, late_blight, healthy, bacterial_wilt, virus) + 2 OOD classes (frost_damage, nutrient_deficiency).
- Outputs (model checkpoints, training history, TensorBoard logs) go to `outputs/<exp_name>/`.
