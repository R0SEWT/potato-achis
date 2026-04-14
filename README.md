# 🥔 Potato-ACHIS

**Multi-source Domain Feature Adaptation Network for Andean Potato Disease Classification**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

This project implements a **Multi-source Domain Feature Adaptation Network (MDFAN)** for potato disease classification, specifically designed to address the domain shift between public datasets and real-world Andean field conditions.

### Key Features

- 🔄 **Multi-source Domain Adaptation**: Leverages multiple source domains (PlantVillage, commercial images) to improve generalization
- 🏔️ **Andean Field Augmentations**: Specialized transforms simulating highland lighting conditions
- 🎯 **Open-Set Recognition**: OOD detection for rejecting unknown disease classes
- ⚡ **Flexible Backbones**: MobileNetV3 (lightweight) and ResNet50 (accuracy) via `timm`
- 🛠️ **Modern Tooling**: `uv` for fast dependency management, `ruff` for linting

## 📁 Project Structure

```
potato-achis/
├── src/
│   ├── models/
│   │   ├── model.py              # Factory: BaselineModel, MDFAN
│   │   ├── backbones/            # MobileNet, ResNet (timm)
│   │   ├── mdfan/                # Domain adaptation components
│   │   ├── heads/                # Classification heads
│   │   └── components/           # GRL, bottleneck
│   ├── data/
│   │   ├── datasets/             # Dataset classes
│   │   ├── transforms/           # Augmentations (Andean)
│   │   └── datamodule.py
│   ├── losses/                   # MMD, adversarial losses
│   ├── utils/                    # Metrics, OOD, visualization
│   ├── train.py
│   └── eval.py
├── configs/                      # Hydra configs
├── data/                         # Data directory
├── notebooks/
├── tests/
├── pyproject.toml               # uv/hatch config
└── uv.lock                      # Lockfile
```

## 🚀 Quick Start

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### Installation

```bash
# Clone repository
git clone https://github.com/R0SEWT/potato-achis.git
cd potato-achis

# Create environment and install dependencies (uv handles everything)
uv sync

# With optional dependencies
uv sync --extra dev --extra viz --extra tracking --extra onnx

# Or install all extras
uv sync --all-extras
```

### Development

```bash
# Run training
uv run python src/train.py --model baseline --backbone mobilenet_v3_small

# Run tests
uv run pytest

# Lint and format
uv run ruff check src/
uv run ruff format src/

# Type checking
uv run mypy src/

# Add a new dependency
uv add pandas

# Add a dev dependency
uv add --dev hypothesis
```

### Training

**Baseline (MobileNet without domain adaptation):**
```bash
uv run python src/train.py \
    --model baseline \
    --backbone mobilenet_v3_small \
    --data_dir ./data/raw/plantvillage \
    --epochs 50 \
    --batch_size 32
```

**MDFAN (Multi-source Domain Adaptation):**
```bash
uv run python src/train.py \
    --model mdfan \
    --backbone resnet50 \
    --source_dirs ./data/raw/plantvillage ./data/raw/local_commercial \
    --target_dir ./data/raw/andean_field \
    --epochs 50 \
    --lambda_mmd 1.0 \
    --lambda_adv 0.5 \
    --lambda_align 0.0
```

**W&B tracking (requires optional `tracking` extra):**
```bash
uv sync --extra tracking
uv run python src/train.py \
    --model mdfan \
    --backbone resnet50 \
    --source_dirs ./data/raw/plantvillage ./data/raw/local_commercial \
    --target_dir ./data/raw/andean_field \
    --use_wandb
```

Set `WANDB_PROJECT` / `WANDB_ENTITY` env vars as needed.

### Evaluation

```bash
uv run python src/eval.py \
    --checkpoint ./outputs/best_model.pt \
    --test_dir ./data/raw/andean_field/test \
    --ood_dir ./data/raw/andean_field/ood_classes \
    --ood_method entropy \
    --visualize
```

**Grad-CAM (requires optional `viz` extra):**
```bash
uv sync --extra viz
uv run python src/eval.py \
    --checkpoint ./outputs/best_model.pt \
    --test_dir ./data/raw/andean_field/test \
    --gradcam
```

### Prediction

```bash
uv run potato-predict \
    --checkpoint ./outputs/exp/best_model.pt \
    --input ./some_image.jpg

uv run potato-predict \
    --checkpoint ./outputs/exp/best_model.pt \
    --input ./some_folder_of_images \
    --output ./outputs/predictions.jsonl
```

### ONNX export

```bash
uv sync --extra onnx
uv run potato-export-onnx \
    --checkpoint ./outputs/exp/best_model.pt \
    --model baseline \
    --backbone mobilenet_v3_small

# If your checkpoint was trained with a non-default class count, override it:
#   --num_classes <N>
```

## 🧪 Disease Classes

| Class | Description |
|-------|-------------|
| `early_blight` | *Alternaria solani* |
| `late_blight` | *Phytophthora infestans* |
| `healthy` | No visible disease |
| `bacterial_wilt` | *Ralstonia solanacearum* |
| `virus` | Various viral infections |

**OOD Classes (Andean-specific):**
- `frost_damage` - High-altitude cold injury
- `nutrient_deficiency` - Mineral deficiencies

## 📊 Model Architecture

### MDFAN Pipeline

```
Input Image
    │
    ▼
┌─────────────────────┐
│  Shared Backbone    │  (MobileNetV3 / ResNet50)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Feature Extractor  │  (Bottleneck: 2048 → 256)
└─────────────────────┘
    │
    ├──────────────────────────────┐
    ▼                              ▼
┌─────────────────────┐    ┌─────────────────────┐
│  Domain Discriminator│   │  Source Classifiers │
│  + GRL               │   │                     │
└─────────────────────┘    └─────────────────────┘
    │                              │
    ▼                              ▼
  Domain Loss               Classification Loss
         │                        │
         └───────┬────────────────┘
                 ▼
           Total Loss = L_cls + λ_adv·L_domain + λ_mmd·L_mmd
```

## 🛠️ Development Commands

```bash
# Environment info
uv python list              # List available Python versions
uv venv --python 3.11       # Create venv with specific Python

# Dependencies
uv add torch --index pytorch  # Add from specific index
uv remove pandas              # Remove dependency
uv lock                       # Update lockfile
uv tree                       # Show dependency tree

# Running
uv run pytest -v              # Run tests
uv run python -m src.train    # Run as module
uv run jupyter lab            # Start Jupyter (with notebooks extra)

# Build
uv build                      # Build wheel/sdist
```

## 📈 Expected Results

| Model | Source Acc | Target Acc | OOD AUROC |
|-------|------------|------------|-----------|
| Baseline (MobileNet) | ~95% | ~70% | ~75% |
| Baseline (ResNet50) | ~97% | ~75% | ~78% |
| **MDFAN (ResNet50)** | ~95% | **~85%** | **~85%** |

## 🗺️ Roadmap

- [x] Baseline MobileNet/ResNet
- [x] MDFAN with GRL + MMD
- [x] OOD detection (MSP, entropy, energy)
- [x] Andean field augmentations
- [x] uv package management
- [x] Classifier alignment (Stage 2)
- [x] Grad-CAM visualization
- [x] W&B integration
- [x] ONNX export

Detailed roadmap: see [docs/roadmap.md](docs/roadmap.md)

## 📝 Citation

```bibtex
@software{potato_achis_2025,
  author = {R0SEWT, Nakato156},
  title = {Potato-ACHIS: Multi-source Domain Adaptation for Andean Potato Disease Classification},
  year = {2025},
  url = {https://github.com/R0SEWT/potato-achis}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE)
