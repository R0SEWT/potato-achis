# 🥔 Potato-ACHIS

**Multi-source Domain Feature Adaptation Network for Andean Potato Disease Classification**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

This project implements a **Multi-source Domain Feature Adaptation Network (MDFAN)** for potato disease classification, specifically designed to address the domain shift between public datasets and real-world Andean field conditions.

### Key Features

- **Multi-source Domain Adaptation**: Leverages multiple source domains (PlantVillage, commercial images) to improve generalization to Andean field conditions
- **Flexible Backbones**: Supports MobileNetV3 (lightweight) and ResNet50 (accuracy) via `timm`
- **Two-stage Alignment**: 
  - Stage 1: Feature-level alignment using MMD loss + adversarial training (GRL)
  - Stage 2: Classifier-level alignment (scaffolded for future work)
- **Open-Set Recognition**: OOD detection for rejecting unknown disease classes (frost damage, nutrient deficiency)
- **Andean Field Augmentations**: Specialized transforms simulating highland lighting conditions

## 📁 Project Structure

```
potato-achis/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config
│   ├── model/
│   │   ├── baseline.yaml      # MobileNet/ResNet baseline
│   │   └── mdfan.yaml         # MDFAN configuration
│   └── data/
│       └── potato.yaml        # Dataset configuration
│
├── src/                       # Source code
│   ├── models/
│   │   ├── model.py          # Model factory (MobileNet, ResNet, MDFAN)
│   │   ├── backbones/        # Feature extractors
│   │   ├── mdfan/            # MDFAN components
│   │   │   ├── mdfan_model.py
│   │   │   ├── domain_discriminator.py
│   │   │   └── feature_extractor.py
│   │   ├── heads/            # Classification heads
│   │   └── components/       # GRL, bottleneck layers
│   │
│   ├── data/
│   │   ├── datasets/         # Dataset classes
│   │   ├── transforms/       # Augmentations (incl. Andean)
│   │   └── datamodule.py     # Data management
│   │
│   ├── losses/               # Loss functions
│   │   ├── mmd_loss.py       # Maximum Mean Discrepancy
│   │   └── domain_adversarial_loss.py
│   │
│   ├── utils/
│   │   ├── metrics.py        # Evaluation metrics
│   │   ├── ood_detection.py  # Open-set recognition
│   │   └── visualization.py  # t-SNE, Grad-CAM
│   │
│   ├── train.py              # Training entry point
│   └── eval.py               # Evaluation entry point
│
├── data/                     # Data directory (gitignored)
│   └── raw/
│       ├── plantvillage/     # Source domain 1
│       ├── local_commercial/ # Source domain 2
│       └── andean_field/     # Target domain
│
├── notebooks/                # Jupyter notebooks
├── scripts/                  # Shell scripts
├── outputs/                  # Model outputs
└── logs/                     # Training logs
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/R0SEWT/potato-achis.git
cd potato-achis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Training

**Baseline (MobileNet without domain adaptation):**
```bash
python src/train.py \
    --model baseline \
    --backbone mobilenet_v3_small \
    --data_dir ./data/raw/plantvillage \
    --epochs 50 \
    --batch_size 32
```

**MDFAN (Multi-source Domain Adaptation):**
```bash
python src/train.py \
    --model mdfan \
    --backbone resnet50 \
    --source_dirs ./data/raw/plantvillage ./data/raw/local_commercial \
    --target_dir ./data/raw/andean_field \
    --epochs 50 \
    --lambda_mmd 1.0 \
    --lambda_adv 0.5
```

### Evaluation

```bash
python src/eval.py \
    --checkpoint ./outputs/best_model.pt \
    --test_dir ./data/raw/andean_field/test \
    --ood_dir ./data/raw/andean_field/ood_classes \
    --ood_method entropy \
    --visualize
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
│  (ImageNet pretrained)
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
│  (per source-target) │   │  (per source domain)│
│  + GRL               │   │                     │
└─────────────────────┘    └─────────────────────┘
    │                              │
    ▼                              ▼
  Domain Loss               Classification Loss
  (Adversarial)                 (CE)
         │                        │
         └───────┬────────────────┘
                 ▼
           Total Loss = L_cls + λ_adv * L_domain + λ_mmd * L_mmd
```

## 📈 Expected Results

| Model | Source Accuracy | Target Accuracy | OOD AUROC |
|-------|-----------------|-----------------|-----------|
| Baseline (MobileNet) | ~95% | ~70% | ~75% |
| Baseline (ResNet50) | ~97% | ~75% | ~78% |
| MDFAN (ResNet50) | ~95% | ~85% | ~85% |

## 🛠️ Development Roadmap

- [x] Baseline MobileNet/ResNet
- [x] MDFAN with GRL + MMD
- [x] OOD detection (MSP, entropy, energy)
- [x] Andean field augmentations
- [ ] Classifier alignment (Stage 2)
- [ ] Grad-CAM visualization
- [ ] W&B integration
- [ ] ONNX export for deployment

## 📝 Citation

If you use this code, please cite:

```bibtex
@software{potato_achis_2026,
  author = {R0SEWT},
  title = {Potato-ACHIS: Multi-source Domain Adaptation for Andean Potato Disease Classification},
  year = {2026},
  url = {https://github.com/R0SEWT/potato-achis}
}
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details