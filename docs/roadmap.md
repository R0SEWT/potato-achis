# Roadmap

Last updated: 2026-04-14

## Completed

- [x] Baseline MobileNet/ResNet classification
- [x] MDFAN Stage 1: feature alignment with GRL + MMD
- [x] OOD detection (MSP, entropy, energy)
- [x] Andean field augmentations
- [x] uv package management
- [x] TensorBoard integration
- [x] Unit tests (MMD, GRL, model factory, transforms, data pipeline)
- [x] Data pipeline validation (no train/val leakage in single-source split)

## In Progress / Next

### 1. Classifier Alignment (Stage 2)

**Priority: medium**

Module implemented at `src/models/mdfan/classifier_alignment.py` and integrated into training.
- [x] Add `--lambda_align` flag to `src/train.py`
- [x] Add unit tests for `ClassifierAlignment`
- [x] Wire `ClassifierAlignment` into `MDFAN.forward_train()`
- [x] Uncomment `classifier_alignment: 0.3` in `configs/model/mdfan.yaml`
- [x] ADR: document alignment strategy choice (L1 vs KL vs MCD)

### 2. Grad-CAM in evaluation

**Priority: medium**

`visualize_gradcam()` exists in `src/utils/visualization.py`. Not connected to eval pipeline.

- [x] Add `--gradcam` flag to `src/eval.py`
- [x] Map default `target_layer` per backbone (MobileNetV3, ResNet50)

### 3. W&B integration

**Priority: low**

Dependency already in `pyproject.toml` (`tracking` extra).

- [x] Add `--use_wandb` flag to `src/train.py`
- [x] Log metrics, hyperparams, and artifacts alongside TensorBoard

### 4. ONNX export

**Priority: low**

- [ ] Export script using `torch.onnx.export()` for inference-only models
- [ ] Validate exported model with `onnxruntime`

## ADRs

Architecture decisions are documented in [`docs/adr/`](adr/) using MADR format. Created incrementally as decisions are made.
