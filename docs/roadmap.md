# Roadmap

Last updated: 2026-04-09

## Completed

- [x] Baseline MobileNet/ResNet classification
- [x] MDFAN Stage 1: feature alignment with GRL + MMD
- [x] OOD detection (MSP, entropy, energy)
- [x] Andean field augmentations
- [x] uv package management
- [x] TensorBoard integration

## In Progress / Next

### 1. Unit tests

**Priority: high**

`tests/` is empty. Minimum coverage needed before adding features.

- [ ] `test_mmd_loss.py` — MMD loss with RBF kernel
- [ ] `test_gradient_reversal.py` — GRL forward/backward behavior
- [ ] `test_model_factory.py` — `create_model()` for baseline and MDFAN
- [ ] `test_transforms.py` — Andean augmentations pipeline

### 2. Data pipeline validation

**Priority: high**

Verify no data leak between train/val/test splits. Ref: commit `9fabe77` reported suspiciously perfect accuracy, partially addressed in PR #3.

- [ ] Audit `PotatoDataModule` split logic
- [ ] Add assertions for non-overlapping splits

### 3. Classifier Alignment (Stage 2)

**Priority: medium**

Module scaffolded at `src/models/mdfan/classifier_alignment.py`. Needs integration.

- [ ] Wire `ClassifierAlignment` into `MDFAN.forward_train()`
- [ ] Add `--lambda_align` flag to `src/train.py`
- [ ] Uncomment `classifier_alignment: 0.3` in `configs/model/mdfan.yaml`
- [ ] ADR: document alignment strategy choice (L1 vs KL vs MCD)

### 4. Grad-CAM in evaluation

**Priority: medium**

`visualize_gradcam()` exists in `src/utils/visualization.py`. Not connected to eval pipeline.

- [ ] Add `--gradcam` flag to `src/eval.py`
- [ ] Map default `target_layer` per backbone (MobileNetV3, ResNet50)

### 5. W&B integration

**Priority: low**

Dependency already in `pyproject.toml` (`tracking` extra).

- [ ] Add `--use_wandb` flag to `src/train.py`
- [ ] Log metrics, hyperparams, and artifacts alongside TensorBoard

### 6. ONNX export

**Priority: low**

- [ ] Export script using `torch.onnx.export()` for inference-only models
- [ ] Validate exported model with `onnxruntime`

## ADRs

Architecture decisions are documented in [`docs/adr/`](adr/) using MADR format. Created incrementally as decisions are made.
