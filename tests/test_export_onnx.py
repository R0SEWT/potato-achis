import pytest
import torch

from src.export_onnx import _infer_num_classes


def test_infer_num_classes_baseline_from_head_classifier_weight():
    state_dict = {"head.classifier.weight": torch.randn(3, 128)}
    assert _infer_num_classes(state_dict, model_type="baseline") == 3


def test_infer_num_classes_baseline_handles_module_prefix():
    state_dict = {"module.head.classifier.weight": torch.randn(7, 128)}
    assert _infer_num_classes(state_dict, model_type="baseline") == 7


def test_infer_num_classes_mdfan_from_combined_classifier_weight():
    state_dict = {"combined_classifier.classifier.weight": torch.randn(5, 128)}
    assert _infer_num_classes(state_dict, model_type="mdfan") == 5


def test_infer_num_classes_mdfan_falls_back_to_source_classifier_weight():
    state_dict = {"source_classifiers.0.classifier.weight": torch.randn(4, 128)}
    assert _infer_num_classes(state_dict, model_type="mdfan") == 4


def test_infer_num_classes_raises_when_not_found():
    with pytest.raises(ValueError, match="Could not infer num_classes"):
        _infer_num_classes({}, model_type="baseline")
