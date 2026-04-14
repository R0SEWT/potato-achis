import pytest
import torch

from src.eval import _infer_num_classes, _infer_num_sources


def test_infer_num_classes_baseline_from_checkpoint_state_dict():
    state_dict = {"head.classifier.weight": torch.randn(3, 128)}
    assert _infer_num_classes(state_dict, model_type="baseline") == 3


def test_infer_num_classes_mdfan_from_checkpoint_state_dict():
    state_dict = {"combined_classifier.classifier.weight": torch.randn(5, 128)}
    assert _infer_num_classes(state_dict, model_type="mdfan") == 5


def test_infer_num_sources_from_checkpoint_state_dict():
    state_dict = {
        "source_classifiers.0.classifier.weight": torch.randn(5, 128),
        "source_classifiers.1.classifier.weight": torch.randn(5, 128),
    }
    assert _infer_num_sources(state_dict) == 2


def test_infer_num_sources_handles_module_prefix():
    state_dict = {"module.source_classifiers.0.classifier.weight": torch.randn(5, 128)}
    assert _infer_num_sources(state_dict) == 1


def test_infer_num_sources_raises_when_not_found():
    with pytest.raises(ValueError, match="Could not infer num_sources"):
        _infer_num_sources({})
