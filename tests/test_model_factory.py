"""Tests for model factory and model creation."""

import pytest
import torch

from src.models import create_model
from src.models.model import BaselineModel, ModelFactory


class TestBaselineModel:
    """Tests for BaselineModel."""

    def test_forward_shape(self):
        model = create_model("baseline", backbone="mobilenet_v3_small", num_classes=5)
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, 5)

    def test_return_features(self):
        model = create_model("baseline", backbone="mobilenet_v3_small", num_classes=5)
        x = torch.randn(2, 3, 224, 224)
        logits, features = model(x, return_features=True)
        assert logits.shape == (2, 5)
        assert features.shape[0] == 2

    def test_extract_features(self):
        model = create_model("baseline", backbone="mobilenet_v3_small", num_classes=5)
        x = torch.randn(2, 3, 224, 224)
        features = model.extract_features(x)
        assert features.shape[0] == 2
        assert features.dim() == 2

    def test_num_classes_attribute(self):
        model = create_model("baseline", backbone="mobilenet_v3_small", num_classes=3)
        assert model.num_classes == 3


class TestMDFANModel:
    """Tests for MDFAN model creation."""

    def test_creation(self):
        model = create_model(
            "mdfan", backbone="mobilenet_v3_small",
            num_classes=5, num_sources=2,
        )
        assert model is not None

    def test_forward_shape(self):
        model = create_model(
            "mdfan", backbone="mobilenet_v3_small",
            num_classes=5, num_sources=2,
        )
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, 5)


class TestModelFactory:
    """Tests for ModelFactory dispatch."""

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported model type"):
            ModelFactory.create({"type": "nonexistent"})

    def test_baseline_default(self):
        model = ModelFactory.create({"type": "baseline", "num_classes": 5})
        assert isinstance(model, BaselineModel)

    def test_create_model_convenience(self):
        model = create_model("baseline", backbone="mobilenet_v3_small", num_classes=5)
        assert isinstance(model, BaselineModel)
