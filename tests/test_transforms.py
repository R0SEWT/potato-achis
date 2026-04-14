"""Tests for augmentation and transform pipelines."""

import pytest
import torch
from PIL import Image

from src.data.transforms.andean_transforms import (
    AndeanFieldAugmentation,
    AndeanFieldTransform,
)
from src.data.transforms.augmentations import (
    Denormalize,
    get_train_transforms,
    get_val_transforms,
)


class TestStandardTransforms:
    """Tests for standard train/val transforms."""

    def test_train_transform_output_shape(self):
        transform = get_train_transforms(image_size=224, strength="medium")
        img = Image.new("RGB", (300, 300))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_transform_output_shape(self):
        transform = get_val_transforms(image_size=224)
        img = Image.new("RGB", (300, 300))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_all_strengths_work(self):
        for strength in ("light", "medium", "strong"):
            transform = get_train_transforms(image_size=128, strength=strength)
            img = Image.new("RGB", (200, 200))
            tensor = transform(img)
            assert tensor.shape == (3, 128, 128)

    def test_invalid_strength_raises(self):
        with pytest.raises(ValueError, match="Unknown strength"):
            get_train_transforms(strength="extreme")


class TestDenormalize:
    """Tests for Denormalize utility."""

    def test_roundtrip(self):
        """Normalize then denormalize should approximate identity."""
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        denorm = Denormalize(mean=mean, std=std)

        original = torch.rand(3, 32, 32)
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)
        normalized = (original - mean_t) / std_t

        recovered = denorm(normalized)
        assert torch.allclose(original, recovered, atol=1e-5)

    def test_batched_roundtrip(self):
        """Denormalize should work correctly on batched (N, C, H, W) inputs."""
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        denorm = Denormalize(mean=mean, std=std)

        originals = torch.rand(4, 3, 32, 32)
        mean_t = torch.tensor(mean).view(1, 3, 1, 1)
        std_t = torch.tensor(std).view(1, 3, 1, 1)
        normalized = (originals - mean_t) / std_t

        recovered = denorm(normalized)
        assert torch.allclose(originals, recovered, atol=1e-5)


class TestAndeanFieldAugmentation:
    """Tests for Andean-specific augmentations."""

    def test_output_shape_preserved(self):
        aug = AndeanFieldAugmentation(p=1.0, intensity="medium")
        x = torch.rand(3, 224, 224)
        y = aug(x)
        assert y.shape == x.shape

    def test_all_intensities_work(self):
        for intensity in ("light", "medium", "strong"):
            aug = AndeanFieldAugmentation(p=1.0, intensity=intensity)
            x = torch.rand(3, 128, 128)
            y = aug(x)
            assert y.shape == x.shape

    def test_invalid_intensity_raises(self):
        with pytest.raises(ValueError, match="Unknown intensity"):
            AndeanFieldAugmentation(intensity="ultra")

    def test_p_zero_is_identity(self):
        """With p=0, no augmentation should be applied."""
        aug = AndeanFieldAugmentation(p=0.0, intensity="medium")
        x = torch.rand(3, 64, 64)
        y = aug(x)
        assert torch.equal(x, y)


class TestAndeanFieldTransform:
    """Tests for the full Andean transform pipeline."""

    def test_train_mode(self):
        transform = AndeanFieldTransform(image_size=224, augment=True)
        img = Image.new("RGB", (300, 300))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_mode(self):
        transform = AndeanFieldTransform(image_size=224, augment=False)
        img = Image.new("RGB", (300, 300))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_mode_is_deterministic(self):
        """Validation transforms should give consistent output."""
        transform = AndeanFieldTransform(image_size=128, augment=False)
        img = Image.new("RGB", (200, 200), color=(100, 150, 200))
        t1 = transform(img)
        t2 = transform(img)
        assert torch.equal(t1, t2)
