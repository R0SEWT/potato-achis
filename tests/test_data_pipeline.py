"""Tests for data pipeline integrity — no train/val leakage."""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.datamodule import PotatoDataModule, TransformSubset


@pytest.fixture
def fake_dataset_dir():
    """Create a temporary dataset with class subfolders and dummy images."""
    tmpdir = Path(tempfile.mkdtemp())
    classes = ["early_blight", "late_blight", "healthy"]
    for cls in classes:
        cls_dir = tmpdir / cls
        cls_dir.mkdir()
        for i in range(20):
            img = Image.new("RGB", (64, 64), color=(i * 10, i * 5, i * 3))
            img.save(cls_dir / f"img_{i:03d}.jpg")
    yield str(tmpdir)
    shutil.rmtree(tmpdir)


class TestTrainValSplit:
    """Verify train/val splits have no overlap and use correct transforms."""

    def test_no_index_overlap(self, fake_dataset_dir):
        """Train and val indices must be disjoint."""
        dm = PotatoDataModule(
            data_dir=fake_dataset_dir,
            batch_size=4,
            num_workers=0,
            val_split=0.2,
            use_andean_aug=False,
        )
        dm.setup_single_source(fake_dataset_dir, class_filter=None)

        assert isinstance(dm.train_dataset, TransformSubset)
        assert isinstance(dm.val_dataset, TransformSubset)

        train_indices = set(dm.train_dataset.indices)
        val_indices = set(dm.val_dataset.indices)

        assert len(train_indices & val_indices) == 0, "Train/val indices overlap!"

    def test_all_samples_covered(self, fake_dataset_dir):
        """Train + val should cover the full dataset."""
        dm = PotatoDataModule(
            data_dir=fake_dataset_dir,
            batch_size=4,
            num_workers=0,
            val_split=0.2,
            use_andean_aug=False,
        )
        dm.setup_single_source(fake_dataset_dir, class_filter=None)

        total = len(dm._full_dataset)
        covered = len(dm.train_dataset) + len(dm.val_dataset)
        assert covered == total

    def test_split_is_deterministic(self, fake_dataset_dir):
        """Same seed should produce same split."""
        dm1 = PotatoDataModule(data_dir=fake_dataset_dir, num_workers=0, use_andean_aug=False)
        dm1.setup_single_source(fake_dataset_dir, class_filter=None)

        dm2 = PotatoDataModule(data_dir=fake_dataset_dir, num_workers=0, use_andean_aug=False)
        dm2.setup_single_source(fake_dataset_dir, class_filter=None)

        assert dm1.train_dataset.indices == dm2.train_dataset.indices
        assert dm1.val_dataset.indices == dm2.val_dataset.indices

    def test_train_and_val_use_different_transforms(self, fake_dataset_dir):
        """Train subset should have train transforms, val should have val transforms."""
        dm = PotatoDataModule(
            data_dir=fake_dataset_dir,
            batch_size=4,
            num_workers=0,
            val_split=0.2,
            use_andean_aug=False,
        )
        dm.setup_single_source(fake_dataset_dir, class_filter=None)

        assert dm.train_dataset.transform is not None
        assert dm.val_dataset.transform is not None
        assert dm.train_dataset.transform is not dm.val_dataset.transform

    def test_underlying_dataset_has_no_transform(self, fake_dataset_dir):
        """The shared base dataset should have no transform applied."""
        dm = PotatoDataModule(
            data_dir=fake_dataset_dir,
            batch_size=4,
            num_workers=0,
            use_andean_aug=False,
        )
        dm.setup_single_source(fake_dataset_dir, class_filter=None)

        assert dm._full_dataset.transform is None

    def test_classes_detected(self, fake_dataset_dir):
        """Classes should be auto-detected from directory structure."""
        dm = PotatoDataModule(data_dir=fake_dataset_dir, num_workers=0, use_andean_aug=False)
        dm.setup_single_source(fake_dataset_dir, class_filter=None)

        assert set(dm.classes) == {"early_blight", "late_blight", "healthy"}
        assert dm.num_classes == 3
