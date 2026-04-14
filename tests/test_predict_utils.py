from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.predict import _iter_image_paths


def test_iter_image_paths_single_file(tmp_path: Path):
    img_path = tmp_path / "x.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(img_path)

    paths = _iter_image_paths(img_path)
    assert paths == [img_path]


def test_iter_image_paths_directory_recursive(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    img1 = tmp_path / "a" / "1.jpg"
    img2 = tmp_path / "b" / "2.png"
    txt = tmp_path / "b" / "note.txt"

    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(img1)
    Image.new("RGB", (8, 8), color=(0, 0, 255)).save(img2)
    txt.write_text("hi", encoding="utf-8")

    paths = _iter_image_paths(tmp_path)
    assert paths == [img1, img2]
