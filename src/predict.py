"""Prediction script.

Runs inference on one image or a directory of images using a training checkpoint.

Examples:
    # Predict a single image
    uv run potato-predict \
        --checkpoint ./outputs/exp/best_model.pt \
        --input ./some_image.jpg \
        --model baseline \
        --backbone mobilenet_v3_small

    # Predict all images under a directory (recursively)
    uv run potato-predict \
        --checkpoint ./outputs/exp/best_model.pt \
        --input ./data/raw/andean_field/test

Notes:
    - If the checkpoint was saved by `src/train.py`, the script will prefer the
      stored checkpoint metadata (model/backbone/num_classes/num_sources/classes)
      when CLI flags are omitted.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add repo root to path so `src.*` imports work when running as a script.
_REPO_ROOT = str(Path(__file__).parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.transforms import get_val_transforms  # noqa: E402
from src.models import create_model  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
    ".WEBP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on images")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a training checkpoint (e.g., best_model.pt)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an image file or a directory (searched recursively)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["baseline", "mdfan"],
        help="Model type (default: infer from checkpoint if available)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Backbone name (default: infer from checkpoint if available)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of output classes (default: infer from checkpoint)",
    )
    parser.add_argument(
        "--num_sources",
        type=int,
        default=None,
        help="Number of source domains for MDFAN (default: infer from checkpoint)",
    )

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of top classes to report",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSONL path to write predictions",
    )

    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def _infer_num_sources(state_dict: dict[str, torch.Tensor]) -> int:
    pattern = re.compile(r"^(?:module\.)?source_classifiers\.(\d+)\.")
    indices: set[int] = set()

    for key in state_dict:
        match = pattern.match(key)
        if match is not None:
            indices.add(int(match.group(1)))

    if not indices:
        raise ValueError(
            "Could not infer num_sources from checkpoint. Pass --num_sources explicitly."
        )

    return max(indices) + 1


def _infer_num_classes(state_dict: dict[str, torch.Tensor], model_type: str) -> int:
    candidates: list[re.Pattern[str]]

    if model_type == "baseline":
        candidates = [re.compile(r"^(?:module\.)?head\.classifier\.weight$")]
    else:
        candidates = [
            re.compile(r"^(?:module\.)?combined_classifier\.classifier\.weight$"),
            re.compile(r"^(?:module\.)?source_classifiers\.\d+\.classifier\.weight$"),
        ]

    for pattern in candidates:
        for key, value in state_dict.items():
            if pattern.match(key) and value.ndim == 2:
                return int(value.shape[0])

    raise ValueError(
        "Could not infer num_classes from checkpoint. Pass --num_classes explicitly."
    )


def _iter_image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    paths = [p for p in input_path.rglob("*") if p.suffix in _IMAGE_EXTENSIONS]
    return sorted(paths)


class _InferenceImageDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(self, image_paths: list[Path], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image) if self.transform is not None else image
        return tensor, str(path)


def _collate_batch(
    batch: list[tuple[torch.Tensor, str]],
) -> tuple[torch.Tensor, list[str]]:
    images, paths = zip(*batch)
    return torch.stack(list(images), dim=0), list(paths)


@dataclass(frozen=True)
class _Prediction:
    path: str
    pred_idx: int
    pred_class: str
    confidence: float
    topk: list[dict[str, float | int | str]]


@torch.inference_mode()
def _predict(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    class_names: list[str],
    topk: int,
) -> list[_Prediction]:
    model.eval()
    model = model.to(device)

    results: list[_Prediction] = []

    for images, paths in tqdm(loader, desc="Predicting"):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        k = min(topk, probs.shape[1])
        scores, indices = torch.topk(probs, k=k, dim=1)

        for i, path in enumerate(paths):
            top_entries = []
            for j in range(k):
                class_idx = int(indices[i, j].item())
                class_name = (
                    class_names[class_idx]
                    if 0 <= class_idx < len(class_names)
                    else f"class_{class_idx}"
                )
                top_entries.append(
                    {
                        "class_idx": class_idx,
                        "class_name": class_name,
                        "prob": float(scores[i, j].item()),
                    }
                )

            pred_idx = int(indices[i, 0].item())
            pred_class = (
                class_names[pred_idx]
                if 0 <= pred_idx < len(class_names)
                else f"class_{pred_idx}"
            )
            confidence = float(scores[i, 0].item())

            results.append(
                _Prediction(
                    path=path,
                    pred_idx=pred_idx,
                    pred_class=pred_class,
                    confidence=confidence,
                    topk=top_entries,
                )
            )

    return results


def main() -> None:
    args = parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path(args.checkpoint)
    input_path = Path(args.input)

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format (expected a dict)")

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    meta = checkpoint.get("meta", {})

    model_type = args.model or meta.get("model") or "baseline"
    backbone = args.backbone or meta.get("backbone") or "mobilenet_v3_small"

    num_classes = args.num_classes
    if num_classes is None:
        meta_num_classes = meta.get("num_classes")
        if isinstance(meta_num_classes, int):
            num_classes = meta_num_classes
        else:
            num_classes = _infer_num_classes(state_dict, model_type)

    num_sources = args.num_sources
    if model_type == "mdfan" and num_sources is None:
        meta_num_sources = meta.get("num_sources")
        if isinstance(meta_num_sources, int):
            num_sources = meta_num_sources
        else:
            num_sources = _infer_num_sources(state_dict)

    classes = meta.get("classes")
    if isinstance(classes, list) and all(isinstance(x, str) for x in classes):
        class_names = classes
    else:
        class_names = [f"class_{i}" for i in range(num_classes)]

    if len(class_names) < num_classes:
        class_names = class_names + [
            f"class_{i}" for i in range(len(class_names), num_classes)
        ]

    logger.info(
        f"Model: {model_type} | backbone={backbone} | num_classes={num_classes} | device={device}"
    )

    if model_type == "baseline":
        model = create_model(
            model_type="baseline",
            backbone=backbone,
            num_classes=num_classes,
            pretrained=False,
        )
    else:
        if num_sources is None:
            raise ValueError("num_sources is required for MDFAN")
        model = create_model(
            model_type="mdfan",
            backbone=backbone,
            num_classes=num_classes,
            num_sources=num_sources,
            pretrained=False,
        )

    model.load_state_dict(state_dict)

    image_paths = _iter_image_paths(input_path)
    if not image_paths:
        raise ValueError(f"No images found under: {input_path}")

    transforms = get_val_transforms(image_size=args.image_size)
    dataset = _InferenceImageDataset(image_paths, transforms)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device != "cpu"),
        collate_fn=_collate_batch,
    )

    predictions = _predict(
        model=model,
        loader=loader,
        device=device,
        class_names=class_names,
        topk=args.topk,
    )

    for pred in predictions:
        logger.info(f"{pred.path} -> {pred.pred_class} ({pred.confidence:.3f})")

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(
                    json.dumps(
                        {
                            "path": pred.path,
                            "pred_idx": pred.pred_idx,
                            "pred_class": pred.pred_class,
                            "confidence": pred.confidence,
                            "topk": pred.topk,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        logger.info(f"Wrote predictions to: {output_path}")


if __name__ == "__main__":
    main()
