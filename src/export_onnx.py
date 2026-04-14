"""ONNX export utility.

Exports a trained Baseline or MDFAN model to ONNX for inference.

Usage:
    uv sync --extra onnx

    # Baseline
    uv run python src/export_onnx.py \
        --checkpoint ./outputs/exp/best_model.pt \
        --model baseline \
        --backbone mobilenet_v3_small \
        --num_classes 5

    # MDFAN
    uv run python src/export_onnx.py \
        --checkpoint ./outputs/exp/best_model.pt \
        --model mdfan \
        --backbone resnet50 \
        --num_classes 5
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add repo root to path so `src.*` imports work when running as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import create_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained model to ONNX")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a training checkpoint (e.g., best_model.pt)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "mdfan"],
        help="Model type",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenet_v3_small",
        help="Backbone network",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        help="Number of output classes",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size (H=W=image_size)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX path (default: <checkpoint>.onnx)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for export/validation (cpu recommended)",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for onnxruntime validation",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for onnxruntime validation",
    )

    return parser.parse_args()


def _infer_num_sources(state_dict: dict[str, torch.Tensor]) -> int:
    pattern = re.compile(r"^source_classifiers\.(\d+)\.")
    indices: set[int] = set()

    for key in state_dict:
        match = pattern.match(key)
        if match is not None:
            indices.add(int(match.group(1)))

    if not indices:
        raise ValueError(
            "Could not infer num_sources from checkpoint. "
            "Pass a checkpoint from an MDFAN training run."
        )

    return max(indices) + 1


class _InferenceWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.model(images)


def export_onnx(
    model: nn.Module,
    output_path: Path,
    image_size: int,
    opset: int,
) -> None:
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting ONNX to: {output_path}")

    torch.onnx.export(
        _InferenceWrapper(model),
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={
            "images": {0: "batch"},
            "logits": {0: "batch"},
        },
        export_params=True,
        do_constant_folding=True,
    )


@torch.inference_mode()
def validate_with_onnxruntime(
    model: nn.Module,
    onnx_path: Path,
    image_size: int,
    device: str,
    rtol: float,
    atol: float,
) -> None:
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required to validate the exported ONNX model. "
            "Install with `uv sync --extra onnx`."
        ) from exc

    model.eval()
    model = model.to(device)

    dummy = torch.randn(2, 3, image_size, image_size, device=device)
    torch_logits = model(dummy).detach().cpu().numpy()

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    input_name = session.get_inputs()[0].name
    ort_logits = session.run(None, {input_name: dummy.cpu().numpy()})[0]

    np.testing.assert_allclose(torch_logits, ort_logits, rtol=rtol, atol=atol)
    max_abs_diff = float(np.max(np.abs(torch_logits - ort_logits)))
    logger.info(f"onnxruntime validation passed (max abs diff={max_abs_diff:.6f})")


def main() -> None:
    args = parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path(args.checkpoint)
    if args.output is None:
        output_path = checkpoint_path.with_suffix(".onnx")
    else:
        output_path = Path(args.output)

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if args.model == "baseline":
        model = create_model(
            model_type="baseline",
            backbone=args.backbone,
            num_classes=args.num_classes,
            pretrained=False,
        )
    else:
        num_sources = _infer_num_sources(state_dict)
        model = create_model(
            model_type="mdfan",
            backbone=args.backbone,
            num_classes=args.num_classes,
            num_sources=num_sources,
            pretrained=False,
        )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    export_onnx(
        model=model,
        output_path=output_path,
        image_size=args.image_size,
        opset=args.opset,
    )

    validate_with_onnxruntime(
        model=model,
        onnx_path=output_path,
        image_size=args.image_size,
        device=device,
        rtol=args.rtol,
        atol=args.atol,
    )


if __name__ == "__main__":
    main()
