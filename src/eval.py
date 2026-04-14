"""
Evaluation Script
=================
Evaluate trained models on test data.
Supports OOD detection and detailed analysis.

Usage:
    python src/eval.py --checkpoint ./outputs/exp/best_model.pt --test_dir ./data/test
    python src/eval.py --checkpoint ./outputs/exp/best_model.pt --test_dir ./data/test --ood_dir ./data/ood
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import PotatoDiseaseDataset
from src.data.transforms import get_val_transforms
from src.models import create_model
from src.utils.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_f1,
    compute_per_class_accuracy,
)
from src.utils.ood_detection import OODDetector, compute_ood_metrics
from src.utils.visualization import plot_confusion_matrix, plot_tsne, visualize_gradcam

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


_GRADCAM_TARGET_LAYER_BY_BACKBONE: dict[str, str] = {
    "mobilenet_v3_small": "backbone.backbone.conv_head",
    "mobilenet_v3_large": "backbone.backbone.conv_head",
    "resnet18": "backbone.backbone.layer4",
    "resnet34": "backbone.backbone.layer4",
    "resnet50": "backbone.backbone.layer4",
    "resnet101": "backbone.backbone.layer4",
}


def get_default_gradcam_target_layer(backbone: str) -> str:
    try:
        return _GRADCAM_TARGET_LAYER_BY_BACKBONE[backbone]
    except KeyError as exc:
        supported = sorted(_GRADCAM_TARGET_LAYER_BY_BACKBONE)
        raise ValueError(
            f"Unsupported backbone '{backbone}' for Grad-CAM. Supported: {supported}"
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate potato disease classifier")

    # Model
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model", type=str, default="baseline", choices=["baseline", "mdfan"]
    )
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small")
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes (default: infer from checkpoint)",
    )
    parser.add_argument(
        "--num_sources",
        type=int,
        default=None,
        help="Number of source domains for MDFAN (default: infer from checkpoint)",
    )

    # Data
    parser.add_argument(
        "--test_dir", type=str, required=True, help="Test data directory"
    )
    parser.add_argument(
        "--ood_dir",
        type=str,
        default=None,
        help="OOD data directory for open-set evaluation",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)

    # OOD detection
    parser.add_argument(
        "--ood_method",
        type=str,
        default="msp",
        choices=["msp", "entropy", "energy"],
        help="OOD detection method",
    )
    parser.add_argument(
        "--ood_threshold", type=float, default=None, help="OOD rejection threshold"
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/eval")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualizations"
    )

    # Explainability
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate a Grad-CAM visualization (requires torchcam; install with --extra viz)",
    )

    # Device
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
            "Could not infer num_sources from checkpoint. "
            "Pass --num_sources explicitly."
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


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    test_loader,
    device: str,
    num_classes: int,
    class_names: list[str],
) -> dict:
    """
    Evaluate classification performance.

    Returns:
        Dictionary with all metrics
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []

    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)

        # Get predictions and features
        if hasattr(model, "forward"):
            outputs = model(images, return_features=True)
            if isinstance(outputs, tuple):
                logits, features = outputs
            else:
                logits = outputs
                features = (
                    model.extract_features(images)
                    if hasattr(model, "extract_features")
                    else None
                )
        else:
            logits = model(images)
            features = None

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels)
        all_probs.append(probs.cpu())
        if features is not None:
            all_features.append(features.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    all_features = torch.cat(all_features) if all_features else None

    # Compute metrics
    results = {
        "accuracy": compute_accuracy(all_preds, all_labels),
        "f1_macro": compute_f1(all_preds, all_labels, num_classes, "macro"),
        "f1_weighted": compute_f1(all_preds, all_labels, num_classes, "weighted"),
        "per_class_accuracy": compute_per_class_accuracy(
            all_preds, all_labels, num_classes, class_names
        ),
        "confusion_matrix": compute_confusion_matrix(
            all_preds, all_labels, num_classes
        ),
        "predictions": all_preds.numpy(),
        "labels": all_labels.numpy(),
        "probabilities": all_probs.numpy(),
        "features": all_features.numpy() if all_features is not None else None,
    }

    return results


@torch.no_grad()
def evaluate_ood(
    model: nn.Module,
    in_dist_loader,
    ood_loader,
    device: str,
    method: str = "msp",
    threshold: float | None = None,
) -> dict:
    """
    Evaluate OOD detection performance.

    Returns:
        Dictionary with OOD metrics
    """
    detector = OODDetector(model, method=method, threshold=threshold)

    # Collect in-distribution scores
    in_scores = []
    for images, _ in tqdm(in_dist_loader, desc="In-dist scores"):
        images = images.to(device)
        scores = detector.compute_scores(images)
        in_scores.append(scores.cpu())
    in_scores = torch.cat(in_scores).numpy()

    # Collect OOD scores
    ood_scores = []
    for images, _ in tqdm(ood_loader, desc="OOD scores"):
        images = images.to(device)
        scores = detector.compute_scores(images)
        ood_scores.append(scores.cpu())
    ood_scores = torch.cat(ood_scores).numpy()

    # Compute metrics
    ood_metrics = compute_ood_metrics(in_scores, ood_scores)

    # Find optimal threshold if not provided
    if threshold is None:
        threshold = np.percentile(in_scores, 5)  # 5% FPR

    # Compute rejection statistics
    in_rejected = (in_scores < threshold).mean()
    ood_rejected = (ood_scores < threshold).mean()

    results = {
        **ood_metrics,
        "threshold": threshold,
        "in_dist_rejection_rate": in_rejected,
        "ood_rejection_rate": ood_rejected,
        "in_scores": in_scores,
        "ood_scores": ood_scores,
    }

    return results


def main():
    args = parse_args()

    # Setup
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    num_classes = args.num_classes
    if num_classes is None:
        num_classes = _infer_num_classes(state_dict, args.model)
        logger.info(f"Inferred num_classes={num_classes} from checkpoint")

    # Create model
    if args.model == "baseline":
        model = create_model(
            model_type="baseline",
            backbone=args.backbone,
            num_classes=num_classes,
            pretrained=False,
        )
    else:
        num_sources = args.num_sources
        if num_sources is None:
            num_sources = _infer_num_sources(state_dict)
            logger.info(f"Inferred num_sources={num_sources} from checkpoint")

        model = create_model(
            model_type="mdfan",
            backbone=args.backbone,
            num_classes=num_classes,
            num_sources=num_sources,
            pretrained=False,
        )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    # Create test dataset
    transforms = get_val_transforms(image_size=args.image_size)

    test_dataset = PotatoDiseaseDataset(
        root=args.test_dir,
        transform=transforms,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    class_names = test_dataset.classes

    if len(class_names) > num_classes:
        raise ValueError(
            f"Test dataset has {len(class_names)} classes but model outputs {num_classes}. "
            "Use a matching test_dir or pass the correct --num_classes."
        )

    if len(class_names) < num_classes:
        class_names = class_names + [
            f"class_{i}" for i in range(len(class_names), num_classes)
        ]

    logger.info(
        f"Test dataset: {len(test_dataset)} samples, {len(test_dataset.classes)} classes"
    )
    logger.info(f"Model output classes: {num_classes}")

    # Evaluate classification
    logger.info("\n=== Classification Evaluation ===")
    cls_results = evaluate_classification(
        model, test_loader, device, num_classes, class_names
    )

    logger.info(f"Accuracy: {cls_results['accuracy']:.4f}")
    logger.info(f"F1 (macro): {cls_results['f1_macro']:.4f}")
    logger.info(f"F1 (weighted): {cls_results['f1_weighted']:.4f}")
    logger.info("\nPer-class accuracy:")
    for cls, acc in cls_results["per_class_accuracy"].items():
        logger.info(f"  {cls}: {acc:.4f}")

    # OOD evaluation
    if args.ood_dir:
        logger.info("\n=== OOD Detection Evaluation ===")

        ood_dataset = PotatoDiseaseDataset(
            root=args.ood_dir,
            transform=transforms,
            labeled=False,  # OOD classes may not have labels
        )

        ood_loader = torch.utils.data.DataLoader(
            ood_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        logger.info(f"OOD dataset: {len(ood_dataset)} samples")

        ood_results = evaluate_ood(
            model,
            test_loader,
            ood_loader,
            device,
            method=args.ood_method,
            threshold=args.ood_threshold,
        )

        logger.info(f"AUROC: {ood_results['auroc']:.4f}")
        logger.info(f"AUPR: {ood_results['aupr']:.4f}")
        logger.info(f"FPR@95TPR: {ood_results['fpr@95tpr']:.4f}")
        logger.info(f"Threshold: {ood_results['threshold']:.4f}")
        logger.info(
            f"In-dist rejection rate: {ood_results['in_dist_rejection_rate']:.4f}"
        )
        logger.info(f"OOD rejection rate: {ood_results['ood_rejection_rate']:.4f}")
    else:
        ood_results = None

    # Generate visualizations
    if args.visualize:
        logger.info("\n=== Generating Visualizations ===")

        # Confusion matrix
        plot_confusion_matrix(
            cls_results["confusion_matrix"],
            class_names,
            save_path=str(output_dir / "confusion_matrix.png"),
            title="Confusion Matrix",
        )
        logger.info(f"Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")

        # t-SNE (if features available)
        if cls_results["features"] is not None:
            plot_tsne(
                cls_results["features"],
                cls_results["labels"],
                class_names=class_names,
                save_path=str(output_dir / "tsne.png"),
                title="t-SNE Feature Visualization",
            )
            logger.info(f"Saved t-SNE to {output_dir / 'tsne.png'}")

    # Grad-CAM (requires gradients)
    if args.gradcam:
        logger.info("\n=== Generating Grad-CAM ===")

        target_layer = get_default_gradcam_target_layer(args.backbone)

        images, _ = next(iter(test_loader))
        images = images.to(device)

        try:
            visualize_gradcam(
                model,
                images,
                target_layer=target_layer,
                save_path=str(output_dir / "gradcam.png"),
            )
        except AttributeError as exc:
            raise AttributeError(
                f"Grad-CAM target_layer '{target_layer}' could not be resolved for backbone '{args.backbone}'. "
                "If you added a new backbone, extend the mapping in eval.py."
            ) from exc
        logger.info(f"Saved Grad-CAM to {output_dir / 'gradcam.png'}")

    # Save results
    results = {
        "classification": {
            "accuracy": cls_results["accuracy"],
            "f1_macro": cls_results["f1_macro"],
            "f1_weighted": cls_results["f1_weighted"],
            "per_class_accuracy": cls_results["per_class_accuracy"],
        },
    }

    if ood_results:
        results["ood_detection"] = {
            "auroc": ood_results["auroc"],
            "aupr": ood_results["aupr"],
            "fpr_at_95tpr": ood_results["fpr@95tpr"],
            "threshold": ood_results["threshold"],
        }

    # Save predictions if requested
    if args.save_predictions:
        np.savez(
            str(output_dir / "predictions.npz"),
            predictions=cls_results["predictions"],
            labels=cls_results["labels"],
            probabilities=cls_results["probabilities"],
        )
        logger.info(f"Saved predictions to {output_dir / 'predictions.npz'}")

    # Save summary
    torch.save(results, str(output_dir / "eval_results.pt"))
    logger.info(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    main()
