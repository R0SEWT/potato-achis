"""
Training Script
===============
Main training entry point for baseline and MDFAN models.

Usage:
    # Baseline training
    python src/train.py --model baseline --backbone mobilenet_v3_small --data_dir ./data/plantvillage
    
    # MDFAN training
    python src/train.py --model mdfan --backbone resnet50 --source_dirs ./data/source1 ./data/source2 --target_dir ./data/target
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import PotatoDataModule
from src.data.datasets import MultiSourceIterator
from src.losses import MMDLoss
from src.losses.domain_adversarial_loss import ClassificationLoss, MultiSourceDomainLoss
from src.models import create_model
from src.models.components.gradient_reversal import get_lambda_schedule
from src.utils.metrics import MetricTracker, compute_accuracy, compute_f1

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train potato disease classifier")

    # Model
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline', 'mdfan'],
                        help='Model type')
    parser.add_argument('--backbone', type=str, default='mobilenet_v3_small',
                        help='Backbone network')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of disease classes')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data/raw/plantvillage',
                        help='Data directory (for baseline)')
    parser.add_argument('--source_dirs', nargs='+', type=str,
                        help='Source domain directories (for MDFAN)')
    parser.add_argument('--target_dir', type=str,
                        help='Target domain directory (for MDFAN)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--use_andean_aug', action='store_true', default=True,
                        help='Use Andean field augmentations')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['none', 'step', 'cosine'])

    # MDFAN specific
    parser.add_argument('--lambda_mmd', type=float, default=1.0,
                        help='MMD loss weight')
    parser.add_argument('--lambda_adv', type=float, default=0.5,
                        help='Adversarial loss weight')
    parser.add_argument('--grl_warmup', type=int, default=10,
                        help='GRL warmup epochs')

    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Checkpoint save frequency')

    # Device
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_baseline_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> dict[str, float]:
    """Train one epoch for baseline model."""
    model.train()

    metrics = MetricTracker(['loss', 'accuracy'])

    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        acc = compute_accuracy(logits, labels)
        metrics.update('loss', loss.item(), images.size(0))
        metrics.update('accuracy', acc, images.size(0))

        pbar.set_postfix(metrics.get_averages())

    return metrics.get_averages()


def train_mdfan_epoch(
    model: nn.Module,
    source_loaders: list[DataLoader],
    target_loader: DataLoader,
    criterion_cls: nn.Module,
    criterion_domain: nn.Module,
    criterion_mmd: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    lambda_adv: float,
    lambda_mmd: float,
    epoch: int,
    max_epochs: int,
    grl_warmup: int,
) -> dict[str, float]:
    """Train one epoch for MDFAN model."""
    model.train()

    # Update GRL lambda
    grl_lambda = get_lambda_schedule(
        epoch=epoch,
        max_epochs=max_epochs,
        initial=0.0,
        final=1.0,
        warmup_epochs=grl_warmup,
    )
    model.set_grl_lambda(grl_lambda)

    metrics = MetricTracker([
        'total_loss', 'cls_loss', 'domain_loss', 'mmd_loss', 'accuracy'
    ])

    # Create synchronized iterator
    iterator = MultiSourceIterator(
        source_loaders, target_loader,
        num_iterations=len(target_loader)
    )

    pbar = tqdm(iterator, desc=f"Training (λ={grl_lambda:.2f})")
    for source_batches, target_batch in pbar:
        # Unpack batches
        source_images = [b[0].to(device) for b in source_batches]
        source_labels = [b[1].to(device) for b in source_batches]
        target_images = target_batch[0].to(device)

        # Forward pass
        outputs = model.forward_train(source_images, source_labels, target_images)

        # Classification loss (source only)
        cls_loss = torch.tensor(0.0, device=device)
        total_correct = 0
        total_samples = 0

        for i, (logits, labels) in enumerate(zip(
            outputs['source_logits'], source_labels
        )):
            cls_loss += criterion_cls(logits, labels)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

        cls_loss /= len(source_labels)
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        # Domain adversarial loss
        domain_loss = criterion_domain(
            outputs['source_domain_preds'],
            outputs['target_domain_preds'],
        )

        # MMD loss
        mmd_loss = criterion_mmd(
            outputs['source_features'],
            outputs['target_features'],
        )

        # Total loss
        total_loss = cls_loss + lambda_adv * domain_loss + lambda_mmd * mmd_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update metrics
        metrics.update('total_loss', total_loss.item())
        metrics.update('cls_loss', cls_loss.item())
        metrics.update('domain_loss', domain_loss.item())
        metrics.update('mmd_loss', mmd_loss.item())
        metrics.update('accuracy', accuracy)

        pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.get_averages().items()})

    return metrics.get_averages()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int,
) -> dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()

    metrics = MetricTracker(['loss', 'accuracy'])
    all_preds = []
    all_labels = []

    for images, labels in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        acc = compute_accuracy(logits, labels)
        metrics.update('loss', loss.item(), images.size(0))
        metrics.update('accuracy', acc, images.size(0))

        all_preds.append(logits.argmax(1))
        all_labels.append(labels)

    # Compute F1
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    f1 = compute_f1(all_preds, all_labels, num_classes)

    results = metrics.get_averages()
    results['f1'] = f1

    return results


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    save_path: str,
):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, save_path)
    logger.info(f"Checkpoint saved: {save_path}")


def main():
    args = parse_args()

    # Setup
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create output directory
    if args.exp_name is None:
        args.exp_name = (
            f"{args.model}_{args.backbone}_{datetime.now():%Y%m%d_%H%M%S}"
        )

    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Setup TensorBoard
    tb_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))
    logger.info(f"TensorBoard logs: {tb_dir}")

    # Create data module FIRST to detect number of classes
    data_module = PotatoDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_andean_aug=args.use_andean_aug,
    )

    # Setup data loaders based on model type
    if args.model == "baseline":
        data_module.setup_single_source(args.data_dir)
        train_loader = data_module.get_train_loader()
        val_loader = data_module.get_val_loader()
    else:
        if not args.source_dirs or not args.target_dir:
            raise ValueError("MDFAN requires --source_dirs and --target_dir")

        data_module.setup_multi_source(
            source_dirs=args.source_dirs,
            target_dir=args.target_dir,
        )
        source_loaders, target_loader = data_module.get_multi_source_loaders()

        # Also get validation loader from first source
        data_module.setup_single_source(args.source_dirs[0])
        val_loader = data_module.get_val_loader()

    # Get num_classes from data module (auto-detected)
    num_classes = (
        data_module.num_classes if hasattr(data_module, "num_classes") else args.num_classes
    )
    logger.info(f"Detected {num_classes} classes: {getattr(data_module, 'classes', [])}")

    # Create model AFTER data setup
    logger.info(f"Creating {args.model} model with {args.backbone} backbone")

    if args.model == "baseline":
        model = create_model(
            model_type="baseline",
            backbone=args.backbone,
            num_classes=num_classes,
            pretrained=args.pretrained,
        )
    else:  # mdfan
        num_sources = len(args.source_dirs) if args.source_dirs else 2
        model = create_model(
            model_type="mdfan",
            backbone=args.backbone,
            num_classes=num_classes,
            num_sources=num_sources,
            pretrained=args.pretrained,
        )

    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss functions
    criterion_cls = ClassificationLoss(num_classes=num_classes)

    if args.model == "mdfan":
        criterion_domain = MultiSourceDomainLoss(num_sources=len(args.source_dirs))
        criterion_mmd = MMDLoss(kernel_type="rbf")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.1
        )
    else:
        scheduler = None

    # Training loop
    best_val_acc = 0.0
    history = {'train': [], 'val': []}

    logger.info("Starting training...")

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        if args.model == 'baseline':
            train_metrics = train_baseline_epoch(
                model, train_loader, criterion_cls, optimizer, device
            )
        else:
            train_metrics = train_mdfan_epoch(
                model, source_loaders, target_loader,
                criterion_cls, criterion_domain, criterion_mmd,
                optimizer, device,
                lambda_adv=args.lambda_adv,
                lambda_mmd=args.lambda_mmd,
                epoch=epoch,
                max_epochs=args.epochs,
                grl_warmup=args.grl_warmup,
            )

        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion_cls, device, args.num_classes
        )

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Log metrics
        logger.info(f"Train: {train_metrics}")
        logger.info(f"Val: {val_metrics}")

        # TensorBoard logging
        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)

        # Log learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/learning_rate", current_lr, epoch)

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                str(output_dir / 'best_model.pt')
            )

        # Periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                str(output_dir / f'checkpoint_epoch_{epoch + 1}.pt')
            )

    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs - 1, val_metrics,
        str(output_dir / 'final_model.pt')
    )

    # Save training history
    torch.save(history, str(output_dir / 'history.pt'))

    # Close TensorBoard writer
    writer.close()

    logger.info(f"\nTraining complete! Best val accuracy: {best_val_acc:.4f}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"View TensorBoard: tensorboard --logdir {tb_dir}")


if __name__ == '__main__':
    main()
