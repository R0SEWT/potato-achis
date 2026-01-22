"""
Visualization Utilities
=======================
Tools for visualizing features, predictions, and model explanations.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix (num_classes, num_classes)
        class_names: List of class names
        save_path: Path to save figure
        title: Plot title
        cmap: Colormap
        normalize: Normalize rows to percentages
        
    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label',
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    domain_labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "t-SNE Feature Visualization",
    perplexity: int = 30,
) -> plt.Figure:
    """
    Plot t-SNE visualization of features.
    
    Args:
        features: Feature array (N, D)
        labels: Class labels (N,)
        class_names: Optional class names
        domain_labels: Optional domain labels for different markers
        save_path: Path to save figure
        title: Plot title
        perplexity: t-SNE perplexity parameter
        
    Returns:
        Matplotlib figure
    """
    from sklearn.manifold import TSNE
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings = tsne.fit_transform(features)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        
        if domain_labels is not None:
            # Different markers for different domains
            for domain in np.unique(domain_labels):
                domain_mask = mask & (domain_labels == domain)
                marker = 'o' if domain == 0 else '^'
                ax.scatter(
                    embeddings[domain_mask, 0],
                    embeddings[domain_mask, 1],
                    c=[colors[i]],
                    marker=marker,
                    label=f"{class_names[label] if class_names else label} (D{domain})",
                    alpha=0.7,
                    s=30,
                )
        else:
            label_name = class_names[label] if class_names else f"Class {label}"
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[colors[i]],
                label=label_name,
                alpha=0.7,
                s=30,
            )
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Training Curves",
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accs: Training accuracies per epoch
        val_accs: Validation accuracies per epoch
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    has_acc = train_accs is not None and val_accs is not None
    
    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(12 if has_acc else 6, 5))
    
    if not has_acc:
        axes = [axes]
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if has_acc:
        axes[1].plot(epochs, train_accs, 'b-', label='Train Acc')
        axes[1].plot(epochs, val_accs, 'r-', label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_domain_distribution(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_name: str = "Source",
    target_name: str = "Target",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot domain distribution comparison.
    
    Args:
        source_features: Source domain features
        target_features: Target domain features
        source_name: Source domain name
        target_name: Target domain name
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    from sklearn.manifold import TSNE
    
    # Combine and run t-SNE
    combined = np.vstack([source_features, target_features])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = tsne.fit_transform(combined)
    
    source_emb = embeddings[:len(source_features)]
    target_emb = embeddings[len(source_features):]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(source_emb[:, 0], source_emb[:, 1], 
               c='blue', alpha=0.5, label=source_name, s=20)
    ax.scatter(target_emb[:, 0], target_emb[:, 1], 
               c='red', alpha=0.5, label=target_name, s=20)
    
    ax.legend()
    ax.set_title("Domain Feature Distribution")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    target_layer: str,
    class_idx: Optional[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize Grad-CAM attention maps.
    
    Requires torchcam library: pip install torchcam
    
    Args:
        model: Trained model
        images: Input images (B, C, H, W)
        target_layer: Name of target layer for Grad-CAM
        class_idx: Target class index (None = predicted class)
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    try:
        from torchcam.methods import GradCAM
        from torchcam.utils import overlay_mask
    except ImportError:
        raise ImportError("Install torchcam: pip install torchcam")
    
    model.eval()
    
    # Get the target layer
    target_module = model
    for name in target_layer.split('.'):
        target_module = getattr(target_module, name)
    
    cam_extractor = GradCAM(model, target_layer=target_module)
    
    # Process images
    with torch.enable_grad():
        outputs = model(images)
    
    if class_idx is None:
        class_idx = outputs.argmax(dim=1)
    
    # Get CAM
    activation_maps = cam_extractor(class_idx.tolist(), outputs)
    
    # Plot
    n_images = min(images.size(0), 8)
    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 3, 6))
    
    for i in range(n_images):
        # Original image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Class: {class_idx[i].item()}")
        axes[0, i].axis('off')
        
        # Grad-CAM overlay
        cam = activation_maps[i].squeeze().cpu().numpy()
        axes[1, i].imshow(img)
        axes[1, i].imshow(cam, cmap='jet', alpha=0.5)
        axes[1, i].set_title("Grad-CAM")
        axes[1, i].axis('off')
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
