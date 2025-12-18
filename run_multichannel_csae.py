"""
Multi-Channel ConvSAE Training Script (Flexible Architecture Support)

Supports multiple CNN architectures (ResNet, VGG, DenseNet, EfficientNet) and layers.
Uses pretrained models to extract activation channels and applies a Convolutional 
Sparse Autoencoder with two-level sparsity.

Usage:
    # ResNet50 layer3 (default)
    python run_multichannel_csae.py
    
    # VGG16 features.23
    python run_multichannel_csae.py --architecture vgg16 --target_layer features.23
    
    # ResNet18 layer2 with custom parameters
    python run_multichannel_csae.py --architecture resnet18 --target_layer layer2 --hidden_multiplier 16
    
    # List available architectures
    python run_multichannel_csae.py --list_architectures
"""

import torch
torch.cuda.init()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import joblib
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import sys
import os
import argparse

sys.path.append('.')
from run_multichannel_csae_resnet50 import (
    MultiChannelConvSAE, 
    LateralInhibitionLoss,
    SpatialCompactnessLoss,
    FeatureChannelSparsityLoss
)

# Try to import fixed GradCAM first, fallback to original
try:
    from src.gradcam_fixed import GradCAM
    print("Using fixed GradCAM (supports in-place operations)")
except ImportError:
    from src.gradcam import GradCAM
    print("Using original GradCAM")

from config import ModelConfig, list_available_architectures


class FlexibleActivationExtractor:
    """
    Flexible activation extractor for any CNN architecture and layer.
    Extracts activation channels with GradCAM-based selection.
    """

    def __init__(self, config: ModelConfig, device='cuda'):
        """
        Args:
            config: ModelConfig instance specifying architecture and layer
            device: Device to run on
        """
        self.config = config
        self.device = device

        # Load model and target layer
        print(f"Loading {config.architecture} on {device}...")
        self.model, self.target_layer = config.get_model_and_layer()
        self.model = self.model.to(device)
        self.model.eval()

        print(f"Target layer: {config.target_layer}")
        
        # Detect spatial size if not specified
        if config.spatial_size is None:
            print(f"  - Detecting spatial size...")
            spatial_size = config.detect_spatial_size(self.model, self.target_layer, device)
            print(f"  - Detected: {spatial_size}x{spatial_size}")
        
        print(f"  - Channels: {config.input_channels}")
        print(f"  - Spatial size: {config.spatial_size}×{config.spatial_size}")

        # GradCAM for channel importance
        self.gradcam = GradCAM(self.model, self.target_layer)

        # Hook for activations
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output.detach()

    def _select_channels_with_gradcam(self, image: torch.Tensor, class_idx: int = None) -> torch.Tensor:
        """
        Use GradCAM to select important channels and create a binary mask.
        
        Args:
            image: [1, 3, 224, 224] - Input image
            class_idx: Target class (None = use predicted class)
        
        Returns:
            channel_mask: [num_channels] - Binary mask
            num_selected: Number of selected channels
        """
        weights, _, pred_class = self.gradcam.forward(image, class_idx=class_idx, verbose=False)

        # Validate weights shape
        if weights.shape[0] != self.config.input_channels:
            raise RuntimeError(
                f"GradCAM output mismatch: got {weights.shape[0]} channels, "
                f"expected {self.config.input_channels}. "
                f"This usually means the detected layer output shape is incorrect."
            )

        # Sort channels by importance
        sorted_indices = torch.argsort(weights, descending=True)
        sorted_weights = weights[sorted_indices]

        # Normalize and compute cumulative sum
        total_score = sorted_weights.sum()
        if total_score > 0:
            cumsum = torch.cumsum(sorted_weights / total_score, dim=0)
            num_selected = (cumsum < self.config.cumulative_threshold).sum().item() + 1
            num_selected = min(num_selected, len(sorted_indices))
        else:
            num_selected = max(1, int(0.1 * len(sorted_indices)))

        # Create binary mask
        channel_mask = torch.zeros(self.config.input_channels, dtype=torch.bool, device=self.device)
        selected_channels = sorted_indices[:num_selected]
        channel_mask[selected_channels] = True

        return channel_mask, num_selected

    def collect_activation_maps(self, data_loader: DataLoader, normalize: bool = True) -> torch.Tensor:
        """
        Collect activation maps with GradCAM-based channel selection.
        
        Returns:
            X: [N, num_channels, H, W] - Activation maps
        """
        all_activations = []
        channel_selection_stats = []

        print(f"\nCollecting activation maps from {self.config.architecture} {self.config.target_layer}...")
        print(f"  Cumulative threshold: {self.config.cumulative_threshold * 100:.0f}%")

        for images, labels in tqdm(data_loader, desc="Extracting activations"):
            batch_activations = []

            for i in range(images.size(0)):
                image = images[i:i+1].to(self.device)

                # Forward pass
                with torch.no_grad():
                    _ = self.model(image)
                    activations = self.activations.clone()

                # Validate activation shape
                if activations.shape[1] != self.config.input_channels:
                    raise RuntimeError(
                        f"Activation shape mismatch at {self.config.target_layer}:\n"
                        f"  Expected channels: {self.config.input_channels}\n"
                        f"  Got: {activations.shape[1]}\n"
                        f"  Full shape: {activations.shape}\n"
                        f"This usually means the layer config is incorrect for this architecture."
                    )
                
                # Update spatial size if not detected yet
                if not self.config._spatial_size_detected:
                    self.config.spatial_size = activations.shape[2]
                    self.config._spatial_size_detected = True
                    print(f"\n  Auto-detected spatial size: {self.config.spatial_size}x{self.config.spatial_size}")

                # Get GradCAM-based channel mask
                channel_mask, num_selected = self._select_channels_with_gradcam(image)
                channel_selection_stats.append(num_selected)

                # Apply mask
                mask_4d = channel_mask.view(1, self.config.input_channels, 1, 1).float()
                masked_activations = activations * mask_4d

                batch_activations.append(masked_activations.cpu())

            all_activations.append(torch.cat(batch_activations, dim=0))

        X = torch.cat(all_activations, dim=0)

        # Print statistics
        avg_selected = np.mean(channel_selection_stats)
        std_selected = np.std(channel_selection_stats)
        print(f"\nChannel selection statistics:")
        print(f"  Average: {avg_selected:.1f} ± {std_selected:.1f} (out of {self.config.input_channels})")
        print(f"  Min: {min(channel_selection_stats)}, Max: {max(channel_selection_stats)}")
        print(f"  Sparsity: {(self.config.input_channels - avg_selected) / self.config.input_channels * 100:.1f}%")

        print(f"\nCollected {X.shape[0]} activation maps:")
        print(f"  Shape: {X.shape}")
        print(f"  Range: [{X.min():.4f}, {X.max():.4f}]")

        # Normalize
        if normalize:
            print("\nApplying robust normalization (per-channel)...")
            for c in range(X.shape[1]):
                channel_data = X[:, c, :, :]
                if channel_data.abs().sum() < 1e-8:
                    continue

                flat = channel_data.flatten()
                non_zero_flat = flat[flat > 1e-8]
                if len(non_zero_flat) > 0:
                    scale_factor = torch.quantile(non_zero_flat, 0.99)
                    if scale_factor > 1e-8:
                        channel_data = torch.clamp(channel_data, min=0.0, max=scale_factor)
                        X[:, c, :, :] = channel_data / (scale_factor + 1e-8)

            print(f"  Normalized range: [{X.min():.4f}, {X.max():.4f}]")

        return X


def plot_training_logs(logs: Dict[str, List], config: ModelConfig, save_path: str):
    """Plot training metrics."""
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'Multi-Channel ConvSAE Training ({config.architecture} - {config.target_layer})',
                 fontsize=14, fontweight='bold')

    # Row 1
    axs[0, 0].plot(logs["recon_loss"], color='blue', linewidth=1.5)
    axs[0, 0].set_title("Reconstruction Loss")
    axs[0, 0].set_ylabel("MSE")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(logs["l1_loss"], color='green', linewidth=1.5)
    axs[0, 1].set_title("L1 Sparsity Loss")
    axs[0, 1].set_ylabel("L1")
    axs[0, 1].grid(True, alpha=0.3)

    axs[0, 2].plot(logs["channel_sparsity_loss"], color='purple', linewidth=1.5)
    axs[0, 2].set_title("Channel Sparsity Loss")
    axs[0, 2].set_ylabel("L1 per feature")
    axs[0, 2].grid(True, alpha=0.3)

    # Row 2
    axs[1, 0].plot(logs["lateral_loss"], color='orange', linewidth=1.5)
    axs[1, 0].set_title("Lateral Inhibition Loss")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(logs["compact_loss"], color='red', linewidth=1.5)
    axs[1, 1].set_title("Spatial Compactness Loss")
    axs[1, 1].grid(True, alpha=0.3)

    axs[1, 2].plot(logs["active_pct"], color='teal', linewidth=1.5)
    axs[1, 2].set_title("Active Channels %")
    axs[1, 2].set_ylabel("Percent (%)")
    axs[1, 2].grid(True, alpha=0.3)

    # Row 3
    axs[2, 0].plot(logs["total_loss"], color='black', linewidth=2)
    axs[2, 0].set_title("Total Loss")
    axs[2, 0].grid(True, alpha=0.3)

    axs[2, 1].plot(logs["recon_loss"], label='Recon', alpha=0.7)
    axs[2, 1].plot(logs["l1_loss"], label='L1', alpha=0.7)
    axs[2, 1].set_title("Loss Components")
    axs[2, 1].set_yscale('log')
    axs[2, 1].legend(fontsize=7)
    axs[2, 1].grid(True, alpha=0.3)

    axs[2, 2].scatter(logs["channel_sparsity_loss"], logs["recon_loss"],
                     c=range(len(logs["recon_loss"])), cmap='viridis', alpha=0.5, s=5)
    axs[2, 2].set_title("Recon vs Channel Sparsity")
    axs[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training logs saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Multi-Channel ConvSAE on any CNN architecture')
    
    # Architecture selection
    parser.add_argument('--architecture', type=str, default='resnet50',
                       help='CNN architecture (resnet50, resnet18, vgg16, etc.)')
    parser.add_argument('--target_layer', type=str, default='layer3',
                       help='Target layer name (layer3, features.23, etc.)')
    parser.add_argument('--list_architectures', action='store_true',
                       help='List available architectures and exit')
    
    # Model parameters
    parser.add_argument('--hidden_multiplier', type=float, default=-1.0,
                       help='Hidden dimension multiplier (hidden_dim = in_channels * multiplier)')
    parser.add_argument('--top_k', type=int, default=20,
                       help='Number of active features per spatial position')
    parser.add_argument('--cumulative_threshold', type=float, default=0.8,
                       help='GradCAM cumulative threshold for channel selection')
    
    # Training parameters
    parser.add_argument('--data_dir', type=str, default='data/imagenette',
                       help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    
    # Loss weights
    parser.add_argument('--lambda_l1', type=float, default=3.0,
                       help='L1 sparsity penalty')
    parser.add_argument('--lambda_lateral', type=float, default=0.01,
                       help='Lateral inhibition penalty')
    parser.add_argument('--lambda_compact', type=float, default=0.01,
                       help='Spatial compactness penalty')
    parser.add_argument('--lambda_channel_sparsity', type=float, default=0.003,
                       help='Feature-channel sparsity penalty')
    
    # Output paths
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory for models and visualizations')
    
    args = parser.parse_args()
    
    # List architectures and exit
    if args.list_architectures:
        list_available_architectures()
        return
    
    # Create model config
    print("="*80)
    print("Multi-Channel ConvSAE Training (Flexible Architecture)")
    print("="*80)
    
    config = ModelConfig(
        architecture=args.architecture,
        target_layer=args.target_layer,
        hidden_dim_multiplier=args.hidden_multiplier,
        top_k=args.top_k,
        cumulative_threshold=args.cumulative_threshold
    )
    
    print(f"\n{config}\n")
    
    # Setup data
    print("Setting up data...")
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(root=args.data_dir, transform=data_transform)
    data_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    
    print(f"Dataset: {len(full_dataset)} images")
    
    # Extract activations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    extractor = FlexibleActivationExtractor(config, device=device)
    X = extractor.collect_activation_maps(data_loader, normalize=True)
    
    # Setup training
    print("\n" + "="*80)
    print("Setting up Multi-Channel ConvSAE training...")
    print("="*80)
    
    csae_model = MultiChannelConvSAE(
        in_channels=config.input_channels,
        hidden_dim=config.hidden_dim,
        kernel_size=1,
        top_k=config.top_k
    ).to(device)
    
    optimizer = optim.Adam(csae_model.parameters(), lr=args.lr, weight_decay=1e-5)
    lat_inhib_loss = LateralInhibitionLoss().to(device)
    compact_loss_fn = SpatialCompactnessLoss().to(device)
    channel_sparsity_loss_fn = FeatureChannelSparsityLoss().to(device)
    
    dataset = TensorDataset(X)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    logs = {
        "total_loss": [], "recon_loss": [], "l1_loss": [],
        "lateral_loss": [], "compact_loss": [], "channel_sparsity_loss": [],
        "active_pct": []
    }
    
    # Training loop
    print("\nStarting Training...")
    print("="*80)
    
    for epoch in range(args.epochs):
        epoch_metrics = {k: 0 for k in logs.keys()}
        n_batches = 0
        
        for batch_idx, (batch_acts,) in enumerate(train_loader):
            batch_acts = batch_acts.to(device)
            
            optimizer.zero_grad()
            
            reconstruction, sparse_features = csae_model(batch_acts, use_topk=True)
            
            loss_recon = F.mse_loss(reconstruction, batch_acts)
            loss_l1 = sparse_features.abs().mean()
            loss_lateral = lat_inhib_loss(sparse_features)
            loss_compact = compact_loss_fn(sparse_features)
            loss_channel_sparsity = channel_sparsity_loss_fn(csae_model.encoder.weight)
            
            loss = (loss_recon +
                   args.lambda_l1 * loss_l1 +
                   args.lambda_lateral * loss_lateral +
                   args.lambda_compact * loss_compact +
                   args.lambda_channel_sparsity * loss_channel_sparsity)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(csae_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            csae_model.normalize_decoder_weights()
            
            with torch.no_grad():
                active_pct = (sparse_features > 0).float().mean().item() * 100
                
                logs["total_loss"].append(loss.item())
                logs["recon_loss"].append(loss_recon.item())
                logs["l1_loss"].append(loss_l1.item())
                logs["lateral_loss"].append(loss_lateral.item())
                logs["compact_loss"].append(loss_compact.item())
                logs["channel_sparsity_loss"].append(loss_channel_sparsity.item())
                logs["active_pct"].append(active_pct)
                
                for k in epoch_metrics.keys():
                    epoch_metrics[k] += logs[k][-1]
                n_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"\rEpoch {epoch+1}/{args.epochs} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | Recon: {loss_recon.item():.4f} | "
                      f"Active: {active_pct:.1f}%", end="")
        
        avg_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
        print(f"\n[Epoch {epoch+1}] Loss: {avg_metrics['total_loss']:.4f} | "
              f"Recon: {avg_metrics['recon_loss']:.4f} | "
              f"Active: {avg_metrics['active_pct']:.2f}%")
    
    # Save outputs
    print("\n" + "="*80)
    print("Saving models...")
    
    output_name = f"multichannel_csae_{args.architecture}_{args.target_layer.replace('.', '_')}"
    os.makedirs(f'{args.output_dir}/weights', exist_ok=True)
    os.makedirs(f'{args.output_dir}/training_info', exist_ok=True)
    
    # Save model
    torch.save(csae_model.state_dict(), f'{args.output_dir}/weights/{output_name}_model.pth')
    joblib.dump(csae_model.cpu(), f'{args.output_dir}/weights/{output_name}_model.pkl')
    
    # Save config and logs
    training_info = {
        'config': config.to_dict(),
        'logs': logs,
        'final_metrics': avg_metrics,
        'args': vars(args)
    }
    joblib.dump(training_info, f'{args.output_dir}/training_info/{output_name}_training_info.pkl')
    
    # Plot logs
    csae_model = csae_model.to(device)
    plot_training_logs(logs, config, f'{args.output_dir}/training_info/{output_name}_logs.png')
    
    print(f"✓ Model saved to: {args.output_dir}/weights/{output_name}_model.pkl")
    print(f"✓ Training info saved to: {args.output_dir}/training_info/{output_name}_training_info.pkl")
    print("="*80)


if __name__ == "__main__":
    main()