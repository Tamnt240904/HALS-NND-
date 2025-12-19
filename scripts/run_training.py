"""
Generic Multi-Channel ConvSAE Training Script

Supports any model and layer configuration.
Can run in two modes:
1. MASKED: Apply GradCAM masking to input activations
2. MASKED LOSS: Use all channels as input, compute loss only on GradCAM-selected channels

Usage:
    python scripts/run_training.py --model vgg16 --layer features.23 
    python scripts/run_training.py --model resnet50 --layer layer3 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import joblib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import Dict, List
from tqdm import tqdm
import sys
import os
from pathlib import Path

sys.path.append('.')
from config.model_configs import get_model_config, list_available_configs, print_available_configs
from src.base_extractor import BaseActivationExtractor

# Giả định các class loss và model được định nghĩa trong src hoặc dùng bản mock dưới đây
# Nếu bạn đã có file riêng, hãy import từ đó.
from src.base_extractor import (
    MultiChannelConvSAE,
    LateralInhibitionLoss,
    SpatialCompactnessLoss,
    FeatureChannelSparsityLoss,
    masked_reconstruction_loss
)

def plot_training_logs(logs: Dict[str, List], save_path: str, model_name: str, layer_path: str):
    """Plot training metrics."""
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'Multi-Channel ConvSAE Training ({model_name}.{layer_path})',
                 fontsize=14, fontweight='bold')
    
    # Row 1: Main losses
    axs[0, 0].plot(logs["recon_loss"], color='blue', linewidth=1.5)
    axs[0, 0].set_title("Reconstruction Loss")
    axs[0, 0].set_ylabel("MSE")
    axs[0, 0].set_xlabel("Batch")
    axs[0, 0].grid(True, alpha=0.3)
    
    axs[0, 1].plot(logs["l1_loss"], color='green', linewidth=1.5)
    axs[0, 1].set_title("L1 Sparsity Loss")
    axs[0, 1].set_ylabel("L1")
    axs[0, 1].set_xlabel("Batch")
    axs[0, 1].grid(True, alpha=0.3)
    
    axs[0, 2].plot(logs["channel_sparsity_loss"], color='purple', linewidth=1.5)
    axs[0, 2].set_title("Channel Sparsity Loss")
    axs[0, 2].set_ylabel("L1 per feature")
    axs[0, 2].set_xlabel("Batch")
    axs[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Regularization
    axs[1, 0].plot(logs["lateral_loss"], color='orange', linewidth=1.5)
    axs[1, 0].set_title("Lateral Inhibition Loss")
    axs[1, 0].set_ylabel("Correlation")
    axs[1, 0].set_xlabel("Batch")
    axs[1, 0].grid(True, alpha=0.3)
    
    axs[1, 1].plot(logs["compact_loss"], color='red', linewidth=1.5)
    axs[1, 1].set_title("Spatial Compactness Loss")
    axs[1, 1].set_ylabel("Total Variation")
    axs[1, 1].set_xlabel("Batch")
    axs[1, 1].grid(True, alpha=0.3)
    
    axs[1, 2].plot(logs["active_pct"], color='teal', linewidth=1.5)
    axs[1, 2].set_title("Active Channels %")
    axs[1, 2].set_ylabel("Percent (%)")
    axs[1, 2].set_xlabel("Batch")
    axs[1, 2].grid(True, alpha=0.3)
    
    # Row 3: Summary
    axs[2, 0].plot(logs["total_loss"], color='black', linewidth=2)
    axs[2, 0].set_title("Total Loss")
    axs[2, 0].set_ylabel("Loss")
    axs[2, 0].set_xlabel("Batch")
    axs[2, 0].grid(True, alpha=0.3)
    
    axs[2, 1].plot(logs["recon_loss"], label='Recon', alpha=0.7)
    axs[2, 1].plot(logs["l1_loss"], label='L1', alpha=0.7)
    axs[2, 1].plot(logs["lateral_loss"], label='Lateral', alpha=0.7)
    axs[2, 1].set_title("Loss Components")
    axs[2, 1].set_ylabel("Loss")
    axs[2, 1].set_xlabel("Batch")
    axs[2, 1].set_yscale('log')
    axs[2, 1].legend()
    axs[2, 1].grid(True, alpha=0.3)
    
    axs[2, 2].scatter(logs["channel_sparsity_loss"], logs["recon_loss"],
                     c=range(len(logs["recon_loss"])), cmap='viridis', alpha=0.5, s=5)
    axs[2, 2].set_title("Recon vs Channel Sparsity")
    axs[2, 2].set_xlabel("Channel Sparsity Loss")
    axs[2, 2].set_ylabel("Reconstruction Loss")
    axs[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training logs saved to {save_path}")
    plt.close()


def train_csae(
    model_config,
    data_loader: DataLoader,
    device: str = 'cuda',
    mode: str = 'masked_loss',
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_dim_multiplier: int = 8,
    top_k_ratio: float = 0.15,
    lambda_l1: float = 0.3,
    lambda_lat: float = 0.01,
    lambda_compact: float = 0.01,
    lambda_channel_sparsity: float = 0.0,
    output_prefix: str = None
):
    """
    Train Multi-Channel ConvSAE on any model/layer.
    """
    print("="*80)
    print(f"Multi-Channel ConvSAE Training: {model_config.model_name}.{model_config.layer_path}")
    print(f"Mode: {mode.upper()}")
    print("="*80)
    
    # Extract activations
    use_masking = (mode == 'masked')
    extractor = BaseActivationExtractor(
        model_config,
        device=device,
        cumulative_threshold=0.85,
        use_masking=use_masking
    )
    
    X, masks = extractor.collect_activation_maps(
        data_loader,
        normalize=True,
        use_cache=True
    )
    
    # Setup training
    INPUT_CHANNELS = model_config.num_channels
    HIDDEN_DIM = INPUT_CHANNELS * hidden_dim_multiplier
    TOP_K = int(HIDDEN_DIM * top_k_ratio)
    
    # Create model
    csae_model = MultiChannelConvSAE(
        in_channels=INPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        kernel_size=1,
        top_k=TOP_K
    ).to(device)
    
    optimizer = optim.Adam(csae_model.parameters(), lr=lr, weight_decay=weight_decay)
    lat_inhib_loss = LateralInhibitionLoss().to(device)
    compact_loss_fn = SpatialCompactnessLoss().to(device)
    channel_sparsity_loss_fn = FeatureChannelSparsityLoss().to(device)
    
    # DataLoader
    if mode == 'masked_loss':
        dataset = TensorDataset(X, masks)
    else:
        dataset = TensorDataset(X, torch.zeros(X.shape[0], INPUT_CHANNELS))
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Logging
    logs = {
        "total_loss": [], "recon_loss": [], "l1_loss": [],
        "lateral_loss": [], "compact_loss": [], "channel_sparsity_loss": [],
        "active_pct": []
    }
    
    print(f"\nStarting Training...")
    
    for epoch in range(epochs):
        epoch_metrics = {k: 0 for k in logs.keys()}
        n_batches = 0
        
        for batch_idx, (batch_acts, batch_masks) in enumerate(train_loader):
            batch_acts = batch_acts.to(device)
            batch_masks = batch_masks.to(device)
            
            optimizer.zero_grad()
            reconstruction, sparse_features = csae_model(batch_acts, use_topk=True)
            
            if mode == 'masked_loss':
                loss_recon = masked_reconstruction_loss(reconstruction, batch_acts, batch_masks)
            else:
                loss_recon = F.mse_loss(reconstruction, batch_acts)
            
            loss_l1 = sparse_features.abs().mean()
            loss_lateral = lat_inhib_loss(sparse_features)
            loss_compact = compact_loss_fn(sparse_features)
            loss_channel_sparsity = channel_sparsity_loss_fn(csae_model.encoder.weight)
            
            loss = (loss_recon + lambda_l1 * loss_l1 + lambda_lat * loss_lateral +
                   lambda_compact * loss_compact + lambda_channel_sparsity * loss_channel_sparsity)
            
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
                for k in epoch_metrics.keys(): epoch_metrics[k] += logs[k][-1]
                n_batches += 1
        
        avg_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_metrics['total_loss']:.4f} | Recon: {avg_metrics['recon_loss']:.4f}")

    # --- CẬP NHẬT LOGIC LƯU FILE ---
    weights_dir = Path("output/weights")
    info_dir = Path("output/training_info")
    weights_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    if output_prefix is None:
        output_prefix = f"{model_config.model_name}_{model_config.layer_path.replace('.', '_')}_csae_{mode}"
    
    # 1. Lưu Model PKL
    model_save_path = weights_dir / f"{output_prefix}_model.pkl"
    joblib.dump(csae_model.cpu(), str(model_save_path))
    
    # 2. Lưu Training Info
    training_info = {
        'config': {
            'model': model_config.model_name, 'layer': model_config.layer_path,
            'mode': mode, 'input_channels': INPUT_CHANNELS, 'hidden_dim': HIDDEN_DIM,
            'top_k': TOP_K, 'epochs': epochs,
        },
        'logs': logs, 'final_metrics': avg_metrics
    }
    info_save_path = info_dir / f"{output_prefix}_training_info.pkl"
    joblib.dump(training_info, str(info_save_path))
    
    # 3. Lưu Logs PNG
    log_img_path = info_dir / f"{output_prefix}_logs.png"
    plot_training_logs(logs, str(log_img_path), model_config.model_name, model_config.layer_path)
    
    print(f"\n✓ Saved outputs:\n  - Model: {model_save_path}\n  - Info:  {info_save_path}\n  - Logs:  {log_img_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Multi-Channel ConvSAE')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--layer', type=str, help='Layer path')
    parser.add_argument('--mode', type=str, default='masked_loss', choices=['masked', 'masked_loss'])
    parser.add_argument('--data_dir', type=str, default='data/imagenette')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--list', action='store_true', help='List configs')
    parser.add_argument('--output_prefix', type=str, default=None)
    
    args = parser.parse_args()
    if args.list: print_available_configs(); return
    if not args.model or not args.layer: print("Error: Specify --model and --layer"); return
    
    try:
        config = get_model_config(args.model, args.layer)
    except ValueError as e: print(f"Error: {e}"); return
    
    data_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = datasets.ImageFolder(root=args.data_dir, transform=data_transform)
    data_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    
    train_csae(config, data_loader, device='cuda' if torch.cuda.is_available() else 'cpu',
              mode=args.mode, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
              output_prefix=args.output_prefix)

if __name__ == "__main__":
    main()