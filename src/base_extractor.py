import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import hashlib
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional
from src.gradcam_fixed import GradCAM
from config.model_configs import ModelLayerConfig
"""
Base Activation Extractor for Multi-Channel ConvSAE

Generic extractor that works with any model/layer configuration.
Supports both masked and non-masked variants.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import hashlib
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional
import sys
sys.path.append('.')

# from src.gradcam import GradCAM


class BaseActivationExtractor:
    """
    Generic activation extractor for any model and layer.
    
    Supports:
    - Any backbone model (VGG, ResNet, EfficientNet, etc.)
    - Any target layer
    - Masked or non-masked activation collection
    - Caching for faster repeated runs
    """
    
    def __init__(
        self,
        model_config,
        device='cuda',
        cumulative_threshold=0.85,
        use_masking=False
    ):
        """
        Args:
            model_config: ModelLayerConfig object from config.model_configs
            device: Device to run on
            cumulative_threshold: GradCAM threshold (default: 0.85 = 85%)
            use_masking: If True, apply GradCAM masking to activations (masked variant)
                        If False, collect ALL channels (masked loss variant)
        """
        self.config = model_config
        self.device = device
        self.cumulative_threshold = cumulative_threshold
        self.use_masking = use_masking
        
        # Load model
        print(f"Loading {model_config.model_name} model...")
        self.model = model_config.model_loader().to(device)
        self.model.eval()
        
        # Get target layer
        self.target_layer = model_config.layer_getter(self.model, model_config.layer_path)
        
        print(f"Target Layer: {model_config.layer_path}")
        print(f"  Output channels: {model_config.num_channels}")
        print(f"  Spatial resolution: {model_config.spatial_size[0]}×{model_config.spatial_size[1]}")
        print(f"  Masking mode: {'MASKED (pre-masking activations)' if use_masking else 'MASKED LOSS (no pre-masking)'}")
        
        # GradCAM for channel importance
        self.gradcam = GradCAM(self.model, self.target_layer)
        
        # Hook for activations
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)
    
    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output.detach()
    
    def _select_channels_with_gradcam(
        self,
        image: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Use GradCAM to select important channels.
        
        Args:
            image: [1, 3, 224, 224] - Input image
            class_idx: Target class (None = use predicted class)
        
        Returns:
            channel_mask: [num_channels] - Binary mask (1 for selected, 0 for others)
            num_selected: Number of selected channels
        """
        # Compute GradCAM channel weights
        weights, _, pred_class = self.gradcam.forward(image, class_idx=class_idx, verbose=False)
        
        # Sort channels by importance
        sorted_indices = torch.argsort(weights, descending=True)
        sorted_weights = weights[sorted_indices]
        
        # Find number of channels for cumulative threshold
        total_score = sorted_weights.sum()
        if total_score > 0:
            cumsum = torch.cumsum(sorted_weights / total_score, dim=0)
            num_selected = (cumsum < self.cumulative_threshold).sum().item() + 1
            num_selected = min(num_selected, len(sorted_indices))
        else:
            num_selected = max(1, int(0.1 * len(sorted_indices)))
        
        # Create binary mask
        channel_mask = torch.zeros(self.config.num_channels, dtype=torch.bool, device=self.device)
        selected_channels = sorted_indices[:num_selected]
        channel_mask[selected_channels] = True
        
        return channel_mask, num_selected
    
    def _get_cache_filename(self, data_loader: DataLoader, normalize: bool) -> str:
        """Generate unique cache filename based on configuration."""
        config_str = (f"{self.config.model_name}_{self.config.layer_path}_"
                     f"threshold{self.cumulative_threshold}_"
                     f"masking{self.use_masking}_"
                     f"normalize{normalize}_"
                     f"nsamples{len(data_loader.dataset)}")
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        cache_dir = "cache_activations"
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_filename = os.path.join(
            cache_dir,
            f"{self.config.model_name}_{self.config.layer_path.replace('.', '_')}_{config_hash}.pt"
        )
        return cache_filename
    
    def collect_activation_maps(
        self,
        data_loader: DataLoader,
        normalize: bool = True,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect activation maps from target layer.
        
        Two variants:
        1. MASKED (use_masking=True): Apply GradCAM mask to activations
        2. MASKED LOSS (use_masking=False): Keep ALL channels, return masks separately
        
        Args:
            data_loader: DataLoader with (image, label) pairs
            normalize: Apply robust normalization (99th percentile)
            use_cache: Use cached activations if available
        
        Returns:
            X: [N, num_channels, H, W] - Activation maps
            masks: [N, num_channels] - GradCAM masks (for masked loss variant)
                   None for masked variant
        """
        # Check cache
        if use_cache:
            cache_filename = self._get_cache_filename(data_loader, normalize)
            if os.path.exists(cache_filename):
                print(f"\n{'='*80}")
                print(f"Loading cached activations from: {cache_filename}")
                print(f"{'='*80}")
                cached_data = torch.load(cache_filename)
                X = cached_data['activations']
                masks = cached_data.get('masks', None)
                print(f"\nLoaded {X.shape[0]} activation maps:")
                print(f"  Shape: {X.shape}")
                if masks is not None:
                    print(f"  Masks shape: {masks.shape}")
                return X, masks
        
        all_activations = []
        all_masks = [] if not self.use_masking else None
        channel_selection_stats = []
        
        mode_str = "with pre-masking" if self.use_masking else "WITHOUT masking (masked loss)"
        print(f"\nCollecting activations from {self.config.model_name}.{self.config.layer_path} {mode_str}...")
        print(f"  GradCAM threshold: {self.cumulative_threshold * 100:.0f}%")
        
        for images, labels in tqdm(data_loader, desc="Extracting activations"):
            batch_activations = []
            batch_masks = [] if not self.use_masking else None
            
            for i in range(images.size(0)):
                image = images[i:i+1].to(self.device)
                
                # Forward pass to get activations
                with torch.no_grad():
                    _ = self.model(image)
                    activations = self.activations.clone()
                
                # Get GradCAM mask
                channel_mask, num_selected = self._select_channels_with_gradcam(image)
                channel_selection_stats.append(num_selected)
                
                if self.use_masking:
                    # MASKED VARIANT: Apply mask to activations
                    mask_4d = channel_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).float()
                    activations = activations * mask_4d
                    batch_activations.append(activations.cpu())
                else:
                    # MASKED LOSS VARIANT: Keep all channels, store mask separately
                    batch_activations.append(activations.cpu())
                    batch_masks.append(channel_mask.cpu())
            
            all_activations.append(torch.cat(batch_activations, dim=0))
            if batch_masks is not None:
                all_masks.append(torch.stack(batch_masks, dim=0))
        
        # Concatenate
        X = torch.cat(all_activations, dim=0)
        masks = torch.cat(all_masks, dim=0) if all_masks is not None else None
        
        # Statistics
        avg_selected = np.mean(channel_selection_stats)
        std_selected = np.std(channel_selection_stats)
        print(f"\nChannel selection statistics:")
        print(f"  Average: {avg_selected:.1f} ± {std_selected:.1f} (out of {self.config.num_channels})")
        print(f"  Min: {min(channel_selection_stats)}, Max: {max(channel_selection_stats)}")
        
        print(f"\nCollected {X.shape[0]} activation maps:")
        print(f"  Shape: {X.shape}")
        if self.use_masking:
            print(f"  Mode: MASKED (pre-masking applied)")
        else:
            print(f"  Mode: MASKED LOSS (all channels kept, masks stored separately)")
        
        # Normalize
        if normalize:
            print("\nApplying robust normalization...")
            for c in range(X.shape[1]):
                channel_data = X[:, c, :, :]
                flat = channel_data.flatten()
                non_zero_flat = flat[flat > 1e-8]
                
                if len(non_zero_flat) > 0:
                    scale_factor = torch.quantile(non_zero_flat, 0.99)
                    if scale_factor > 1e-8:
                        channel_data = torch.clamp(channel_data, min=0.0, max=scale_factor)
                        X[:, c, :, :] = channel_data / (scale_factor + 1e-8)
            
            print(f"  Normalized range: [{X.min():.4f}, {X.max():.4f}]")
        
        # Cache
        if use_cache:
            cache_filename = self._get_cache_filename(data_loader, normalize)
            print(f"\nSaving to cache: {cache_filename}")
            torch.save({'activations': X, 'masks': masks}, cache_filename)
            print("✓ Cached successfully!")
        
        return X, masks
"""
Multi-Channel ConvSAE Training Script with VGG16 Backbone (Masked Loss Variant)

VARIANT: Uses ALL activation channels as input, but computes reconstruction loss
only on the top 85% GradCAM-selected channels (masking others in loss computation).

Key Design Differences from run_csae_vgg.py:
- Input: ALL 256 activation channels (no pre-masking)
- CSAE: Processes all channels with two-level sparsity
- Loss: Reconstruction error computed ONLY on top 85% GradCAM-selected channels
- Goal: Learn from full representation, but focus reconstruction on important channels

This approach allows the model to:
1. Learn from richer input (all channels)
2. Focus reconstruction quality on class-discriminative channels
3. Potentially discover useful patterns in "less important" channels

Usage:
    python run_vgg_mask.py
"""

import torch
torch.cuda.init()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from torchvision import datasets, transforms
import joblib
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import sys
import os
import hashlib
sys.path.append('.')
from src.gradcam_fixed import GradCAM

# ==========================================
# Multi-Channel ConvSAE Architecture
# ==========================================

class MultiChannelConvSAE(nn.Module):
    """
    Convolutional Sparse Autoencoder for multi-channel input with Two-Level Sparsity.

    Uses 1×1 convolutions to learn sparse features across the channel dimension.
    Each learned feature corresponds to a specific combination of input channels.

    Two-Level Sparsity Mechanism:
        1. Channel-level sparsity (Top-K): For each sample, rank channels by their total
           spatial activation (sum over H×W), then select only top-k channels (hard selection)
        2. Spatial-level sparsity (L1): Among the selected top-k channels, apply L1
           regularization to make each channel's spatial activations sparse

    Key Features:
        - Top-K channel selection: Only the top-k channels activate per sample (hard channel sparsity)
        - L1 spatial regularization: Encourages sparse spatial activations within selected channels
        - Spatial compactness: Optional TV loss for localized feature activations

    Architecture:
        - Encoder: Conv2d(in_channels → hidden_dim, kernel_size=1×1)
        - Channel Selection: z_channels = TopK_channels(ReLU(W_enc * x + b), k)
        - Spatial Sparsity: Applied via L1 regularization during training
        - Decoder: Conv2d(hidden_dim → in_channels, kernel_size=1×1)

    Args:
        in_channels: Number of input channels (e.g., 256 for VGG16 features[16])
        hidden_dim: Number of sparse features (e.g., 2048 for 8× expansion)
        kernel_size: Convolution kernel size (default: 1 for channel-wise features)
        top_k: Number of channels to keep active per sample (default: 10)
    """

    def __init__(self, in_channels: int = 256, hidden_dim: int = 2048,
                 kernel_size: int = 1, top_k: int = 10):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.top_k = top_k

        # Encoder: Projects input channels to sparse feature space
        self.encoder = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True
        )

        # Decoder: Reconstructs input from sparse features
        self.decoder = nn.Conv2d(
            hidden_dim,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False  # No bias for decoder (standard in SAE)
        )

        # Initialize encoder with small weights
        nn.init.kaiming_normal_(self.encoder.weight, mode='fan_out', nonlinearity='relu')
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)

        # Initialize decoder weights
        nn.init.kaiming_normal_(self.decoder.weight, mode='fan_in')

    def topk_activation(self, x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """
        Apply Top-K channel selection based on spatial activation sum (hard channel sparsity).

        Two-level sparsity mechanism:
        1. Channel-level: Select top-k channels based on sum of spatial activations (H×W)
        2. Spatial-level: Keep original spatial pattern for selected channels
                         (L1 regularization in training loop enforces spatial sparsity)

        Args:
            x: [B, C, H, W] - Feature activations after ReLU
            threshold: Minimum activation value to keep (default: 0.0, disabled)

        Returns:
            x_topk: [B, C, H, W] - Sparse features with only top-k channels active
        """
        B, C, H, W = x.shape

        # 1. CHANNEL-LEVEL SPARSITY: Rank channels by total spatial activation
        # Sum over spatial dimensions (H×W) for each channel
        channel_importance = x.sum(dim=[2, 3])  # [B, C] - sum over H×W

        # Select top-k channels based on importance (per sample)
        topk_vals, topk_indices = torch.topk(channel_importance, k=self.top_k, dim=1)  # [B, k]

        # Apply threshold to importance scores if needed (optional)
        if threshold > 0:
            # Create mask for channels above threshold
            threshold_mask = topk_vals > threshold  # [B, k]
        else:
            threshold_mask = None

        # Create channel selection mask [B, C]
        channel_mask = torch.zeros(B, C, device=x.device, dtype=torch.bool)
        channel_mask.scatter_(1, topk_indices, True)  # Mark top-k channels as True

        # Apply threshold mask if specified
        if threshold_mask is not None:
            # Zero out channels that didn't meet threshold
            for b in range(B):
                valid_channels = topk_indices[b][threshold_mask[b]]
                temp_mask = torch.zeros(C, device=x.device, dtype=torch.bool)
                temp_mask[valid_channels] = True
                channel_mask[b] = temp_mask

        # 2. SPATIAL-LEVEL: Keep original spatial activations for selected channels
        # Expand mask to [B, C, H, W] for broadcasting
        channel_mask_4d = channel_mask.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]

        # Zero out non-selected channels (keep full spatial pattern for selected channels)
        result = x * channel_mask_4d.float()  # [B, C, H, W]

        return result

    def forward(self, x: torch.Tensor, use_topk: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.

        Args:
            x: [B, in_channels, H, W] - Input activation maps
            use_topk: Whether to apply top-k activation (default: True)

        Returns:
            reconstruction: [B, in_channels, H, W] - Reconstructed activation maps
            sparse_features: [B, hidden_dim, H, W] - Sparse feature activations
        """
        # Encode: Project to feature space
        features = self.encoder(x)  # [B, hidden_dim, H, W]

        # Apply ReLU
        features = F.relu(features)

        # Apply Top-K activation for hard sparsity
        if use_topk:
            sparse_features = self.topk_activation(features)  # [B, hidden_dim, H, W]
        else:
            sparse_features = features

        # Decode: Reconstruct input
        reconstruction = self.decoder(sparse_features)  # [B, in_channels, H, W]

        return reconstruction, sparse_features

    def normalize_decoder_weights(self):
        """
        Normalize decoder weights to have unit norm per feature.
        Standard practice in SAE to prevent scale ambiguity.
        """
        with torch.no_grad():
            # Decoder weight shape: [in_channels, hidden_dim, kernel_size, kernel_size]
            weight = self.decoder.weight.data

            # Compute L2 norm per output feature (across input dimension)
            # For 1×1 conv: [in_channels, hidden_dim, 1, 1]
            norm = weight.norm(p=2, dim=(0, 2, 3), keepdim=True).clamp(min=1e-8)

            # Normalize
            self.decoder.weight.data = weight / norm


class LateralInhibitionLoss(nn.Module):
    """
    Penalizes neighboring features from activating together.
    Encourages spatial diversity in feature activations.
    """

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] - Feature activations

        Returns:
            loss: Scalar - Lateral inhibition penalty
        """
        # Compute spatial autocorrelation
        # Apply Gaussian blur to features
        B, C, H, W = features.shape

        # Simple approximation: penalize correlation between neighboring spatial locations
        # Shift features and compute correlation
        feat_center = features[:, :, 1:-1, 1:-1]
        feat_left = features[:, :, 1:-1, :-2]
        feat_right = features[:, :, 1:-1, 2:]
        feat_up = features[:, :, :-2, 1:-1]
        feat_down = features[:, :, 2:, 1:-1]

        # Compute correlation
        corr = (
            (feat_center * feat_left).mean() +
            (feat_center * feat_right).mean() +
            (feat_center * feat_up).mean() +
            (feat_center * feat_down).mean()
        ) / 4.0

        return corr


class SpatialCompactnessLoss(nn.Module):
    """
    Spatial Compactness Regularization using Total Variation (TV) loss.

    Encourages feature activations to form compact, localized spatial regions
    by penalizing spatial gradients. This prevents scattered/noisy activations
    across the feature map.

    Total Variation = sum of absolute differences between neighboring pixels.

    Lower TV = smoother, more compact activations
    Higher TV = scattered, noisy activations
    """

    def __init__(self):
        super().__init__()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Total Variation loss for spatial compactness.

        Args:
            features: [B, C, H, W] - Feature activations

        Returns:
            tv_loss: Scalar - Total variation penalty
        """
        # Compute absolute differences between neighboring spatial positions

        # Horizontal differences: |f(x, y) - f(x+1, y)|
        diff_h = torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :])

        # Vertical differences: |f(x, y) - f(x, y+1)|
        diff_w = torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1])

        # Total variation = sum of all differences
        tv_loss = diff_h.mean() + diff_w.mean()

        return tv_loss


class FeatureChannelSparsityLoss(nn.Module):
    """
    Feature-Channel Sparsity Loss.

    Encourages each learned feature to respond to only a few of the input channels.
    This makes features more interpretable by ensuring they correspond to specific
    combinations of input channels, rather than using all channels.

    Similar to how SAE features in LLM interpretability are encouraged to activate
    for specific token patterns, we encourage features to activate for specific
    channel combinations.

    Implementation:
        For each feature (row in encoder weight matrix), compute L1 norm across
        input channels. Penalize features that have large L1 norms (use many channels).
    """

    def __init__(self):
        super().__init__()

    def forward(self, encoder_weight: torch.Tensor) -> torch.Tensor:
        """
        Compute feature-channel sparsity loss.

        Args:
            encoder_weight: [out_channels, in_channels, k, k] - Encoder weights
                           For 1×1 conv: [hidden_dim, in_channels, 1, 1]

        Returns:
            sparsity_loss: Scalar - Feature-channel sparsity penalty
        """
        # Squeeze spatial dimensions for 1×1 conv: [hidden_dim, in_channels]
        weight = encoder_weight.squeeze()

        # For each feature (row), compute L1 norm across input channels
        # This measures how many input channels each feature uses
        feature_channel_usage = weight.abs().sum(dim=1)  # [hidden_dim]

        # Penalize features that use many channels
        # Mean over all features
        sparsity_loss = feature_channel_usage.mean()

        return sparsity_loss


# ==========================================
# VGG16 Activation Extractor (NO MASKING)
# ==========================================


# ==========================================
# Masked Reconstruction Loss
# ==========================================

def masked_reconstruction_loss(reconstruction: torch.Tensor,
                               target: torch.Tensor,
                               masks: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE reconstruction loss only on GradCAM-selected channels.

    Args:
        reconstruction: [B, C, H, W] - Reconstructed activation maps
        target: [B, C, H, W] - Target activation maps
        masks: [B, C] - Binary masks (1 for selected channels, 0 for others)

    Returns:
        loss: Scalar - Masked MSE loss
    """
    # Expand masks to [B, C, 1, 1] for broadcasting
    masks_4d = masks.unsqueeze(2).unsqueeze(3).float()  # [B, C, 1, 1]

    # Compute squared error
    squared_error = (reconstruction - target) ** 2  # [B, C, H, W]

    # Apply mask (only compute error on selected channels)
    masked_squared_error = squared_error * masks_4d  # [B, C, H, W]

    # Compute mean over selected elements
    # Count number of selected elements per sample
    num_selected = masks.sum(dim=1, keepdim=True).float()  # [B, 1]
    num_selected = num_selected.clamp(min=1.0)  # Avoid division by zero

    # Sum over channels, spatial dims, then normalize by number of selected channels
    # [B, C, H, W] -> [B]
    loss_per_sample = masked_squared_error.sum(dim=(1, 2, 3)) / (num_selected.squeeze() * reconstruction.shape[2] * reconstruction.shape[3])

    # Mean over batch
    loss = loss_per_sample.mean()

    return loss


# ==========================================
# Visualization Functions
# ==========================================

def plot_training_logs(logs: Dict[str, List], save_path: str = 'multichannel_csae_vgg_mask_logs.png'):
    """Plot training metrics."""
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Multi-Channel ConvSAE Training (VGG16 - Masked Loss Variant)',
                 fontsize=14, fontweight='bold')

    # Row 1: Main losses
    # Reconstruction loss
    axs[0, 0].plot(logs["recon_loss"], color='blue', linewidth=1.5)
    axs[0, 0].set_title("Masked Reconstruction Loss")
    axs[0, 0].set_ylabel("MSE (masked)")
    axs[0, 0].set_xlabel("Batch")
    axs[0, 0].grid(True, alpha=0.3)

    # L1 sparsity loss
    axs[0, 1].plot(logs["l1_loss"], color='green', linewidth=1.5)
    axs[0, 1].set_title("L1 Sparsity Loss (Feature Activation)")
    axs[0, 1].set_ylabel("L1")
    axs[0, 1].set_xlabel("Batch")
    axs[0, 1].grid(True, alpha=0.3)

    # Channel sparsity loss
    axs[0, 2].plot(logs["channel_sparsity_loss"], color='purple', linewidth=1.5)
    axs[0, 2].set_title("Channel Sparsity Loss (Encoder Weights)")
    axs[0, 2].set_ylabel("L1 per feature")
    axs[0, 2].set_xlabel("Batch")
    axs[0, 2].grid(True, alpha=0.3)

    # Row 2: Regularization losses
    # Lateral inhibition loss
    axs[1, 0].plot(logs["lateral_loss"], color='orange', linewidth=1.5)
    axs[1, 0].set_title("Lateral Inhibition Loss")
    axs[1, 0].set_ylabel("Correlation")
    axs[1, 0].set_xlabel("Batch")
    axs[1, 0].grid(True, alpha=0.3)

    # Spatial compactness loss
    axs[1, 1].plot(logs["compact_loss"], color='red', linewidth=1.5)
    axs[1, 1].set_title("Spatial Compactness Loss (TV)")
    axs[1, 1].set_ylabel("Total Variation")
    axs[1, 1].set_xlabel("Batch")
    axs[1, 1].grid(True, alpha=0.3)

    # Active neurons percentage (channel-level sparsity)
    axs[1, 2].plot(logs["active_pct"], color='teal', linewidth=1.5)
    axs[1, 2].set_title("Active Channels % (Top-K Channel Selection)")
    axs[1, 2].set_ylabel("Percent (%)")
    axs[1, 2].set_xlabel("Batch")
    axs[1, 2].set_ylim(0, 10)
    axs[1, 2].grid(True, alpha=0.3)

    # Row 3: Summary metrics
    # Total loss
    axs[2, 0].plot(logs["total_loss"], color='black', linewidth=2)
    axs[2, 0].set_title("Total Loss")
    axs[2, 0].set_ylabel("Loss")
    axs[2, 0].set_xlabel("Batch")
    axs[2, 0].grid(True, alpha=0.3)

    # Loss components (log scale)
    axs[2, 1].plot(logs["recon_loss"], label='Recon (masked)', alpha=0.7)
    axs[2, 1].plot(logs["l1_loss"], label='L1', alpha=0.7)
    axs[2, 1].plot(logs["lateral_loss"], label='Lateral', alpha=0.7)
    axs[2, 1].plot(logs["compact_loss"], label='Compact', alpha=0.7)
    axs[2, 1].plot(logs["channel_sparsity_loss"], label='Ch-Sp', alpha=0.7)
    axs[2, 1].set_title("Loss Components (Log Scale)")
    axs[2, 1].set_ylabel("Loss")
    axs[2, 1].set_xlabel("Batch")
    axs[2, 1].set_yscale('log')
    axs[2, 1].legend(fontsize=7)
    axs[2, 1].grid(True, alpha=0.3)

    # Reconstruction vs Channel Sparsity trade-off
    axs[2, 2].scatter(logs["channel_sparsity_loss"], logs["recon_loss"],
                     c=range(len(logs["recon_loss"])), cmap='viridis',
                     alpha=0.5, s=5)
    axs[2, 2].set_title("Reconstruction vs Channel Sparsity")
    axs[2, 2].set_xlabel("Channel Sparsity Loss")
    axs[2, 2].set_ylabel("Masked Reconstruction Loss")
    axs[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training logs saved to {save_path}")
    plt.close()


def visualize_learned_features(model: MultiChannelConvSAE,
                               num_features: int = 32,
                               save_path: str = 'multichannel_csae_vgg_mask_features.png'):
    """Visualize learned decoder features."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Multi-Channel ConvSAE Learned Features (VGG16 - Masked Loss)', fontsize=14, fontweight='bold')

    # Get decoder weights [in_channels, hidden_dim, kernel_size, kernel_size]
    decoder_weights = model.decoder.weight.detach().cpu()

    # For 1×1 conv, shape is [in_channels, hidden_dim, 1, 1]
    # Each feature is a in_channels-dimensional vector
    decoder_weights = decoder_weights.squeeze()  # [in_channels, hidden_dim]

    # 1. Decoder weight distribution
    weights_flat = decoder_weights.flatten().numpy()
    axes[0, 0].hist(weights_flat, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(weights_flat), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(weights_flat):.3f}')
    axes[0, 0].set_title('Decoder Weight Distribution')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Feature L2 norms (how "strong" each feature is)
    feature_norms = decoder_weights.norm(dim=0).numpy()  # [hidden_dim]
    n_show = min(num_features, len(feature_norms))
    axes[0, 1].bar(range(n_show), feature_norms[:n_show], color='green', alpha=0.7)
    axes[0, 1].set_title(f'Feature Magnitudes (Top {n_show})')
    axes[0, 1].set_xlabel('Feature Index')
    axes[0, 1].set_ylabel('L2 Norm')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Channel usage distribution (which input channels are most important)
    channel_importance = decoder_weights.abs().sum(dim=1).numpy()  # [in_channels]
    axes[1, 0].bar(range(len(channel_importance)), channel_importance, color='orange', alpha=0.7)
    axes[1, 0].set_title('Input Channel Importance (Sum of Absolute Weights)')
    axes[1, 0].set_xlabel(f'Input Channel Index (0-{len(channel_importance)-1})')
    axes[1, 0].set_ylabel('Importance')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Feature sparsity (how many input channels each feature uses)
    # Count non-zero weights per feature
    feature_sparsity = (decoder_weights.abs() > 1e-3).float().sum(dim=0).numpy()  # [hidden_dim]
    axes[1, 1].hist(feature_sparsity, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(feature_sparsity), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(feature_sparsity):.1f}')
    axes[1, 1].set_title('Feature Sparsity (# Input Channels Used)')
    axes[1, 1].set_xlabel('Number of Active Input Channels')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Feature visualization saved to {save_path}")
    plt.close()


# ==========================================
# Main Training Script
# ==========================================

