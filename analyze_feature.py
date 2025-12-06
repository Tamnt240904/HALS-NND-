"""
Feature Analyzer for Multi-Channel ConvSAE (ResNet50)

Phân tích ngược: từ 1 feature ID → decoder weights → top-k channels

Cho một feature ID (0-8191), script này sẽ:
1. Trích xuất decoder weights của feature đó (1024 giá trị)
2. Vẽ histogram phân phối weights
3. Chọn top-k channels theo |weight|
4. Load activation maps mẫu của các channels đó
5. Visualize theo grid 3×3 với feature ở giữa

Usage:
    python analyze_feature.py --feature_id 137 --top_k 8
    python analyze_feature.py --feature_id 2048 --top_k 16
    python analyze_feature.py --feature_id 137 --image_path data/imagenette/tench/n01440764_1.JPEG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.append('.')
from run_multichannel_csae_resnet50 import MultiChannelConvSAE, ResNet50ActivationExtractor
from src.gradcam import GradCAM


class FeatureAnalyzer:
    """
    Analyzer cho learned features của Multi-Channel ConvSAE.
    
    Pipeline:
    1. Load trained CSAE model
    2. Extract decoder weights cho feature ID
    3. Rank channels by |weight|
    4. Load sample activations để visualize
    """
    
    def __init__(self, 
                 csae_model_path: str = 'multichannel_csae_resnet50_model.pkl',
                 device: str = 'cuda'):
        """
        Args:
            csae_model_path: Path to trained CSAE model
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load CSAE model
        print(f"Loading Multi-Channel ConvSAE from {csae_model_path}...")
        self.csae_model = joblib.load(csae_model_path).to(self.device)
        self.csae_model.eval()
        
        print(f"  ✓ Model loaded:")
        print(f"    - Input channels: {self.csae_model.in_channels}")
        print(f"    - Hidden dim: {self.csae_model.hidden_dim}")
        print(f"    - Top-K: {self.csae_model.top_k}")
        
        # Load ResNet50 for activation extraction
        print("Loading ResNet50 backbone...")
        self.resnet = models.resnet50(pretrained=True).to(self.device)
        self.resnet.eval()
        
        # Setup activation extraction
        self.layer3_activations = None
        self.resnet.layer3.register_forward_hook(self._save_layer3_activation)
        
        # GradCAM for channel selection
        self.gradcam = GradCAM(self.resnet, self.resnet.layer3)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✓ Analyzer ready!\n")
    
    def _save_layer3_activation(self, module, input, output):
        """Hook to save layer3 activations."""
        self.layer3_activations = output.detach()
    
    def get_feature_weights(self, feature_id: int, use_encoder: bool = True) -> Dict:
        """
        Trích xuất weights cho một feature ID.
        
        Args:
            feature_id: Feature index
            use_encoder: 
                - True: Dùng Encoder weights (phân tích Detection/Kích thích)
                - False: Dùng Decoder weights (phân tích Reconstruction/Tái tạo)
        """
        if feature_id < 0 or feature_id >= self.csae_model.hidden_dim:
            raise ValueError(f"Feature ID must be in [0, {self.csae_model.hidden_dim-1}]")
        
        if use_encoder:
            # Encoder shape: [hidden_dim, in_channels, 1, 1]
            # Lấy hàng feature_id: [1024]
            weights_tensor = self.csae_model.encoder.weight.detach().cpu()
            feature_weights = weights_tensor[feature_id, :, 0, 0].numpy()
            print("Using Encoder weights (Detection Analysis)")
        else:
            # Decoder shape: [in_channels, hidden_dim, 1, 1]
            # Lấy cột feature_id: [1024]
            weights_tensor = self.csae_model.decoder.weight.detach().cpu()
            feature_weights = weights_tensor[:, feature_id, 0, 0].numpy()
            print("Using Decoder weights (Reconstruction Analysis)")
            
        # --- PHẦN QUAN TRỌNG: TÍNH TOÁN ĐẦY ĐỦ CÁC CHỈ SỐ STATS ---
        stats = {
            'mean': float(np.mean(feature_weights)),
            'std': float(np.std(feature_weights)),  # Đây là dòng bị thiếu gây lỗi
            'min': float(np.min(feature_weights)),
            'max': float(np.max(feature_weights)),
            'abs_max': float(np.max(np.abs(feature_weights))),
            'non_zero_count': int(np.sum(np.abs(feature_weights) > 1e-5)),
            'sparsity': float(1.0 - np.sum(np.abs(feature_weights) > 1e-5) / len(feature_weights)),
        }
        
        return {
            'feature_id': feature_id,
            'weights': feature_weights,
            'stats': stats,
        }
        
    def get_top_k_channels(self, feature_weights: np.ndarray, k: int = 8) -> List[Tuple[int, float]]:
        """
        Lấy top-k channels theo absolute weight.
        
        Args:
            feature_weights: [1024] weights
            k: Number of top channels
        
        Returns:
            List of (channel_idx, weight) sorted by |weight| descending
        """
        abs_weights = np.abs(feature_weights)
        top_k_indices = np.argsort(abs_weights)[::-1][:k]
        
        return [(int(idx), float(feature_weights[idx])) for idx in top_k_indices]
    
    def extract_channel_activations(self, image_path: str) -> torch.Tensor:
        """
        Extract layer3 activations cho một image.
        
        Args:
            image_path: Path to image
        
        Returns:
            activations: [1, 1024, 14, 14]
        """
        # Load and preprocess
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract activations
        with torch.no_grad():
            _ = self.resnet(image_tensor)
            activations = self.layer3_activations.clone()
        
        # Apply GradCAM channel selection (same as training)
        channel_mask, num_selected, _ = self._select_channels_with_gradcam(image_tensor)
        mask_4d = channel_mask.view(1, 1024, 1, 1).float()
        activations = activations * mask_4d
        
        # Normalize (same as training)
        activations = self._normalize_activations(activations)
        
        return activations.cpu()
    
    def _select_channels_with_gradcam(self, image: torch.Tensor, threshold: float = 0.8):
        """GradCAM channel selection (same as training)."""
        weights, _, _ = self.gradcam.forward(image, class_idx=None, verbose=False)
        
        sorted_indices = torch.argsort(weights, descending=True)
        sorted_weights = weights[sorted_indices]
        
        total_score = sorted_weights.sum()
        if total_score > 0:
            cumsum = torch.cumsum(sorted_weights / total_score, dim=0)
            num_selected = (cumsum < threshold).sum().item() + 1
            num_selected = min(num_selected, len(sorted_indices))
        else:
            num_selected = max(1, int(0.1 * len(sorted_indices)))
        
        channel_mask = torch.zeros(1024, dtype=torch.bool, device=self.device)
        channel_mask[sorted_indices[:num_selected]] = True
        
        return channel_mask, num_selected, weights
    
    def _normalize_activations(self, acts: torch.Tensor) -> torch.Tensor:
        """Normalize activations (same as training)."""
        normalized = acts.clone()
        
        for c in range(acts.shape[1]):
            channel_data = acts[0, c, :, :]
            if channel_data.abs().sum() < 1e-8:
                continue
            
            non_zero_vals = channel_data[channel_data > 1e-8]
            if len(non_zero_vals) > 0:
                scale_factor = torch.quantile(non_zero_vals, 0.99)
                if scale_factor > 1e-8:
                    channel_data = torch.clamp(channel_data, min=0.0, max=scale_factor)
                    normalized[0, c, :, :] = channel_data / (scale_factor + 1e-8)
        
        return normalized
    
    def compute_feature_activation(self, 
                                   channel_activations: torch.Tensor,
                                   feature_weights: np.ndarray,
                                   feature_id: int) -> torch.Tensor:
        """
        Tính feature activation map từ channel activations và weights.
        
        Feature = Σ(weight[i] × channel[i]) for i in all channels
        
        Args:
            channel_activations: [1, 1024, 14, 14]
            feature_weights: [1024]
            feature_id: Feature index
        
        Returns:
            feature_map: [14, 14]
        """
        # Convert weights to tensor
        weights_tensor = torch.from_numpy(feature_weights).float()
        weights_tensor = weights_tensor.view(1, 1024, 1, 1)  # [1, 1024, 1, 1]
        
        # Weighted sum across channels
        feature_map = (channel_activations * weights_tensor).sum(dim=1).squeeze()  # [14, 14]
        
        return feature_map
    
    def visualize_feature_analysis(self,
                                   feature_id: int,
                                   image_path: str,
                                   top_k: int = 8,
                                   save_path: str = None):
        """
        Main visualization function.
        
        Tạo 2 plots:
        1. Histogram phân phối weights (all 1024 channels)
        2. Grid 3×3 showing top-k channels + feature map ở giữa
        
        Args:
            feature_id: Feature to analyze (0-8191)
            image_path: Path to sample image
            top_k: Number of top channels to show
            save_path: Where to save visualization
        """
        print(f"="*80)
        print(f"Analyzing Feature {feature_id}")
        print(f"="*80)
        
        # 1. Get feature weights
        print(f"\n1. Extracting decoder weights for feature {feature_id}...")
        weight_data = self.get_feature_weights(feature_id)
        weights = weight_data['weights']
        stats = weight_data['stats']
        
        print(f"  ✓ Statistics:")
        print(f"    - Mean: {stats['mean']:.6f}")
        print(f"    - Std: {stats['std']:.6f}")
        print(f"    - Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"    - Non-zero: {stats['non_zero_count']}/1024 ({(1-stats['sparsity'])*100:.1f}%)")
        
        # 2. Get top-k channels
        print(f"\n2. Finding top-{top_k} channels by |weight|...")
        top_channels = self.get_top_k_channels(weights, k=top_k)
        
        print(f"  ✓ Top-{top_k} channels:")
        for i, (ch_idx, weight) in enumerate(top_channels[:5]):
            print(f"    #{i+1}: Channel {ch_idx:4d} | weight = {weight:+.6f}")
        if len(top_channels) > 5:
            print(f"    ... and {len(top_channels)-5} more")
        
        # 3. Extract channel activations from sample image
        print(f"\n3. Extracting channel activations from image...")
        print(f"  Image: {image_path}")
        channel_acts = self.extract_channel_activations(image_path)
        print(f"  ✓ Activations shape: {channel_acts.shape}")
        
        # 4. Compute feature activation
        print(f"\n4. Computing feature activation map...")
        feature_map = self.compute_feature_activation(channel_acts, weights, feature_id)
        print(f"  ✓ Feature map shape: {feature_map.shape}")
        print(f"  ✓ Feature activation range: [{feature_map.min():.4f}, {feature_map.max():.4f}]")
        
        # 5. Visualize
        print(f"\n5. Creating visualization...")
        self._plot_analysis(
            feature_id=feature_id,
            weights=weights,
            stats=stats,
            top_channels=top_channels,
            channel_activations=channel_acts,
            feature_map=feature_map,
            image_path=image_path,
            save_path=save_path
        )
        
        print(f"\n{'='*80}")
        print(f"✓ Analysis complete!")
        if save_path:
            print(f"  Saved to: {save_path}")
        print(f"{'='*80}")
    
    def _plot_analysis(self,
                      feature_id: int,
                      weights: np.ndarray,
                      stats: Dict,
                      top_channels: List[Tuple[int, float]],
                      channel_activations: torch.Tensor,
                      feature_map: torch.Tensor,
                      image_path: str,
                      save_path: str = None):
        """
        Plot full analysis with histogram and 3×3 grid.
        """
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.1, 2.5], hspace=0.3)
        
        # ===== Section 1: Weight Distribution =====
        ax_hist = fig.add_subplot(gs[0])
        
        # Histogram
        counts, bins, patches = ax_hist.hist(weights, bins=60, color='steelblue', 
                                             alpha=0.7, edgecolor='navy')
        
        # Color bars based on sign
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor('#3b82f6')  # Blue for negative
            else:
                patch.set_facecolor('#10b981')  # Green for positive
        
        # Zero line
        ax_hist.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        
        # Mean line
        ax_hist.axvline(stats['mean'], color='orange', linestyle='--', 
                       linewidth=2, label=f"Mean = {stats['mean']:.4f}")
        
        # ax_hist.set_xlabel('Weight Value', fontsize=12, fontweight='bold')
        ax_hist.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax_hist.set_title(f'1. Weight Distribution for Feature {feature_id} (All 1024 Channels)',
                         fontsize=14, fontweight='bold', pad=15)
        ax_hist.legend(fontsize=10)
        ax_hist.grid(True, alpha=0.3, axis='y')
        
        # Stats text
        stats_text = (f"Mean: {stats['mean']:.4f} | Std: {stats['std']:.4f} | "
                     f"Range: [{stats['min']:.4f}, {stats['max']:.4f}] | "
                     f"Active: {stats['non_zero_count']}/1024 ({(1-stats['sparsity'])*100:.1f}%)")
        ax_hist.text(0.5, -0.15, stats_text, transform=ax_hist.transAxes,
                    fontsize=10, ha='center', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.3))
        
        # ===== Section 2: 3×3 Grid =====
        gs_grid = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[2],
                                                    hspace=0.3, wspace=0.3)
        
        # Prepare activation maps for top-k channels
        top_k_maps = []
        for ch_idx, weight in top_channels:
            ch_map = channel_activations[0, ch_idx, :, :].numpy()
            top_k_maps.append((ch_idx, weight, ch_map))
        
        # Ensure we have at least 8 for 3×3 grid
        while len(top_k_maps) < 8:
            top_k_maps.append((0, 0.0, np.zeros((14, 14))))
        
        # Layout: 
        # [0] [1] [2]
        # [3] [F] [4]  <- F = Feature map in center
        # [5] [6] [7]
        
        positions = [
            (0, 0), (0, 1), (0, 2),  # Top row
            (1, 0),         (1, 2),  # Middle row (skip center)
            (2, 0), (2, 1), (2, 2)   # Bottom row
        ]
        
        # Plot top-8 channels
        for idx, (row, col) in enumerate(positions):
            ax = fig.add_subplot(gs_grid[row, col])
            
            if idx < len(top_k_maps):
                ch_idx, weight, ch_map = top_k_maps[idx]
                
                # Plot heatmap
                im = ax.imshow(ch_map, cmap='hot', interpolation='bilinear')
                
                # Title with channel info
                sign = '+' if weight >= 0 else ''
                color = 'green' if weight >= 0 else 'red'
                ax.set_title(f"Channel {ch_idx}\nw={sign}{weight:.4f}",
                           fontsize=10, fontweight='bold', color=color)
                
                # Colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis('off')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Plot feature map in center (1, 1)
        ax_center = fig.add_subplot(gs_grid[1, 1])
        feature_map_np = feature_map.numpy()
        im_feature = ax_center.imshow(feature_map_np, cmap='hot', interpolation='bilinear')
        ax_center.set_title(f"FEATURE {feature_id}\n(Weighted Sum)",
                          fontsize=12, fontweight='bold', 
                          color='blue', bbox=dict(boxstyle='round', 
                          facecolor='lightblue', alpha=0.8))
        plt.colorbar(im_feature, ax=ax_center, fraction=0.046, pad=0.04)
        ax_center.set_xticks([])
        ax_center.set_yticks([])
        
        # Add border to center
        for spine in ax_center.spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(3)
        
        # # Main title for grid
        # fig.text(0.5, 0.37, 
        #         f'2. Top-{len(top_channels)} Channel Contributions (Feature {feature_id})',
        #         ha='center', fontsize=14, fontweight='bold')
        
        # # Image source info
        # fig.text(0.5, 0.34,
        #         f'Activation maps from: {Path(image_path).name}',
        #         ha='center', fontsize=10, style='italic', color='gray')
        
        # Overall title
        plt.suptitle(f'Multi-Channel ConvSAE Feature Analysis: Feature {feature_id}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Multi-Channel ConvSAE learned features'
    )
    parser.add_argument('--csae_model', type=str,
                       default='multichannel_csae_resnet50_model.pkl',
                       help='Path to trained CSAE model')
    parser.add_argument('--feature_id', type=int, required=True,
                       help='Feature ID to analyze (0-8191)')
    parser.add_argument('--image_path', type=str,
                       default='data/imagenette/tench/n01440764_1.JPEG',
                       help='Sample image for activation visualization')
    parser.add_argument('--top_k', type=int, default=8,
                       help='Number of top channels to visualize (default: 8)')
    parser.add_argument('--output_dir', type=str, 
                       default='feature_analysis',
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create analyzer
    print("="*80)
    print("Multi-Channel ConvSAE Feature Analyzer")
    print("="*80 + "\n")
    
    analyzer = FeatureAnalyzer(
        csae_model_path=args.csae_model,
        device=args.device
    )
    
    # Generate save path
    save_path = output_dir / f"feature_{args.feature_id}_analysis.png"
    
    # Run analysis
    analyzer.visualize_feature_analysis(
        feature_id=args.feature_id,
        image_path=args.image_path,
        top_k=args.top_k,
        save_path=str(save_path)
    )


if __name__ == "__main__":
    main()