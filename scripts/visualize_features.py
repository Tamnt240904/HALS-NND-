"""
Generic Multi-Channel ConvSAE Feature Visualization

Supports any trained CSAE model (any backbone, any layer).

Usage:
    # Visualize single image
    python scripts/visualize_features.py --model vgg16 --layer features.16 --csae_model vgg16_features_16_csae_masked_loss_model.pkl --image_path data/imagenette/tench/n01440764_1.JPEG
    
    # Visualize multiple images from class
    python scripts/visualize_features.py --model resnet50 --layer layer3 --csae_model resnet50_layer3_csae_masked_loss_model.pkl --class_name golf_ball --num_images 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import joblib
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import sys

sys.path.append('.')
from config.model_configs import get_model_config
from src.gradcam_fixed import GradCAM


class GenericCSAEVisualizer:
    """
    Generic visualizer for Multi-Channel ConvSAE (any model/layer).
    """
    
    def __init__(
        self,
        model_config,
        csae_model_path: str,
        device='cuda',
        cumulative_threshold=0.85
    ):
        """
        Args:
            model_config: ModelLayerConfig object
            csae_model_path: Path to trained CSAE model
            device: Device to use
            cumulative_threshold: GradCAM threshold for visualization
        """
        self.config = model_config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.cumulative_threshold = cumulative_threshold
        
        # Load CSAE
        print(f"Loading CSAE model from {csae_model_path}...")
        self.csae_model = joblib.load(csae_model_path).to(self.device)
        self.csae_model.eval()
        print(f"  ✓ Model: {self.csae_model.in_channels}→{self.csae_model.hidden_dim}, top_k={self.csae_model.top_k}")
        
        # Load backbone
        print(f"Loading {model_config.model_name} backbone...")
        self.model = model_config.model_loader().to(self.device)
        self.model.eval()
        
        # Get target layer
        self.target_layer = model_config.layer_getter(self.model, model_config.layer_path)
        print(f"  ✓ Target layer: {model_config.layer_path}")
        
        # Hook for activations
        self.layer_activations = None
        self.target_layer.register_forward_hook(self._save_activation)
        
        # GradCAM
        self.gradcam = GradCAM(self.model, self.target_layer)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✓ Visualizer ready!\n")
    
    def _save_activation(self, module, input, output):
        """Hook to save activations."""
        self.layer_activations = output.detach()
    
    def _select_channels_with_gradcam(self, image: torch.Tensor):
        """Get GradCAM channel selection (for visualization reference)."""
        weights, _, pred_class = self.gradcam.forward(image, class_idx=None, verbose=False)
        
        sorted_indices = torch.argsort(weights, descending=True)
        sorted_weights = weights[sorted_indices]
        
        total_score = sorted_weights.sum()
        if total_score > 0:
            cumsum = torch.cumsum(sorted_weights / total_score, dim=0)
            num_selected = (cumsum < self.cumulative_threshold).sum().item() + 1
        else:
            num_selected = max(1, int(0.1 * len(sorted_indices)))
        
        channel_mask = torch.zeros(self.config.num_channels, dtype=torch.bool, device=self.device)
        channel_mask[sorted_indices[:num_selected]] = True
        
        return channel_mask, num_selected, weights
    
    def _normalize_activations(self, acts: torch.Tensor) -> torch.Tensor:
        """Normalize activations (per-channel 99th percentile)."""
        normalized = acts.clone()
        
        for c in range(acts.shape[1]):
            channel_data = acts[0, c, :, :]
            if channel_data.abs().sum() < 1e-8:
                continue
            
            non_zero = channel_data[channel_data > 1e-8]
            if len(non_zero) > 0:
                scale = torch.quantile(non_zero, 0.99)
                if scale > 1e-8:
                    channel_data = torch.clamp(channel_data, min=0.0, max=scale)
                    normalized[0, c, :, :] = channel_data / (scale + 1e-8)
        
        return normalized
    
    def extract_features(self, image_path: str, top_k: int = 16) -> Dict:
        """
        Extract top-k activated CSAE features.
        
        Returns:
            Dictionary with extracted features and metadata
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract activations
        with torch.no_grad():
            _ = self.model(image_tensor)
            layer_acts = self.layer_activations.clone()
        
        # GradCAM reference
        channel_mask, num_selected, channel_weights = self._select_channels_with_gradcam(image_tensor)
        
        print(f"  GradCAM: {num_selected}/{self.config.num_channels} channels selected")
        
        # Normalize
        layer_acts_norm = self._normalize_activations(layer_acts)
        
        # CSAE forward
        with torch.no_grad():
            _, sparse_features = self.csae_model(layer_acts_norm, use_topk=True)
        
        # Get top-k features
        feature_importance = sparse_features.sum(dim=(2, 3)).squeeze()
        top_k_values, top_k_indices = torch.topk(feature_importance, k=min(top_k, len(feature_importance)))
        
        top_features = []
        for idx, imp in zip(top_k_indices, top_k_values):
            activation_map = sparse_features[0, idx, :, :].cpu()
            top_features.append((idx.item(), imp.item(), activation_map))
        
        print(f"  ✓ Extracted {len(top_features)} top features")
        
        return {
            'image': image,
            'image_tensor': image_tensor,
            'layer_acts': layer_acts.cpu(),
            'num_selected_channels': num_selected,
            'channel_mask': channel_mask.cpu(),
            'channel_weights': channel_weights.cpu(),
            'sparse_features': sparse_features.cpu(),
            'top_features': top_features
        }
    
    def visualize_features(self, image_path: str, top_k: int = 16, save_path: str = None):
        """Visualize top-k CSAE features."""
        print(f"Processing: {image_path}")
        
        results = self.extract_features(image_path, top_k=top_k)
        
        self._plot_feature_grid(
            results['image'],
            results['top_features'],
            results['num_selected_channels'],
            results['channel_weights'],
            save_path
        )
        
        print(f"✓ Visualization complete!")
        if save_path:
            print(f"  Saved to: {save_path}")
    
    def _plot_feature_grid(
        self,
        image: Image.Image,
        top_features: List[Tuple],
        num_selected_channels: int,
        channel_weights: torch.Tensor,
        save_path: str = None
    ):
        """Plot grid of features."""
        n_features = len(top_features)
        n_cols = 8
        n_rows = 1 + (n_features + 3) // 4
        
        fig = plt.figure(figsize=(24, 3.5 * n_rows))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)
        
        # Row 0: Overview
        ax_img = fig.add_subplot(gs[0, 0:2])
        ax_img.imshow(image)
        ax_img.set_title("Input Image", fontsize=12, fontweight='bold')
        ax_img.axis('off')
        
        # Info
        ax_info = fig.add_subplot(gs[0, 2:4])
        ax_info.axis('off')
        
        top_5_indices = torch.argsort(channel_weights, descending=True)[:5].tolist()
        top_5_scores = [channel_weights[i].item() for i in top_5_indices]
        
        info_text = f"Model: {self.config.model_name}\n"
        info_text += f"Layer: {self.config.layer_path}\n"
        info_text += f"Channels: {self.config.num_channels}\n\n"
        info_text += f"GradCAM Reference:\n"
        info_text += f"  Selected: {num_selected_channels}/{self.config.num_channels}\n"
        info_text += f"  Top 3 channels:\n"
        for idx, score in zip(top_5_indices[:3], top_5_scores[:3]):
            info_text += f"    #{idx}: {score:.4f}\n"
        
        ax_info.text(0.05, 0.5, info_text, fontsize=9, family='monospace',
                    verticalalignment='center', transform=ax_info.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Feature importance
        ax_bar = fig.add_subplot(gs[0, 4:])
        importances = [imp for _, imp, _ in top_features]
        feature_indices = [f"F{idx}" for idx, _, _ in top_features]
        ax_bar.bar(range(len(importances)), importances, color='steelblue', alpha=0.8)
        ax_bar.set_xlabel('Feature Index', fontsize=10)
        ax_bar.set_ylabel('Importance', fontsize=10)
        ax_bar.set_title(f'Top-{n_features} Feature Importance', fontsize=12, fontweight='bold')
        ax_bar.set_xticks(range(len(importances)))
        ax_bar.set_xticklabels(feature_indices, rotation=45, ha='right', fontsize=8)
        ax_bar.grid(True, alpha=0.3, axis='y')
        
        # Feature maps
        for i, (feat_idx, importance, activation_map) in enumerate(top_features):
            row = 1 + i // 4
            col = (i % 4) * 2
            
            if row >= n_rows:
                break
            
            ax = fig.add_subplot(gs[row, col:col+2])
            im = ax.imshow(activation_map.numpy(), cmap='hot', interpolation='bilinear')
            ax.set_title(f"Feature {feat_idx}\nImp: {importance:.2f}", fontsize=10, fontweight='bold')
            ax.axis('off')
            
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)
        
        title = f'Multi-Channel ConvSAE Features: {self.config.model_name}.{self.config.layer_path}\n'
        title += f'Top-{n_features} Activated Features ({self.config.spatial_size[0]}×{self.config.spatial_size[1]})'
        plt.suptitle(title, fontsize=13, fontweight='bold', y=0.998)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize CSAE Features')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., vgg16, resnet50)')
    parser.add_argument('--layer', type=str, required=True,
                       help='Layer path (e.g., features.16, layer3)')
    parser.add_argument('--csae_model', type=str, required=True,
                       help='Path to trained CSAE model')
    parser.add_argument('--image_path', type=str,
                       help='Path to single image')
    parser.add_argument('--class_name', type=str,
                       help='Class name for multiple images')
    parser.add_argument('--num_images', type=int, default=1,
                       help='Number of images (if using --class_name)')
    parser.add_argument('--data_dir', type=str, default='data/imagenette',
                       help='Dataset directory')
    parser.add_argument('--top_k_features', type=int, default=16,
                       help='Number of features to visualize')
    parser.add_argument('--output_dir', type=str, default='output/visualizations',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Get config
    try:
        config = get_model_config(args.model, args.layer)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Create visualizer
    visualizer = GenericCSAEVisualizer(
        config,
        args.csae_model,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Get images
    image_paths = []
    if args.image_path:
        image_paths = [args.image_path]
    elif args.class_name:
        class_dir = Path(args.data_dir) / args.class_name
        if not class_dir.exists():
            print(f"Error: Class directory not found: {class_dir}")
            return
        all_images = list(class_dir.glob('*.JPEG')) + list(class_dir.glob('*.jpg'))
        image_paths = all_images[:args.num_images]
    else:
        print("Error: Specify --image_path or --class_name")
        return
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Process images
    print(f"Processing {len(image_paths)} image(s)...\n")
    
    for i, img_path in enumerate(image_paths):
        print(f"\n{'='*80}")
        print(f"Image {i+1}/{len(image_paths)}")
        print(f"{'='*80}\n")
        
        img_name = Path(img_path).stem
        save_path = output_dir / f"{img_name}_{args.model}_{args.layer.replace('.', '_')}_features.png"
        
        visualizer.visualize_features(
            str(img_path),
            top_k=args.top_k_features,
            save_path=str(save_path)
        )
    
    print(f"\n{'='*80}")
    print(f"✓ Complete! Saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()