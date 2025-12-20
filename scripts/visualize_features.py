"""
Generic Multi-Channel ConvSAE Feature Visualization

Supports any trained CSAE model (any backbone, any layer).
Now supports finetuned classifiers and batch processing.

Usage:
    # Visualize single image
    python scripts/visualize_features.py --model vgg16 --layer features.16 --csae_model vgg16_features_16_csae_masked_loss_model.pkl --image_path data/imagenette/tench/n01440764_1.JPEG
    
    # Visualize with finetuned classifier
    python scripts/visualize_features.py --model resnet50 --layer layer3 --csae_model resnet50_layer3_csae_masked_loss_model.pkl --image_path data/imagenette/tench/n01440764_1.JPEG --mode finetuned
    
    # Visualize all models
    python scripts/visualize_features.py --all --num_images 5
    
    # Visualize multiple images from class
    python scripts/visualize_features.py --model resnet50 --layer layer3 --csae_model resnet50_layer3_csae_masked_loss_model.pkl --class_name golf_ball --num_images 10

    # Visualize all classes and model 
    python scripts/visualize_features.py --all --num_images 10 --mode finetuned

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
import random

sys.path.append('.')
from config.model_configs import get_model_config, list_available_configs
from src.gradcam_fixed import GradCAM


RANDOM_SEED = 42


class GenericCSAEVisualizer:
    """
    Generic visualizer for Multi-Channel ConvSAE (any model/layer).
    """
    
    def __init__(
        self,
        model_config,
        csae_model_path: str,
        device='cuda',
        cumulative_threshold=0.85,
        classifier_path=None
    ):
        """
        Args:
            model_config: ModelLayerConfig object
            csae_model_path: Path to trained CSAE model
            device: Device to use
            cumulative_threshold: GradCAM threshold for visualization
            classifier_path: Path to finetuned classifier weights (optional)
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
        
        # Determine model type
        self.model_type = 'unknown'
        if 'vgg' in self.config.model_name:
            self.model_type = 'sequential'
        elif 'resnet' in self.config.model_name:
            self.model_type = 'resnet'
        elif 'densenet' in self.config.model_name:
            self.model_type = 'sequential'
        elif 'efficientnet' in self.config.model_name:
            self.model_type = 'sequential'
        elif 'alexnet' in self.config.model_name:
            self.model_type = 'sequential'
        
        # Load finetuned classifier if provided
        if classifier_path and Path(classifier_path).exists():
            print(f"   -> Loading finetuned classifier: {Path(classifier_path).name}")
            try:
                classifier = torch.load(classifier_path, map_location=self.device)
                if self.model_type == 'resnet':
                    self.model.fc = classifier
                else:
                    self.model.classifier = classifier
                print(f"   ✓ Finetuned classifier loaded")
            except Exception as e:
                print(f"   ⚠ Error loading classifier: {e}, using original weights")
        else:
            print(f"   -> Using Original ImageNet Weights")
        
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
    
    def visualize_features(self, image_path: str, top_k: int = 16, save_path: str = None, is_finetuned: bool = False):
        """Visualize top-k CSAE features."""
        print(f"Processing: {image_path}")
        
        results = self.extract_features(image_path, top_k=top_k)
        
        self._plot_feature_grid(
            results['image'],
            results['top_features'],
            results['num_selected_channels'],
            results['channel_weights'],
            save_path,
            is_finetuned
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
        save_path: str = None,
        is_finetuned: bool = False
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
        
        ft_status = "[Finetuned]" if is_finetuned else "[Original]"
        info_text = f"Model: {self.config.model_name} {ft_status}\n"
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
        
        title = f'Multi-Channel ConvSAE Features: {self.config.model_name}.{self.config.layer_path} {ft_status}\n'
        title += f'Top-{n_features} Activated Features ({self.config.spatial_size[0]}×{self.config.spatial_size[1]})'
        plt.suptitle(title, fontsize=13, fontweight='bold', y=0.998)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def find_trained_models(weights_dir="output/weights/original", filter_str=None):
    """Find all trained models in weights directory."""
    trained = []
    available = list_available_configs()
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        return []
    
    for pkl in weights_path.glob("*_model.pkl"):
        if filter_str and filter_str not in pkl.name:
            continue
        for m_name in available.keys():
            if pkl.name.startswith(m_name):
                for l_path in available[m_name]:
                    if l_path.replace('.', '_') in pkl.name:
                        trained.append({
                            'model': m_name,
                            'layer': l_path,
                            'path': str(pkl)
                        })
    return trained


def main():
    parser = argparse.ArgumentParser(description='Visualize CSAE Features')
    parser.add_argument('--all', action='store_true', help='Visualize all models in output/weights')
    parser.add_argument('--filter', type=str, help='Filter models (e.g. resnet)')
    parser.add_argument('--model', type=str, help='Model name (e.g., vgg16, resnet50)')
    parser.add_argument('--layer', type=str, help='Layer path (e.g., features.16, layer3)')
    parser.add_argument('--csae_model', type=str, help='Path to trained CSAE model')
    
    # Mode control
    parser.add_argument('--mode', type=str, default='both', choices=['both', 'original', 'finetuned', 'smart'],
                       help="Visualization mode: 'both' (default), 'original', 'finetuned', or 'smart' (prefer ft)")
    
    parser.add_argument('--image_path', type=str, help='Path to single image')
    parser.add_argument('--class_name', type=str, help='Class name for multiple images')
    parser.add_argument('--num_images', type=int, default=1,
                       help='Number of images per class')
    parser.add_argument('--data_dir', type=str, default='data/imagenette',
                       help='Dataset directory')
    parser.add_argument('--top_k_features', type=int, default=16,
                       help='Number of features to visualize')
    parser.add_argument('--output_dir', type=str, default='output/visualizations',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Build list of models to process
    to_visualize = []
    if args.all:
        to_visualize = find_trained_models(filter_str=args.filter)
        print(f"Found {len(to_visualize)} models to visualize.")
    else:
        if not (args.model and args.layer and args.csae_model):
            print("Error: Specify --model, --layer, --csae_model or use --all")
            return
        to_visualize = [{
            'model': args.model,
            'layer': args.layer,
            'path': args.csae_model
        }]
    
    if not to_visualize:
        print("No models to visualize.")
        return
    
    # Process each model
    for cfg in to_visualize:
        print("\n" + "="*80)
        print(f"Model: {cfg['model']}.{cfg['layer']}")
        print("="*80 + "\n")
        
        try:
            config = get_model_config(cfg['model'], cfg['layer'])
        except ValueError as e:
            print(f"Error loading config: {e}")
            continue
        
        # Determine classifier path
        csae_path_obj = Path(cfg['path'])
        finetune_filename = csae_path_obj.name.replace('_model.pkl', '_finetuned.pth')
        expected_ft_path = csae_path_obj.parent.parent / 'finetuned' / finetune_filename
        has_finetune = expected_ft_path.exists()
        
        # Determine run queue based on mode
        run_queue = []
        
        if args.mode == 'both':
            run_queue.append((None, False))  # Original
            if has_finetune:
                run_queue.append((str(expected_ft_path), True))  # Finetuned
        elif args.mode == 'original':
            run_queue.append((None, False))
        elif args.mode == 'finetuned':
            if has_finetune:
                run_queue.append((str(expected_ft_path), True))
            else:
                print(f"   [Skip] No finetuned weights found (Mode: finetuned)")
                continue
        elif args.mode == 'smart':
            if has_finetune:
                run_queue.append((str(expected_ft_path), True))
            else:
                run_queue.append((None, False))
        
        # Prepare images to process
        data_path = Path(args.data_dir)
        if not data_path.exists():
            print(f"Error: Data directory not found: {data_path}")
            continue
        
        images_to_process: List[Tuple[str, str]] = []
        
        # Case 1: Single image
        if args.image_path:
            img_path = Path(args.image_path)
            parent_name = img_path.parent.name
            class_name_for_output = parent_name if parent_name and parent_name != data_path.name else "single_image_run"
            images_to_process.append((str(img_path), class_name_for_output))
        
        # Case 2 & 3: Class-based or all classes
        else:
            if args.class_name:
                classes_to_run = [args.class_name]
            else:
                classes_to_run = [d.name for d in data_path.iterdir() if d.is_dir()]
            
            for class_name in classes_to_run:
                class_dir = data_path / class_name
                if not class_dir.exists():
                    continue
                
                all_images = [p for p in class_dir.iterdir() 
                            if p.is_file() and p.suffix.lower() in ('.jpeg', '.jpg', '.png')]
                
                if not all_images:
                    continue
                
                num_to_sample = min(args.num_images, len(all_images))
                rng = random.Random(RANDOM_SEED)
                sampled_images = rng.sample(all_images, num_to_sample)
                
                for img_path in sampled_images:
                    images_to_process.append((str(img_path), class_name))
        
        if not images_to_process:
            print("No images found to process.")
            continue
        
        # Process each classifier variant (original/finetuned)
        for clf_path, is_ft in run_queue:
            ft_label = "finetuned" if is_ft else "original"
            print(f"\n--- Processing with {ft_label} classifier ---")
            
            # Setup visualizer
            visualizer = GenericCSAEVisualizer(
                config,
                cfg['path'],
                device='cuda' if torch.cuda.is_available() else 'cpu',
                classifier_path=clf_path
            )
            
            # Process each image
            for i, (img_path_str, class_name) in enumerate(images_to_process):
                img_path = Path(img_path_str)
                
                # Output directory structure: output_dir/model_name/ft_label/class_name/
                output_dir_path = Path(args.output_dir) / cfg['model'] / ft_label / class_name
                output_dir_path.mkdir(parents=True, exist_ok=True)
                
                print(f"\nImage {i+1}/{len(images_to_process)} | Class: {class_name} | File: {img_path.name}")
                
                img_name = img_path.stem
                save_path = output_dir_path / f"{img_name}_{cfg['model']}_{cfg['layer'].replace('.', '_')}_features.png"
                
                visualizer.visualize_features(
                    str(img_path),
                    top_k=args.top_k_features,
                    save_path=str(save_path),
                    is_finetuned=is_ft
                )
    
    print(f"\n{'='*80}")
    print(f"✓ All visualizations complete!")
    print(f"  Output directory: {Path(args.output_dir)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()