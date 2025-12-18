"""
Fidelity Evaluation Script for Multi-Channel ConvSAE (Batch Multi-Model Support)

Evaluates multiple trained CSAE models across different architectures and layers in one run.
Automatically discovers all trained models in a directory and evaluates them.

Usage:
    # Evaluate all models in weights directory
    python evaluate_fidelity.py --weights_dir output/weights --data_dir data/subset
    
    # Evaluate specific models
    python evaluate_fidelity.py --models output/weights/model1.pkl output/weights/model2.pkl
    
    # Quick test with fewer batches
    python evaluate_fidelity.py --weights_dir output/weights --limit_batches 20
    
    # Grid search (optional, uses first config by default)
    python evaluate_fidelity.py --weights_dir output/weights \
        --threshold 0.8,0.9 --noise_threshold 3.0,4.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
import sys
import os
import joblib
import itertools
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')
from run_multichannel_csae_resnet50 import MultiChannelConvSAE

# Try to import fixed GradCAM first, fallback to original
try:
    from src.gradcam_fixed import GradCAM
except ImportError:
    from src.gradcam import GradCAM

from config import ModelConfig


class FlexibleFidelityEvaluator:
    """Fidelity evaluator for any CNN architecture and layer."""
    
    def __init__(self, 
                 csae_path: str,
                 config: ModelConfig,
                 device='cuda', 
                 cumulative_threshold=0.8, 
                 amplification_factor=1.0,
                 latent_threshold=0.01,
                 noise_threshold=3.0,
                 verbose=False):
        
        self.device = device
        self.config = config
        self.cumulative_threshold = cumulative_threshold
        self.amplification_factor = amplification_factor
        self.latent_threshold = latent_threshold 
        self.noise_threshold = noise_threshold
        self.verbose = verbose
        
        # Load CNN model and target layer
        self.model, self.target_layer = config.get_model_and_layer()
        self.model = self.model.to(device)
        self.model.eval()
        
        # Load CSAE
        csae_config = self._try_load_config(csae_path)
        self.csae = MultiChannelConvSAE(
            in_channels=csae_config['input_channels'],
            hidden_dim=csae_config['hidden_dim'],
            kernel_size=csae_config.get('kernel_size', 1),
            top_k=csae_config['top_k']
        ).to(device)
        
        self._load_weights(csae_path)
        self.csae.eval()
        
        # Validate channels
        if csae_config['input_channels'] != config.input_channels:
            raise ValueError(
                f"Channel mismatch: CSAE expects {csae_config['input_channels']} "
                f"but {config.architecture} {config.target_layer} has {config.input_channels}"
            )
        
        # Setup GradCAM
        self.gradcam = GradCAM(self.model, self.target_layer)
        self.original_feats = None
        self.patched_feats = None

    def _try_load_config(self, csae_path):
        """Load config from training_info file."""
        base_name = os.path.basename(csae_path).replace('_model.pkl', '').replace('_model.pth', '')
        # base_name = base_name.replace('model', '')
        possible_paths = [
            f'output/training_info/{base_name}_training_info.pkl',
            f'training_info/{base_name}_training_info.pkl',
        ]
        
        for info_path in possible_paths:
            if os.path.exists(info_path):
                try:
                    training_info = joblib.load(info_path)
                    return training_info.get('config', {})
                except:
                    pass
        
        # Fallback
        return {
            'input_channels': self.config.input_channels,
            'hidden_dim': self.config.hidden_dim,
            'kernel_size': 1,
            'top_k': self.config.top_k
        }

    def _load_weights(self, csae_path):
        """Load CSAE weights."""
        if csae_path.endswith('.pkl'):
            try:
                loaded_model = joblib.load(csae_path)
                if isinstance(loaded_model, MultiChannelConvSAE):
                    self.csae = loaded_model.to(self.device)
                    return
            except:
                csae_path = csae_path.replace('.pkl', '.pth')
        
        if not os.path.exists(csae_path):
            raise FileNotFoundError(f"Model not found: {csae_path}")
        
        sd = torch.load(csae_path, map_location=self.device)
        if list(sd.keys())[0].startswith('module.'):
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
        self.csae.load_state_dict(sd)

    def _hook_fn(self, module, input, output):
        self.original_feats = output.detach()

    def _patching_hook_fn(self, module, input, output):
        if self.patched_feats is not None:
            return self.patched_feats 
        return output

    def robust_normalize_and_get_scale(self, x):
        """Robust normalization using max value per channel."""
        x_norm = x.clone()
        B, C, H, W = x.shape
        flat = x.view(B, C, -1)
        max_vals, _ = flat.max(dim=2, keepdim=True)
        scale = max_vals.view(B, C, 1, 1)
        scale = torch.clamp(scale, min=1e-8)
        x_norm = x / scale
        return torch.clamp(x_norm, 0, 1), scale

    def _get_gradcam_mask(self, images, cumulative_threshold=0.8):
        """Get GradCAM-based channel selection mask."""
        B = images.shape[0]
        C = self.config.input_channels
        masks = []
        
        for i in range(B):
            img_tensor = images[i:i+1]
            weights, _, _ = self.gradcam.forward(img_tensor, class_idx=None, verbose=False)
            sorted_vals, sorted_indices = torch.sort(weights, descending=True)
            total_score = sorted_vals.sum() + 1e-8
            cumsum = torch.cumsum(sorted_vals / total_score, dim=0)
            num_selected = (cumsum < cumulative_threshold).sum().item() + 1
            num_selected = min(num_selected, C)
            
            mask = torch.zeros(C, device=self.device)
            mask[sorted_indices[:num_selected]] = 1.0
            masks.append(mask)
        
        return torch.stack(masks).view(B, C, 1, 1)

    def evaluate(self, dataloader, num_batches=None):
        """Evaluate fidelity on dataset."""
        metrics = {
            "orig_loss": [], "orig_acc": [],
            "pruned_loss": [], "pruned_acc": [],
            "mse_recon": [], "sparsity_latent": [], "avg_channels_kept": []
        }
        
        criterion = nn.CrossEntropyLoss()
        C = self.config.input_channels
        
        batch_count = 0
        for batch_idx, (images, _) in enumerate(dataloader):
            if num_batches and batch_idx >= num_batches:
                break
            
            images = images.to(self.device)
            B = images.size(0)
            
            # Extract features
            handle_extract = self.target_layer.register_forward_hook(self._hook_fn)
            with torch.no_grad():
                logits_orig = self.model(images)
                preds_orig = logits_orig.argmax(dim=1)
                loss_orig = criterion(logits_orig, preds_orig)
                acc_orig = 1.0
            feats_raw = self.original_feats.clone()
            handle_extract.remove()

            # Input masking
            mask_gradcam = self._get_gradcam_mask(images, self.cumulative_threshold)
            channel_energy = feats_raw.view(B, C, -1).abs().sum(dim=2).view(B, C, 1, 1)
            mask_noise = (channel_energy > self.noise_threshold).float()
            mask_final = mask_gradcam * mask_noise
            
            num_kept_per_img = mask_final.view(B, -1).sum(dim=1)
            avg_k_final = num_kept_per_img.mean().item()
            metrics["avg_channels_kept"].append(avg_k_final)
            feats_masked = feats_raw * mask_final

            # SAE reconstruction
            with torch.no_grad():
                feats_norm, scale_factors = self.robust_normalize_and_get_scale(feats_masked)
                recon_norm, sparse_codes = self.csae(feats_norm, use_topk=True)
                recon_restored = recon_norm * scale_factors
                
                energy_in = feats_masked.view(B, C, -1).norm(p=2, dim=2, keepdim=True)
                energy_out = recon_restored.view(B, C, -1).norm(p=2, dim=2, keepdim=True)
                comp_factor = energy_in / (energy_out + 1e-8)
                comp_factor = comp_factor.view(B, C, 1, 1)
                
                recon_compensated = recon_restored * comp_factor * self.amplification_factor
                
                metrics["sparsity_latent"].append(
                    (sparse_codes.abs() > self.latent_threshold).float().mean().item() * 100
                )
                metrics["mse_recon"].append(F.mse_loss(recon_compensated, feats_raw).item())
                self.patched_feats = recon_compensated

            # Fidelity check
            handle_patch = self.target_layer.register_forward_hook(self._patching_hook_fn)
            with torch.no_grad():
                logits_pruned = self.model(images)
                loss_pruned = criterion(logits_pruned, preds_orig)
                preds_pruned = logits_pruned.argmax(dim=1)
                acc_pruned = (preds_pruned == preds_orig).float().mean().item()
                
                metrics["orig_loss"].append(loss_orig.item())
                metrics["orig_acc"].append(acc_orig)
                metrics["pruned_loss"].append(loss_pruned.item())
                metrics["pruned_acc"].append(acc_pruned)
            
            handle_patch.remove()
            self.patched_feats = None
            batch_count += 1

        return metrics

    def print_report(self, metrics):
        """Print evaluation report."""
        avg = {k: np.mean(v) for k, v in metrics.items()}
        acc_drop = (avg['orig_acc'] - avg['pruned_acc']) * 100
        input_sparsity_pct = (avg['avg_channels_kept'] / self.config.input_channels) * 100
        
        print("="*70)
        print(f"{'FIDELITY EVALUATION REPORT':^70}")
        print("="*70)
        print(f"MODEL: {self.config.architecture} - {self.config.target_layer}")
        print(f"  Channels: {self.config.input_channels}, Spatial: {self.config.spatial_size}x{self.config.spatial_size}")
        print(f"CONFIG: Threshold={self.cumulative_threshold:.2f} | "
              f"Noise={self.noise_threshold:.1f} | Amp={self.amplification_factor:.1f}")
        print("-" * 70)
        print("1. FIDELITY (Model Performance Retention)")
        print(f"   Original:     Loss={avg['orig_loss']:.4f} | Acc={avg['orig_acc']*100:.2f}%")
        print(f"   Reconstructed: Loss={avg['pruned_loss']:.4f} | Acc={avg['pruned_acc']*100:.2f}%")
        print(f"   → Accuracy Drop: {acc_drop:.2f}% (lower is better)")
        print(f"2. INPUT SPARSITY (Channel Selection)")
        print(f"   Avg Channels Kept: {avg['avg_channels_kept']:.1f} / {self.config.input_channels}")
        print(f"   → Sparsity: {100 - input_sparsity_pct:.1f}% channels removed")
        print(f"   → Active: {input_sparsity_pct:.1f}% channels")
        print("3. LATENT SPARSITY (CSAE Features)")
        print(f"   Threshold: t={self.latent_threshold}")
        print(f"   Active Features: {avg['sparsity_latent']:.2f}%")
        print("4. RECONSTRUCTION QUALITY")
        print(f"   MSE Loss: {avg['mse_recon']:.6f}")
        print("="*70 + "\n")


def discover_models(weights_dir: str) -> List[Tuple[str, ModelConfig]]:
    """
    Discover all trained CSAE models in a directory.
    
    Returns:
        List of (model_path, config) tuples
    """
    models = []
    weights_path = Path(weights_dir)
    
    if not weights_path.exists():
        return models
    
    # Find all .pkl model files
    for model_file in weights_path.glob('multichannel_csae_*.pkl'):
        # Try to load corresponding training_info
        base_name = model_file.stem
        base_name = base_name.replace('_model', '')
        info_file = weights_path.parent / 'training_info' / f'{base_name}_training_info.pkl'
        
        if info_file.exists():
            try:
                training_info = joblib.load(info_file)
                config = ModelConfig.from_dict(training_info['config'])
                models.append((str(model_file), config))
            except Exception as e:
                print(f"Warning: Could not load {model_file}: {e}")
        else:
            print(f"Warning: No training_info found for {model_file}")
    
    return models


def parse_float_list(arg):
    """Parse comma-separated float list."""
    if isinstance(arg, list):
        return arg
    return [float(x) for x in arg.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description='Batch evaluate multiple CSAE models across architectures'
    )
    
    # Model discovery
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--weights_dir', type=str,
                       help='Directory containing trained models (auto-discover all)')
    group.add_argument('--models', type=str, nargs='+',
                       help='Specific model paths to evaluate')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/subset',
                       help='Path to evaluation dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--limit_batches', type=int, default=100,
                       help='Limit batches per model (None = all)')
    
    # Evaluation parameters (single config or grid search)
    parser.add_argument('--threshold', type=parse_float_list, default=[0.7, 0.8, 0.9, 1.0],
                       help="GradCAM thresholds (comma-separated)")
    parser.add_argument('--noise_threshold', type=parse_float_list, default=[3.0],
                       help="Noise thresholds (comma-separated)")
    parser.add_argument('--amplify', type=parse_float_list, default=[1.0],
                       help="Amplification factors (comma-separated)")
    parser.add_argument('--latent_threshold', type=float, default=0.01,
                       help='Threshold for counting active latent features')
    
    # Output
    parser.add_argument('--save_results', type=str, default=None,
                       help='Save results to file (e.g., results.pkl)')
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        sys.exit(1)
    
    print("="*80)
    print("Multi-Model CSAE Fidelity Evaluation")
    print("="*80)
    
    # Discover models
    if args.weights_dir:
        print(f"\nDiscovering models in: {args.weights_dir}")
        model_configs = discover_models(args.weights_dir)
        print(f"Found {len(model_configs)} trained models\n")
    else:
        print(f"\nEvaluating {len(args.models)} specified models\n")
        model_configs = []
        for model_path in args.models:
            # Try to load config
            base_name = Path(model_path).stem
            base_name = base_name.replace('_model', '')
            print(f"Base name: {base_name}")
            info_paths = [
                f'output/training_info/{base_name}_training_info.pkl',
                f'training_info/{base_name}_training_info.pkl',
            ]
            
            config = None
            for info_path in info_paths:
                if os.path.exists(info_path):
                    try:
                        training_info = joblib.load(info_path)
                        config = ModelConfig.from_dict(training_info['config'])
                        break
                    except:
                        pass
            
            if config is None:
                print(f"Warning: Could not load config for {model_path}, skipping...")
                continue
            
            model_configs.append((model_path, config))
    
    if not model_configs:
        print("Error: No models found to evaluate")
        sys.exit(1)
    
    # Setup data
    print(f"Loading dataset: {args.data_dir}")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"  Images: {len(dataset)}, Batches: {len(loader)}\n")
    
    # Grid search setup
    combinations = list(itertools.product(args.threshold, args.noise_threshold, args.amplify))
    use_grid_search = len(combinations) > 1
    
    if use_grid_search:
        print(f"Grid search enabled: {len(combinations)} configurations per model")
    else:
        print(f"Single configuration: Threshold={args.threshold[0]}, "
              f"Noise={args.noise_threshold[0]}, Amp={args.amplify[0]}")
    
    print("="*80 + "\n")
    
    # Evaluate all models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = []
    
    for model_idx, (model_path, config) in enumerate(model_configs):
        print(f"\n{'='*80}")
        print(f"[{model_idx + 1}/{len(model_configs)}] Evaluating: {config.architecture} - {config.target_layer}")
        print(f"Model: {Path(model_path).name}")
        print(f"Channels: {config.input_channels}, Spatial: {config.spatial_size}x{config.spatial_size}")
        print('='*80)
        
        model_results = []
        
        for config_idx, (th, nth, amp) in enumerate(combinations):
            if use_grid_search:
                print(f"\n[Config {config_idx + 1}/{len(combinations)}] "
                      f"Threshold={th}, Noise={nth}, Amp={amp}")
            
            try:
                evaluator = FlexibleFidelityEvaluator(
                    csae_path=model_path,
                    config=config,
                    device=device,
                    cumulative_threshold=th,
                    amplification_factor=amp,
                    latent_threshold=args.latent_threshold,
                    noise_threshold=nth,
                    verbose=False
                )
                
                metrics = evaluator.evaluate(loader, num_batches=args.limit_batches)
                
                avg_acc = np.mean(metrics['pruned_acc']) * 100
                acc_drop = (1.0 - np.mean(metrics['pruned_acc'])) * 100
                
                if use_grid_search:
                    print(f"  → Accuracy: {avg_acc:.2f}% (drop: {acc_drop:.2f}%)")
                
                model_results.append({
                    'model_path': model_path,
                    'config': config.to_dict(),
                    'eval_config': (th, nth, amp),
                    'metrics': metrics,
                    'accuracy': avg_acc,
                    'acc_drop': acc_drop
                })
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if not model_results:
            print(f"  No successful evaluations for this model")
            continue
        
        # Find best config for this model
        best_result = sorted(model_results, key=lambda x: -x['accuracy'])[0]
        best_th, best_nth, best_amp = best_result['eval_config']
        
        if use_grid_search:
            print(f"\n→ Best config: Threshold={best_th}, Noise={best_nth}, Amp={best_amp}")
        
        # Print report for best config
        print()
        final_evaluator = FlexibleFidelityEvaluator(
            csae_path=model_path,
            config=config,
            device=device,
            cumulative_threshold=best_th,
            amplification_factor=best_amp,
            latent_threshold=args.latent_threshold,
            noise_threshold=best_nth,
            verbose=False
        )
        final_evaluator.print_report(best_result['metrics'])
        
        all_results.append({
            'model': model_path,
            'architecture': config.architecture,
            'layer': config.target_layer,
            'best_result': best_result,
            'all_configs': model_results
        })
    
    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Model':<35} {'Accuracy':>10} {'Acc Drop':>10} {'Channels':>10}")
    print("-"*80)
    
    for result in all_results:
        model_name = f"{result['architecture']}-{result['layer']}"
        acc = result['best_result']['accuracy']
        drop = result['best_result']['acc_drop']
        channels = result['best_result']['config']['input_channels']
        print(f"{model_name:<35} {acc:>9.2f}% {drop:>9.2f}% {channels:>10}")
    
    print("="*80)
    
    # Save results
    if args.save_results:
        joblib.dump(all_results, args.save_results)
        print(f"\n✓ Results saved to: {args.save_results}")


if __name__ == "__main__":
    main()