"""
Fidelity Evaluation Script for Multi-Channel ConvSAE on ResNet50 (Auto Grid Search & Best Select)

Evaluates the quality of feature reconstruction by the Multi-Channel ConvSAE
at a specific target layer (e.g., layer3). It implements a complete reconstruction
pipeline including dynamic input filtering, sparse encoding, and energy compensation
to measure model performance retention (Fidelity).

Key Design:
- Baseline Comparison: Direct comparison between original ResNet predictions
  and predictions using SAE-reconstructed features.
- Two-Stage Input Filtering:
  1. Semantic: GradCAM-based selection (Top-K cumulative importance).
  2. Magnitude: Noise thresholding to remove weak background signals.
- Advanced Reconstruction: Normalize -> SAE -> Denormalize -> Energy Compensation
  (Channel-wise amplification to restore signal strength).
- Metrics: Fidelity (Accuracy Drop), Input Sparsity (Channels kept),
  Latent Sparsity (Active neurons), and Reconstruction Error (MSE).

Usage for Grid Search:
    python evaluate_fidelity.py --threshold 0.8 --noise_threshold 3.0,3.5,4.0,4.5,5.0 --amplify 1.0,1.5,2.0,2.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import sys
import os
import joblib
import itertools

sys.path.append('.')
from run_multichannel_csae_resnet50 import MultiChannelConvSAE
from src.gradcam import GradCAM

class FidelityEvaluatorFinal:
    def __init__(self, csae_path, device='cuda', 
                 cumulative_threshold=0.8, 
                 amplification_factor=1.0,
                 latent_threshold=0.01,
                 noise_threshold=3.0): 
        
        self.device = device
        self.cumulative_threshold = cumulative_threshold
        self.amplification_factor = amplification_factor
        self.latent_threshold = latent_threshold 
        self.noise_threshold = noise_threshold
        
        # 1. Load ResNet50
        print(f"Loading ResNet50 on {device}...")
        self.resnet = models.resnet50(pretrained=True).to(device)
        self.resnet.eval()
        
        # 2. Config & SAE
        config = self._try_load_config()
        self.csae = MultiChannelConvSAE(
            in_channels=config['input_channels'],
            hidden_dim=config['hidden_dim'],
            kernel_size=config['kernel_size'],
            top_k=config['top_k']
        ).to(device)
        
        # 3. Weights
        self._load_weights(csae_path)
        self.csae.eval()
        
        # 4. GradCAM Setup
        self.gradcam = GradCAM(self.resnet, self.resnet.layer3)
        
        self.original_feats = None
        self.patched_feats = None

    def _try_load_config(self):
        info_path = 'training_info/multichannel_csae_resnet50_training_info.pkl'
        default = {'input_channels': 1024, 'hidden_dim': 8192, 'kernel_size': 1, 'top_k': 20}
        if os.path.exists(info_path):
            try: return joblib.load(info_path).get('config', default)
            except: pass
        return default

    def _load_weights(self, csae_path):
        if csae_path.endswith('.pkl'): csae_path = csae_path.replace('.pkl', '.pth')
        if not os.path.exists(csae_path):
            print(f"Error: {csae_path} not found."); sys.exit(1)
        try:
            sd = torch.load(csae_path, map_location=self.device)
            if list(sd.keys())[0].startswith('module.'):
                sd = {k.replace('module.', ''): v for k, v in sd.items()}
            self.csae.load_state_dict(sd)
        except Exception as e: print(f"Error: {e}"); sys.exit(1)

    def _hook_fn(self, module, input, output):
        self.original_feats = output.detach()

    def _patching_hook_fn(self, module, input, output):
        if self.patched_feats is not None: return self.patched_feats 
        return output

    def robust_normalize_and_get_scale(self, x):
        x_norm = x.clone()
        B, C, H, W = x.shape
        flat = x.view(B, C, -1)
        max_vals, _ = flat.max(dim=2, keepdim=True)
        scale = max_vals.view(B, C, 1, 1)
        scale = torch.clamp(scale, min=1e-8)
        x_norm = x / scale
        return torch.clamp(x_norm, 0, 1), scale

    def _get_gradcam_mask(self, images, cumulative_threshold=0.8):
        B = images.shape[0]
        masks = []
        for i in range(B):
            img_tensor = images[i:i+1]
            weights, _, _ = self.gradcam.forward(img_tensor, class_idx=None, verbose=False)
            sorted_vals, sorted_indices = torch.sort(weights, descending=True)
            total_score = sorted_vals.sum() + 1e-8
            cumsum = torch.cumsum(sorted_vals / total_score, dim=0)
            num_selected = (cumsum < cumulative_threshold).sum().item() + 1
            num_selected = min(num_selected, 1024)
            
            mask = torch.zeros(1024, device=self.device)
            mask[sorted_indices[:num_selected]] = 1.0
            masks.append(mask)
        return torch.stack(masks).view(B, 1024, 1, 1)

    def evaluate(self, dataloader, num_batches=None):
        metrics = {
            "orig_loss": [], "orig_acc": [],
            "pruned_loss": [], "pruned_acc": [],
            "mse_recon": [], "sparsity_latent": [], "avg_channels_kept": []
        }
        
        criterion = nn.CrossEntropyLoss()
        
        disable_tqdm = num_batches is not None and num_batches < 5
        
        for batch_idx, (images, _) in enumerate(dataloader):
            if num_batches and batch_idx >= num_batches: break
            
            images = images.to(self.device)
            B = images.size(0)
            
            # 1. ResNet Original Forward
            handle_extract = self.resnet.layer3.register_forward_hook(self._hook_fn)
            with torch.no_grad():
                logits_orig = self.resnet(images)
                preds_orig = logits_orig.argmax(dim=1)
                loss_orig = criterion(logits_orig, preds_orig)
                acc_orig = 1.0
            
            feats_raw = self.original_feats.clone()
            handle_extract.remove() 

            # 2. INPUT MASKING
            mask_gradcam = self._get_gradcam_mask(images, cumulative_threshold=self.cumulative_threshold)
            
            channel_energy = feats_raw.view(B, 1024, -1).abs().sum(dim=2).view(B, 1024, 1, 1)
            mask_noise = (channel_energy > self.noise_threshold).float()
            
            mask_final = mask_gradcam * mask_noise
            
            # Count actual kept channels
            num_kept_per_img = mask_final.view(B, -1).sum(dim=1) 
            avg_k_final = num_kept_per_img.mean().item()
            metrics["avg_channels_kept"].append(avg_k_final)

            feats_masked = feats_raw * mask_final

            # 3. SAE Pipeline
            with torch.no_grad():
                feats_norm, scale_factors = self.robust_normalize_and_get_scale(feats_masked)
                recon_norm, sparse_codes = self.csae(feats_norm, use_topk=True)
                recon_restored = recon_norm * scale_factors
                
                # Channel-wise Energy Compensation
                energy_in = feats_masked.view(B, 1024, -1).norm(p=2, dim=2, keepdim=True)
                energy_out = recon_restored.view(B, 1024, -1).norm(p=2, dim=2, keepdim=True)
                comp_factor = energy_in / (energy_out + 1e-8)
                comp_factor = comp_factor.view(B, 1024, 1, 1)
                
                # Apply Amplification
                recon_compensated = recon_restored * comp_factor * self.amplification_factor
                
                metrics["sparsity_latent"].append((sparse_codes.abs() > self.latent_threshold).float().mean().item() * 100)
                metrics["mse_recon"].append(F.mse_loss(recon_compensated, feats_raw).item())
                
                self.patched_feats = recon_compensated

            # 4. Check Fidelity
            handle_patch = self.resnet.layer3.register_forward_hook(self._patching_hook_fn)
            with torch.no_grad():
                logits_pruned = self.resnet(images)
                loss_pruned = criterion(logits_pruned, preds_orig)
                preds_pruned = logits_pruned.argmax(dim=1)
                acc_pruned = (preds_pruned == preds_orig).float().mean().item()
                
                metrics["orig_loss"].append(loss_orig.item())
                metrics["orig_acc"].append(acc_orig)
                metrics["pruned_loss"].append(loss_pruned.item())
                metrics["pruned_acc"].append(acc_pruned)
            
            handle_patch.remove()
            self.patched_feats = None

        return metrics

    def print_report(self, metrics):
        avg = {k: np.mean(v) for k, v in metrics.items()}
        acc_drop = (avg['orig_acc'] - avg['pruned_acc']) * 100
        input_sparsity_pct = (avg['avg_channels_kept'] / 1024) * 100
        
        print("\n" + "="*60)
        title = "EVALUATION REPORT (BEST CONFIG)"
        print(f"{title:^60}") 
        print("="*60)
        
        print(f"BEST CONFIG: Threshold={self.cumulative_threshold} | Noise={self.noise_threshold} | Amp={self.amplification_factor}")
        print("-" * 60)
        
        print("1. FIDELITY")
        print(f"   Original Loss: {avg['orig_loss']:.4f} | Accuracy: {avg['orig_acc']*100:.2f}%")
        print(f"   Pruned Loss:   {avg['pruned_loss']:.4f} | Accuracy: {avg['pruned_acc']*100:.2f}%")
        print(f"   --> Accuracy Drop: {acc_drop:.2f}%")
        
        print(f"\n2. INPUT STATISTICS")
        print(f"   Avg Channels Kept: {avg['avg_channels_kept']:.1f} / 1024")
        print(f"   --> Size of non-zeros: {input_sparsity_pct:.2f}%")
        
        print("\n3. LATENT SPARSITY")
        print(f"   Threshold t={self.latent_threshold}") 
        print(f"   Non-zero Activations: {avg['sparsity_latent']:.4f}%")
        
        print("\n4. RECONSTRUCTION ERROR (MSE)")
        print(f"   MSE Loss: {avg['mse_recon']:.6f}")
        print("="*60 + "\n")

def parse_float_list(arg):
    return [float(x) for x in arg.split(',')]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csae_model', type=str, default='weights/multichannel_csae_resnet50_model.pkl')
    parser.add_argument('--data_dir', type=str, default='data/subset') 
    parser.add_argument('--limit_batches', type=int, default=100) 
    
    parser.add_argument('--threshold', type=parse_float_list, default=[0.8], help="List of thresholds (comma separated)")
    parser.add_argument('--noise_threshold', type=parse_float_list, default=[3.0], help="List of noise thresholds (comma separated)")
    parser.add_argument('--amplify', type=parse_float_list, default=[1.0], help="List of amplification factors (comma separated)")
    parser.add_argument('--latent_threshold', type=float, default=0.01)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found."); sys.exit(1)

    print(f"Evaluating on dataset: {args.data_dir}")
    
    combinations = list(itertools.product(args.threshold, args.noise_threshold, args.amplify))
    print(f"Total configurations to test: {len(combinations)}")

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    results = []

    for idx, (th, nth, amp) in enumerate(combinations):
        evaluator = FidelityEvaluatorFinal(
            args.csae_model, 
            cumulative_threshold=th,
            amplification_factor=amp,
            latent_threshold=args.latent_threshold,
            noise_threshold=nth 
        )
        
        metrics = evaluator.evaluate(loader, num_batches=args.limit_batches)
        
        avg_acc = np.mean(metrics['pruned_acc']) * 100
        
        print(f"[{idx+1}/{len(combinations)}] Config(Th={th}, Noise={nth}, Amp={amp}) -> Accuracy: {avg_acc:.2f}%")
        
        results.append({
            'config': (th, nth, amp),
            'metrics': metrics,
            'accuracy': avg_acc,
            'threshold': th,
            'noise': nth,
            'amplify': amp
        })
        
    
    best_result = sorted(results, key=lambda x: (
        -x['accuracy'],  
        x['threshold'],  
        -x['noise'],     
        x['amplify']     
    ))[0] 

    print("\n" + "="*20 + " GRID SEARCH FINISHED " + "="*20)
    
    best_th, best_nth, best_amp = best_result['config']
    final_evaluator = FidelityEvaluatorFinal(
        args.csae_model, 
        cumulative_threshold=best_th,
        amplification_factor=best_amp,
        latent_threshold=args.latent_threshold,
        noise_threshold=best_nth 
    )
    final_evaluator.print_report(best_result['metrics'])