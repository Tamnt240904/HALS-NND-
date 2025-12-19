"""
Script to check accuracy drop between the Original Model and the CSAE Reconstructed Model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import joblib
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, Tuple, List
import sys
import os
from pathlib import Path
import pandas as pd
import random

sys.path.append('.')
from config.model_configs import get_model_config, list_available_configs
from src.base_extractor import MultiChannelConvSAE

RANDOM_SEED = 42 
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

IMAGENETTE_TO_IMAGENET = {
    'n01440764': 0, 'tench': 0,
    'n02102040': 217, 'English_springer': 217,
    'n02979186': 482, 'cassette_player': 482,
    'n03000684': 491, 'chain_saw': 491,
    'n03028079': 497, 'church': 497,
    'n03394916': 566, 'French_horn': 566,
    'n03417042': 569, 'garbage_truck': 569,
    'n03425413': 571, 'gas_pump': 571,
    'n03445777': 574, 'golf_ball': 574,
    'n03888257': 701, 'parachute': 701
}

class GenericModelWithCSAEReconstruction:
    def __init__(self, model_config, csae_model, device='cuda', classifier_path=None):
        self.config = model_config
        self.device = device
        self.model = model_config.model_loader().to(device)
        self.model.eval()
        self.csae = csae_model.to(device)
        self.csae.eval()
        
        self.target_layer = model_config.layer_getter(self.model, model_config.layer_path)
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)

        self.model_type = 'unknown'
        if 'vgg' in self.config.model_name: self.model_type = 'sequential'
        elif 'resnet' in self.config.model_name: self.model_type = 'resnet'
        elif 'densenet' in self.config.model_name: self.model_type = 'sequential'
        elif 'efficientnet' in self.config.model_name: self.model_type = 'sequential'
        elif 'alexnet' in self.config.model_name: self.model_type = 'sequential'

        if classifier_path and os.path.exists(classifier_path):
            print(f"-> Loading fine-tuned classifier from: {classifier_path}")
            state_dict = torch.load(classifier_path, map_location=device)
            try:
                if self.model_type == 'resnet':
                    self.model.fc.load_state_dict(state_dict)
                else:
                    self.model.classifier.load_state_dict(state_dict)
                print("-> Classifier weights loaded successfully!")
            except Exception as e:
                print(f"Error loading classifier weights: {e}")
                print("Continuing with original weights...")

    def _save_activation(self, module, input, output): 
        self.activations = output

    def _normalize_activations(self, acts):
        """
        Normalize activations to [0, 1].
        """
        B, C, H, W = acts.shape
        flat = acts.view(B, C, -1)
        
        if 'efficientnet' in self.config.model_name:
            # EfficientNet has unbounded activations (Swish). 
            # Extreme outliers (e.g., value=50 when mean=1) distort Min-Max scaling.
            # Use Robust Scaling (Quantiles) to focus on the main data distribution.
            min_vals = torch.quantile(flat, 0.01, dim=2, keepdim=True)
            max_vals = torch.quantile(flat, 0.99, dim=2, keepdim=True)
        else:
            min_vals = flat.min(dim=2, keepdim=True)[0]
            max_vals = flat.max(dim=2, keepdim=True)[0]
        
        range_vals = max_vals - min_vals
        range_vals[range_vals < 1e-8] = 1.0 
        
        min_vals_4d = min_vals.view(B, C, 1, 1)
        range_vals_4d = range_vals.view(B, C, 1, 1)
        
        normalized = (acts - min_vals_4d) / range_vals_4d
        return torch.clamp(normalized, 0.0, 1.0), min_vals_4d, range_vals_4d

    def _forward_to_layer(self, x):
        if self.model_type == 'resnet':
            x = self.model.conv1(x); x = self.model.bn1(x); x = self.model.relu(x); x = self.model.maxpool(x)
            if 'layer1' in self.config.layer_path: return self.model.layer1(x)
            x = self.model.layer1(x)
            if 'layer2' in self.config.layer_path: return self.model.layer2(x)
            x = self.model.layer2(x)
            if 'layer3' in self.config.layer_path: return self.model.layer3(x)
            x = self.model.layer3(x)
            return self.model.layer4(x)
        elif hasattr(self.model, 'features'):
            for name, layer in self.model.features.named_children():
                x = layer(x)
                if layer is self.target_layer: return x
            return x
        return x

    def _forward_from_layer(self, features):
        if self.model_type == 'resnet':
            if 'layer1' in self.config.layer_path: features = self.model.layer2(features); features = self.model.layer3(features); features = self.model.layer4(features)
            elif 'layer2' in self.config.layer_path: features = self.model.layer3(features); features = self.model.layer4(features)
            elif 'layer3' in self.config.layer_path: features = self.model.layer4(features)
            features = self.model.avgpool(features); features = torch.flatten(features, 1)
            return self.model.fc(features)
        elif hasattr(self.model, 'features'):
            found_target = False
            for name, layer in self.model.features.named_children():
                if not found_target:
                    if layer is self.target_layer: found_target = True
                    continue 
                features = layer(features)
            
            if 'densenet' in self.config.model_name:
                features = F.relu(features, inplace=True)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)
                return self.model.classifier(features)
            elif 'efficientnet' in self.config.model_name:
                features = self.model.avgpool(features)
                features = torch.flatten(features, 1)
                return self.model.classifier(features)
            elif self.model_type in ['sequential', 'alexnet']:
                features = self.model.avgpool(features)
                features = torch.flatten(features, 1)
                return self.model.classifier(features)
        return features

    def forward_original(self, x):
        with torch.no_grad(): return self.model(x)

    def forward_with_reconstruction(self, x):
        with torch.no_grad():
            features = self._forward_to_layer(x)
            original_acts = features.clone()
            
            normalized_acts, min_vals, range_vals = self._normalize_activations(original_acts)
            
            reconstruction, sparse_features = self.csae(normalized_acts, use_topk=False)
            reconstructed_acts = reconstruction * range_vals + min_vals
            
            logits = self._forward_from_layer(reconstructed_acts)
            
            mse = F.mse_loss(reconstructed_acts, original_acts).item()
            mse_norm = F.mse_loss(reconstruction, normalized_acts).item()
            
            relative_error = ((reconstructed_acts - original_acts).abs().mean() / (original_acts.abs().mean() + 1e-8)).item()
            sparsity = (sparse_features > 0).float().mean().item()
            
            return logits, {'mse': mse, 'mse_norm': mse_norm, 'relative_error': relative_error, 'sparsity': sparsity}

def evaluate_accuracy_drop(model, data_loader, device='cuda', imagenette_to_imagenet=None):
    local_to_imagenet_map = {}
    if imagenette_to_imagenet is not None:
        if hasattr(data_loader.dataset, 'classes'): class_names = data_loader.dataset.classes
        elif hasattr(data_loader.dataset, 'dataset'): class_names = data_loader.dataset.dataset.classes
        else: class_names = [] 

        for idx, name in enumerate(class_names):
            if name in imagenette_to_imagenet:
                local_to_imagenet_map[idx] = imagenette_to_imagenet[name]
            else:
                local_to_imagenet_map[idx] = -1

    total_samples = 0
    correct_original = 0
    correct_reconstructed = 0
    mse_list = []
    mse_norm_list = []

    for images, labels in tqdm(data_loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        if imagenette_to_imagenet is not None:
            valid_mask = []
            mapped_labels = []
            for i in range(len(labels)):
                lbl = labels[i].item()
                if local_to_imagenet_map.get(lbl, -1) != -1:
                    valid_mask.append(i)
                    mapped_labels.append(local_to_imagenet_map[lbl])
            if not valid_mask: continue 
            images = images[valid_mask]
            imagenet_labels = torch.tensor(mapped_labels, device=device)
        else:
            imagenet_labels = labels

        batch_size = images.size(0)
        
        # Forward pass
        logits_orig = model.forward_original(images)
        logits_recon, stats = model.forward_with_reconstruction(images)
        
        correct_original += (logits_orig.argmax(1) == imagenet_labels).sum().item()
        correct_reconstructed += (logits_recon.argmax(1) == imagenet_labels).sum().item()
        
        total_samples += batch_size
        mse_list.append(stats['mse'])
        mse_norm_list.append(stats['mse_norm'])

    if total_samples == 0: return {'acc_drop': 0.0}

    return {
        'total_samples': total_samples,
        'acc_original': correct_original / total_samples * 100,
        'acc_reconstructed': correct_reconstructed / total_samples * 100,
        'acc_drop': (correct_original - correct_reconstructed) / total_samples * 100,
        'avg_mse': np.mean(mse_list),
        'avg_mse_norm': np.mean(mse_norm_list)
    }

def print_results(results, model_name, layer_path):
    print(f"\nRESULTS: {model_name}.{layer_path}")
    print(f"  Samples: {results.get('total_samples', 0)}")
    print(f"  Acc Orig: {results.get('acc_original', 0):.2f}% | Acc Recon: {results.get('acc_reconstructed', 0):.2f}%")
    print(f"  -> Drop: {results.get('acc_drop', 0):.2f}%")
    
    if 'efficientnet' in model_name:
        # EfficientNet: Print Normalized MSE (Stable metric)
        print(f"  MSE: {results.get('avg_mse_norm', 0):.6f}")
    else:
        # Others (ResNet, VGG, DenseNet): Print Absolute MSE (Standard metric)
        print(f"  MSE: {results.get('avg_mse', 0):.6f}")

def find_trained_models(weights_dir="output/weights"):
    trained = []
    available = list_available_configs()
    weights_path = Path(weights_dir)
    if not weights_path.exists(): return []
    for pkl in weights_path.glob("*_model.pkl"):
        for m_name in available.keys():
            if pkl.name.startswith(m_name):
                for l_path in available[m_name]:
                    if l_path.replace('.', '_') in pkl.name:
                        trained.append({'model': m_name, 'layer': l_path, 'path': str(pkl)})
    return trained

def main():
    parser = argparse.ArgumentParser(description='Evaluate Accuracy Drop')
    parser.add_argument('--all', action='store_true', help='Evaluate all models in output/weights')
    parser.add_argument('--model', type=str)
    parser.add_argument('--layer', type=str)
    parser.add_argument('--csae_model', type=str)
    parser.add_argument('--load_classifier', type=str, help='Path to fine-tuned classifier weights')
    parser.add_argument('--data_dir', type=str, default='data/imagenette')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--output_csv', type=str, default='output/accuracy_results.csv')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    original_data_dir = args.data_dir
    if os.path.exists(os.path.join(original_data_dir, 'val')):
        args.data_dir = os.path.join(original_data_dir, 'val')
        print(f"-> Switching data_dir to: {args.data_dir}")
    elif os.path.exists(os.path.join(original_data_dir, 'train')):
        args.data_dir = os.path.join(original_data_dir, 'train')
        print(f"-> Switching data_dir to: {args.data_dir}")

    try:
        data_transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        full_dataset = datasets.ImageFolder(root=args.data_dir, transform=data_transform)
    except Exception as e:
        print(f"Error loading dataset: {e}"); return

    to_evaluate = []
    if args.all:
        to_evaluate = find_trained_models()
    else:
        if not (args.model and args.layer and args.csae_model):
            print("Error: Specify params or --all"); return
        to_evaluate = [{'model': args.model, 'layer': args.layer, 'path': args.csae_model}]

    all_results = []
    for cfg in to_evaluate:
        print(f"\nTarget: {cfg['model']}.{cfg['layer']}")
        try:
            m_cfg = get_model_config(cfg['model'], cfg['layer'])
            csae = joblib.load(cfg['path'])
            
            classifier_path = args.load_classifier if args.load_classifier else None
            
            if not classifier_path and os.path.exists(f"output/weights/{cfg['model']}_finetuned_classifier.pth"):
                 classifier_path = f"output/weights/{cfg['model']}_finetuned_classifier.pth"

            hybrid = GenericModelWithCSAEReconstruction(m_cfg, csae, device=device, classifier_path=classifier_path)
            
            indices = torch.randperm(len(full_dataset))[:args.num_samples].tolist() if args.num_samples else None
            ds = Subset(full_dataset, indices) if indices else full_dataset
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
            
            res = evaluate_accuracy_drop(hybrid, loader, device, IMAGENETTE_TO_IMAGENET)
            res.update({'model': cfg['model'], 'layer': cfg['layer']})
            all_results.append(res)
            print_results(res, cfg['model'], cfg['layer'])
        except Exception as e:
            print(f"Failed {cfg['path']}: {e}")

    if args.all and all_results:
        df = pd.DataFrame(all_results).round(4) 
        os.makedirs("output", exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\nBatch results saved to {args.output_csv}")

if __name__ == "__main__":
    main()