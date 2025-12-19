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

sys.path.append('.')
from config.model_configs import get_model_config, list_available_configs
from src.base_extractor import MultiChannelConvSAE

# Giữ nguyên mapping
IMAGENETTE_TO_IMAGENET = {
    'tench': 0, 'English_springer': 217, 'cassette_player': 482,
    'chain_saw': 491, 'church': 497, 'French_horn': 566,
    'garbage_truck': 569, 'gas_pump': 571, 'golf_ball': 574, 'parachute': 701
}

# --- Giữ nguyên Class GenericModelWithCSAEReconstruction và hàm evaluate_accuracy_drop của bạn ---
# (Tôi lược bớt phần thân hàm bên dưới để tiết kiệm không gian, bạn giữ nguyên code cũ của mình)
class GenericModelWithCSAEReconstruction:
    def __init__(self, model_config, csae_model, device='cuda'):
        self.config = model_config
        self.device = device
        self.model = model_config.model_loader().to(device)
        self.model.eval()
        self.csae = csae_model.to(device)
        self.csae.eval()
        self.target_layer = model_config.layer_getter(self.model, model_config.layer_path)
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output): self.activations = output

    def _normalize_activations(self, acts):
        B, C, H, W = acts.shape
        normalized = acts.clone()
        scale_factors = torch.ones(B, C, device=acts.device)
        for b in range(B):
            for c in range(C):
                channel_data = acts[b, c, :, :]
                if channel_data.abs().sum() < 1e-8: continue
                non_zero = channel_data[channel_data > 1e-8]
                if len(non_zero) > 0:
                    scale = torch.quantile(non_zero, 0.99)
                    if scale > 1e-8:
                        channel_data = torch.clamp(channel_data, min=0.0, max=scale)
                        normalized[b, c, :, :] = channel_data / (scale + 1e-8)
                        scale_factors[b, c] = scale
        return normalized, scale_factors

    def _forward_to_layer(self, x):
        if 'vgg' in self.config.model_name:
            layer_idx = int(self.config.layer_path.split('.')[-1])
            features = x
            for i in range(layer_idx + 1):
                features = self.model.features[i](features)
            return features
        
        elif 'resnet' in self.config.model_name:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            
            x = self.model.layer1(x)
            if 'layer1' in self.config.layer_path:
                return x
            
            x = self.model.layer2(x)
            if 'layer2' in self.config.layer_path:
                return x
            
            x = self.model.layer3(x)
            if 'layer3' in self.config.layer_path:
                return x
            
            x = self.model.layer4(x)
            return x
        
        elif 'efficientnet' in self.config.model_name:
            layer_idx = int(self.config.layer_path.split('.')[-1])
            features = x
            for i in range(layer_idx + 1):
                features = self.model.features[i](features)
            return features
        
        elif 'alexnet' in self.config.model_name:
            layer_idx = int(self.config.layer_path.split('.')[-1])
            features = x
            for i in range(layer_idx + 1):
                features = self.model.features[i](features)
            return features
        
        elif 'densenet' in self.config.model_name:
            features = self.model.features.conv0(x)
            features = self.model.features.norm0(features)
            features = self.model.features.relu0(features)
            features = self.model.features.pool0(features)
            
            if 'denseblock1' in self.config.layer_path:
                features = self.model.features.denseblock1(features)
                if self.config.layer_path.endswith('denseblock1'):
                    return features
            
            if 'transition1' in self.config.layer_path:
                features = self.model.features.transition1(features)
                if self.config.layer_path.endswith('transition1'):
                    return features
            
            if 'denseblock2' in self.config.layer_path:
                features = self.model.features.denseblock2(features)
                if self.config.layer_path.endswith('denseblock2'):
                    return features
            
            if 'transition2' in self.config.layer_path:
                features = self.model.features.transition2(features)
                if self.config.layer_path.endswith('transition2'):
                    return features
            
            if 'denseblock3' in self.config.layer_path:
                features = self.model.features.denseblock3(features)
                if self.config.layer_path.endswith('denseblock3'):
                    return features
            
            if 'transition3' in self.config.layer_path:
                features = self.model.features.transition3(features)
                if self.config.layer_path.endswith('transition3'):
                    return features
            
            if 'denseblock4' in self.config.layer_path:
                features = self.model.features.denseblock4(features)
                if self.config.layer_path.endswith('denseblock4'):
                    return features
            
            features = self.model.features.norm5(features)
            return features
        
        else:
            raise NotImplementedError(f"Model {self.config.model_name} not supported")

    def _forward_from_layer(self, features):
        if 'vgg' in self.config.model_name:
            layer_idx = int(self.config.layer_path.split('.')[-1])
            for i in range(layer_idx + 1, len(self.model.features)): features = self.model.features[i](features)
            features = self.model.avgpool(features); features = torch.flatten(features, 1)
            return self.model.classifier(features)
        elif 'resnet' in self.config.model_name:
            if 'layer1' in self.config.layer_path: features = self.model.layer2(features); features = self.model.layer3(features); features = self.model.layer4(features)
            elif 'layer2' in self.config.layer_path: features = self.model.layer3(features); features = self.model.layer4(features)
            elif 'layer3' in self.config.layer_path: features = self.model.layer4(features)
            features = self.model.avgpool(features); features = torch.flatten(features, 1)
            return self.model.fc(features)
        elif 'efficientnet' in self.config.model_name:
            layer_idx = int(self.config.layer_path.split('.')[-1])
            for i in range(layer_idx + 1, len(self.model.features)): features = self.model.features[i](features)
            features = self.model.avgpool(features); features = torch.flatten(features, 1)
            return self.model.classifier(features)
        elif 'alexnet' in self.config.model_name:
            layer_idx = int(self.config.layer_path.split('.')[-1])
            for i in range(layer_idx + 1, len(self.model.features)): features = self.model.features[i](features)
            features = torch.flatten(features, 1)
            return self.model.classifier(features)
        elif 'densenet' in self.config.model_name:
            layer_parts = self.config.layer_path.split('.')
            for part in layer_parts:
                if part.startswith('denseblock'):
                    block_idx = int(part.replace('denseblock', ''))
                    for i in range(block_idx + 1, 5): features = getattr(self.model.features, f'denseblock{i}')(features)
                elif part.startswith('transition'):
                    trans_idx = int(part.replace('transition', '')) 
                    for i in range(trans_idx + 1, 4): features = getattr(self.model.features, f'transition{i}')(features)
            features = self.model.features.norm5(features)
            features = F.relu(features, inplace=True)
            features = self.model.avgpool(features)
            features = torch.flatten(features, 1)
            return self.model.classifier(features)
        else: raise NotImplementedError(f"Model {self.config.model_name} not supported")

    def forward_original(self, x):
        with torch.no_grad(): return self.model(x)

    def forward_with_reconstruction(self, x):
        with torch.no_grad():
            features = self._forward_to_layer(x)
            original_acts = features.clone()
            normalized_acts, scale_factors = self._normalize_activations(original_acts)
            reconstruction, sparse_features = self.csae(normalized_acts, use_topk=False)
            scale_factors_4d = scale_factors.unsqueeze(2).unsqueeze(3)
            reconstructed_acts = reconstruction * scale_factors_4d
            logits = self._forward_from_layer(reconstructed_acts)
            mse = F.mse_loss(reconstructed_acts, original_acts).item()
            relative_error = ((reconstructed_acts - original_acts).abs().mean() / (original_acts.abs().mean() + 1e-8)).item()
            sparsity = (sparse_features > 0).float().mean().item()
            return logits, {'mse': mse, 'relative_error': relative_error, 'sparsity': sparsity}

def evaluate_accuracy_drop(model, data_loader, device='cuda', imagenette_to_imagenet=None):
    if imagenette_to_imagenet is not None:
        class_names = data_loader.dataset.dataset.classes if hasattr(data_loader.dataset, 'dataset') else data_loader.dataset.classes
        local_to_imagenet = torch.tensor([imagenette_to_imagenet[name] for name in class_names])
    else: local_to_imagenet = None
    
    total_samples, correct_original, correct_reconstructed = 0, 0, 0
    top5_correct_original, top5_correct_reconstructed = 0, 0
    mse_list, rel_err_list, sparsity_list = [], [], []

    for images, labels in tqdm(data_loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        imagenet_labels = local_to_imagenet[labels.cpu()].to(device) if local_to_imagenet is not None else labels
        
        logits_orig = model.forward_original(images)
        logits_recon, stats = model.forward_with_reconstruction(images)
        
        correct_original += (logits_orig.argmax(1) == imagenet_labels).sum().item()
        correct_reconstructed += (logits_recon.argmax(1) == imagenet_labels).sum().item()
        
        _, t5_orig = logits_orig.topk(5, 1); _, t5_recon = logits_recon.topk(5, 1)
        top5_correct_original += sum([imagenet_labels[i] in t5_orig[i] for i in range(batch_size)])
        top5_correct_reconstructed += sum([imagenet_labels[i] in t5_recon[i] for i in range(batch_size)])
        
        total_samples += batch_size
        mse_list.append(stats['mse']); rel_err_list.append(stats['relative_error']); sparsity_list.append(stats['sparsity'])

    acc_orig = correct_original / total_samples * 100
    acc_recon = correct_reconstructed / total_samples * 100
    return {
        'total_samples': total_samples, 'acc_original': acc_orig, 'acc_reconstructed': acc_recon,
        'acc_drop': acc_orig - acc_recon, 'top5_acc_original': top5_correct_original/total_samples*100,
        'top5_acc_reconstructed': top5_correct_reconstructed/total_samples*100,
        'acc5_drop': (top5_correct_original/total_samples*100) - (top5_correct_reconstructed/total_samples*100),
        # 'avg_mse': np.mean(mse_list), 'avg_relative_error': np.mean(rel_err_list), 'avg_sparsity': np.mean(sparsity_list)
    }

def print_results(results, model_name, layer_path):
    print(f"\nRESULTS: {model_name}.{layer_path} | Drop: {results['acc_drop']:.2f}% | MSE: {results['avg_mse']:.6f}")

# --- LOGIC MỚI: QUÉT TẤT CẢ MODEL ---
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
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--layer', type=str, help='Layer path')
    parser.add_argument('--csae_model', type=str, help='Path to CSAE model')
    parser.add_argument('--data_dir', type=str, default='data/imagenette')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--output_csv', type=str, default='output/accuracy_results.csv')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup Data
    data_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = datasets.ImageFolder(root=args.data_dir, transform=data_transform)

    # Xác định danh sách cần evaluate
    to_evaluate = []
    if args.all:
        to_evaluate = find_trained_models()
        print(f"Found {len(to_evaluate)} models for batch evaluation.")
    else:
        if not (args.model and args.layer and args.csae_model):
            print("Error: Specify --model, --layer, and --csae_model OR use --all")
            return
        to_evaluate = [{'model': args.model, 'layer': args.layer, 'path': args.csae_model}]

    all_results = []
    for cfg in to_evaluate:
        print(f"\nTarget: {cfg['model']}.{cfg['layer']}")
        try:
            m_cfg = get_model_config(cfg['model'], cfg['layer'])
            csae = joblib.load(cfg['path'])
            hybrid = GenericModelWithCSAEReconstruction(m_cfg, csae, device=device)
            
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
        df = pd.DataFrame(all_results)
        df = df.round(2)
        os.makedirs("output", exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\nBatch results saved to {args.output_csv}")

if __name__ == "__main__":
    main()