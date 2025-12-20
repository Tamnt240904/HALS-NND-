"""
Classifier Fine-tuning Script.
Saves the FULL fine-tuned hybrid model to: output/weights/finetuned/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import joblib
import argparse
import sys
import os
from tqdm import tqdm
from pathlib import Path
from types import SimpleNamespace  

sys.path.append('.')
from config.model_configs import get_model_config
from src.base_extractor import MultiChannelConvSAE

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

class TrainableHybridModel(nn.Module):
    def __init__(self, model_config, csae_model, device='cuda'):
        super().__init__()
        self.config = model_config
        self.device = device
        self.model = model_config.model_loader().to(device)
        self.csae = csae_model.to(device)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.csae.parameters():
            param.requires_grad = False
            
        self.target_layer = model_config.layer_getter(
            self.model, model_config.layer_path
        )
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)

        self.model_type = 'unknown'
        if 'vgg' in self.config.model_name:
            self.model_type = 'sequential'
            self._unfreeze_params(self.model.classifier)
        elif 'resnet' in self.config.model_name:
            self.model_type = 'resnet'
            self._unfreeze_params(self.model.fc)
        elif 'densenet' in self.config.model_name:
            self.model_type = 'sequential'
            self._unfreeze_params(self.model.classifier)
        elif 'efficientnet' in self.config.model_name:
            self.model_type = 'sequential'
            self._unfreeze_params(self.model.classifier)
        elif 'alexnet' in self.config.model_name:
            self.model_type = 'sequential'
            self._unfreeze_params(self.model.classifier)

    def _unfreeze_params(self, module):
        print(f"-> Unfreezing module: {module}")
        for param in module.parameters():
            param.requires_grad = True

    def _save_activation(self, module, input, output):
        self.activations = output

    def _normalize_activations(self, acts):
        B, C, H, W = acts.shape
        flat = acts.view(B, C, -1)
        
        if 'efficientnet' in self.config.model_name:
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

    def forward(self, x):
        features = self._forward_to_layer(x)
        normalized_acts, min_vals, range_vals = self._normalize_activations(features)
        reconstruction, _ = self.csae(normalized_acts, use_topk=True)
        reconstructed_acts = reconstruction * range_vals + min_vals
        logits = self._forward_from_layer(reconstructed_acts)
        return logits

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Classifier with CSAE')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--layer', type=str, required=True)
    parser.add_argument('--csae_model', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/imagenette')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(os.path.join(args.data_dir, 'train')):
        args.data_dir = os.path.join(args.data_dir, 'train')
    
    print(f"Loading data from: {args.data_dir}")
    data_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=args.data_dir, transform=data_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    class_names = dataset.classes
    local_to_imagenet_map = {}
    for idx, name in enumerate(class_names):
        if name in IMAGENETTE_TO_IMAGENET:
            local_to_imagenet_map[idx] = IMAGENETTE_TO_IMAGENET[name]
        else:
            local_to_imagenet_map[idx] = -1

    try:
        config = get_model_config(args.model, args.layer)
        csae = joblib.load(args.csae_model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Building hybrid model and freezing backbone...")
    hybrid_model = TrainableHybridModel(config, csae, device=device)
    
    trainable_params = [p for p in hybrid_model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Start fine-tuning for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        hybrid_model.train()
        hybrid_model.model.eval()
        hybrid_model.csae.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            mapped_labels = []
            valid_indices = []
            for i, lbl in enumerate(labels):
                mapped = local_to_imagenet_map.get(lbl.item(), -1)
                if mapped != -1:
                    mapped_labels.append(mapped)
                    valid_indices.append(i)
            
            if not valid_indices: continue
            
            images = images[valid_indices]
            target_labels = torch.tensor(mapped_labels, device=device)

            optimizer.zero_grad()
            outputs = hybrid_model(images)
            loss = criterion(outputs, target_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target_labels.size(0)
            correct += predicted.eq(target_labels).sum().item()
            
            pbar.set_postfix({'Loss': running_loss / total, 'Acc': 100. * correct / total})
            
    print("Fine-tuning complete.")
    
    weights_dir = Path("output/weights/finetuned")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    csae_filename = Path(args.csae_model).stem
    base_name = csae_filename.replace('_model', '')
    
    save_path = weights_dir / f"{base_name}_finetuned_model.pkl"
    
    hybrid_model.cpu()
    hybrid_model.config = SimpleNamespace(
        model_name=config.model_name, 
        layer_path=config.layer_path
    )
    
    print(f"Saving FULL hybrid model to: {save_path}...")
    try:
        joblib.dump(hybrid_model, str(save_path))
        print(f"✓ Model saved successfully!")
    except Exception as e:
        print(f"Failed to save model: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()