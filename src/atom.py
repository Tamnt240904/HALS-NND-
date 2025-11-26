import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple, List
from .utils import denormalize_tensor, create_heatmap_overlay

class Deconvolution:
    """
    Thực hiện Zeiler & Fergus Deconvolution để visualize features.
    Khác với Guided Backprop, Deconv bỏ qua dấu của input forward, 
    chỉ giữ lại gradient dương (Reconstruction).
    """
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.model.eval()

    def register_hooks(self):
        def deconv_relu_backward_hook(module, grad_in, grad_out):
            # Deconv logic: Backward của ReLU hoạt động như Forward của ReLU
            # grad_out[0] là gradient từ lớp trên truyền xuống.
            # Ta áp dụng ReLU lên chính gradient đó để lọc bỏ phần âm.
            # Lưu ý: Return value của hook sẽ thay thế grad_in (gradient truyền xuống lớp dưới).
            
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_out[0], min=0.0),)
        
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                # Sử dụng register_backward_hook để can thiệp vào dòng chảy gradient
                self.hooks.append(module.register_backward_hook(deconv_relu_backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_gradients(self, input_image, target_layer_idx, atom_tensor):
        """
        Tạo Deconv map cho atom cụ thể.
        """
        self.register_hooks()
        
        # Clone để không ảnh hưởng ảnh gốc và bật gradient
        input_tensor = input_image.clone().detach()
        input_tensor.requires_grad = True
        
        self.model.zero_grad()
        
        # Hook để lấy activation output tại layer đích
        activations = []
        def hook_fn(module, input, output):
            activations.append(output)
            
        handle = self.model.feature_extractor[target_layer_idx].register_forward_hook(hook_fn)
        
        try:
            # 1. Forward Pass
            _ = self.model(input_tensor)
            
            # 2. Tính Loss để Backward
            # Mục tiêu: Tìm vùng ảnh input tạo ra sự tương đồng cao nhất với Atom
            act = activations[0].squeeze(0) # [C, H, W]
            spatial_act = torch.mean(act, dim=0) # [H, W]
            
            # Loss = Projection của activation lên Atom pattern
            loss = torch.sum(spatial_act * atom_tensor)
            
            # 3. Backward Pass (qua các Deconv hooks)
            loss.backward()
            
            # 4. Lấy Gradient tại Input (chính là ảnh tái tạo Deconv)
            gradients = input_tensor.grad.data.cpu().numpy()[0] # [3, H, W]
            
            # Xử lý kết quả Deconv để hiển thị đẹp hơn
            # Deconv thường tạo ra ảnh có dynamic range lớn, cần chuẩn hóa
            deconv_img = np.transpose(gradients, (1, 2, 0)) # [H, W, 3]
            
            # Chuẩn hóa về 0-1 dùng Abs max (giống paper gốc thường làm cho saliency)
            # Hoặc min-max scaling
            deconv_img = np.abs(deconv_img).max(axis=2) # Lấy max qua channels để thành grayscale heatmap
            
            if deconv_img.max() > 0:
                deconv_img = (deconv_img - deconv_img.min()) / (deconv_img.max() - deconv_img.min())
            
            return deconv_img
            
        finally:
            handle.remove()
            self.remove_hooks()
            if input_tensor.grad is not None:
                input_tensor.grad.data.zero_()


class AtomVisualizer:
    """Class quản lý việc visualize Atom."""
    
    def __init__(self, model: nn.Module, target_layer_idx: int, device: str = 'cuda'):
        self.model = model
        self.target_layer_idx = target_layer_idx
        self.device = device
        self.model.eval()
        # Thay đổi: Sử dụng Deconvolution thay vì GuidedBackprop
        self.deconv = Deconvolution(model)
        
    def optimize_input(self, atom: torch.Tensor, steps=200, lr=0.1):
        """
        Tạo ảnh tối ưu (Activation Maximization) cho Atom.
        """
        # Tạo ảnh nhiễu ngẫu nhiên
        random_img = torch.randn(1, 3, 224, 224, device=self.device) * 0.01
        random_img.requires_grad_(True)
        optimizer = optim.Adam([random_img], lr=lr)
        
        atom_tensor = atom.to(self.device).float()
        
        activations = []
        def hook_fn(module, input, output):
            activations.append(output)
        handle = self.model.feature_extractor[self.target_layer_idx].register_forward_hook(hook_fn)
        
        try:
            for _ in range(steps):
                optimizer.zero_grad()
                activations.clear()
                _ = self.model(random_img)
                
                # Loss: Maximize spatial correlation
                act = activations[0].squeeze(0)
                spatial_act = torch.mean(act, dim=0)
                loss = -torch.sum(spatial_act * atom_tensor)
                
                loss.backward()
                optimizer.step()
                
                # Regularization nhẹ để ảnh mượt hơn
                with torch.no_grad():
                    random_img.data = torch.clamp(random_img.data, -2.5, 2.5)
                    
        finally:
            handle.remove()
            
        return denormalize_tensor(random_img)

    def get_deconv(self, image_tensor, atom: torch.Tensor):
        """Wrapper gọi Deconvolution."""
        return self.deconv.generate_gradients(image_tensor, self.target_layer_idx, atom)

    def visualize_comprehensive(self, atom_idx, D, class_name, sample_img_path=None, save_path=None):
        """
        Vẽ biểu đồ toàn diện gồm 4 phần:
        1. Atom Pattern (Heatmap gốc)
        2. Optimized Input (Atom "thích" nhìn gì nhất - Global)
        3. Deconvolution (Atom nhìn thấy gì trên ảnh thật - Local)
        4. Overlay (Vị trí Atom kích hoạt trên ảnh thật)
        """
        atom_data = D[atom_idx]
        if torch.is_tensor(atom_data):
            atom_data = atom_data.to(self.device)
        
        # 1. Optimize Input
        opt_img = self.optimize_input(atom_data)
        
        # Chuẩn bị plot
        cols = 4 if sample_img_path else 2
        fig, axes = plt.subplots(1, cols, figsize=(4*cols, 4))
        
        # Plot 1: Atom Heatmap
        im = axes[0].imshow(atom_data.cpu().numpy(), cmap='viridis')
        axes[0].set_title(f'Atom {atom_idx} (Pattern)', fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Plot 2: Optimized Input
        axes[1].imshow(opt_img)
        axes[1].set_title('Optimized Input\n(Global Concept)', fontweight='bold')
        axes[1].axis('off')
        
        # Nếu có ảnh mẫu thì chạy Deconv và Overlay
        if sample_img_path and os.path.exists(sample_img_path):
            from .utils import load_image # Import ở đây tránh circular import
            
            # Load ảnh
            img_tensor = load_image(sample_img_path).to(self.device)
            
            # Plot 3: Deconvolution
            deconv_map = self.get_deconv(img_tensor, atom_data)
            axes[2].imshow(deconv_map, cmap='gray')
            axes[2].set_title('Deconvolution\n(Reconstruction)', fontweight='bold')
            axes[2].axis('off')
            
            # Plot 4: Activation Overlay
            # Overlay heatmap lên ảnh gốc để thấy vị trí
            overlay = create_heatmap_overlay(sample_img_path, deconv_map)
            axes[3].imshow(overlay)
            axes[3].set_title(f'Overlay on Sample\n({os.path.basename(sample_img_path)})', fontweight='bold')
            axes[3].axis('off')
            
        plt.suptitle(f'Visual Analysis: Class "{class_name}" - Atom {atom_idx}', fontsize=14, y=1.05)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved visualization to {save_path}")
            
        plt.show()

def visualize_all_unique_atoms(atom_analysis, D, model, target_layer_idx, subset_paths, device='cuda'):
    """Hàm wrapper để chạy visualize cho tất cả các unique atoms tìm được."""
    visualizer = AtomVisualizer(model, target_layer_idx, device)
    
    unique_atoms = atom_analysis['unique_atoms']
    class_names = atom_analysis['class_names']
    
    # Group lại theo class
    class_unique = {}
    for atom_idx, class_idx in unique_atoms.items():
        if class_idx not in class_unique:
            class_unique[class_idx] = []
        class_unique[class_idx].append(atom_idx)
        
    for class_idx in sorted(class_unique.keys()):
        # Lấy atom đầu tiên đặc trưng cho class này
        atom_idx = class_unique[class_idx][0]
        class_name = class_names[class_idx]
        
        # Tìm ảnh sample
        sample_path = None
        for path, label in subset_paths:
            if label == class_idx:
                sample_path = path
                break
        
        print(f"Processing Class: {class_name} | Atom: {atom_idx}")
        visualizer.visualize_comprehensive(
            atom_idx=atom_idx,
            D=D,
            class_name=class_name,
            sample_img_path=sample_path,
            save_path=f"figures/comprehensive/class_{class_name}_atom_{atom_idx}.png"
        )