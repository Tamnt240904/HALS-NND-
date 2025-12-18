"""
Fixed GradCAM implementation that handles in-place operations.

Fixes RuntimeError with AlexNet and other models using inplace ReLU:
"Output is a view and is being modified inplace"

Usage:
    from src.gradcam_fixed import GradCAM
    
    model = models.alexnet(pretrained=True)
    gradcam = GradCAM(model, model.features[6])
    weights, cam, pred = gradcam.forward(image)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Fixed version that handles in-place operations correctly.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: CNN model
            target_layer: Target layer to compute CAM for
        """
        self.model = model
        self.target_layer = target_layer
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)
        
        # Store original inplace settings
        self._original_inplace = {}
        self._disable_inplace_operations()
    
    def _disable_inplace_operations(self):
        """
        Disable in-place operations in ReLU layers to avoid gradient issues.
        This is necessary for GradCAM to work with models like AlexNet.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                self._original_inplace[name] = module.inplace
                module.inplace = False
    
    def _restore_inplace_operations(self):
        """Restore original in-place settings."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU) and name in self._original_inplace:
                module.inplace = self._original_inplace[name]
    
    def _forward_hook(self, module, input, output):
        """Hook to capture forward activations."""
        # Clone to avoid in-place modification issues
        self.activations = output.detach().clone()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Hook to capture backward gradients."""
        # Clone to avoid in-place modification issues
        self.gradients = grad_output[0].detach().clone()
    
    def forward(self, 
                x: torch.Tensor, 
                class_idx: Optional[int] = None,
                verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Compute Grad-CAM.
        
        Args:
            x: Input image [1, 3, H, W]
            class_idx: Target class index (None = use predicted class)
            verbose: Print information
            
        Returns:
            weights: Channel importance weights [num_channels]
            cam: Class activation map [H, W]
            pred_class: Predicted class index
        """
        # Ensure gradients are enabled
        x = x.requires_grad_(True)
        
        # Forward pass
        self.model.eval()  # Keep in eval mode but enable gradients
        logits = self.model(x)
        
        # Get predicted class
        pred_class = logits.argmax(dim=1).item()
        target_class = class_idx if class_idx is not None else pred_class
        
        if verbose:
            print(f"Predicted class: {pred_class}, Target class: {target_class}")
        
        # Backward pass for target class
        self.model.zero_grad()
        
        # Create one-hot encoded target
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        
        # Backward
        logits.backward(gradient=one_hot, retain_graph=False)
        
        # Check if gradients were captured
        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Gradients or activations not captured. "
                "Make sure the target layer is in the forward path."
            )
        
        # Compute channel weights (global average pooling of gradients)
        # Shape: [1, C, H, W] -> [C]
        weights = self.gradients.mean(dim=[0, 2, 3])  # Average over batch and spatial dims
        
        # Apply ReLU to weights (only positive influences)
        weights = F.relu(weights)
        
        # Compute weighted combination of activation maps
        # activations: [1, C, H, W]
        # weights: [C]
        cam = torch.zeros(self.activations.shape[2:], device=x.device)
        
        for i in range(len(weights)):
            cam += weights[i] * self.activations[0, i]
        
        # Apply ReLU to CAM
        cam = F.relu(cam)
        
        # Normalize CAM to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return weights, cam, pred_class
    
    def __del__(self):
        """Clean up hooks and restore settings."""
        try:
            self.forward_handle.remove()
            self.backward_handle.remove()
            self._restore_inplace_operations()
        except:
            pass


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ variant with improved localization.
    Uses weighted combination of gradients.
    """
    
    def forward(self, 
                x: torch.Tensor, 
                class_idx: Optional[int] = None,
                verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Compute Grad-CAM++.
        
        Args:
            x: Input image [1, 3, H, W]
            class_idx: Target class index (None = use predicted class)
            verbose: Print information
            
        Returns:
            weights: Channel importance weights [num_channels]
            cam: Class activation map [H, W]
            pred_class: Predicted class index
        """
        x = x.requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        logits = self.model(x)
        
        pred_class = logits.argmax(dim=1).item()
        target_class = class_idx if class_idx is not None else pred_class
        
        if verbose:
            print(f"Predicted class: {pred_class}, Target class: {target_class}")
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot, retain_graph=False)
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured")
        
        # Grad-CAM++ weighting
        # Alpha: pixel-wise weights
        gradients_power_2 = self.gradients ** 2
        gradients_power_3 = gradients_power_2 * self.gradients
        
        sum_activations = self.activations.sum(dim=(2, 3), keepdim=True)
        alpha = gradients_power_2 / (2 * gradients_power_2 + sum_activations * gradients_power_3 + 1e-8)
        
        # Weighted ReLU gradients
        weighted_gradients = F.relu(self.gradients) * alpha
        
        # Channel weights
        weights = weighted_gradients.sum(dim=(0, 2, 3))
        weights = F.relu(weights)
        
        # Compute CAM
        cam = torch.zeros(self.activations.shape[2:], device=x.device)
        for i in range(len(weights)):
            cam += weights[i] * self.activations[0, i]
        
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return weights, cam, pred_class


def test_gradcam():
    """Test GradCAM with different architectures."""
    import torchvision.models as models
    from torchvision import transforms
    from PIL import Image
    
    print("="*80)
    print("Testing Fixed GradCAM")
    print("="*80)
    
    # Test models with in-place operations
    test_configs = [
        ('AlexNet', models.alexnet(pretrained=True), 'features.6'),
        ('ResNet50', models.resnet50(pretrained=True), 'layer3'),
        ('VGG16', models.vgg16(pretrained=True), 'features.23'),
    ]
    
    # Create dummy image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create random image for testing
    dummy_img = torch.randn(1, 3, 224, 224)
    
    for name, model, layer_name in test_configs:
        print(f"\n{name} - {layer_name}")
        print("-"*80)
        
        try:
            # Get target layer
            if '.' in layer_name:
                parts = layer_name.split('.')
                target_layer = model
                for part in parts:
                    if part.isdigit():
                        target_layer = target_layer[int(part)]
                    else:
                        target_layer = getattr(target_layer, part)
            else:
                target_layer = getattr(model, layer_name)
            
            # Create GradCAM
            gradcam = GradCAM(model, target_layer)
            
            # Test forward
            weights, cam, pred = gradcam.forward(dummy_img, verbose=False)
            
            print(f"✓ Success!")
            print(f"  Weights shape: {weights.shape}")
            print(f"  CAM shape: {cam.shape}")
            print(f"  Predicted class: {pred}")
            print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    print("\n" + "="*80)
    print("✓ All tests completed!")
    print("="*80)


if __name__ == "__main__":
    test_gradcam()