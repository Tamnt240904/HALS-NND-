import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from copy import deepcopy

class AtomVisualizer:
    """Visualize dictionary atoms using deconvolution and activation maximization."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: The neural network model
            target_layer: The layer where atoms are defined (e.g., model.feature_extractor[5])
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.model.eval()
        
    def _get_layer_index(self) -> int:
        """Find the index of target layer in feature_extractor."""
        for idx, layer in enumerate(self.model.feature_extractor):
            if layer is self.target_layer:
                return idx
        raise ValueError("Target layer not found in model.feature_extractor")
    
    # ============================================================
    # Method 1: Deconvolution / Guided Backpropagation
    # ============================================================
    
    def deconvolve_atom(self, atom: torch.Tensor, 
                        input_size: Tuple[int, int, int] = (3, 224, 224),
                        use_guided: bool = True) -> torch.Tensor:
        """
        Deconvolve an atom back to input space.
        
        Args:
            atom: Dictionary atom of shape (H, W) or (1, H, W)
            input_size: Target input size (C, H, W)
            use_guided: If True, use Guided Backprop (ReLU gradient modification)
            
        Returns:
            Reconstructed image in input space (C, H, W)
        """
        # Prepare atom as activation map
        if atom.dim() == 2:
            atom = atom.unsqueeze(0)  # (1, H, W)
        
        atom = atom.to(self.device).unsqueeze(0)  # (1, 1, H, W)
        atom.requires_grad_(True)
        
        # Get layer index
        layer_idx = self._get_layer_index()
        
        # Build decoder (reverse of encoder up to target layer)
        decoder = self._build_decoder(layer_idx, input_size)
        
        # Register hooks for guided backprop if needed
        if use_guided:
            hooks = self._register_guided_backprop_hooks(decoder)
        
        # Forward pass through decoder
        reconstruction = decoder(atom)
        
        # Backward pass
        reconstruction.sum().backward()
        
        # Get gradient w.r.t. reconstruction (this gives us the deconv result)
        deconv_result = reconstruction.detach()
        
        # Remove hooks
        if use_guided:
            for hook in hooks:
                hook.remove()
        
        return deconv_result.squeeze(0)  # (C, H, W)
    
    def _build_decoder(self, layer_idx: int, input_size: Tuple[int, int, int]) -> nn.Module:
        """
        Build a decoder that reverses the feature extractor up to layer_idx.
        
        This is a simplified version - you may need to customize based on your model architecture.
        """
        layers = []
        
        # Get layers from target layer back to input
        encoder_layers = list(self.model.feature_extractor[:layer_idx + 1])
        
        # Reverse and create transpose operations
        for layer in reversed(encoder_layers):
            if isinstance(layer, nn.Conv2d):
                # Create transposed convolution
                transpose_conv = nn.ConvTranspose2d(
                    in_channels=layer.out_channels,
                    out_channels=layer.in_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    bias=False
                )
                # Copy weights (transposed)
                transpose_conv.weight.data = layer.weight.data.transpose(0, 1)
                layers.append(transpose_conv)
                
            elif isinstance(layer, nn.ReLU):
                layers.append(nn.ReLU(inplace=False))
                
            elif isinstance(layer, nn.MaxPool2d):
                # Use unpooling or upsampling
                layers.append(nn.Upsample(scale_factor=layer.kernel_size, mode='nearest'))
                
            elif isinstance(layer, nn.BatchNorm2d):
                # Skip batch norm in decoder or use reverse
                pass
        
        return nn.Sequential(*layers).to(self.device)
    
    def _register_guided_backprop_hooks(self, module: nn.Module):
        """Register hooks to modify ReLU gradients for Guided Backprop."""
        hooks = []
        
        def guided_relu_hook(module, grad_in, grad_out):
            # Only pass positive gradients
            return (F.relu(grad_in[0]),)
        
        for layer in module.modules():
            if isinstance(layer, nn.ReLU):
                hook = layer.register_backward_hook(guided_relu_hook)
                hooks.append(hook)
        
        return hooks
    
    # ============================================================
    # Method 2: Activation Maximization
    # ============================================================
    
    def optimize_input_for_atom(self, 
                                 atom_idx: int,
                                 D: torch.Tensor,
                                 input_size: Tuple[int, int, int] = (3, 224, 224),
                                 n_iterations: int = 300,
                                 lr: float = 1.0,
                                 l2_reg: float = 1e-4,
                                 tv_reg: float = 1e-2,
                                 blur_every: int = 4,
                                 verbose: bool = True) -> torch.Tensor:
        """
        Optimize input image to maximize activation of a specific atom.
        
        Args:
            atom_idx: Index of atom in dictionary D
            D: Dictionary of atoms (n_atoms, H, W)
            input_size: Input image size (C, H, W)
            n_iterations: Number of optimization steps
            lr: Learning rate
            l2_reg: L2 regularization weight
            tv_reg: Total variation regularization weight
            blur_every: Apply Gaussian blur every N iterations
            verbose: Print progress
            
        Returns:
            Optimized input image (C, H, W)
        """
        # Initialize random input
        input_img = torch.randn(1, *input_size, device=self.device) * 0.1
        input_img.requires_grad_(True)
        
        # Get target atom
        target_atom = D[atom_idx].to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam([input_img], lr=lr)
        
        # Get layer index
        layer_idx = self._get_layer_index()
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # Forward pass to target layer
            activations = self._get_activations_at_layer(input_img, layer_idx)
            
            # Compute similarity with target atom
            # activations: (1, C, H, W), target_atom: (H, W)
            # We want to maximize the response that matches the atom pattern
            
            # Method 1: Cosine similarity across spatial dimensions
            batch_size, n_channels, h, w = activations.shape
            activations_flat = activations.view(batch_size, n_channels, -1)  # (1, C, H*W)
            atom_flat = target_atom.view(-1).unsqueeze(0).unsqueeze(0)  # (1, 1, H*W)
            
            # Compute cosine similarity for each channel
            cos_sim = F.cosine_similarity(activations_flat, atom_flat, dim=2)  # (1, C)
            activation_loss = -cos_sim.mean()  # Negative to maximize
            
            # Regularization terms
            l2_loss = l2_reg * torch.norm(input_img)
            tv_loss = tv_reg * self._total_variation(input_img)
            
            # Total loss
            loss = activation_loss + l2_loss + tv_loss
            
            loss.backward()
            optimizer.step()
            
            # Apply Gaussian blur periodically to reduce high-frequency noise
            if (iteration + 1) % blur_every == 0:
                with torch.no_grad():
                    input_img.data = self._gaussian_blur(input_img.data)
            
            # Clip values to reasonable range
            with torch.no_grad():
                input_img.data = torch.clamp(input_img.data, -3, 3)
            
            if verbose and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}/{n_iterations}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Activation: {-activation_loss.item():.4f}")
        
        return input_img.detach().squeeze(0)  # (C, H, W)
    
    def _get_activations_at_layer(self, input_img: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Forward pass to get activations at specific layer."""
        x = input_img
        for idx, layer in enumerate(self.model.feature_extractor):
            x = layer(x)
            if idx == layer_idx:
                return x
        return x
    
    def _total_variation(self, img: torch.Tensor) -> torch.Tensor:
        """Compute total variation loss for smoothness."""
        tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
        tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()
        return tv_h + tv_w
    
    def _gaussian_blur(self, img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """Apply Gaussian blur to reduce noise."""
        # Simple box blur as approximation
        return F.avg_pool2d(img, kernel_size, stride=1, padding=kernel_size // 2)
    
    # ============================================================
    # Visualization
    # ============================================================
    
    def visualize_atom_comparison(self, 
                                   atom_idx: int,
                                   D: torch.Tensor,
                                   input_size: Tuple[int, int, int] = (3, 224, 224),
                                   save_path: Optional[str] = None):
        """
        Visualize an atom using both methods side by side.
        
        Args:
            atom_idx: Index of atom to visualize
            D: Dictionary of atoms
            input_size: Input image size
            save_path: Path to save figure (optional)
        """
        print(f"\nVisualizing Atom {atom_idx}...")
        
        # Method 1: Deconvolution
        print("Running deconvolution...")
        atom = D[atom_idx]
        deconv_result = self.deconvolve_atom(atom, input_size, use_guided=True)
        
        # Method 2: Activation Maximization
        print("Running activation maximization...")
        optim_result = self.optimize_input_for_atom(atom_idx, D, input_size, n_iterations=300)
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original atom
        axes[0].imshow(atom.cpu().numpy(), cmap='viridis')
        axes[0].set_title(f'Atom {atom_idx}\n(Original Pattern)')
        axes[0].axis('off')
        
        # Deconvolution result
        deconv_vis = self._normalize_for_display(deconv_result)
        axes[1].imshow(deconv_vis)
        axes[1].set_title('Method 1:\nGuided Backprop')
        axes[1].axis('off')
        
        # Activation maximization result
        optim_vis = self._normalize_for_display(optim_result)
        axes[2].imshow(optim_vis)
        axes[2].set_title('Method 2:\nActivation Maximization')
        axes[2].axis('off')
        
        # Side by side comparison
        axes[3].imshow(np.concatenate([deconv_vis, optim_vis], axis=1))
        axes[3].set_title('Comparison\n(Deconv | Optim)')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def _normalize_for_display(self, img: torch.Tensor) -> np.ndarray:
        """Normalize image tensor for display."""
        img_np = img.cpu().numpy()
        
        if img_np.shape[0] == 3:  # RGB
            img_np = np.transpose(img_np, (1, 2, 0))
        else:  # Grayscale
            img_np = img_np[0]
        
        # Normalize to [0, 1]
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        return img_np

