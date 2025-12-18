"""
Configuration module for Multi-Channel ConvSAE training.
Supports different CNN architectures and target layers.
"""

import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Tuple, Optional


# ==========================================
# Supported Architectures Configuration
# ==========================================

SUPPORTED_ARCHITECTURES = {
    'resnet50': {
        'model_fn': lambda: models.resnet50(pretrained=True),
        'layers': {
            'layer1': {'channels': 256, 'spatial_size': 56},
            'layer2': {'channels': 512, 'spatial_size': 28},
            'layer3': {'channels': 1024, 'spatial_size': 14},
            'layer4': {'channels': 2048, 'spatial_size': 7},
        }
    },
    'resnet18': {
        'model_fn': lambda: models.resnet18(pretrained=True),
        'layers': {
            'layer1': {'channels': 64, 'spatial_size': 56},
            'layer2': {'channels': 128, 'spatial_size': 28},
            'layer3': {'channels': 256, 'spatial_size': 14},
            'layer4': {'channels': 512, 'spatial_size': 7},
        }
    },
    'resnet34': {
        'model_fn': lambda: models.resnet34(pretrained=True),
        'layers': {
            'layer1': {'channels': 64, 'spatial_size': 56},
            'layer2': {'channels': 128, 'spatial_size': 28},
            'layer3': {'channels': 256, 'spatial_size': 14},
            'layer4': {'channels': 512, 'spatial_size': 7},
        }
    },
    'resnet101': {
        'model_fn': lambda: models.resnet101(pretrained=True),
        'layers': {
            'layer1': {'channels': 256, 'spatial_size': 56},
            'layer2': {'channels': 512, 'spatial_size': 28},
            'layer3': {'channels': 1024, 'spatial_size': 14},
            'layer4': {'channels': 2048, 'spatial_size': 7},
        }
    },
    'vgg16': {
        'model_fn': lambda: models.vgg16(pretrained=True),
        'layers': {
            'features.16': {'channels': 256, 'spatial_size': 28},
            'features.23': {'channels': 512, 'spatial_size': 14},
            'features.30': {'channels': 512, 'spatial_size': 7},
        }
    },
    'vgg19': {
        'model_fn': lambda: models.vgg19(pretrained=True),
        'layers': {
            'features.18': {'channels': 256, 'spatial_size': 28},
            'features.27': {'channels': 512, 'spatial_size': 14},
            'features.36': {'channels': 512, 'spatial_size': 7},
        }
    },
    'densenet121': {
        'model_fn': lambda: models.densenet121(pretrained=True),
        'layers': {
            'features.denseblock2': {'channels': 512, 'spatial_size': 28},
            'features.denseblock3': {'channels': 1024, 'spatial_size': 14},
            'features.denseblock4': {'channels': 1024, 'spatial_size': 7},
        }
    },
    'efficientnet_b0': {
        'model_fn': lambda: models.efficientnet_b0(pretrained=True),
        'layers': {
            'features.2': {'channels': 24, 'spatial_size': None},   # MBConv, 56x56
            'features.3': {'channels': 40, 'spatial_size': None},   # MBConv, 28x28
            'features.4': {'channels': 80, 'spatial_size': None},   # MBConv, 14x14
            'features.5': {'channels': 112, 'spatial_size': None},  # MBConv, 14x14
            'features.6': {'channels': 192, 'spatial_size': None},  # MBConv, 7x7
            'features.7': {'channels': 320, 'spatial_size': None},  # MBConv, 7x7
        }
    },
    'efficientnet_b1': {
        'model_fn': lambda: models.efficientnet_b1(pretrained=True),
        'layers': {
            'features.4': {'channels': 80, 'spatial_size': None},
            'features.5': {'channels': 112, 'spatial_size': None},
            'features.6': {'channels': 192, 'spatial_size': None},
        }
    },
    'alexnet': {
        'model_fn': lambda: models.alexnet(pretrained=True),
        'layers': {
            'features.6': {'channels': 384, 'spatial_size': 13},
        }
    },

}


class ModelConfig:
    """Configuration class for CNN architecture and target layer."""
    
    def __init__(self, 
                 architecture: str = 'resnet50',
                 target_layer: str = 'layer3',
                 hidden_dim_multiplier: float = -1.0,
                 top_k: int = 20,
                 cumulative_threshold: float = 0.8):
        """
        Initialize model configuration.
        
        Args:
            architecture: CNN architecture name (e.g., 'resnet50', 'vgg16')
            target_layer: Target layer name (e.g., 'layer3', 'features.23')
            hidden_dim_multiplier: Multiplier for hidden dimension (hidden_dim = in_channels * multiplier)
            top_k: Number of active features per spatial position
            cumulative_threshold: GradCAM cumulative threshold for channel selection
        """
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Architecture '{architecture}' not supported. "
                f"Available: {list(SUPPORTED_ARCHITECTURES.keys())}"
            )
        
        self.architecture = architecture
        self.target_layer = target_layer
        
        # Get architecture config
        arch_config = SUPPORTED_ARCHITECTURES[architecture]
        
        # Validate target layer
        if target_layer not in arch_config['layers']:
            raise ValueError(
                f"Layer '{target_layer}' not supported for {architecture}. "
                f"Available: {list(arch_config['layers'].keys())}"
            )
        
        # Get layer info
        layer_info = arch_config['layers'][target_layer]
        self.input_channels = layer_info['channels']
        self.spatial_size = layer_info['spatial_size']  # May be None for auto-detect
        
        # Compute hidden dimension
        if hidden_dim_multiplier <= 0:
            self.hidden_dim = 8192
        else:
            self.hidden_dim = int(self.input_channels * hidden_dim_multiplier)
        self.top_k = top_k
        self.cumulative_threshold = cumulative_threshold
        
        # Store model function
        self.model_fn = arch_config['model_fn']
        
        # Flag for dynamic spatial size detection
        self._spatial_size_detected = self.spatial_size is not None
    
    def get_model_and_layer(self) -> Tuple[nn.Module, nn.Module]:
        """
        Get the CNN model and target layer module.
        
        Returns:
            model: CNN model
            target_layer_module: Target layer module for hooking
        """
        model = self.model_fn()
        
        # Get target layer module by name
        target_layer_module = self._get_layer_by_name(model, self.target_layer)
        
        return model, target_layer_module
    
    def detect_spatial_size(self, model: nn.Module, target_layer: nn.Module, device='cuda') -> int:
        """
        Detect actual spatial size by running a forward pass.
        
        Args:
            model: CNN model
            target_layer: Target layer module
            device: Device to run on
            
        Returns:
            spatial_size: Detected spatial dimension (H or W, assumed square)
        """
        if self._spatial_size_detected:
            return self.spatial_size
        
        import torch
        
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        # Hook to capture output
        activation = None
        def hook(module, input, output):
            nonlocal activation
            activation = output
        
        handle = target_layer.register_forward_hook(hook)
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        handle.remove()
        
        if activation is not None:
            # Shape: [1, C, H, W]
            spatial_size = activation.shape[2]  # Assume square
            self.spatial_size = spatial_size
            self._spatial_size_detected = True
            return spatial_size
        
        raise RuntimeError("Could not detect spatial size")
    
    def _get_layer_by_name(self, model: nn.Module, layer_name: str) -> nn.Module:
        """Get layer module by name (supports nested attributes like 'features.23')."""
        parts = layer_name.split('.')
        module = model
        
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        
        return module
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving."""
        return {
            'architecture': self.architecture,
            'target_layer': self.target_layer,
            'input_channels': self.input_channels,
            'spatial_size': self.spatial_size,
            'hidden_dim': self.hidden_dim,
            'top_k': self.top_k,
            'cumulative_threshold': self.cumulative_threshold,
        }
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        hidden_dim_multiplier = config_dict['hidden_dim'] / config_dict['input_channels']
        
        return ModelConfig(
            architecture=config_dict['architecture'],
            target_layer=config_dict['target_layer'],
            hidden_dim_multiplier=hidden_dim_multiplier,
            top_k=config_dict['top_k'],
            cumulative_threshold=config_dict.get('cumulative_threshold', 0.8)
        )
    
    def __repr__(self):
        spatial_str = f"{self.spatial_size}x{self.spatial_size}" if self.spatial_size else "auto-detect"
        return (f"ModelConfig(architecture={self.architecture}, "
                f"target_layer={self.target_layer}, "
                f"input_channels={self.input_channels}, "
                f"hidden_dim={self.hidden_dim}, "
                f"spatial_size={spatial_str})")


def list_available_architectures():
    """Print available architectures and their layers."""
    print("Available Architectures and Layers:")
    print("=" * 80)
    
    for arch_name, arch_config in SUPPORTED_ARCHITECTURES.items():
        print(f"\n{arch_name.upper()}:")
        for layer_name, layer_info in arch_config['layers'].items():
            print(f"  - {layer_name:20s} : {layer_info['channels']:4d} channels, "
                  f"{layer_info['spatial_size']:2d}×{layer_info['spatial_size']:2d} spatial")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Demo usage
    list_available_architectures()
    
    print("\n" + "=" * 80)
    print("Example Configurations:")
    print("=" * 80)
    
    # Example 1: ResNet50 layer3 (default)
    config1 = ModelConfig('resnet50', 'layer3')
    print(f"\n1. {config1}")
    
    # Example 2: VGG16 features.23
    config2 = ModelConfig('vgg16', 'features.23', hidden_dim_multiplier=4.0)
    print(f"2. {config2}")
    
    # Example 3: ResNet18 layer2
    config3 = ModelConfig('resnet18', 'layer2', hidden_dim_multiplier=16.0)
    print(f"3. {config3}")