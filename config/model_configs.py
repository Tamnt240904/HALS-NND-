"""
Model Configuration System for Multi-Channel ConvSAE

Focused on layers with spatial resolution 13×13 or 14×14.
Supports: VGG, ResNet, EfficientNet, DenseNet, AlexNet.

Usage:
    from config.model_configs import get_model_config, list_available_configs
    
    config = get_model_config('vgg16', 'features.23')
    configs = list_available_configs()
"""

from typing import Dict, Tuple
import torchvision.models as models


class ModelLayerConfig:
    """Configuration for a specific model and layer."""
    
    def __init__(
        self,
        model_name: str,
        layer_path: str,
        num_channels: int,
        spatial_size: Tuple[int, int],
        model_loader: callable,
        description: str = ""
    ):
        self.model_name = model_name
        self.layer_path = layer_path
        self.num_channels = num_channels
        self.spatial_size = spatial_size
        self.model_loader = model_loader
        self.description = description
    
    def __repr__(self):
        return (f"ModelLayerConfig({self.model_name}.{self.layer_path}, "
                f"{self.num_channels}ch, {self.spatial_size[0]}×{self.spatial_size[1]})")



# ==========================================
# Unified Layer Getter Function
# ==========================================

def get_layer_by_path(model, layer_path: str):
    """
    Generic layer getter that works for all models.
    
    Examples:
        'features.23' → model.features[23]
        'layer3' → model.layer3
        'features.denseblock3' → model.features.denseblock3
    """
    parts = layer_path.split('.')
    obj = model
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


# ==========================================
# Model Configurations (14×14 or 13×13 only)
# ==========================================

# VGG Models
VGG16_CONFIGS = {
    'features.23': ModelLayerConfig(
        model_name='vgg16',
        layer_path='features.23',
        num_channels=512,
        spatial_size=(14, 14),
        model_loader=lambda: models.vgg16(pretrained=True),
        description="VGG16 after 4th conv block"
    ),
}

VGG19_CONFIGS = {
    'features.27': ModelLayerConfig(
        model_name='vgg19',
        layer_path='features.27',
        num_channels=512,
        spatial_size=(14, 14),
        model_loader=lambda: models.vgg19(pretrained=True),
        description="VGG19 after 4th conv block"
    ),
}

# ResNet Models
RESNET50_CONFIGS = {
    'layer3': ModelLayerConfig(
        model_name='resnet50',
        layer_path='layer3',
        num_channels=1024,
        spatial_size=(14, 14),
        model_loader=lambda: models.resnet50(pretrained=True),
        description="ResNet50 layer3 output"
    ),
}

RESNET101_CONFIGS = {
    'layer3': ModelLayerConfig(
        model_name='resnet101',
        layer_path='layer3',
        num_channels=1024,
        spatial_size=(14, 14),
        model_loader=lambda: models.resnet101(pretrained=True),
        description="ResNet101 layer3 output"
    ),
}

RESNET152_CONFIGS = {
    'layer3': ModelLayerConfig(
        model_name='resnet152',
        layer_path='layer3',
        num_channels=1024,
        spatial_size=(14, 14),
        model_loader=lambda: models.resnet152(pretrained=True),
        description="ResNet152 layer3 output"
    ),
}

# EfficientNet Models
EFFICIENTNET_B0_CONFIGS = {
    'features.6': ModelLayerConfig(
        model_name='efficientnet_b0',
        layer_path='features.6',
        num_channels=192,
        spatial_size=(14, 14),
        model_loader=lambda: models.efficientnet_b0(pretrained=True),
        description="EfficientNet-B0 MBConv block 6"
    ),
}

EFFICIENTNET_B1_CONFIGS = {
    'features.6': ModelLayerConfig(
        model_name='efficientnet_b1',
        layer_path='features.6',
        num_channels=192,
        spatial_size=(14, 14),
        model_loader=lambda: models.efficientnet_b1(pretrained=True),
        description="EfficientNet-B1 MBConv block 6"
    ),
}

EFFICIENTNET_B2_CONFIGS = {
    'features.6': ModelLayerConfig(
        model_name='efficientnet_b2',
        layer_path='features.6',
        num_channels=208,
        spatial_size=(14, 14),
        model_loader=lambda: models.efficientnet_b2(pretrained=True),
        description="EfficientNet-B2 MBConv block 6"
    ),
}

EFFICIENTNET_B3_CONFIGS = {
    'features.6': ModelLayerConfig(
        model_name='efficientnet_b3',
        layer_path='features.6',
        num_channels=232,
        spatial_size=(14, 14),
        model_loader=lambda: models.efficientnet_b3(pretrained=True),
        description="EfficientNet-B3 MBConv block 6"
    ),
}

EFFICIENTNET_B4_CONFIGS = {
    'features.6': ModelLayerConfig(
        model_name='efficientnet_b4',
        layer_path='features.6',
        num_channels=272,
        spatial_size=(14, 14),
        model_loader=lambda: models.efficientnet_b4(pretrained=True),
        description="EfficientNet-B4 MBConv block 6"
    ),
}

# DenseNet Models
DENSENET121_CONFIGS = {
    'features.denseblock3': ModelLayerConfig(
        model_name='densenet121',
        layer_path='features.denseblock3',
        num_channels=1024,
        spatial_size=(14, 14),
        model_loader=lambda: models.densenet121(pretrained=True),
        description="DenseNet121 denseblock2 output"
    ),
}

DENSENET169_CONFIGS = {
    'features.denseblock2': ModelLayerConfig(
        model_name='densenet169',
        layer_path='features.denseblock2',
        num_channels=512,
        spatial_size=(14, 14),
        model_loader=lambda: models.densenet169(pretrained=True),
        description="DenseNet169 denseblock2 output"
    ),
}

DENSENET201_CONFIGS = {
    'features.denseblock2': ModelLayerConfig(
        model_name='densenet201',
        layer_path='features.denseblock2',
        num_channels=512,
        spatial_size=(14, 14),
        model_loader=lambda: models.densenet201(pretrained=True),
        description="DenseNet201 denseblock2 output"
    ),
}

# AlexNet
ALEXNET_CONFIGS = {
    'features.6': ModelLayerConfig(
        model_name='alexnet',
        layer_path='features.6',
        num_channels=384,
        spatial_size=(13, 13),
        model_loader=lambda: models.alexnet(pretrained=True),
        description="AlexNet conv5 output (13×13)"
    ),
}


# ==========================================
# Registry
# ==========================================

MODEL_REGISTRY = {
    'vgg16': VGG16_CONFIGS,
    'vgg19': VGG19_CONFIGS,
    'resnet50': RESNET50_CONFIGS,
    'resnet101': RESNET101_CONFIGS,
    'resnet152': RESNET152_CONFIGS,
    'efficientnet_b0': EFFICIENTNET_B0_CONFIGS,
    'efficientnet_b1': EFFICIENTNET_B1_CONFIGS,
    'efficientnet_b2': EFFICIENTNET_B2_CONFIGS,
    'efficientnet_b3': EFFICIENTNET_B3_CONFIGS,
    'efficientnet_b4': EFFICIENTNET_B4_CONFIGS,
    'densenet121': DENSENET121_CONFIGS,
    'densenet169': DENSENET169_CONFIGS,
    'densenet201': DENSENET201_CONFIGS,
    'alexnet': ALEXNET_CONFIGS,
}



# ==========================================
# Public API
# ==========================================

def get_model_config(model_name: str, layer_path: str = None) -> ModelLayerConfig:
    """
    Get configuration for a specific model and layer.
    
    Args:
        model_name: Model name (e.g., 'vgg16', 'resnet50')
        layer_path: Layer path (optional, returns first if not specified)
    
    Returns:
        ModelLayerConfig object
    
    Raises:
        ValueError: If model or layer not found
    
    Examples:
        >>> config = get_model_config('vgg16', 'features.23')
        >>> config = get_model_config('resnet50')  # Returns first layer
    """
    model_name = model_name.lower()
    
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")
    
    model_configs = MODEL_REGISTRY[model_name]
    
    # If no layer specified, return first available
    if layer_path is None:
        layer_path = list(model_configs.keys())[0]
        print(f"ℹ No layer specified, using: {layer_path}")
    
    if layer_path not in model_configs:
        available = list(model_configs.keys())
        raise ValueError(f"Layer '{layer_path}' not found for {model_name}. Available: {available}")
    
    config = model_configs[layer_path]
    # Add layer getter function
    config.layer_getter = get_layer_by_path
    return config


def list_available_configs() -> Dict[str, Dict[str, Tuple[int, Tuple[int, int]]]]:
    """
    List all available configurations with channel counts and spatial sizes.
    
    Returns:
        Dict mapping model → {layer: (channels, spatial_size)}
    
    Example:
        >>> configs = list_available_configs()
        >>> print(configs['vgg16'])
        {'features.23': (512, (14, 14))}
    """
    result = {}
    for model_name, layers in MODEL_REGISTRY.items():
        result[model_name] = {
            layer_path: (config.num_channels, config.spatial_size)
            for layer_path, config in layers.items()
        }
    return result


def print_available_configs():
    """Pretty-print all available configurations."""
    print("=" * 80)
    print("Available Model & Layer Configurations")
    print("Focus: Layers with 13×13 or 14×14 spatial resolution")
    print("=" * 80)
    
    for model_name, layers in MODEL_REGISTRY.items():
        print(f"\n{model_name.upper()}:")
        for layer_path, config in layers.items():
            print(f"  {layer_path:25s} | {config.num_channels:4d} channels | "
                  f"{config.spatial_size[0]:2d}×{config.spatial_size[1]:2d} | {config.description}")
    
    print("\n" + "=" * 80)
    print(f"Total: {sum(len(layers) for layers in MODEL_REGISTRY.values())} configurations across "
          f"{len(MODEL_REGISTRY)} model families")
    print("=" * 80)


def validate_config(model_name: str, layer_path: str = None, device: str = 'cpu') -> bool:
    """
    Validate configuration by testing actual model output.
    
    Args:
        model_name: Model name
        layer_path: Layer path (optional)
        device: Device to test on
    
    Returns:
        True if configuration is valid
    """
    try:
        import torch
        
        config = get_model_config(model_name, layer_path)
        
        # Load model
        print(f"Loading {config.model_name}...")
        model = config.model_loader().to(device)
        model.eval()
        
        # Get target layer
        target_layer = config.layer_getter(model, config.layer_path)
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        # Extract activations
        activations = None
        def hook(module, input, output):
            nonlocal activations
            activations = output
        
        handle = target_layer.register_forward_hook(hook)
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        handle.remove()
        
        # Validate shape
        expected_shape = (1, config.num_channels, config.spatial_size[0], config.spatial_size[1])
        actual_shape = tuple(activations.shape)
        
        if actual_shape != expected_shape:
            print(f"❌ Shape mismatch for {model_name}.{config.layer_path}:")
            print(f"   Expected: {expected_shape}")
            print(f"   Got:      {actual_shape}")
            return False
        
        print(f"✓ Valid: {model_name}.{config.layer_path} → {actual_shape}")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed for {model_name}.{layer_path}: {e}")
        return False


def validate_all_configs(device: str = 'cpu') -> Dict[str, bool]:
    """
    Validate all configurations.
    
    Args:
        device: Device to test on
    
    Returns:
        Dict mapping config names to validation results
    """
    print("=" * 80)
    print("Validating All Configurations")
    print("=" * 80 + "\n")
    
    results = {}
    for model_name, layers in MODEL_REGISTRY.items():
        for layer_path in layers.keys():
            config_name = f"{model_name}.{layer_path}"
            print(f"Testing {config_name}...")
            results[config_name] = validate_config(model_name, layer_path, device)
            print()
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print("=" * 80)
    print(f"Validation Summary: {passed}/{total} passed")
    print("=" * 80)
    
    if passed < total:
        print("\nFailed configurations:")
        for config_name, result in results.items():
            if not result:
                print(f"  ❌ {config_name}")
    
    return results



# ==========================================
# Example Usage
# ==========================================

if __name__ == "__main__":
    import sys
    
    # Print all configurations
    print_available_configs()
    
    # Example: Get a specific config
    print("\n" + "="*80)
    print("Example: Get VGG16 config")
    print("="*80)
    config = get_model_config('vgg16', 'features.23')
    print(f"Config: {config}")
    print(f"  Model: {config.model_name}")
    print(f"  Layer: {config.layer_path}")
    print(f"  Channels: {config.num_channels}")
    print(f"  Spatial: {config.spatial_size}")
    
    # Validate if torch is available
    try:
        import torch
        if '--validate' in sys.argv:
            print("\n" + "="*80)
            print("Running validation (use --validate flag)")
            print("="*80 + "\n")
            validate_all_configs(device='cpu')
    except ImportError:
        print("\nℹ PyTorch not installed, skipping validation")
        print("  Install PyTorch to use validation: pip install torch torchvision")