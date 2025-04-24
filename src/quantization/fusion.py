# Implements module fusion for better quantization:
#     - Conv-BN-ReLU fusion
#     - Other common fusion patterns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization.fuse_modules import fuse_modules
import re


def fuse_conv_bn(conv, bn):
    """
    Fuse Conv2d and BatchNorm2d modules.
    
    Args:
        conv: Conv2d module
        bn: BatchNorm2d module
        
    Returns:
        Fused Conv2d module
    """
    # Get parameters
    w_conv = conv.weight.clone().detach()
    
    # Handle bias
    if conv.bias is not None:
        b_conv = conv.bias.clone().detach()
    else:
        b_conv = torch.zeros_like(bn.running_mean)
    
    # BatchNorm parameters
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    gamma = bn.weight
    beta = bn.bias
    
    # Fuse parameters
    w_fused = w_conv * (gamma / torch.sqrt(var + eps)).reshape(-1, 1, 1, 1)
    b_fused = beta + (b_conv - mean) * gamma / torch.sqrt(var + eps)
    
    # Create new Conv2d module
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    )
    
    # Set weights and bias
    fused_conv.weight.data = w_fused
    fused_conv.bias.data = b_fused
    
    return fused_conv


def fuse_conv_bn_relu(conv, bn, relu):
    """
    Fuse Conv2d, BatchNorm2d, and ReLU modules.
    
    Args:
        conv: Conv2d module
        bn: BatchNorm2d module
        relu: ReLU module
        
    Returns:
        Fused Conv2d module
    """
    # Fuse Conv and BN first
    fused_conv = fuse_conv_bn(conv, bn)
    
    # Create a new module that includes ReLU
    class ConvBnReLU(nn.Module):
        def __init__(self, conv):
            super(ConvBnReLU, self).__init__()
            self.conv = conv
        
        def forward(self, x):
            return F.relu(self.conv(x))
    
    return ConvBnReLU(fused_conv)


def fuse_conv_bn_silu(conv, bn, silu):
    """
    Fuse Conv2d, BatchNorm2d, and SiLU (Swish) modules.
    
    Args:
        conv: Conv2d module
        bn: BatchNorm2d module
        silu: SiLU module
        
    Returns:
        Fused Conv2d module
    """
    # Fuse Conv and BN first
    fused_conv = fuse_conv_bn(conv, bn)
    
    # Create a new module that includes SiLU
    class ConvBnSiLU(nn.Module):
        def __init__(self, conv):
            super(ConvBnSiLU, self).__init__()
            self.conv = conv
        
        def forward(self, x):
            x = self.conv(x)
            return x * torch.sigmoid(x)  # SiLU/Swish activation
    
    return ConvBnSiLU(fused_conv)


def find_modules_to_fuse(model, fusion_patterns):
    """
    Find modules to fuse in model based on fusion patterns.
    
    Args:
        model: Model to search
        fusion_patterns: List of patterns to search for
        
    Returns:
        List of lists of module names to fuse
    """
    modules_to_fuse = []
    named_modules = dict(model.named_modules())
    
    # Helper function to check if a module matches a pattern
    def matches_pattern(name, pattern):
        return re.match(pattern, name) is not None
    
    # Check each module
    for name, module in named_modules.items():
        for pattern in fusion_patterns:
            if matches_pattern(name, pattern["pattern"]):
                # Check if module has required components
                module_names = []
                current_name = name
                
                # Try to find all modules in the pattern
                all_found = True
                for module_type in pattern["modules"]:
                    if current_name in named_modules and isinstance(named_modules[current_name], get_module_class(module_type)):
                        module_names.append(current_name)
                        # For sequential modules, move to the next one
                        if current_name + ".1" in named_modules:
                            current_name = current_name + ".1"
                        elif current_name + ".bn" in named_modules and module_type == "conv":
                            current_name = current_name + ".bn"
                        elif current_name + ".act" in named_modules and (module_type == "bn" or module_type == "conv"):
                            current_name = current_name + ".act"
                        else:
                            # If no standard pattern is found, break the chain
                            all_found = False
                            break
                    else:
                        all_found = False
                        break
                
                if all_found and len(module_names) == len(pattern["modules"]):
                    modules_to_fuse.append(module_names)
    
    return modules_to_fuse


def get_module_class(module_type):
    """
    Get module class from type string.
    
    Args:
        module_type: String representation of module type
        
    Returns:
        Module class
    """
    if module_type == "conv":
        return nn.Conv2d
    elif module_type == "bn":
        return nn.BatchNorm2d
    elif module_type == "relu":
        return nn.ReLU
    elif module_type == "silu":
        # SiLU (Swish) could be implemented in different ways
        return (nn.SiLU, nn.Hardswish)
    else:
        raise ValueError(f"Unknown module type: {module_type}")


def fuse_model_modules(model, fusion_patterns, inplace=False):
    """
    Fuse modules in model based on fusion patterns.
    
    Args:
        model: Model to fuse
        fusion_patterns: List of fusion patterns
        inplace: Whether to modify model inplace
        
    Returns:
        Fused model
    """
    if not inplace:
        model = model.deepcopy()
    
    # Find modules to fuse
    modules_to_fuse = find_modules_to_fuse(model, fusion_patterns)
    
    # Apply fuser function for each pattern
    for module_names in modules_to_fuse:
        # Get first module type to determine fuser function
        first_module = model
        for name in module_names[0].split('.'):
            if name.isdigit():
                first_module = first_module[int(name)]
            else:
                first_module = getattr(first_module, name)
        
        # Determine fuser function based on modules
        if len(module_names) == 2:
            fuser_function = fuse_conv_bn
        elif len(module_names) == 3:
            # Check if third module is ReLU or SiLU
            third_module = model
            for name in module_names[2].split('.'):
                if name.isdigit():
                    third_module = third_module[int(name)]
                else:
                    third_module = getattr(third_module, name)
            
            if isinstance(third_module, nn.ReLU):
                fuser_function = fuse_conv_bn_relu
            else:
                fuser_function = fuse_conv_bn_silu
        else:
            continue
        
        # Apply fuser function
        fuse_modules(model, module_names, inplace=True, fuser_func=fuser_function)
    
    return model


def fuse_yolov8_modules(model, fusion_patterns=None):
    """
    Fuse modules in YOLOv8 model for better quantization.
    
    Args:
        model: YOLOv8 model
        fusion_patterns: Optional fusion patterns override
        
    Returns:
        Fused model
    """
    if fusion_patterns is None:
        # Default fusion patterns for YOLOv8
        fusion_patterns = [
            {
                "pattern": r"model\.\d+\.conv",
                "modules": ["conv", "bn"],
                "fuser_method": "fuse_conv_bn"
            },
            {
                "pattern": r"model\.\d+\.cv\d+\.conv",
                "modules": ["conv", "bn", "silu"],
                "fuser_method": "fuse_conv_bn_silu"
            },
            # Add more specific patterns as needed
        ]
    
    return fuse_model_modules(model, fusion_patterns, inplace=True)