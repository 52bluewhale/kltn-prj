# Handles layers sensitive to quantization:
#     - Identifies critical layers that affect accuracy
#     - Implements special quantization schemes for these layers

"""
Critical layer management for YOLOv8 QAT.

This module provides utilities for identifying and handling layers that
are critical for model accuracy during quantization-aware training.
"""

import torch
import torch.nn as nn
import re
import logging
from typing import Dict, List, Optional, Union, Tuple, Any

from ..quantization.qconfig import (
    get_qconfig_by_name,
    create_qconfig_mapping
)

# Setup logging
logger = logging.getLogger(__name__)

def get_critical_layers(
    model: nn.Module
) -> List[Tuple[str, nn.Module]]:
    """
    Get list of layers critical for model accuracy.
    
    Args:
        model: YOLOv8 model
        
    Returns:
        List of (name, module) tuples for critical layers
    """
    critical_layers = []
    
    # Get layer patterns
    patterns = _get_critical_layer_patterns()
    
    # Find layers matching patterns
    for name, module in model.named_modules():
        for pattern in patterns:
            if re.match(pattern, name):
                critical_layers.append((name, module))
                break
    
    logger.info(f"Found {len(critical_layers)} critical layers")
    
    return critical_layers


def apply_special_quantization(
    model: nn.Module,
    qconfig_mapping: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Apply special quantization to critical layers.
    
    Args:
        model: YOLOv8 model
        qconfig_mapping: Mapping of layer names to QConfigs
        
    Returns:
        Model with special quantization applied to critical layers
    """
    # Get critical layers
    critical_layers = get_critical_layers(model)
    
    # Get configuration for each category of critical layers
    if qconfig_mapping is None:
        qconfig_mapping = _get_default_qconfig_mapping()
    
    # Apply specialized QConfig to each critical layer
    for name, module in critical_layers:
        # Find appropriate QConfig
        qconfig = None
        for pattern, config in qconfig_mapping.items():
            if re.match(pattern, name):
                qconfig = config
                break
        
        # Apply QConfig if found
        if qconfig is not None and hasattr(module, 'qconfig'):
            module.qconfig = qconfig
            logger.info(f"Applied special quantization to {name}")
    
    return model


def skip_critical_layers_from_quantization(
    model: nn.Module,
    skip_patterns: Optional[List[str]] = None
) -> nn.Module:
    """
    Skip specified critical layers from quantization.
    
    Args:
        model: YOLOv8 model
        skip_patterns: List of regex patterns for layers to skip
        
    Returns:
        Model with layers skipped from quantization
    """
    if skip_patterns is None:
        skip_patterns = _get_layers_to_skip()
    
    # Find layers matching patterns
    for name, module in model.named_modules():
        for pattern in skip_patterns:
            if re.match(pattern, name):
                # Remove QConfig to skip quantization
                if hasattr(module, 'qconfig'):
                    module.qconfig = None
                    logger.info(f"Skipped layer from quantization: {name}")
                break
    
    return model


def analyze_layer_sensitivity(
    model: nn.Module,
    calibration_fn: callable,
    test_fn: callable,
    num_trials: int = 3
) -> Dict[str, float]:
    """
    Analyze sensitivity of model layers to quantization.
    
    This function quantizes each layer individually and measures the
    impact on model accuracy to identify the most sensitive layers.
    
    Args:
        model: YOLOv8 model
        calibration_fn: Function to calibrate model
        test_fn: Function to test model accuracy
        num_trials: Number of trials for each layer
        
    Returns:
        Dictionary mapping layer names to sensitivity scores
    """
    # Get baseline accuracy
    model_copy = torch.quantization.quantize_dynamic(
        model.to('cpu').eval(),
        {nn.Conv2d, nn.Linear},
        dtype=torch.qint8
    )
    baseline_accuracy = test_fn(model_copy)
    
    # Test each layer
    sensitivity = {}
    critical_layers = get_critical_layers(model)
    
    for name, module in critical_layers:
        # Skip layers that can't be quantized
        if not isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            continue
        
        # Measure accuracy impact
        accuracies = []
        for _ in range(num_trials):
            # Create model with only this layer quantized
            model_copy = model.copy()
            
            # Skip all layers except this one
            for n, m in model_copy.named_modules():
                if hasattr(m, 'qconfig'):
                    m.qconfig = None
            
            # Enable quantization for this layer
            for n, m in model_copy.named_modules():
                if n == name and hasattr(m, 'qconfig'):
                    m.qconfig = torch.quantization.default_qconfig
            
            # Calibrate and test
            model_copy = calibration_fn(model_copy)
            accuracy = test_fn(model_copy)
            accuracies.append(accuracy)
        
        # Calculate average accuracy drop
        avg_accuracy = sum(accuracies) / len(accuracies)
        sensitivity[name] = baseline_accuracy - avg_accuracy
        
        logger.info(f"Layer {name} sensitivity: {sensitivity[name]:.4f}")
    
    return sensitivity


def _get_critical_layer_patterns() -> List[str]:
    """
    Get regex patterns for critical layers.
    
    Returns:
        List of regex patterns
    """
    return [
        # First layer is critical for accuracy
        r"model\.0\.conv",
        # Detection head is critical for accuracy
        r"model\.\d+\.detect",
        r"model\.\d+\.Detect",
        # Bottleneck layers in detection head are critical
        r"model\.\d+\.cv\d+",
        # Spatial pyramid pooling layers are critical
        r"model\.\d+\.m\.\d+\.cv\d+",
        r"model\.\d+\.SPPF",
        # Channel attention layers are critical
        r".*\.ca",
        # C2f blocks are critical connection points
        r"model\.\d+\.C2f"
    ]


def _get_layers_to_skip() -> List[str]:
    """
    Get regex patterns for layers to skip from quantization.
    
    Returns:
        List of regex patterns
    """
    return [
        # Skip detection head to preserve accuracy
        r"model\.\d+\.detect",
        r"model\.\d+\.Detect",
        # Skip forward method
        r"model\.\d+\.forward",
        # Skip distribution focal loss
        r"model\.\d+\.dfl\.conv"
    ]


def _get_default_qconfig_mapping() -> Dict[str, Any]:
    """
    Get default QConfig mapping for critical layers.
    
    Returns:
        Dictionary mapping patterns to QConfigs
    """
    return {
        # First convolution layer uses sensitive layer QConfig
        r"model\.0\.conv": get_qconfig_by_name("first_layer"),
        # Detection head uses sensitive layer QConfig
        r"model\.\d+\.detect": get_qconfig_by_name("sensitive"),
        r"model\.\d+\.Detect": get_qconfig_by_name("sensitive"),
        # DFL needs higher precision
        r"model\.\d+\.dfl\.conv": get_qconfig_by_name("sensitive"),
        # Other critical layers use default QConfig
        r"model\.\d+\.m\.\d+\.cv\d+": get_qconfig_by_name("default"),
        r"model\.\d+\.SPPF": get_qconfig_by_name("default"),
        r"model\.\d+\.C2f": get_qconfig_by_name("default")
    }