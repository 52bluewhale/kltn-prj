# Functions to prepare models for QAT:
#     - Inserts fake quantization nodes
#     - Performs necessary fusions
#     - Configures layer-specific quantization parameters

"""
Model transformation utilities for YOLOv8 QAT.

This module provides functions to transform YOLOv8 models for quantization-aware 
training, including fusion of operations and insertion of fake quantization nodes.
"""

import torch
import torch.nn as nn
import re
import logging
from typing import Dict, List, Optional, Union, Tuple, Any

from ..quantization.fusion import (
    fuse_conv_bn,
    fuse_conv_bn_relu,
    fuse_conv_bn_silu,
    fuse_yolov8_modules,
    find_modules_to_fuse
)
from ..quantization.qconfig import (
    create_qconfig_mapping,
    get_qconfig_by_name
)
from .yolov8_qat_modules import (
    QATDetectionHead,
    QATCSPLayer,
    QATBottleneckCSP
)

# Setup logging
logger = logging.getLogger(__name__)

def apply_qat_transforms(
    model: nn.Module,
    qconfig_dict: Optional[Dict] = None,
    fusion_patterns: Optional[List[Dict]] = None,
    convert_custom_modules: bool = True
) -> nn.Module:
    """
    Apply all necessary transformations for QAT to a YOLOv8 model.
    
    Args:
        model: YOLOv8 model to transform
        qconfig_dict: Quantization configuration dictionary
        fusion_patterns: List of fusion patterns
        convert_custom_modules: Whether to convert YOLOv8-specific modules
        
    Returns:
        Transformed model
    """
    # Step 1: Fuse modules (Conv+BN, etc.)
    if fusion_patterns is None:
        fusion_patterns = _get_default_fusion_patterns()
    
    logger.info("Fusing modules for quantization...")
    model = fuse_yolov8_model_modules(model, fusion_patterns)
    
    # Step 2: Insert fake quantize nodes
    logger.info("Inserting fake quantization nodes...")
    model = insert_fake_quantize_nodes(model, qconfig_dict)
    
    # Step 3: Convert custom YOLOv8 modules to QAT versions
    if convert_custom_modules:
        logger.info("Converting custom YOLOv8 modules to QAT versions...")
        model = convert_yolov8_modules_to_qat(model, qconfig_dict)
    
    return model


def fuse_yolov8_model_modules(
    model: nn.Module,
    fusion_patterns: Optional[List[Dict]] = None
) -> nn.Module:
    """
    Fuse modules in YOLOv8 model for better quantization.
    
    Args:
        model: YOLOv8 model
        fusion_patterns: List of fusion patterns
        
    Returns:
        Model with fused modules
    """
    # Use default patterns if none provided
    if fusion_patterns is None:
        fusion_patterns = _get_default_fusion_patterns()
    
    # Fuse modules using fusion utility
    return fuse_yolov8_modules(model, fusion_patterns)


def insert_fake_quantize_nodes(
    model: nn.Module,
    qconfig_dict: Optional[Dict] = None
) -> nn.Module:
    """
    Insert fake quantization nodes into the model.
    
    Args:
        model: YOLOv8 model
        qconfig_dict: Quantization configuration dictionary
        
    Returns:
        Model with fake quantization nodes
    """
    from torch.quantization import prepare_qat
    
    # Use default QConfig if not provided
    if qconfig_dict is None:
        from torch.quantization import get_default_qat_qconfig
        qconfig_dict = {"": get_default_qat_qconfig()}
    
    # Prepare model for QAT using PyTorch's utility
    model = prepare_qat(model, qconfig_dict, inplace=True)
    
    return model


def convert_yolov8_modules_to_qat(
    model: nn.Module,
    qconfig_dict: Optional[Dict] = None
) -> nn.Module:
    """
    Convert YOLOv8-specific modules to their QAT versions.
    
    Args:
        model: YOLOv8 model
        qconfig_dict: Quantization configuration dictionary
        
    Returns:
        Model with QAT modules
    """
    # Get default QConfig if not provided
    default_qconfig = None
    if qconfig_dict is not None and "" in qconfig_dict:
        default_qconfig = qconfig_dict[""]
    
    # Process model to replace custom modules
    for name, module in list(model.named_children()):
        # Recursively process children
        if len(list(module.children())) > 0:
            model._modules[name] = convert_yolov8_modules_to_qat(module, qconfig_dict)
        
        # Convert detection head
        elif _is_detection_head(module):
            # Get specific QConfig for detection head if available
            qconfig = _get_qconfig_for_module(name, qconfig_dict, default_qconfig)
            model._modules[name] = QATDetectionHead.from_float(module, qconfig)
            logger.info(f"Converted detection head: {name}")
        
        # Convert CSP layer
        elif _is_csp_layer(module):
            # Get specific QConfig for CSP layer if available
            qconfig = _get_qconfig_for_module(name, qconfig_dict, default_qconfig)
            model._modules[name] = QATCSPLayer.from_float(module, qconfig)
            logger.info(f"Converted CSP layer: {name}")
        
        # Convert BottleneckCSP
        elif _is_bottleneck_csp(module):
            # Get specific QConfig for BottleneckCSP if available
            qconfig = _get_qconfig_for_module(name, qconfig_dict, default_qconfig)
            model._modules[name] = QATBottleneckCSP.from_float(module, qconfig)
            logger.info(f"Converted BottleneckCSP: {name}")
    
    return model


def remove_fake_quantize_nodes(model: nn.Module) -> nn.Module:
    """
    Remove fake quantization nodes from the model.
    
    Args:
        model: YOLOv8 model
        
    Returns:
        Model without fake quantization nodes
    """
    # Helper function to remove attributes related to quantization
    def _remove_qat_attributes(module):
        if hasattr(module, 'qconfig'):
            delattr(module, 'qconfig')
        if hasattr(module, 'weight_fake_quant'):
            delattr(module, 'weight_fake_quant')
        if hasattr(module, 'activation_post_process'):
            delattr(module, 'activation_post_process')
    
    # Process all modules
    for module in model.modules():
        _remove_qat_attributes(module)
    
    return model


def _get_default_fusion_patterns() -> List[Dict]:
    """
    Get default fusion patterns for YOLOv8.
    
    Returns:
        List of fusion patterns
    """
    return [
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
        {
            "pattern": r"model\.\d+\.m\.\d+\.cv\d+\.conv",
            "modules": ["conv", "bn", "silu"],
            "fuser_method": "fuse_conv_bn_silu"
        }
    ]


def _get_qconfig_for_module(
    name: str,
    qconfig_dict: Dict,
    default_qconfig: Optional[Any] = None
) -> Optional[Any]:
    """
    Get QConfig for a specific module based on name.
    
    Args:
        name: Module name
        qconfig_dict: Dictionary of QConfigs
        default_qconfig: Default QConfig
        
    Returns:
        QConfig for the module
    """
    if qconfig_dict is None:
        return default_qconfig
    
    # Check if there's a direct match
    if name in qconfig_dict:
        return qconfig_dict[name]
    
    # Check for pattern matches
    for pattern, qconfig in qconfig_dict.items():
        if pattern != "" and re.match(pattern, name):
            return qconfig
    
    # Return default
    return default_qconfig


def _is_detection_head(module: nn.Module) -> bool:
    """
    Check if a module is a YOLOv8 detection head.
    
    Args:
        module: Module to check
        
    Returns:
        True if module is a detection head, False otherwise
    """
    # Check class name
    if 'Detect' in module.__class__.__name__:
        return True
    
    # Check for typical detection head attributes
    has_typical_attrs = hasattr(module, 'nc') and hasattr(module, 'no')
    has_cv2 = hasattr(module, 'cv2') and isinstance(module.cv2, nn.ModuleList)
    
    return has_typical_attrs and has_cv2


def _is_csp_layer(module: nn.Module) -> bool:
    """
    Check if a module is a YOLOv8 CSP layer.
    
    Args:
        module: Module to check
        
    Returns:
        True if module is a CSP layer, False otherwise
    """
    # Check class name
    if 'CSP' in module.__class__.__name__ and 'Bottleneck' not in module.__class__.__name__:
        return True
    
    # Check for typical CSP layer attributes
    has_cv1_cv2 = hasattr(module, 'cv1') and hasattr(module, 'cv2')
    has_m_or_cv3 = hasattr(module, 'm') or hasattr(module, 'cv3')
    
    return has_cv1_cv2 and has_m_or_cv3


def _is_bottleneck_csp(module: nn.Module) -> bool:
    """
    Check if a module is a YOLOv8 BottleneckCSP.
    
    Args:
        module: Module to check
        
    Returns:
        True if module is a BottleneckCSP, False otherwise
    """
    # Check class name
    if 'BottleneckCSP' in module.__class__.__name__:
        return True
    
    # Check for typical BottleneckCSP attributes
    has_cv1_cv2_cv3_cv4 = (
        hasattr(module, 'cv1') and hasattr(module, 'cv2') and
        hasattr(module, 'cv3') and hasattr(module, 'cv4')
    )
    
    return has_cv1_cv2_cv3_cv4