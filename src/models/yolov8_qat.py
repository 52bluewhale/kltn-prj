"""
YOLOv8 model with quantization-aware training support.

This module extends the base YOLOv8 model with QAT capabilities,
including specialized handling for detection heads and critical layers.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
import yaml
import os
import re

from .yolov8_base import YOLOv8BaseModel
from ..quantization.utils import (
    load_quantization_config,
    prepare_model_for_qat,
    convert_qat_model_to_quantized,
    apply_layer_specific_quantization,
    skip_layers_from_quantization
)
from ..quantization.qconfig import (
    create_qconfig_mapping,
    prepare_qat_config_from_yaml,
    get_qconfig_by_name
)
from ..quantization.fusion import fuse_yolov8_modules

# Setup logging
logger = logging.getLogger(__name__)

class YOLOv8QATModel(nn.Module):
    """
    YOLOv8 model adapted for Quantization-Aware Training.
    
    This class extends the base YOLOv8 model with quantization-aware training
    capabilities, including specialized handling for detection heads and critical layers.
    """
    
    def __init__(
        self,
        model: nn.Module,
        qconfig_dict: Optional[Dict] = None,
        skip_layers: Optional[List[Dict]] = None
    ):
        """
        Initialize QAT-ready YOLOv8 model.
        
        Args:
            model: Base YOLOv8 model
            qconfig_dict: Quantization configuration dictionary
            skip_layers: List of layer patterns to skip from quantization
        """
        super().__init__()
        
        self.model = model
        self.qconfig_dict = qconfig_dict
        self.skip_layers = skip_layers
        
        # Cache important layers for special handling
        self._cache_important_layers()
        
        # Copy metadata from original model if available
        if hasattr(model, 'model_variant'):
            self.model_variant = model.model_variant
        if hasattr(model, 'num_classes'):
            self.num_classes = model.num_classes
    
    def _cache_important_layers(self) -> None:
        """
        Cache important layers for special handling during QAT.
        
        Identifies and stores references to layers that need special treatment,
        such as the first layer, detection head, and other critical components.
        """
        self.first_conv = None
        self.detection_head = None
        self.critical_blocks = []
        
        # If model is a YOLOv8BaseModel, use its cached layers
        if isinstance(self.model, YOLOv8BaseModel):
            critical_layers = self.model.get_critical_layers()
            
            for name, module in critical_layers:
                if name.endswith('.conv') and self.first_conv is None:
                    self.first_conv = (name, module)
                elif 'Detect' in name or name.endswith('.detect'):
                    self.detection_head = (name, module)
                else:
                    self.critical_blocks.append((name, module))
        else:
            # Find first convolutional layer
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    self.first_conv = (name, module)
                    break
            
            # Find detection head
            for name, module in self.model.named_modules():
                if 'Detect' in name or name.endswith('.detect'):
                    self.detection_head = (name, module)
                    break
            
            # Find critical blocks (e.g., CSP blocks)
            for name, module in self.model.named_modules():
                if any(x in name for x in ['C2f', 'SPPF']):
                    self.critical_blocks.append((name, module))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        return self.model(x)
    
    def get_critical_layers(self) -> List[Tuple[str, nn.Module]]:
        """
        Get list of layers critical for model accuracy.
        
        Returns:
            List of (name, module) tuples for critical layers
        """
        critical_layers = []
        
        # First conv is critical
        if self.first_conv:
            critical_layers.append(self.first_conv)
        
        # Detection head is critical
        if self.detection_head:
            critical_layers.append(self.detection_head)
        
        # Add all critical blocks
        critical_layers.extend(self.critical_blocks)
        
        return critical_layers
    
    def save(self, path: str, include_qconfig: bool = True) -> None:
        """
        Save model to file.
        
        Args:
            path: Path to save model
            include_qconfig: Whether to include quantization configuration
            
        Returns:
            None
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {
            "model": self.model,
            "qat_config": self.qconfig_dict if include_qconfig else None,
            "skip_layers": self.skip_layers if include_qconfig else None
        }
        
        torch.save(save_dict, path)
        logger.info(f"QAT model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load model from file.
        
        Args:
            path: Path to load model from
            
        Returns:
            None
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        if "model" in checkpoint:
            self.model = checkpoint["model"]
        else:
            # Try to load as state dict
            self.model.load_state_dict(checkpoint)
        
        # Load QAT configuration if available
        if "qat_config" in checkpoint and checkpoint["qat_config"]:
            self.qconfig_dict = checkpoint["qat_config"]
        
        # Load skip layers if available
        if "skip_layers" in checkpoint and checkpoint["skip_layers"]:
            self.skip_layers = checkpoint["skip_layers"]
        
        # Re-cache important layers
        self._cache_important_layers()
        
        logger.info(f"QAT model loaded from {path}")
    
    def to_quantized(self) -> nn.Module:
        """
        Convert QAT model to fully quantized model.
        
        Returns:
            Quantized model
        """
        from torch.quantization import convert
        
        # Ensure model is in eval mode
        self.eval()
        
        # Convert model
        quantized_model = convert(self.model)
        
        return quantized_model


def prepare_yolov8_for_qat(
    model: Union[YOLOv8BaseModel, nn.Module],
    config_path: Optional[str] = None,
    qconfig_dict: Optional[Dict] = None
) -> YOLOv8QATModel:
    """
    Prepare YOLOv8 model for quantization-aware training.
    
    Args:
        model: YOLOv8 model to prepare
        config_path: Path to quantization configuration file
        qconfig_dict: Quantization configuration dictionary
        
    Returns:
        QAT-ready YOLOv8 model
    """
    # Extract model if wrapper
    if isinstance(model, YOLOv8BaseModel):
        yolo_model = model.model
        skip_patterns = _get_default_skip_patterns()
    else:
        yolo_model = model
        skip_patterns = _get_default_skip_patterns()
    
    # Load configuration if path provided
    if config_path is not None:
        config = load_quantization_config(config_path)
        qconfig_dict = prepare_qat_config_from_yaml(config)
        
        # Extract skip patterns from config if available
        if "quantization" in config and "skip_layers" in config["quantization"]:
            skip_patterns = config["quantization"]["skip_layers"]
    
    # Use PyTorch's QAT preparation
    prepared_model = prepare_model_for_qat(yolo_model, qconfig_dict=qconfig_dict)
    
    # Skip specific layers from quantization
    if skip_patterns:
        prepared_model = skip_layers_from_quantization(prepared_model, skip_patterns)
    
    # Create QAT model
    qat_model = YOLOv8QATModel(
        model=prepared_model,
        qconfig_dict=qconfig_dict,
        skip_layers=skip_patterns
    )
    
    return qat_model


def convert_yolov8_to_quantized(qat_model: YOLOv8QATModel) -> nn.Module:
    """
    Convert YOLOv8 QAT model to fully quantized model.
    
    Args:
        qat_model: QAT model to convert
        
    Returns:
        Quantized model
    """
    # Extract model if wrapper
    if isinstance(qat_model, YOLOv8QATModel):
        model = qat_model.model
    else:
        model = qat_model
    
    # Ensure model is in eval mode
    model.eval()
    
    # Convert model
    quantized_model = convert_qat_model_to_quantized(model)
    
    return quantized_model


def apply_qat_config_to_yolov8(
    model: Union[YOLOv8QATModel, nn.Module],
    config_path: str
) -> YOLOv8QATModel:
    """
    Apply quantization configuration to YOLOv8 model.
    
    Args:
        model: YOLOv8 model
        config_path: Path to configuration file
        
    Returns:
        YOLOv8 model with quantization configuration applied
    """
    # Load configuration
    config = load_quantization_config(config_path)
    
    # Extract model
    if isinstance(model, YOLOv8QATModel):
        yolo_model = model.model
    else:
        yolo_model = model
    
    # Apply layer-specific quantization based on config
    if "quantization" in config and "layer_configs" in config["quantization"]:
        layer_configs = config["quantization"]["layer_configs"]
        default_qconfig = config["quantization"].get("default_qconfig", "default")
        
        yolo_model = apply_layer_specific_quantization(
            yolo_model, layer_configs, default_qconfig
        )
    
    # Create/update QAT model
    if isinstance(model, YOLOv8QATModel):
        model.model = yolo_model
        model.qconfig_dict = prepare_qat_config_from_yaml(config)
        
        # Update skip layers if available
        if "quantization" in config and "skip_layers" in config["quantization"]:
            model.skip_layers = config["quantization"]["skip_layers"]
        
        return model
    else:
        return YOLOv8QATModel(
            model=yolo_model,
            qconfig_dict=prepare_qat_config_from_yaml(config),
            skip_layers=config.get("quantization", {}).get("skip_layers", None)
        )


def _get_default_skip_patterns() -> List[Dict]:
    """
    Get default patterns for layers to skip from quantization.
    
    Returns:
        List of skip patterns
    """
    return [
        {
            "pattern": r"model\.\d+\.forward",
            "reason": "Skip forward method"
        },
        {
            "pattern": r"model\.\d+\.detect",
            "reason": "Special handling for detection layers"
        }
    ]