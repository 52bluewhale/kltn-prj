"""
Base YOLOv8 model implementation that serves as the foundation for QAT.

This module provides wrapper classes and utilities for working with YOLOv8 models
from the Ultralytics library, preparing them for quantization-aware training.
"""

import os
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
import yaml

# Try to import ultralytics
try:
    from ultralytics.models.yolo.model import YOLO, YOLOWorld
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import intersect_dicts
except ImportError:
    logging.warning("Ultralytics not found. Please install with: pip install ultralytics")

# Setup logging
logger = logging.getLogger(__name__)

class YOLOv8BaseModel(nn.Module):
    """
    Base wrapper for YOLOv8 models from Ultralytics.
    
    This class provides a standardized interface for working with YOLOv8 models,
    making them compatible with the quantization-aware training pipeline.
    """
    
    def __init__(
        self,
        model_variant: str = "yolov8n",
        num_classes: int = 80,
        pretrained: bool = True,
        pretrained_weights: Optional[str] = None
    ):
        """
        Initialize YOLOv8 model.
        
        Args:
            model_variant: YOLOv8 variant ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
            num_classes: Number of classes for detection
            pretrained: Whether to use pretrained weights
            pretrained_weights: Path to pretrained weights file
        """
        super().__init__()
        
        self.model_variant = model_variant
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        # Load model
        self.model = self._load_model(model_variant, num_classes, pretrained, pretrained_weights)
        
        # Cache important layers for later use in quantization
        self._cache_important_layers()
    
    def _load_model(
        self,
        model_variant: str,
        num_classes: int,
        pretrained: bool,
        pretrained_weights: Optional[str]
    ) -> nn.Module:
        """
        Load YOLOv8 model from Ultralytics.
        
        Args:
            model_variant: YOLOv8 variant
            num_classes: Number of classes
            pretrained: Whether to use pretrained weights
            pretrained_weights: Path to pretrained weights
            
        Returns:
            YOLOv8 model
        """
        # Create model
        if pretrained and pretrained_weights is None:
            # Use Ultralytics' pretrained model
            model = YOLO(f"{model_variant}.pt").model
            
            # If num_classes doesn't match pretrained, modify the detection head
            if num_classes != 80:  # COCO has 80 classes
                self._modify_detection_head(model, num_classes)
        else:
            # Create from scratch or custom weights
            model = DetectionModel(cfg=f"{model_variant}.yaml", nc=num_classes)
            
            # Load custom weights if provided
            if pretrained_weights is not None and os.path.exists(pretrained_weights):
                ckpt = torch.load(pretrained_weights, map_location='cpu')
                csd = ckpt['model'].float().state_dict()
                csd = intersect_dicts(csd, model.state_dict())  # Intersect
                model.load_state_dict(csd, strict=False)
                logger.info(f"Loaded weights from {pretrained_weights}")
        
        return model
    
    def _modify_detection_head(self, model: nn.Module, num_classes: int) -> None:
        """
        Modify detection head for custom number of classes.
        
        Args:
            model: YOLOv8 model
            num_classes: New number of classes
            
        Returns:
            None, modifies model in place
        """
        # Find detection head
        detection_head = model.model[-1]
        
        # Check if it's a detection head
        if hasattr(detection_head, 'nc'):
            # Create new head with the right number of classes
            old_nc = detection_head.nc
            detection_head.nc = num_classes
            
            # Adjust output layers for new number of classes
            for i, layer in enumerate(detection_head.cv2):
                if isinstance(layer, nn.Conv2d):
                    # Calculate new output channels
                    old_out = layer.weight.shape[0]
                    new_out = old_out - old_nc + num_classes
                    
                    # Create new conv layer with adjusted output size
                    new_layer = nn.Conv2d(
                        in_channels=layer.in_channels,
                        out_channels=new_out,
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                        bias=layer.bias is not None
                    )
                    
                    # Copy weights for shared dimensions
                    min_out = min(old_out, new_out)
                    new_layer.weight.data[:min_out] = layer.weight.data[:min_out]
                    if layer.bias is not None:
                        new_layer.bias.data[:min_out] = layer.bias.data[:min_out]
                    
                    # Replace layer
                    detection_head.cv2[i] = new_layer
            
            logger.info(f"Modified detection head from {old_nc} to {num_classes} classes")
    
    def _cache_important_layers(self) -> None:
        """
        Cache important layers for quantization-aware training.
        
        Identifies and stores references to layers that need special handling during QAT,
        such as the first layer, detection head, and other critical components.
        """
        self.first_conv = None
        self.detection_head = None
        self.critical_blocks = []
        
        # Find first convolutional layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.first_conv = (name, module)
                break
        
        # Find detection head
        if hasattr(self.model, 'model'):
            self.detection_head = ('model.' + str(len(self.model.model) - 1), self.model.model[-1])
        
        # Find critical blocks (e.g., CSP blocks)
        for name, module in self.model.named_modules():
            if name.startswith('model.') and any(x in name for x in ['C2f', 'SPPF', 'Detect']):
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
    
    def save(self, path: str) -> None:
        """
        Save model to file.
        
        Args:
            path: Path to save model
            
        Returns:
            None
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"model": self.model}, path)
        logger.info(f"Model saved to {path}")
    
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
            self.model.load_state_dict(checkpoint)
        
        # Re-cache important layers
        self._cache_important_layers()
        
        logger.info(f"Model loaded from {path}")


def load_yolov8_from_ultralytics(
    model_path: str,
    num_classes: Optional[int] = None
) -> YOLOv8BaseModel:
    """
    Load YOLOv8 model from Ultralytics checkpoint.
    
    Args:
        model_path: Path to model file (.pt)
        num_classes: Number of classes (override checkpoint value)
        
    Returns:
        YOLOv8BaseModel instance
    """
    # Determine model variant from checkpoint
    ckpt = torch.load(model_path, map_location='cpu')
    
    if "model" in ckpt:
        if hasattr(ckpt["model"], "yaml"):
            model_variant = os.path.splitext(os.path.basename(ckpt["model"].yaml["model"]))[0]
        else:
            # Try to infer from model structure
            model_variant = "yolov8n"  # Default
            model_size = sum(p.numel() for p in ckpt["model"].parameters())
            
            if model_size > 90_000_000:
                model_variant = "yolov8x"
            elif model_size > 40_000_000:
                model_variant = "yolov8l"
            elif model_size > 20_000_000:
                model_variant = "yolov8m"
            elif model_size > 10_000_000:
                model_variant = "yolov8s"
    else:
        model_variant = "yolov8n"  # Default
    
    # Get number of classes from checkpoint if not provided
    if num_classes is None:
        if "model" in ckpt and hasattr(ckpt["model"], "nc"):
            num_classes = ckpt["model"].nc
        else:
            num_classes = 80  # Default to COCO
    
    # Create model
    model = YOLOv8BaseModel(
        model_variant=model_variant,
        num_classes=num_classes,
        pretrained=False
    )
    
    # Load checkpoint
    model.load(model_path)
    
    return model


def get_yolov8_model(
    model_name: str = "yolov8n",
    num_classes: int = 80,
    pretrained: bool = True,
    pretrained_path: Optional[str] = None
) -> YOLOv8BaseModel:
    """
    Get YOLOv8 model with specified parameters.
    
    Args:
        model_name: YOLOv8 model variant (yolov8n, yolov8s, yolov8m, etc.)
        num_classes: Number of classes for classification head
        pretrained: Whether to use pretrained weights
        pretrained_path: Path to pretrained weights
        
    Returns:
        YOLOv8BaseModel instance
    """
    # Check if using custom weights
    if pretrained_path is not None:
        return load_yolov8_from_ultralytics(pretrained_path, num_classes)
    
    # Create new model
    return YOLOv8BaseModel(
        model_variant=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_weights=None
    )