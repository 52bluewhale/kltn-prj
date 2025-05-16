"""
YOLOv8 model implementations with quantization-aware training support.
This module provides the core model components for YOLOv8 QAT implementation.
"""

from .yolov8_base import (
    YOLOv8BaseModel,
    load_yolov8_from_ultralytics,
    get_yolov8_model
)

from .yolov8_qat import (
    QuantizedYOLOv8
)

from .yolov8_qat_modules import (
    QATDetectionHead,
    QATCSPLayer,
    QATBottleneckCSP
)

from .model_transforms import (
    apply_qat_transforms,
    insert_fake_quantize_nodes,
    fuse_yolov8_model_modules
)

from .critical_layers import (
    get_critical_layers,
    apply_special_quantization,
    skip_critical_layers_from_quantization
)

# Main API functions for easy access
def create_yolov8_model(model_name="yolov8n", num_classes=10, pretrained=True, pretrained_path=None):
    """
    Create YOLOv8 model with specified parameters.
    
    Args:
        model_name: YOLOv8 model variant (yolov8n, yolov8s, yolov8m, etc.)
        num_classes: Number of classes for classification head
        pretrained: Whether to use pretrained weights
        pretrained_path: Path to pretrained weights
        
    Returns:
        YOLOv8BaseModel instance
    """
    return get_yolov8_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_path=pretrained_path
    )

def prepare_model_for_qat(model, config_path=None, critical_layers=True, fuse_modules=True):
    """
    Prepare YOLOv8 model for QAT.
    
    Args:
        model: YOLOv8 model to prepare
        config_path: Path to QAT configuration file
        critical_layers: Whether to handle critical layers
        fuse_modules: Whether to fuse modules (Conv+BN, etc.)
        
    Returns:
        YOLOv8QATModel instance
    """
    if fuse_modules:
        model = fuse_yolov8_model_modules(model)
    
    qat_model = prepare_yolov8_for_qat(model, config_path)
    
    if critical_layers:
        qat_model = apply_special_quantization(qat_model)
    
    return qat_model

def convert_to_quantized(qat_model):
    """
    Convert QAT model to fully quantized model.
    
    Args:
        qat_model: QAT model to convert
        
    Returns:
        Fully quantized model
    """
    return convert_yolov8_to_quantized(qat_model)