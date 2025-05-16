import torch
import torch.nn as nn
import re
import yaml
import logging
import os
from collections import OrderedDict
from torch.quantization import get_default_qconfig, prepare_qat, convert

from .qconfig import create_qconfig_mapping, prepare_qat_config_from_yaml
from .fusion import fuse_yolov8_modules

import copy

# Setup logging
logger = logging.getLogger(__name__)

def load_quantization_config(config_path):
    """
    Load quantization configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading quantization config: {e}")
        return {}

def get_layer_names_matching_pattern(model, pattern):
    """
    Get names of layers matching regex pattern.
    
    Args:
        model: Model to search
        pattern: Regex pattern to match
        
    Returns:
        List of layer names
    """
    matching_layers = []
    
    for name, _ in model.named_modules():
        if re.match(pattern, name):
            matching_layers.append(name)
    
    return matching_layers

def prepare_model_for_qat(model, qconfig_dict=None, config_path=None, inplace=False):
    """
    Prepare model for quantization-aware training.
    
    Args:
        model: Model to prepare
        qconfig_dict: Quantization configuration dictionary
        config_path: Path to configuration file
        inplace: Whether to modify model inplace
        
    Returns:
        Prepared model
    """
    import copy
    import logging
    from torch.quantization import prepare_qat
    
    logger = logging.getLogger('utils')
    
    # Make a copy if not modifying inplace
    if not inplace:
        logger.info("Creating a copy of the model...")
        model_copy = copy.deepcopy(model)
        model = model_copy
    
    # Load config if path is provided
    if config_path is not None:
        logger.info(f"Loading configuration from {config_path}...")
        config = load_quantization_config(config_path)
        qconfig_dict = prepare_qat_config_from_yaml(config)
    
    # Fuse modules if possible
    try:
        logger.info("Attempting to fuse modules for quantization...")
        model = fuse_yolov8_modules(model)
    except Exception as e:
        logger.warning(f"Module fusion failed with error: {e}")
        logger.warning("Continuing without module fusion")
    
    # Create QConfig mapping
    if qconfig_dict is None:
        logger.info("Using default QConfig as none was provided")
        # Use default QConfig if not provided
        qconfig_dict = {"": get_default_qconfig()}

    # Set model to training mode - THIS IS THE KEY FIX
    logger.info("Setting model to training mode...")
    model.train()
    
    # Prepare model for QAT
    logger.info("Preparing model for quantization-aware training...")
    try:
        model = prepare_qat(model, qconfig_dict, inplace=True)
        logger.info("Model successfully prepared for QAT")
    except Exception as e:
        logger.error(f"Error preparing model for QAT: {e}")
        raise
    
    return model

def convert_qat_model_to_quantized(model, inplace=False):
    """
    Convert QAT model to quantized model.
    
    Args:
        model: QAT model to convert
        inplace: Whether to modify model inplace
        
    Returns:
        Quantized model
    """
    import copy
    import logging
    from torch.quantization import convert
    
    logger = logging.getLogger('utils')

    if not inplace:
        logger.info("Creating a copy of the model...")
        model = copy.deepcopy(model)  # Use Python's copy.deepcopy() instead of model.deepcopy()
    
    # Convert model to quantized version
    logger.info("Converting QAT model to quantized model...")
    try:
        model = convert(model, inplace=True)
        logger.info("Model successfully converted to INT8")
    except Exception as e:
        logger.error(f"Error converting model to quantized: {e}")
        raise
    
    return model

def apply_layer_specific_quantization(model, layer_configs, default_qconfig):
    """
    Apply layer-specific quantization settings.
    
    Args:
        model: Model to modify
        layer_configs: Layer-specific configurations
        default_qconfig: Default QConfig
        
    Returns:
        Model with layer-specific quantization
    """
    # Create mapping of layer names to QConfigs
    qconfig_mapping = create_qconfig_mapping(model, layer_configs, default_qconfig)
    
    # Apply QConfigs to layers
    for name, module in model.named_modules():
        if name in qconfig_mapping:
            module.qconfig = qconfig_mapping[name]
    
    return model

def skip_layers_from_quantization(model, skip_patterns):
    """
    Skip specified layers from quantization.
    
    Args:
        model: Model to modify
        skip_patterns: List of patterns for layers to skip
        
    Returns:
        Model with layers skipped
    """
    for pattern in skip_patterns:
        # Find layers matching pattern
        matching_layers = get_layer_names_matching_pattern(model, pattern["pattern"])
        
        # Remove qconfig from matching layers
        for name, module in model.named_modules():
            if name in matching_layers:
                module.qconfig = None
    
    return model

def get_model_size(model):
    """
    Get model size in megabytes.
    
    Args:
        model: Model to measure
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    # Calculate parameter size
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    # Calculate buffer size
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return total_size / 1024 / 1024  # Convert to MB

def compare_model_sizes(fp32_model, int8_model):
    """
    Compare sizes of FP32 and INT8 models.
    
    Args:
        fp32_model: FP32 model
        int8_model: INT8 model
        
    Returns:
        Dictionary with size information
    """
    fp32_size = get_model_size(fp32_model)
    int8_size = get_model_size(int8_model)
    
    compression_ratio = fp32_size / int8_size if int8_size > 0 else 0
    
    return {
        "fp32_size_mb": fp32_size,
        "int8_size_mb": int8_size,
        "compression_ratio": compression_ratio,
        "size_reduction_percent": (1 - int8_size / fp32_size) * 100 if fp32_size > 0 else 0
    }

def save_quantized_model(model, save_path, metadata=None):
    """
    Save quantized model to file.
    
    Args:
        model: Quantized model to save
        save_path: Path to save model
        metadata: Optional metadata to save with model
        
    Returns:
        Success flag
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Prepare model state dict
        state_dict = model.state_dict()
        
        # Add metadata if provided
        if metadata is not None:
            save_dict = {
                "state_dict": state_dict,
                "metadata": metadata
            }
        else:
            save_dict = state_dict
        
        # Save model
        torch.save(save_dict, save_path)
        logger.info(f"Quantized model saved to {save_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving quantized model: {e}")
        return False

def load_quantized_model(model, load_path):
    """
    Load quantized model from file.
    
    Args:
        model: Model to load into
        load_path: Path to load model from
        
    Returns:
        Loaded model and metadata
    """
    try:
        # Load saved dict
        saved_dict = torch.load(load_path, map_location=torch.device('cpu'))
        
        # Extract state dict and metadata
        if isinstance(saved_dict, dict) and "state_dict" in saved_dict:
            state_dict = saved_dict["state_dict"]
            metadata = saved_dict.get("metadata", None)
        else:
            state_dict = saved_dict
            metadata = None
        
        # Load state dict
        model.load_state_dict(state_dict)
        logger.info(f"Quantized model loaded from {load_path}")
        
        return model, metadata
    except Exception as e:
        logger.error(f"Error loading quantized model: {e}")
        return None, None

def analyze_quantization_effects(model):
    """
    Analyze the effects of quantization on model.
    
    Args:
        model: Quantized model to analyze
        
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    # Count quantized and non-quantized modules
    quantized_modules = 0
    total_modules = 0
    
    for name, module in model.named_modules():
        total_modules += 1
        if hasattr(module, 'qconfig') and module.qconfig is not None:
            quantized_modules += 1
    
    results["quantized_ratio"] = quantized_modules / total_modules if total_modules > 0 else 0
    results["total_modules"] = total_modules
    results["quantized_modules"] = quantized_modules
    
    return results

def get_quantization_parameters(model):
    """
    Get quantization parameters from model.
    
    Args:
        model: Quantized model
        
    Returns:
        Dictionary with quantization parameters
    """
    params = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'scale') and hasattr(module, 'zero_point'):
            if isinstance(module.scale, torch.Tensor) and isinstance(module.zero_point, torch.Tensor):
                params[name] = {
                    "scale": module.scale.detach().cpu().numpy().tolist(),
                    "zero_point": module.zero_point.detach().cpu().numpy().tolist()
                }
    
    return params

def measure_layer_wise_quantization_error(fp32_model, int8_model, test_input):
    """
    Measure layer-wise quantization error.
    
    Args:
        fp32_model: FP32 model
        int8_model: INT8 model
        test_input: Test input tensor
        
    Returns:
        Dictionary with layer-wise error metrics
    """
    errors = {}
    
    # Register hooks to collect outputs
    fp32_outputs = {}
    int8_outputs = {}
    
    def fp32_hook(name):
        def hook(module, input, output):
            fp32_outputs[name] = output.detach().cpu()
        return hook
    
    def int8_hook(name):
        def hook(module, input, output):
            int8_outputs[name] = output.detach().cpu()
        return hook
    
    # Register hooks for each module
    fp32_handles = []
    int8_handles = []
    
    for name, module in fp32_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            handle = module.register_forward_hook(fp32_hook(name))
            fp32_handles.append(handle)
    
    for name, module in int8_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            handle = module.register_forward_hook(int8_hook(name))
            int8_handles.append(handle)
    
    # Run inference
    with torch.no_grad():
        fp32_model(test_input)
        int8_model(test_input)
    
    # Calculate errors
    for name in fp32_outputs:
        if name in int8_outputs:
            fp32_out = fp32_outputs[name]
            int8_out = int8_outputs[name]
            
            # Ensure same shape
            if fp32_out.shape == int8_out.shape:
                # Calculate error metrics
                abs_error = torch.abs(fp32_out - int8_out).mean().item()
                rel_error = (torch.abs(fp32_out - int8_out) / (torch.abs(fp32_out) + 1e-8)).mean().item()
                
                errors[name] = {
                    "abs_error": abs_error,
                    "rel_error": rel_error
                }
    
    # Remove hooks
    for handle in fp32_handles:
        handle.remove()
    
    for handle in int8_handles:
        handle.remove()
    
    return errors