# Defines quantization configurations:
#   - QConfig objects that pair weight and activation observers
#   - Predefined configurations for common scenarios

import torch
from torch.quantization import QConfig
from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver
from torch.quantization.fake_quantize import FakeQuantize

from .observers import CustomMinMaxObserver, PerChannelMinMaxObserver, HistogramObserver
from .fake_quantize import CustomFakeQuantize, PerChannelFakeQuantize, LSQFakeQuantize

# Helper function to create QConfig from parameters
def create_qconfig(
    activation_observer=MovingAverageMinMaxObserver,
    weight_observer=MinMaxObserver,
    activation_quantize=FakeQuantize,
    weight_quantize=FakeQuantize,
    activation_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    activation_qscheme=torch.per_tensor_affine,
    weight_qscheme=torch.per_channel_symmetric,
    activation_reduce_range=False,
    weight_reduce_range=False,
    weight_ch_axis=0,
):
    """
    Create custom QConfig with specified parameters.
    
    Args:
        activation_observer: Observer for activations
        weight_observer: Observer for weights
        activation_quantize: Fake quantize module for activations
        weight_quantize: Fake quantize module for weights
        activation_dtype: Data type for quantized activations
        weight_dtype: Data type for quantized weights
        activation_qscheme: Quantization scheme for activations
        weight_qscheme: Quantization scheme for weights
        activation_reduce_range: Reduce range for activations
        weight_reduce_range: Reduce range for weights
        weight_ch_axis: Channel axis for per-channel quantization
        
    Returns:
        QConfig object
    """
    # Define activation fake quantize
    activation_fake_quant = activation_quantize.with_args(
        observer=activation_observer,
        quant_min=0 if activation_dtype == torch.quint8 else -128,
        quant_max=255 if activation_dtype == torch.quint8 else 127,
        dtype=activation_dtype,
        qscheme=activation_qscheme,
        reduce_range=activation_reduce_range,
    )
    
    # Define weight fake quantize
    if weight_qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]:
        weight_fake_quant = weight_quantize.with_args(
            observer=weight_observer,
            quant_min=-128,
            quant_max=127,
            dtype=weight_dtype,
            qscheme=weight_qscheme,
            ch_axis=weight_ch_axis,
            reduce_range=weight_reduce_range,
        )
    else:
        weight_fake_quant = weight_quantize.with_args(
            observer=weight_observer,
            quant_min=-128,
            quant_max=127,
            dtype=weight_dtype,
            qscheme=weight_qscheme,
            reduce_range=weight_reduce_range,
        )
    
    return QConfig(activation=activation_fake_quant, weight=weight_fake_quant)

# Default QAT configuration
def get_default_qat_qconfig():
    """
    Returns the default QConfig for QAT.
    Uses per-channel quantization for weights and per-tensor for activations.
    """
    return create_qconfig(
        activation_observer=MovingAverageMinMaxObserver,
        weight_observer=MinMaxObserver,
        activation_quantize=CustomFakeQuantize,
        weight_quantize=PerChannelFakeQuantize,
        activation_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        activation_qscheme=torch.per_tensor_affine,
        weight_qscheme=torch.per_channel_symmetric,
        weight_ch_axis=0,
    )

# QConfig for sensitive layers (e.g., detection heads)
def get_sensitive_layer_qconfig():
    """
    Returns a QConfig for layers sensitive to quantization.
    Uses histogram observer for activations for more precise statistics.
    """
    return create_qconfig(
        activation_observer=HistogramObserver,
        weight_observer=PerChannelMinMaxObserver,
        activation_quantize=CustomFakeQuantize,
        weight_quantize=PerChannelFakeQuantize,
        activation_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        activation_qscheme=torch.per_tensor_affine,
        weight_qscheme=torch.per_channel_symmetric,
        weight_ch_axis=0,
    )

# QConfig for first convolutional layer
def get_first_layer_qconfig():
    """
    Returns a QConfig for the first layer.
    Uses higher precision for first layer which is critical for model accuracy.
    """
    return create_qconfig(
        activation_observer=HistogramObserver,
        weight_observer=PerChannelMinMaxObserver,
        activation_quantize=CustomFakeQuantize,
        weight_quantize=PerChannelFakeQuantize,
        activation_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        activation_qscheme=torch.per_tensor_affine,
        weight_qscheme=torch.per_channel_symmetric,
        weight_ch_axis=0,
    )

# QConfig for last layer (output layer)
def get_last_layer_qconfig():
    """
    Returns a QConfig for the last layer.
    Uses higher precision for output layer which is critical for model accuracy.
    """
    return create_qconfig(
        activation_observer=HistogramObserver,
        weight_observer=PerChannelMinMaxObserver,
        activation_quantize=CustomFakeQuantize,
        weight_quantize=PerChannelFakeQuantize,
        activation_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        activation_qscheme=torch.per_tensor_affine,
        weight_qscheme=torch.per_channel_symmetric,
        weight_ch_axis=0,
    )

# QConfig using advanced LSQ quantization
def get_lsq_qconfig():
    """
    Returns a QConfig using Learned Step Size Quantization (LSQ).
    LSQ typically improves model accuracy by learning optimal quantization steps.
    """
    return create_qconfig(
        activation_observer=MovingAverageMinMaxObserver,
        weight_observer=MinMaxObserver,
        activation_quantize=LSQFakeQuantize,
        weight_quantize=LSQFakeQuantize,
        activation_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        activation_qscheme=torch.per_tensor_affine,
        weight_qscheme=torch.per_channel_symmetric,
        weight_ch_axis=0,
    )

# Dictionary of available QConfigs
QAT_CONFIGS = {
    "default": get_default_qat_qconfig(),
    "sensitive": get_sensitive_layer_qconfig(),
    "first_layer": get_first_layer_qconfig(),
    "last_layer": get_last_layer_qconfig(),
    "lsq": get_lsq_qconfig(),
}

# Function to get QConfig by name
def get_qconfig_by_name(name):
    """
    Get QConfig by name.
    
    Args:
        name: Name of QConfig
        
    Returns:
        QConfig object
    """
    if name not in QAT_CONFIGS:
        raise ValueError(f"Unknown QConfig: {name}")
    return QAT_CONFIGS[name]

# Function to map layer names to QConfigs using regex patterns
def create_qconfig_mapping(model, layer_configs, default_qconfig="default"):
    """
    Create QConfig mapping for model layers.
    
    Args:
        model: Model to create mapping for
        layer_configs: Layer-specific configurations
        default_qconfig: Default QConfig name
        
    Returns:
        Dictionary mapping regex patterns to QConfigs
    """
    import re
    qconfig_mapping = {}
    
    # Map layer names to QConfigs
    for layer_name, _ in model.named_modules():
        # Default QConfig
        qconfig_mapping[layer_name] = get_qconfig_by_name(default_qconfig)
        
        # Check if layer matches any pattern
        for config in layer_configs:
            pattern = config["pattern"]
            if re.match(pattern, layer_name):
                # Get QConfig type
                qconfig_type = config.get("qconfig", "sensitive")
                qconfig_mapping[layer_name] = get_qconfig_by_name(qconfig_type)
                break
    
    return qconfig_mapping

# Function to prepare QAT config from YAML configuration
def prepare_qat_config_from_yaml(config):
    """
    Prepare QAT configuration from YAML.
    
    Args:
        config: YAML configuration
        
    Returns:
        Dictionary of QConfigs
    """
    qat_config = {}
    
    # Extract quantization settings
    quant_config = config.get("quantization", {})
    
    # Prepare default QConfig
    qat_config["default"] = create_qconfig(
        activation_observer=get_observer_by_name(
            quant_config.get("activation", {}).get("observer", "moving_average_minmax")
        ),
        weight_observer=get_observer_by_name(
            quant_config.get("weight", {}).get("observer", "minmax")
        ),
        activation_dtype=get_dtype_by_name(
            quant_config.get("activation", {}).get("dtype", "quint8")
        ),
        weight_dtype=get_dtype_by_name(
            quant_config.get("weight", {}).get("dtype", "qint8")
        ),
        activation_qscheme=get_qscheme_by_name(
            quant_config.get("activation", {}).get("qscheme", "per_tensor_affine")
        ),
        weight_qscheme=get_qscheme_by_name(
            quant_config.get("weight", {}).get("qscheme", "per_channel_symmetric")
        ),
    )
    
    # Prepare layer-specific QConfigs
    for layer_config in quant_config.get("layer_configs", []):
        layer_name = layer_config.get("pattern", "")
        layer_config_dict = layer_config.get("config", {})
        
        qat_config[layer_name] = create_qconfig(
            activation_observer=get_observer_by_name(
                layer_config_dict.get("activation", {}).get("observer", "moving_average_minmax")
            ),
            weight_observer=get_observer_by_name(
                layer_config_dict.get("weight", {}).get("observer", "minmax")
            ),
            activation_dtype=get_dtype_by_name(
                layer_config_dict.get("activation", {}).get("dtype", "quint8")
            ),
            weight_dtype=get_dtype_by_name(
                layer_config_dict.get("weight", {}).get("dtype", "qint8")
            ),
            activation_qscheme=get_qscheme_by_name(
                layer_config_dict.get("activation", {}).get("qscheme", "per_tensor_affine")
            ),
            weight_qscheme=get_qscheme_by_name(
                layer_config_dict.get("weight", {}).get("qscheme", "per_channel_symmetric")
            ),
        )
    
    return qat_config

# Helper functions
def get_observer_by_name(name):
    """
    Get observer class by name.
    
    Args:
        name: Observer name
        
    Returns:
        Observer class
    """
    observers = {
        "minmax": MinMaxObserver,
        "moving_average_minmax": MovingAverageMinMaxObserver,
        "histogram": HistogramObserver,
        "custom_minmax": CustomMinMaxObserver,
        "per_channel_minmax": PerChannelMinMaxObserver,
    }
    return observers.get(name, MovingAverageMinMaxObserver)

def get_dtype_by_name(name):
    """
    Get dtype by name.
    
    Args:
        name: Dtype name
        
    Returns:
        Torch dtype
    """
    dtypes = {
        "quint8": torch.quint8,
        "qint8": torch.qint8,
    }
    return dtypes.get(name, torch.quint8)

def get_qscheme_by_name(name):
    """
    Get quantization scheme by name.
    
    Args:
        name: Scheme name
        
    Returns:
        Torch qscheme
    """
    qschemes = {
        "per_tensor_affine": torch.per_tensor_affine,
        "per_tensor_symmetric": torch.per_tensor_symmetric,
        "per_channel_affine": torch.per_channel_affine,
        "per_channel_symmetric": torch.per_channel_symmetric,
    }
    return qschemes.get(name, torch.per_tensor_affine)