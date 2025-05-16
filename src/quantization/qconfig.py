# Defines quantization configurations:
#   - QConfig objects that pair weight and activation observers
#   - Predefined configurations for common scenarios

import torch
from torch.quantization import QConfig
from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver, HistogramObserver
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
        )
    else:
        weight_fake_quant = weight_quantize.with_args(
            observer=weight_observer,
            quant_min=-128,
            quant_max=127,
            dtype=weight_dtype,
            qscheme=weight_qscheme,
        )
    
    return QConfig(activation=activation_fake_quant, weight=weight_fake_quant)

# Default QAT configuration
def get_default_qat_qconfig():
    """
    Returns the default QConfig for QAT.
    Uses per-channel quantization for weights and per-tensor for activations.
    """
    # Import necessary modules
    from torch.quantization import QConfig
    from torch.quantization.observer import MovingAverageMinMaxObserver, PerChannelMinMaxObserver
    from torch.quantization.fake_quantize import FakeQuantize
    
    # Create weight fake quantize with PerChannelMinMaxObserver (not MinMaxObserver)
    weight_fake_quantize = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,  # Changed from MinMaxObserver
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0
    )
    
    # Create activation fake quantize
    activation_fake_quantize = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine  # Use per_tensor for activations
    )
    
    # Create and return QConfig
    return QConfig(
        activation=activation_fake_quantize,
        weight=weight_fake_quantize
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
    Compatible with PyTorch 2.4.1.
    """
    from torch.quantization import QConfig
    from torch.quantization.observer import HistogramObserver, PerChannelMinMaxObserver
    from torch.quantization.fake_quantize import FakeQuantize
    
    # Create basic observer instances directly
    activation_observer = HistogramObserver
    weight_observer = PerChannelMinMaxObserver
    
    # Activation fake quantize for first layer
    activation_fake_quant = FakeQuantize.with_args(
        observer=activation_observer,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine
    )
    
    # Weight fake quantize for first layer
    weight_fake_quant = FakeQuantize.with_args(
        observer=weight_observer,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0
    )
    
    return QConfig(activation=activation_fake_quant, weight=weight_fake_quant)

# QConfig for last layer (output layer)
def get_last_layer_qconfig():
    """
    Returns a QConfig for the last layer.
    Uses higher precision for output layer which is critical for model accuracy.
    Compatible with PyTorch 2.4.1.
    """
    from torch.quantization import QConfig
    from torch.quantization.observer import HistogramObserver, PerChannelMinMaxObserver
    from torch.quantization.fake_quantize import FakeQuantize
    
    # Activation fake quantize for last layer - using histogram for better precision
    activation_fake_quant = FakeQuantize.with_args(
        observer=HistogramObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine
    )
    
    # Weight fake quantize for last layer
    weight_fake_quant = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0
    )
    
    return QConfig(activation=activation_fake_quant, weight=weight_fake_quant)

# QConfig using advanced LSQ quantization
def get_lsq_qconfig():
    """
    Returns a QConfig using Learned Step Size Quantization (LSQ).
    LSQ typically improves model accuracy by learning optimal quantization steps.
    Compatible with PyTorch 2.4.1.
    """
    from torch.quantization import QConfig
    from torch.quantization.observer import MovingAverageMinMaxObserver, MinMaxObserver
    
    # Create basic observer instances directly
    activation_observer = MovingAverageMinMaxObserver
    weight_observer = MinMaxObserver
    
    # For LSQ, we'll use your custom implementation
    # but ensure it's properly imported and compatible
    try:
        from .fake_quantize import LSQFakeQuantize
        
        # Activation fake quantize using LSQ
        activation_fake_quant = LSQFakeQuantize.with_args(
            observer=activation_observer,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine
        )
        
        # Weight fake quantize using LSQ
        weight_fake_quant = LSQFakeQuantize.with_args(
            observer=weight_observer,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        )
        
        return QConfig(activation=activation_fake_quant, weight=weight_fake_quant)
    
    except (ImportError, TypeError) as e:
        # Fallback to standard FakeQuantize if LSQ has issues
        print(f"Warning: LSQ initialization failed ({e}), falling back to standard quantization")
        from torch.quantization.fake_quantize import FakeQuantize
        
        activation_fake_quant = FakeQuantize.with_args(
            observer=activation_observer,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine
        )
        
        weight_fake_quant = FakeQuantize.with_args(
            observer=weight_observer,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        )
        
        return QConfig(activation=activation_fake_quant, weight=weight_fake_quant)

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
    
    # Create a default mapping
    default_qconf = get_qconfig_by_name(default_qconfig)
    qconfig_mapping = {}
    
    # Add global default with empty string key
    qconfig_mapping[""] = default_qconf
    
    # Map specific layers based on patterns
    for config in layer_configs:
        pattern = config["pattern"]
        qconfig_type = config.get("qconfig", "sensitive")
        qconfig_obj = get_qconfig_by_name(qconfig_type)
        
        # Check each module against the pattern
        for name, module in model.named_modules():
            if re.match(pattern, name):
                qconfig_mapping[name] = qconfig_obj
    
    # Log the mapping for debugging
    logger.info(f"Created QConfig mapping with {len(qconfig_mapping)} entries")
    
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

if __name__ == "__main__":
    # Test that we can create these configs without errors
    default_config = get_default_qat_qconfig()
    first_layer_config = get_first_layer_qconfig()
    
    print("Default QAT config created successfully:", default_config)
    print("First layer config created successfully:", first_layer_config)
    
    # Create a test module and apply config
    test_conv = torch.nn.Conv2d(3, 3, 3)
    test_conv.qconfig = default_config
    
    print("Config applied to test module:", test_conv.qconfig is not None)