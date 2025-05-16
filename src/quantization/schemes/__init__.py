from .symmetric import (
    symmetric_quantize,
    symmetric_quantize_per_tensor,
    symmetric_quantize_per_channel,
    get_symmetric_qparams,
    create_symmetric_quantization_config,
)

from .asymmetric import (
    asymmetric_quantize,
    asymmetric_quantize_per_tensor,
    asymmetric_quantize_per_channel,
    get_asymmetric_qparams,
    create_asymmetric_quantization_config,
)

from .per_tensor import (
    PerTensorQuantizer,
    create_per_tensor_quantizer,
    UINT8_ASYMMETRIC,
    INT8_SYMMETRIC,
    INT8_ASYMMETRIC,
)

from .per_channel import (
    PerChannelQuantizer,
    create_per_channel_quantizer,
    INT8_SYMMETRIC_PER_CHANNEL,
    INT8_ASYMMETRIC_PER_CHANNEL,
    get_default_weight_quantizer,
    get_default_activation_quantizer,
)

# Common factory functions for easy access
def get_weight_quantizer(config):
    """
    Get weight quantizer from configuration.
    
    Args:
        config: Quantization configuration dictionary
        
    Returns:
        Weight quantizer
    """
    # Extract parameters from config
    weight_config = config.get("weight", {})
    bit_width = weight_config.get("bit_width", 8)
    scheme = weight_config.get("scheme", "per_channel")
    symmetric = weight_config.get("symmetric", True)
    
    if scheme == "per_channel":
        return create_per_channel_quantizer(
            bit_width=bit_width,
            symmetric=symmetric,
            is_signed=True,
            ch_axis=0
        )
    else:
        return create_per_tensor_quantizer(
            bit_width=bit_width,
            symmetric=symmetric,
            is_signed=True
        )

def get_activation_quantizer(config):
    """
    Get activation quantizer from configuration.
    
    Args:
        config: Quantization configuration dictionary
        
    Returns:
        Activation quantizer
    """
    # Extract parameters from config
    act_config = config.get("activation", {})
    bit_width = act_config.get("bit_width", 8)
    scheme = act_config.get("scheme", "per_tensor")
    symmetric = act_config.get("symmetric", False)
    
    if scheme == "per_channel":
        return create_per_channel_quantizer(
            bit_width=bit_width,
            symmetric=symmetric,
            is_signed=False,
            ch_axis=0
        )
    else:
        return create_per_tensor_quantizer(
            bit_width=bit_width,
            symmetric=symmetric,
            is_signed=False
        )