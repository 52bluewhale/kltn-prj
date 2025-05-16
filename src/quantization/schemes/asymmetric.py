# Range-based quantization

import torch
import torch.nn as nn
import torch.nn.functional as F

# Required constants
QUANT_MIN_INT8 = -128
QUANT_MAX_INT8 = 127
QUANT_MIN_UINT8 = 0
QUANT_MAX_UINT8 = 255

def asymmetric_quantize(x, scale, zero_point, quant_min, quant_max):
    """
    Apply asymmetric quantization to tensor.
    
    Args:
        x: Input tensor
        scale: Scale factor
        zero_point: Zero point offset
        quant_min: Minimum quantized value
        quant_max: Maximum quantized value
        
    Returns:
        Quantized tensor
    """
    # Quantize
    x_q = torch.round(x / scale + zero_point)
    
    # Clamp
    x_q = torch.clamp(x_q, quant_min, quant_max)
    
    # Dequantize
    x_dq = (x_q - zero_point) * scale
    
    return x_dq

def asymmetric_quantize_per_tensor(x, scale, zero_point, quant_min=QUANT_MIN_UINT8, quant_max=QUANT_MAX_UINT8):
    """
    Apply asymmetric per-tensor quantization.
    
    Args:
        x: Input tensor
        scale: Scale factor
        zero_point: Zero point offset
        quant_min: Minimum quantized value
        quant_max: Maximum quantized value
        
    Returns:
        Quantized tensor
    """
    return asymmetric_quantize(x, scale, zero_point, quant_min, quant_max)

def asymmetric_quantize_per_channel(x, scale, zero_point, axis=0, quant_min=QUANT_MIN_UINT8, quant_max=QUANT_MAX_UINT8):
    """
    Apply asymmetric per-channel quantization.
    
    Args:
        x: Input tensor
        scale: Scale factor per channel
        zero_point: Zero point offset per channel
        axis: Channel axis
        quant_min: Minimum quantized value
        quant_max: Maximum quantized value
        
    Returns:
        Quantized tensor
    """
    # Get shape for broadcasting
    shape = [1] * x.dim()
    shape[axis] = -1
    
    # Reshape scale and zero_point for broadcasting
    scale_reshaped = scale.reshape(shape)
    zero_point_reshaped = zero_point.reshape(shape)
    
    # Quantize
    x_q = torch.round(x / scale_reshaped + zero_point_reshaped)
    
    # Clamp
    x_q = torch.clamp(x_q, quant_min, quant_max)
    
    # Dequantize
    x_dq = (x_q - zero_point_reshaped) * scale_reshaped
    
    return x_dq

def calculate_qparams_asymmetric(min_val, max_val, quant_min, quant_max):
    """
    Calculate quantization parameters for asymmetric quantization.
    
    Args:
        min_val: Minimum value in tensor
        max_val: Maximum value in tensor
        quant_min: Minimum quantized value
        quant_max: Maximum quantized value
        
    Returns:
        Scale and zero_point
    """
    # Ensure min_val and max_val are different
    if min_val.item() == max_val.item():
        # Handle constant tensor
        if min_val.item() == 0:
            # All zeros tensor
            scale = torch.tensor(1.0)
            zero_point = torch.tensor(0, dtype=torch.int64)
        else:
            # Non-zero constant tensor
            scale = max_val.abs() / (quant_max - quant_min) * 2.0
            zero_point = torch.tensor((quant_min + quant_max) // 2, dtype=torch.int64)
        return scale, zero_point
    
    # Calculate scale
    scale = (max_val - min_val) / (quant_max - quant_min)
    
    # Calculate zero point
    zero_point = quant_min - torch.round(min_val / scale)
    
    # Clamp zero_point
    zero_point = torch.clamp(zero_point, quant_min, quant_max).to(torch.int64)
    
    return scale, zero_point

def get_asymmetric_qparams(min_val, max_val, quant_min=QUANT_MIN_UINT8, quant_max=QUANT_MAX_UINT8):
    """
    Get quantization parameters for asymmetric quantization.
    
    Args:
        min_val: Minimum value in tensor
        max_val: Maximum value in tensor
        quant_min: Minimum quantized value
        quant_max: Maximum quantized value
        
    Returns:
        Scale and zero_point
    """
    return calculate_qparams_asymmetric(min_val, max_val, quant_min, quant_max)

class AsymmetricQuantizationConfig:
    """
    Configuration for asymmetric quantization.
    """
    
    def __init__(self, bit_width=8, is_signed=False, per_channel=False, channel_axis=0):
        """
        Initialize configuration.
        
        Args:
            bit_width: Bit width for quantization
            is_signed: Whether to use signed quantization
            per_channel: Whether to use per-channel quantization
            channel_axis: Channel axis for per-channel quantization
        """
        self.bit_width = bit_width
        self.is_signed = is_signed
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        
        # Calculate quantization range
        if is_signed:
            self.quant_min = -2 ** (bit_width - 1)
            self.quant_max = 2 ** (bit_width - 1) - 1
        else:
            self.quant_min = 0
            self.quant_max = 2 ** bit_width - 1
    
    def quantize(self, x, scale, zero_point):
        """
        Quantize tensor using this configuration.
        
        Args:
            x: Input tensor
            scale: Scale factor
            zero_point: Zero point offset
            
        Returns:
            Quantized tensor
        """
        if self.per_channel:
            return asymmetric_quantize_per_channel(
                x, scale, zero_point, self.channel_axis, self.quant_min, self.quant_max
            )
        else:
            return asymmetric_quantize_per_tensor(
                x, scale, zero_point, self.quant_min, self.quant_max
            )
    
    def calculate_qparams(self, min_val, max_val):
        """
        Calculate quantization parameters.
        
        Args:
            min_val: Minimum value in tensor
            max_val: Maximum value in tensor
            
        Returns:
            Scale and zero_point
        """
        return get_asymmetric_qparams(min_val, max_val, self.quant_min, self.quant_max)

# Factory function to create quantization config
def create_asymmetric_quantization_config(bit_width=8, is_signed=False, per_channel=False, channel_axis=0):
    """
    Create asymmetric quantization configuration.
    
    Args:
        bit_width: Bit width for quantization
        is_signed: Whether to use signed quantization
        per_channel: Whether to use per-channel quantization
        channel_axis: Channel axis for per-channel quantization
        
    Returns:
        AsymmetricQuantizationConfig object
    """
    return AsymmetricQuantizationConfig(bit_width, is_signed, per_channel, channel_axis)