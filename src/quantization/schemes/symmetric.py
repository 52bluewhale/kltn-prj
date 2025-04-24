# Zero-centered quantization

import torch
import torch.nn as nn
import torch.nn.functional as F

# Required constants
QUANT_MIN_INT8 = -128
QUANT_MAX_INT8 = 127
QUANT_MIN_UINT8 = 0
QUANT_MAX_UINT8 = 255

def symmetric_quantize(x, scale, quant_min, quant_max):
    """
    Apply symmetric quantization to tensor.
    
    Args:
        x: Input tensor
        scale: Scale factor
        quant_min: Minimum quantized value
        quant_max: Maximum quantized value
        
    Returns:
        Quantized tensor
    """
    # Quantize
    x_q = torch.round(x / scale)
    
    # Clamp
    x_q = torch.clamp(x_q, quant_min, quant_max)
    
    # Dequantize
    x_dq = x_q * scale
    
    return x_dq

def symmetric_quantize_per_tensor(x, scale, quant_min=QUANT_MIN_INT8, quant_max=QUANT_MAX_INT8):
    """
    Apply symmetric per-tensor quantization.
    
    Args:
        x: Input tensor
        scale: Scale factor
        quant_min: Minimum quantized value
        quant_max: Maximum quantized value
        
    Returns:
        Quantized tensor
    """
    return symmetric_quantize(x, scale, quant_min, quant_max)

def symmetric_quantize_per_channel(x, scale, axis=0, quant_min=QUANT_MIN_INT8, quant_max=QUANT_MAX_INT8):
    """
    Apply symmetric per-channel quantization.
    
    Args:
        x: Input tensor
        scale: Scale factor per channel
        axis: Channel axis
        quant_min: Minimum quantized value
        quant_max: Maximum quantized value
        
    Returns:
        Quantized tensor
    """
    # Get shape for broadcasting
    shape = [1] * x.dim()
    shape[axis] = -1
    
    # Reshape scale for broadcasting
    scale_reshaped = scale.reshape(shape)
    
    # Quantize
    x_q = torch.round(x / scale_reshaped)
    
    # Clamp
    x_q = torch.clamp(x_q, quant_min, quant_max)
    
    # Dequantize
    x_dq = x_q * scale_reshaped
    
    return x_dq

def calculate_scale_symmetric(min_val, max_val, quant_min=QUANT_MIN_INT8, quant_max=QUANT_MAX_INT8):
    """
    Calculate scale for symmetric quantization.
    
    Args:
        min_val: Minimum value in tensor
        max_val: Maximum value in tensor
        quant_min: Minimum quantized value
        quant_max: Maximum quantized value
        
    Returns:
        Scale factor
    """
    # Get max absolute value
    max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
    
    # Calculate scale
    scale = max_abs / torch.max(torch.abs(torch.tensor(quant_min)), torch.abs(torch.tensor(quant_max)))
    
    return scale

def get_symmetric_qparams(min_val, max_val, quant_min=QUANT_MIN_INT8, quant_max=QUANT_MAX_INT8):
    """
    Get quantization parameters for symmetric quantization.
    
    Args:
        min_val: Minimum value in tensor
        max_val: Maximum value in tensor
        quant_min: Minimum quantized value
        quant_max: Maximum quantized value
        
    Returns:
        Scale and zero_point
    """
    scale = calculate_scale_symmetric(min_val, max_val, quant_min, quant_max)
    zero_point = torch.zeros_like(scale, dtype=torch.int64)
    
    return scale, zero_point

class SymmetricQuantizationConfig:
    """
    Configuration for symmetric quantization.
    """
    
    def __init__(self, bit_width=8, is_signed=True, per_channel=False, channel_axis=0):
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
    
    def quantize(self, x, scale):
        """
        Quantize tensor using this configuration.
        
        Args:
            x: Input tensor
            scale: Scale factor
            
        Returns:
            Quantized tensor
        """
        if self.per_channel:
            return symmetric_quantize_per_channel(
                x, scale, self.channel_axis, self.quant_min, self.quant_max
            )
        else:
            return symmetric_quantize_per_tensor(
                x, scale, self.quant_min, self.quant_max
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
        return get_symmetric_qparams(min_val, max_val, self.quant_min, self.quant_max)

# Factory function to create quantization config
def create_symmetric_quantization_config(bit_width=8, is_signed=True, per_channel=False, channel_axis=0):
    """
    Create symmetric quantization configuration.
    
    Args:
        bit_width: Bit width for quantization
        is_signed: Whether to use signed quantization
        per_channel: Whether to use per-channel quantization
        channel_axis: Channel axis for per-channel quantization
        
    Returns:
        SymmetricQuantizationConfig object
    """
    return SymmetricQuantizationConfig(bit_width, is_signed, per_channel, channel_axis)