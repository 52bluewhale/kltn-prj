# Channel-wise quantization

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization.fake_quantize import FakeQuantize

from .symmetric import symmetric_quantize_per_channel, get_symmetric_qparams
from .asymmetric import asymmetric_quantize_per_channel, get_asymmetric_qparams

class PerChannelQuantizer:
    """
    Quantizer for per-channel quantization.
    """
    
    def __init__(self, bit_width=8, symmetric=True, is_signed=None, ch_axis=0, reduce_range=False):
        """
        Initialize per-channel quantizer.
        
        Args:
            bit_width: Bit width for quantization
            symmetric: Whether to use symmetric quantization
            is_signed: Whether to use signed quantization (if None, determined from symmetric)
            ch_axis: Channel axis for per-channel quantization
            reduce_range: Whether to reduce quantization range for compatibility
        """
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.ch_axis = ch_axis
        
        # Determine if using signed quantization
        if is_signed is None:
            # Default: unsigned for asymmetric, signed for symmetric
            self.is_signed = symmetric
        else:
            self.is_signed = is_signed
        
        self.reduce_range = reduce_range
        
        # Calculate quantization range
        if self.is_signed:
            self.quant_min = -2 ** (bit_width - 1 - (1 if reduce_range else 0))
            self.quant_max = 2 ** (bit_width - 1 - (1 if reduce_range else 0)) - 1
        else:
            self.quant_min = 0
            self.quant_max = 2 ** (bit_width - (1 if reduce_range else 0)) - 1
    
    def quantize(self, x, scale, zero_point=None):
        """
        Quantize tensor per-channel.
        
        Args:
            x: Input tensor
            scale: Scale factor per channel
            zero_point: Zero point offset per channel (for asymmetric quantization)
            
        Returns:
            Quantized tensor
        """
        if self.symmetric:
            return symmetric_quantize_per_channel(x, scale, self.ch_axis, self.quant_min, self.quant_max)
        else:
            if zero_point is None:
                raise ValueError("zero_point must be provided for asymmetric quantization")
            return asymmetric_quantize_per_channel(x, scale, zero_point, self.ch_axis, self.quant_min, self.quant_max)
    
    def calculate_qparams(self, min_val, max_val):
        """
        Calculate quantization parameters per-channel.
        
        Args:
            min_val: Minimum value in tensor per channel
            max_val: Maximum value in tensor per channel
            
        Returns:
            Scale and zero_point per channel
        """
        if self.symmetric:
            return get_symmetric_qparams(min_val, max_val, self.quant_min, self.quant_max)
        else:
            return get_asymmetric_qparams(min_val, max_val, self.quant_min, self.quant_max)
    
    def to_fake_quantize(self, observer_class):
        """
        Create FakeQuantize module with this quantizer's settings.
        
        Args:
            observer_class: Observer class to use
            
        Returns:
            FakeQuantize module
        """
        if self.symmetric:
            qscheme = torch.per_channel_symmetric
        else:
            qscheme = torch.per_channel_affine
        
        dtype = torch.qint8 if self.is_signed else torch.quint8
        
        # Create and return FakeQuantize module
        return FakeQuantize.with_args(
            observer=observer_class,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            dtype=dtype,
            qscheme=qscheme,
            ch_axis=self.ch_axis,
            reduce_range=self.reduce_range
        )

# Factory function to create per-channel quantizer
def create_per_channel_quantizer(bit_width=8, symmetric=True, is_signed=None, ch_axis=0, reduce_range=False):
    """
    Create per-channel quantizer.
    
    Args:
        bit_width: Bit width for quantization
        symmetric: Whether to use symmetric quantization
        is_signed: Whether to use signed quantization
        ch_axis: Channel axis for per-channel quantization
        reduce_range: Whether to reduce quantization range for compatibility
        
    Returns:
        PerChannelQuantizer object
    """
    return PerChannelQuantizer(bit_width, symmetric, is_signed, ch_axis, reduce_range)

# Create common quantizer configurations
INT8_SYMMETRIC_PER_CHANNEL = create_per_channel_quantizer(bit_width=8, symmetric=True, is_signed=True, ch_axis=0)
INT8_ASYMMETRIC_PER_CHANNEL = create_per_channel_quantizer(bit_width=8, symmetric=False, is_signed=True, ch_axis=0)

# Helper function to get appropriate quantizer for weights or activations
def get_default_weight_quantizer(per_channel=True, ch_axis=0, bit_width=8):
    """
    Get default quantizer for weights.
    Typically uses symmetric per-channel quantization.
    
    Args:
        per_channel: Whether to use per-channel quantization
        ch_axis: Channel axis
        bit_width: Bit width
        
    Returns:
        Weight quantizer
    """
    if per_channel:
        return create_per_channel_quantizer(bit_width=bit_width, symmetric=True, is_signed=True, ch_axis=ch_axis)
    else:
        # Fallback to per-tensor for special cases
        from .per_tensor import create_per_tensor_quantizer
        return create_per_tensor_quantizer(bit_width=bit_width, symmetric=True, is_signed=True)

def get_default_activation_quantizer(per_channel=False, ch_axis=0, bit_width=8):
    """
    Get default quantizer for activations.
    Typically uses asymmetric per-tensor quantization.
    
    Args:
        per_channel: Whether to use per-channel quantization
        ch_axis: Channel axis
        bit_width: Bit width
        
    Returns:
        Activation quantizer
    """
    if per_channel:
        return create_per_channel_quantizer(bit_width=bit_width, symmetric=False, is_signed=False, ch_axis=ch_axis)
    else:
        # Standard per-tensor for activations
        from .per_tensor import create_per_tensor_quantizer
        return create_per_tensor_quantizer(bit_width=bit_width, symmetric=False, is_signed=False)