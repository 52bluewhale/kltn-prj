# Whole tensor quantization

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization.fake_quantize import FakeQuantize

from .symmetric import symmetric_quantize_per_tensor, get_symmetric_qparams
from .asymmetric import asymmetric_quantize_per_tensor, get_asymmetric_qparams

class PerTensorQuantizer:
    """
    Quantizer for per-tensor quantization.
    """
    
    def __init__(self, bit_width=8, symmetric=False, is_signed=None, reduce_range=False):
        """
        Initialize per-tensor quantizer.
        
        Args:
            bit_width: Bit width for quantization
            symmetric: Whether to use symmetric quantization
            is_signed: Whether to use signed quantization (if None, determined from symmetric)
            reduce_range: Whether to reduce quantization range for compatibility
        """
        self.bit_width = bit_width
        self.symmetric = symmetric
        
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
        Quantize tensor.
        
        Args:
            x: Input tensor
            scale: Scale factor
            zero_point: Zero point offset (for asymmetric quantization)
            
        Returns:
            Quantized tensor
        """
        if self.symmetric:
            return symmetric_quantize_per_tensor(x, scale, self.quant_min, self.quant_max)
        else:
            if zero_point is None:
                raise ValueError("zero_point must be provided for asymmetric quantization")
            return asymmetric_quantize_per_tensor(x, scale, zero_point, self.quant_min, self.quant_max)
    
    def calculate_qparams(self, min_val, max_val):
        """
        Calculate quantization parameters.
        
        Args:
            min_val: Minimum value in tensor
            max_val: Maximum value in tensor
            
        Returns:
            Scale and zero_point
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
            qscheme = torch.per_tensor_symmetric
        else:
            qscheme = torch.per_tensor_affine
        
        dtype = torch.qint8 if self.is_signed else torch.quint8
        
        # Create and return FakeQuantize module
        return FakeQuantize.with_args(
            observer=observer_class,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=self.reduce_range
        )

# Factory function to create per-tensor quantizer
def create_per_tensor_quantizer(bit_width=8, symmetric=False, is_signed=None, reduce_range=False):
    """
    Create per-tensor quantizer.
    
    Args:
        bit_width: Bit width for quantization
        symmetric: Whether to use symmetric quantization
        is_signed: Whether to use signed quantization (if None, determined from symmetric)
        reduce_range: Whether to reduce quantization range for compatibility
        
    Returns:
        PerTensorQuantizer object
    """
    return PerTensorQuantizer(bit_width, symmetric, is_signed, reduce_range)

# Create common quantizer configurations
UINT8_ASYMMETRIC = create_per_tensor_quantizer(bit_width=8, symmetric=False, is_signed=False)
INT8_SYMMETRIC = create_per_tensor_quantizer(bit_width=8, symmetric=True, is_signed=True)
INT8_ASYMMETRIC = create_per_tensor_quantizer(bit_width=8, symmetric=False, is_signed=True)