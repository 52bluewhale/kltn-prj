# Implements observer classes that collect statistics about tensor values:
#   - MinMaxObserver:               Records min/max values
#   - MovingAverageMinMaxObserver:  Tracks running min/max
#   - HistogramObserver:            For more sophisticated calibration

import torch
import torch.nn as nn
from torch.quantization.observer import _ObserverBase, MinMaxObserver, MovingAverageMinMaxObserver
import torch.nn.functional as F

class CustomMinMaxObserver(_ObserverBase):
    """
    Custom MinMax Observer for more precise quantization.
    Tracks min and max values with momentum.
    """
    
    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                ch_axis=0, momentum=0.1, eps=1e-5):
        """
        Initialize custom observer.
        
        Args:
            dtype: Quantized data type
            qscheme: Quantization scheme
            ch_axis: Channel axis for per-channel quantization
            momentum: Momentum for moving average
            eps: Small value for numerical stability
        """
        super().__init__(dtype=dtype)
        self.qscheme = qscheme
        self.ch_axis = ch_axis
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.bool))
    
    def forward(self, x_orig):
        """
        Forward pass to observe tensor values.
        
        Args:
            x_orig: Input tensor
            
        Returns:
            Input tensor (unchanged)
        """
        x = x_orig.detach()
        
        if x.numel() == 0:
            return x_orig
        
        min_val = torch.min(x)
        max_val = torch.max(x)
        
        if not self.initialized:
            self.min_val.copy_(min_val)
            self.max_val.copy_(max_val)
            self.initialized.copy_(torch.tensor(1, dtype=torch.bool))
        else:
            self.min_val.copy_(torch.min(self.min_val * (1 - self.momentum) + min_val * self.momentum, min_val))
            self.max_val.copy_(torch.max(self.max_val * (1 - self.momentum) + max_val * self.momentum, max_val))
        
        return x_orig
    
    def calculate_qparams(self):
        """
        Calculate quantization parameters.
        
        Returns:
            scale and zero_point
        """
        if not self.initialized:
            return torch.tensor([1.0]), torch.tensor([0])
        
        min_val = self.min_val
        max_val = self.max_val
        
        # Handle case where min=max
        if min_val == max_val:
            scale = torch.tensor(1.0)
            zero_point = torch.tensor(0)
            return scale, zero_point
        
        if self.qscheme == torch.per_tensor_symmetric or self.qscheme == torch.per_channel_symmetric:
            # Symmetric quantization
            max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
            scale = max_abs / ((self.quant_max - self.quant_min) / 2)
            zero_point = torch.zeros_like(scale, dtype=torch.int32)
        else:
            # Affine quantization
            scale = (max_val - min_val) / (self.quant_max - self.quant_min)
            zero_point = self.quant_min - torch.round(min_val / scale)
            zero_point = torch.clamp(zero_point, self.quant_min, self.quant_max).to(torch.int32)
        
        return scale, zero_point
    
    def extra_repr(self):
        return f"dtype={self.dtype}, qscheme={self.qscheme}, ch_axis={self.ch_axis}, momentum={self.momentum}"


class PerChannelMinMaxObserver(_ObserverBase):
    """
    Per-channel min-max observer for weights.
    """
    
    def __init__(self, dtype=torch.qint8, qscheme=torch.per_channel_symmetric,
                ch_axis=0, eps=1e-5):
        """
        Initialize per-channel observer.
        
        Args:
            dtype: Quantized data type
            qscheme: Quantization scheme
            ch_axis: Channel axis for per-channel quantization
            eps: Small value for numerical stability
        """
        super().__init__(dtype=dtype)
        self.qscheme = qscheme
        self.ch_axis = ch_axis
        self.eps = eps
        self.register_buffer('min_vals', None)
        self.register_buffer('max_vals', None)
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.bool))
    
    def forward(self, x_orig):
        """
        Forward pass to observe tensor values.
        
        Args:
            x_orig: Input tensor
            
        Returns:
            Input tensor (unchanged)
        """
        x = x_orig.detach()
        
        if x.numel() == 0:
            return x_orig
        
        # Reshape tensor to get min/max per channel
        x_dim = x.size()
        new_shape = [1] * len(x_dim)
        new_shape[self.ch_axis] = x_dim[self.ch_axis]
        x_reshaped = x.reshape(x_dim[self.ch_axis], -1)
        
        # Get min and max per channel
        min_vals = torch.min(x_reshaped, dim=1)[0]
        max_vals = torch.max(x_reshaped, dim=1)[0]
        
        if not self.initialized:
            self.min_vals = min_vals
            self.max_vals = max_vals
            self.initialized.copy_(torch.tensor(1, dtype=torch.bool))
        else:
            self.min_vals = torch.min(self.min_vals, min_vals)
            self.max_vals = torch.max(self.max_vals, max_vals)
        
        return x_orig
    
    def calculate_qparams(self):
        """
        Calculate per-channel quantization parameters.
        
        Returns:
            scales and zero_points
        """
        if not self.initialized:
            return torch.tensor([1.0]), torch.tensor([0])
        
        min_vals = self.min_vals
        max_vals = self.max_vals
        
        # Handle case where min=max for each channel
        same_vals = min_vals == max_vals
        if same_vals.any():
            max_vals[same_vals] = min_vals[same_vals] + 1e-5
        
        if self.qscheme == torch.per_channel_symmetric:
            # Symmetric quantization
            max_abs = torch.max(torch.abs(min_vals), torch.abs(max_vals))
            scales = max_abs / ((self.quant_max - self.quant_min) / 2)
            zero_points = torch.zeros_like(scales, dtype=torch.int32)
        else:
            # Affine quantization
            scales = (max_vals - min_vals) / (self.quant_max - self.quant_min)
            zero_points = self.quant_min - torch.round(min_vals / scales)
            zero_points = torch.clamp(zero_points, self.quant_min, self.quant_max).to(torch.int32)
        
        return scales, zero_points
    
    def extra_repr(self):
        return f"dtype={self.dtype}, qscheme={self.qscheme}, ch_axis={self.ch_axis}"


class HistogramObserver(_ObserverBase):
    """
    Histogram-based observer for more precise quantization.
    Uses histogram of values to determine optimal scale and zero-point.
    """
    
    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 bins=2048, upsample_rate=1, eps=1e-5):
        """
        Initialize histogram observer.
        
        Args:
            dtype: Quantized data type
            qscheme: Quantization scheme
            bins: Number of histogram bins
            upsample_rate: Bin upsampling for higher precision
            eps: Small value for numerical stability
        """
        super().__init__(dtype=dtype)
        self.qscheme = qscheme
        self.bins = bins
        self.upsample_rate = upsample_rate
        self.eps = eps
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))
        self.register_buffer('histogram', torch.zeros(self.bins))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.bool))
    
    def _compute_histogram(self, x, min_val, max_val):
        """
        Compute histogram of tensor values.
        
        Args:
            x: Input tensor
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Histogram tensor
        """
        # Handle case where min=max
        if min_val == max_val:
            min_val = min_val - 0.5
            max_val = max_val + 0.5
        
        hist_range = max_val - min_val
        # Compute histogram
        hist = torch.histc(x, self.bins, min=min_val, max=max_val)
        return hist
    
    def forward(self, x_orig):
        """
        Forward pass to observe tensor values.
        
        Args:
            x_orig: Input tensor
            
        Returns:
            Input tensor (unchanged)
        """
        x = x_orig.detach()
        
        if x.numel() == 0:
            return x_orig
        
        min_val = torch.min(x)
        max_val = torch.max(x)
        
        if not self.initialized:
            self.min_val.copy_(min_val)
            self.max_val.copy_(max_val)
            self.histogram.copy_(self._compute_histogram(x, min_val, max_val))
            self.initialized.copy_(torch.tensor(1, dtype=torch.bool))
        else:
            self.min_val.copy_(torch.min(self.min_val, min_val))
            self.max_val.copy_(torch.max(self.max_val, max_val))
            # Update histogram with new range
            self.histogram.copy_(self._compute_histogram(x, self.min_val, self.max_val))
        
        return x_orig
    
    def _compute_quantization_params(self, hist, min_val, max_val):
        """
        Compute optimal quantization parameters from histogram.
        
        Args:
            hist: Histogram tensor
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            scale and zero_point
        """
        if self.qscheme == torch.per_tensor_symmetric:
            # For symmetric quantization, use max absolute value
            max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
            scale = max_abs / ((self.quant_max - self.quant_min) / 2)
            zero_point = torch.zeros_like(scale, dtype=torch.int32)
        else:
            # For affine quantization, use histogram to find optimal params
            bin_width = (max_val - min_val) / self.bins
            scale = bin_width
            
            # Compute cumulative histogram
            cumsum = torch.cumsum(hist, dim=0)
            
            # Find threshold that minimizes quantization error
            threshold = cumsum[-1] * 0.95  # 95th percentile
            zero_point_idx = torch.nonzero(cumsum >= threshold)[0]
            zero_point = self.quant_min + zero_point_idx
            zero_point = torch.clamp(zero_point, self.quant_min, self.quant_max).to(torch.int32)
        
        return scale, zero_point
    
    def calculate_qparams(self):
        """
        Calculate quantization parameters using histogram.
        
        Returns:
            scale and zero_point
        """
        if not self.initialized:
            return torch.tensor([1.0]), torch.tensor([0])
        
        return self._compute_quantization_params(self.histogram, self.min_val, self.max_val)
    
    def extra_repr(self):
        return f"dtype={self.dtype}, qscheme={self.qscheme}, bins={self.bins}"


# Helper function to get the correct observer based on configuration
def get_observer(observer_type, dtype, qscheme, ch_axis=0):
    """
    Get observer instance based on type.
    
    Args:
        observer_type: Type of observer
        dtype: Quantized data type
        qscheme: Quantization scheme
        ch_axis: Channel axis for per-channel quantization
        
    Returns:
        Observer instance
    """
    if observer_type == "minmax":
        if qscheme == torch.per_channel_symmetric or qscheme == torch.per_channel_affine:
            return PerChannelMinMaxObserver(dtype=dtype, qscheme=qscheme, ch_axis=ch_axis)
        else:
            return MinMaxObserver(dtype=dtype, qscheme=qscheme)
    elif observer_type == "moving_average_minmax":
        return MovingAverageMinMaxObserver(dtype=dtype, qscheme=qscheme)
    elif observer_type == "histogram":
        return HistogramObserver(dtype=dtype, qscheme=qscheme)
    else:
        raise ValueError(f"Unknown observer type: {observer_type}")