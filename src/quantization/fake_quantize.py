# Contains fake quantization modules that simulate quantization:
#   - FakeQuantize: Base class for fake quantization
#   - Specialized variants for different quantization schemes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization.fake_quantize import FakeQuantize
from .observers import get_observer

class CustomFakeQuantize(FakeQuantize):
    """
    Custom FakeQuantize module with improved gradient approximation.
    Uses Straight-Through Estimator (STE) with a smoother gradient.
    """
    
    def __init__(self, observer, quant_min, quant_max, **observer_kwargs):
        """
        Initialize custom fake quantize module.
        
        Args:
            observer: Observer class for collecting statistics
            quant_min: Minimum quantized value
            quant_max: Maximum quantized value
            observer_kwargs: Additional arguments for observer
        """
        super().__init__(observer, quant_min, quant_max, **observer_kwargs)
        self.grad_factor = 1.0  # Factor for gradient scaling
    
    def forward(self, x):
        """
        Forward pass with quantization and dequantization.
        
        Args:
            x: Input tensor
            
        Returns:
            Quantized and dequantized tensor
        """
        if self.training:
            # Update observer statistics
            self.activation_post_process(x)
            
            # Get quantization parameters
            scale, zero_point = self.calculate_qparams()
            self.scale.copy_(scale)
            self.zero_point.copy_(zero_point)
            
            # Quantize and dequantize in forward pass
            x_q = torch.fake_quantize_per_tensor_affine(
                x, scale.item(), int(zero_point.item()), 
                self.quant_min, self.quant_max)
            
            # Apply STE with smoother gradient in backward pass
            return x_q + (x - x_q).detach() * self.grad_factor
        else:
            # In eval mode, just apply quantization
            return torch.fake_quantize_per_tensor_affine(
                x, self.scale.item(), int(self.zero_point.item()),
                self.quant_min, self.quant_max)


class PerChannelFakeQuantize(FakeQuantize):
    """
    Per-channel fake quantization for weights.
    """
    
    def __init__(self, observer, quant_min, quant_max, ch_axis=0, **observer_kwargs):
        """
        Initialize per-channel fake quantize module.
        
        Args:
            observer: Observer class for collecting statistics
            quant_min: Minimum quantized value
            quant_max: Maximum quantized value
            ch_axis: Channel axis for per-channel quantization
            observer_kwargs: Additional arguments for observer
        """
        super().__init__(observer, quant_min, quant_max, ch_axis=ch_axis, **observer_kwargs)
        self.ch_axis = ch_axis
        self.grad_factor = 1.0  # Factor for gradient scaling
    
    def forward(self, x):
        """
        Forward pass with per-channel quantization.
        
        Args:
            x: Input tensor
            
        Returns:
            Quantized and dequantized tensor
        """
        if self.training:
            # Update observer statistics
            self.activation_post_process(x)
            
            # Get quantization parameters
            scales, zero_points = self.calculate_qparams()
            self.scale.copy_(scales)
            self.zero_point.copy_(zero_points)
            
            # Apply per-channel fake quantization
            x_q = torch.fake_quantize_per_channel_affine(
                x, scales, zero_points.to(torch.int32), 
                self.ch_axis, self.quant_min, self.quant_max)
            
            # Apply STE with smoother gradient in backward pass
            return x_q + (x - x_q).detach() * self.grad_factor
        else:
            # In eval mode, just apply quantization
            return torch.fake_quantize_per_channel_affine(
                x, self.scale, self.zero_point.to(torch.int32),
                self.ch_axis, self.quant_min, self.quant_max)


class LSQFakeQuantize(FakeQuantize):
    """
    Learned Step Size Quantization (LSQ).
    LSQ learns the quantization step size as a model parameter.
    """
    
    def __init__(self, observer, quant_min, quant_max, **observer_kwargs):
        """
        Initialize LSQ fake quantize module.
        
        Args:
            observer: Observer class for collecting statistics
            quant_min: Minimum quantized value
            quant_max: Maximum quantized value
            observer_kwargs: Additional arguments for observer
        """
        super().__init__(observer, quant_min, quant_max, **observer_kwargs)
        self.register_parameter('step_size', 
                                nn.Parameter(torch.tensor([1.0])))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.bool))
    
    def _initialize_step_size(self, x):
        """
        Initialize step size based on the range of input tensor.
        
        Args:
            x: Input tensor
        """
        with torch.no_grad():
            min_val = torch.min(x)
            max_val = torch.max(x)
            
            # Handle constant tensor case
            if min_val == max_val:
                min_val = torch.tensor(-1.0)
                max_val = torch.tensor(1.0)
            
            if self.quant_min < 0:
                # Symmetric quantization (for weights)
                max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
                self.step_size.copy_(2 * max_abs / (self.quant_max - self.quant_min))
            else:
                # Asymmetric quantization (for activations)
                self.step_size.copy_((max_val - min_val) / (self.quant_max - self.quant_min))
                
            # Ensure step size is positive and non-zero
            if self.step_size <= 0:
                self.step_size.copy_(torch.tensor([0.1]))
                
            self.initialized.copy_(torch.tensor(1, dtype=torch.bool))
    
    def forward(self, x):
        """
        Forward pass with learned step size quantization.
        
        Args:
            x: Input tensor
            
        Returns:
            Quantized and dequantized tensor
        """
        if self.training:
            if not self.initialized:
                self._initialize_step_size(x)
            
            # Calculate zero point
            if self.quant_min < 0:
                # Symmetric quantization
                zero_point = torch.zeros_like(self.step_size)
            else:
                # Asymmetric quantization
                zero_point = self.quant_min - torch.min(x) / self.step_size
                zero_point = torch.clamp(zero_point, self.quant_min, self.quant_max)
            
            # Quantize
            x_scaled = x / self.step_size
            x_clipped = torch.clamp(x_scaled, self.quant_min, self.quant_max)
            x_rounded = torch.round(x_clipped)
            x_q = x_rounded * self.step_size
            
            # STE with gradient scaling
            x_q = x_q - x_scaled.detach() + x_scaled
            
            return x_q
        else:
            # In eval mode
            x_scaled = x / self.step_size
            x_clipped = torch.clamp(x_scaled, self.quant_min, self.quant_max)
            x_rounded = torch.round(x_clipped)
            return x_rounded * self.step_size

class AdaptiveSTE(torch.autograd.Function):
    """
    Adaptive Straight-Through Estimator with improved gradient scaling.
    Fixes static grad_factor issue mentioned in PDF.
    """
    
    @staticmethod
    def forward(ctx, input, scale_factor=1.0):
        # Store context for backward pass
        ctx.scale_factor = scale_factor
        
        # Forward pass is identity
        return input
    
    @staticmethod  
    def backward(ctx, grad_output):
        # Adaptive gradient scaling based on quantization error
        scale_factor = ctx.scale_factor
        
        # Scale gradients based on quantization error magnitude
        return grad_output * scale_factor, None

class ImprovedCustomFakeQuantize(FakeQuantize):
    """
    Improved FakeQuantize with adaptive STE and better gradient flow.
    """
    
    def __init__(self, observer, quant_min, quant_max, **observer_kwargs):
        super().__init__(observer, quant_min, quant_max, **observer_kwargs)
        self.adaptive_grad_scaling = True
        self.grad_factor = 1.0
        self.register_buffer('quantization_error_ema', torch.tensor(0.0))
        self.ema_momentum = 0.9
    
    def forward(self, x):
        """Forward pass with adaptive STE and improved gradient handling."""
        if self.training:
            # Update observer statistics
            self.activation_post_process(x)
            
            # Get quantization parameters
            scale, zero_point = self.calculate_qparams()
            self.scale.copy_(scale)
            self.zero_point.copy_(zero_point)
            
            # Perform quantization
            x_q = torch.fake_quantize_per_tensor_affine(
                x, scale.item(), int(zero_point.item()), 
                self.quant_min, self.quant_max
            )
            
            # Calculate quantization error for adaptive scaling
            quantization_error = torch.mean(torch.abs(x - x_q))
            
            # Update EMA of quantization error
            self.quantization_error_ema.data = (
                self.ema_momentum * self.quantization_error_ema.data + 
                (1 - self.ema_momentum) * quantization_error.detach()
            )
            
            # Adaptive gradient scaling based on quantization error
            if self.adaptive_grad_scaling:
                # Higher error = lower gradient scaling to avoid instability
                adaptive_factor = torch.clamp(
                    1.0 / (1.0 + self.quantization_error_ema), 
                    min=0.1, max=2.0
                )
                grad_factor = self.grad_factor * adaptive_factor
            else:
                grad_factor = self.grad_factor
            
            # Apply adaptive STE
            x_q = AdaptiveSTE.apply(x_q, grad_factor) + x - x.detach()
            
            return x_q
            
        else:
            # Eval mode - standard quantization
            return torch.fake_quantize_per_tensor_affine(
                x, self.scale.item(), int(self.zero_point.item()),
                self.quant_min, self.quant_max
            )
    
    def get_quantization_stats(self):
        """Return quantization statistics for monitoring."""
        return {
            'quantization_error_ema': self.quantization_error_ema.item(),
            'current_grad_factor': self.grad_factor,
            'scale': self.scale.item() if hasattr(self, 'scale') else 0.0,
            'zero_point': self.zero_point.item() if hasattr(self, 'zero_point') else 0
        }

# Enhanced LSQ with better step size learning
class ImprovedLSQFakeQuantize(LSQFakeQuantize):
    """
    Improved Learned Step Size Quantization with better convergence.
    """
    
    def __init__(self, observer, quant_min, quant_max, **observer_kwargs):
        super().__init__(observer, quant_min, quant_max, **observer_kwargs)
        self.step_size_lr_factor = 0.1  # Separate learning rate for step size
        self.register_buffer('step_size_momentum', torch.tensor(0.0))
        self.momentum_factor = 0.9
    
    def forward(self, x):
        """Forward pass with improved LSQ and momentum-based step size updates."""
        if self.training:
            if not self.initialized:
                self._initialize_step_size(x)
            
            # Gradient computation for step size with momentum
            if self.step_size.grad is not None:
                # Apply momentum to step size gradients
                self.step_size_momentum.data = (
                    self.momentum_factor * self.step_size_momentum.data + 
                    (1 - self.momentum_factor) * self.step_size.grad.data
                )
                
                # Update step size with momentum
                self.step_size.data -= self.step_size_lr_factor * self.step_size_momentum
                
                # Ensure step size remains positive
                self.step_size.data.clamp_(min=1e-8)
            
            # Standard LSQ forward pass
            return super().forward(x)
        else:
            return super().forward(x)

# Factory function to create fake quantizer based on configuration
def create_fake_quantizer(observer_type, quant_min, quant_max, dtype=torch.quint8, 
                         qscheme=torch.per_tensor_affine, ch_axis=0, is_weight=False):
    """
    Create appropriate fake quantizer based on configuration.
    
    Args:
        observer_type: Type of observer ("minmax", "moving_average_minmax", "histogram")
        quant_min: Minimum quantized value
        quant_max: Maximum quantized value
        dtype: Quantized data type
        qscheme: Quantization scheme
        ch_axis: Channel axis for per-channel quantization
        is_weight: Whether quantizing weights (True) or activations (False)
        
    Returns:
        FakeQuantize module
    """
    observer_class = get_observer(observer_type, dtype, qscheme, ch_axis)
    
    # Check if per-channel quantization is needed
    if qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]:
        return PerChannelFakeQuantize.with_args(
            observer=observer_class,
            quant_min=quant_min,
            quant_max=quant_max,
            ch_axis=ch_axis
        )
    else:
        return CustomFakeQuantize.with_args(
            observer=observer_class,
            quant_min=quant_min,
            quant_max=quant_max
        )


# Helper function to get fake quantize module from config
def get_fake_quantize_from_config(config, is_weight=False):
    """
    Create fake quantize module from configuration.
    
    Args:
        config: Quantization configuration
        is_weight: Whether quantizing weights (True) or activations (False)
        
    Returns:
        FakeQuantize module
    """
    if is_weight:
        # Weight quantization
        cfg = config["weight"]
        dtype = torch.qint8 if cfg["dtype"] == "qint8" else torch.quint8
        qscheme = torch.per_channel_symmetric if cfg["scheme"] == "per_channel" else torch.per_tensor_symmetric
        quant_min, quant_max = -128, 127
    else:
        # Activation quantization
        cfg = config["activation"]
        dtype = torch.quint8 if cfg["dtype"] == "quint8" else torch.qint8
        qscheme = torch.per_tensor_affine
        quant_min, quant_max = (0, 255) if dtype == torch.quint8 else (-128, 127)
    
    return create_fake_quantizer(
        observer_type=cfg["observer"],
        quant_min=quant_min,
        quant_max=quant_max,
        dtype=dtype,
        qscheme=qscheme,
        ch_axis=0 if is_weight else -1
    )