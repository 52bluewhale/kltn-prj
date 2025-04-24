# QAT-ready versions of PyTorch modules:
#     - QATConv2d: Quantization-aware Conv2d
#     - QATLinear: Quantization-aware Linear
#     - And other specialized modules

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization.fake_quantize import FakeQuantize
from .fake_quantize import CustomFakeQuantize, PerChannelFakeQuantize

class QATConv2d(nn.Conv2d):
    """
    Quantization-aware training version of Conv2d.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', qconfig=None):
        """
        Initialize QAT Conv2d.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            stride: Stride of convolution
            padding: Padding of convolution
            dilation: Dilation of convolution
            groups: Number of groups
            bias: Whether to include bias
            padding_mode: Mode of padding
            qconfig: Quantization configuration
        """
        super().__init__(in_channels, out_channels, kernel_size, stride,
                          padding, dilation, groups, bias, padding_mode)
        
        # Initialize quantization parameters if qconfig is provided
        if qconfig is not None:
            self.qconfig = qconfig
            self.weight_fake_quant = qconfig.weight()
            self.activation_post_process = qconfig.activation()
        else:
            self.weight_fake_quant = None
            self.activation_post_process = None
    
    def forward(self, x):
        """
        Forward pass with quantization.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Quantize weights if weight_fake_quant is available
        if self.weight_fake_quant is not None:
            weight = self.weight_fake_quant(self.weight)
        else:
            weight = self.weight
        
        # Perform convolution
        output = F.conv2d(
            x, weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )
        
        # Quantize activations if activation_post_process is available
        if self.activation_post_process is not None:
            output = self.activation_post_process(output)
        
        return output
    
    @classmethod
    def from_float(cls, mod):
        """
        Create QATConv2d from float Conv2d.
        
        Args:
            mod: Float Conv2d module
            
        Returns:
            QATConv2d module
        """
        qat_conv = cls(
            mod.in_channels, mod.out_channels, mod.kernel_size,
            mod.stride, mod.padding, mod.dilation, mod.groups,
            mod.bias is not None, mod.padding_mode
        )
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        qat_conv.qconfig = mod.qconfig
        
        if hasattr(mod, 'weight_fake_quant'):
            qat_conv.weight_fake_quant = mod.weight_fake_quant
        
        if hasattr(mod, 'activation_post_process'):
            qat_conv.activation_post_process = mod.activation_post_process
        
        return qat_conv
    
    def to_float(self):
        """
        Convert to float Conv2d.
        
        Returns:
            Float Conv2d module
        """
        float_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, self.dilation, self.groups,
            self.bias is not None, self.padding_mode
        )
        float_conv.weight = self.weight
        float_conv.bias = self.bias
        
        return float_conv


class QATBatchNorm2d(nn.BatchNorm2d):
    """
    Quantization-aware training version of BatchNorm2d.
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, qconfig=None):
        """
        Initialize QAT BatchNorm2d.
        
        Args:
            num_features: Number of features
            eps: Epsilon value for numerical stability
            momentum: Momentum for running statistics
            affine: Whether to use learnable affine parameters
            track_running_stats: Whether to track running statistics
            qconfig: Quantization configuration
        """
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        
        # Initialize quantization parameters if qconfig is provided
        if qconfig is not None:
            self.qconfig = qconfig
            self.activation_post_process = qconfig.activation()
        else:
            self.activation_post_process = None
    
    def forward(self, x):
        """
        Forward pass with quantization.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Perform batch normalization
        output = super().forward(x)
        
        # Quantize activations if activation_post_process is available
        if self.activation_post_process is not None:
            output = self.activation_post_process(output)
        
        return output
    
    @classmethod
    def from_float(cls, mod):
        """
        Create QATBatchNorm2d from float BatchNorm2d.
        
        Args:
            mod: Float BatchNorm2d module
            
        Returns:
            QATBatchNorm2d module
        """
        qat_bn = cls(
            mod.num_features, mod.eps, mod.momentum,
            mod.affine, mod.track_running_stats
        )
        qat_bn.weight = mod.weight
        qat_bn.bias = mod.bias
        qat_bn.running_mean = mod.running_mean
        qat_bn.running_var = mod.running_var
        qat_bn.num_batches_tracked = mod.num_batches_tracked
        qat_bn.qconfig = mod.qconfig
        
        if hasattr(mod, 'activation_post_process'):
            qat_bn.activation_post_process = mod.activation_post_process
        
        return qat_bn
    
    def to_float(self):
        """
        Convert to float BatchNorm2d.
        
        Returns:
            Float BatchNorm2d module
        """
        float_bn = nn.BatchNorm2d(
            self.num_features, self.eps, self.momentum,
            self.affine, self.track_running_stats
        )
        float_bn.weight = self.weight
        float_bn.bias = self.bias
        float_bn.running_mean = self.running_mean
        float_bn.running_var = self.running_var
        float_bn.num_batches_tracked = self.num_batches_tracked
        
        return float_bn


class QATLinear(nn.Linear):
    """
    Quantization-aware training version of Linear.
    """
    
    def __init__(self, in_features, out_features, bias=True, qconfig=None):
        """
        Initialize QAT Linear.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias
            qconfig: Quantization configuration
        """
        super().__init__(in_features, out_features, bias)
        
        # Initialize quantization parameters if qconfig is provided
        if qconfig is not None:
            self.qconfig = qconfig
            self.weight_fake_quant = qconfig.weight()
            self.activation_post_process = qconfig.activation()
        else:
            self.weight_fake_quant = None
            self.activation_post_process = None
    
    def forward(self, x):
        """
        Forward pass with quantization.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Quantize weights if weight_fake_quant is available
        if self.weight_fake_quant is not None:
            weight = self.weight_fake_quant(self.weight)
        else:
            weight = self.weight
        
        # Perform linear operation
        output = F.linear(x, weight, self.bias)
        
        # Quantize activations if activation_post_process is available
        if self.activation_post_process is not None:
            output = self.activation_post_process(output)
        
        return output
    
    @classmethod
    def from_float(cls, mod):
        """
        Create QATLinear from float Linear.
        
        Args:
            mod: Float Linear module
            
        Returns:
            QATLinear module
        """
        qat_linear = cls(
            mod.in_features, mod.out_features, mod.bias is not None
        )
        qat_linear.weight = mod.weight
        qat_linear.bias = mod.bias
        qat_linear.qconfig = mod.qconfig
        
        if hasattr(mod, 'weight_fake_quant'):
            qat_linear.weight_fake_quant = mod.weight_fake_quant
        
        if hasattr(mod, 'activation_post_process'):
            qat_linear.activation_post_process = mod.activation_post_process
        
        return qat_linear
    
    def to_float(self):
        """
        Convert to float Linear.
        
        Returns:
            Float Linear module
        """
        float_linear = nn.Linear(
            self.in_features, self.out_features, self.bias is not None
        )
        float_linear.weight = self.weight
        float_linear.bias = self.bias
        
        return float_linear


class QATReLU(nn.ReLU):
    """
    Quantization-aware training version of ReLU.
    """
    
    def __init__(self, inplace=False, qconfig=None):
        """
        Initialize QAT ReLU.
        
        Args:
            inplace: Whether to modify input inplace
            qconfig: Quantization configuration
        """
        super().__init__(inplace)
        
        # Initialize quantization parameters if qconfig is provided
        if qconfig is not None:
            self.qconfig = qconfig
            self.activation_post_process = qconfig.activation()
        else:
            self.activation_post_process = None
    
    def forward(self, x):
        """
        Forward pass with quantization.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Perform ReLU
        output = F.relu(x, self.inplace)
        
        # Quantize activations if activation_post_process is available
        if self.activation_post_process is not None:
            output = self.activation_post_process(output)
        
        return output
    
    @classmethod
    def from_float(cls, mod):
        """
        Create QATReLU from float ReLU.
        
        Args:
            mod: Float ReLU module
            
        Returns:
            QATReLU module
        """
        qat_relu = cls(mod.inplace)
        qat_relu.qconfig = mod.qconfig
        
        if hasattr(mod, 'activation_post_process'):
            qat_relu.activation_post_process = mod.activation_post_process
        
        return qat_relu
    
    def to_float(self):
        """
        Convert to float ReLU.
        
        Returns:
            Float ReLU module
        """
        return nn.ReLU(self.inplace)