"""
Specialized QAT modules for YOLOv8.

This module provides quantization-aware versions of YOLOv8-specific modules,
such as detection heads and CSP blocks, which need special handling during QAT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import math

from ..quantization.qat_modules import QATConv2d, QATBatchNorm2d, QATReLU

# Setup logging
logger = logging.getLogger(__name__)

class QATDetectionHead(nn.Module):
    """
    Quantization-aware detection head for YOLOv8.
    
    The detection head is a critical component that requires special handling
    during quantization to maintain accuracy.
    """
    
    def __init__(self, original_head, qconfig=None):
        """
        Initialize QAT detection head.
        
        Args:
            original_head: Original YOLOv8 detection head
            qconfig: Quantization configuration
        """
        super().__init__()
        
        self.nc = original_head.nc  # Number of classes
        self.reg_max = getattr(original_head, 'reg_max', 16)  # Maximum box regression value
        self.no = self.nc + self.reg_max * 4  # Number of outputs per detection
        
        # Copy and convert convolution layers
        self.cv2 = nn.ModuleList()
        for conv in original_head.cv2:
            if isinstance(conv, nn.Conv2d):
                qat_conv = QATConv2d.from_float(conv)
                if qconfig is not None:
                    qat_conv.qconfig = qconfig
                self.cv2.append(qat_conv)
        
        # Copy other attributes from original head
        if hasattr(original_head, 'stride'):
            self.stride = original_head.stride
        if hasattr(original_head, 'anchors'):
            self.anchors = original_head.anchors
        if hasattr(original_head, 'dfl'):
            # Handle distribution focal loss (DFL) conv layer
            if isinstance(original_head.dfl, nn.Conv2d):
                self.dfl = QATConv2d.from_float(original_head.dfl)
                if qconfig is not None:
                    self.dfl.qconfig = qconfig
            else:
                self.dfl = original_head.dfl
        
        # Additional attributes for QAT
        self.activation_post_process = None
        if qconfig is not None:
            self.qconfig = qconfig
            self.activation_post_process = qconfig.activation()
    
    def forward(self, x):
        """
        Forward pass of detection head.
        
        Args:
            x: Input features (list of tensors)
            
        Returns:
            Detection outputs
        """
        for i in range(len(x)):
            x[i] = self.cv2[i](x[i])  # Apply convolutions
            
            # Apply activation post-process if available
            if self.activation_post_process is not None:
                x[i] = self.activation_post_process(x[i])
        
        # Apply decode logic (to be compatible with original head)
        if hasattr(self, 'dfl') and self.training:
            # Apply DFL to regression outputs
            b, _, h, w = x[0].shape
            y = torch.cat([xi.view(b, self.reg_max * 4, -1) for xi in x], -1)
            y = self.dfl(y.view(b, 4, self.reg_max, -1))
            return y
        
        return x
    
    @classmethod
    def from_float(cls, module, qconfig=None):
        """
        Create QAT detection head from float module.
        
        Args:
            module: Float detection head
            qconfig: Quantization configuration
            
        Returns:
            QAT detection head
        """
        assert isinstance(module, type(module)), f"Expected {type(module)}, got {type(module)}"
        qat_head = cls(module, qconfig)
        return qat_head


class QATCSPLayer(nn.Module):
    """
    Quantization-aware CSP (Cross Stage Partial) layer for YOLOv8.
    
    This is a QAT version of the CSP layer used in YOLOv8, which is a key
    building block of the network.
    """
    
    def __init__(self, original_csp, qconfig=None):
        """
        Initialize QAT CSP layer.
        
        Args:
            original_csp: Original YOLOv8 CSP layer
            qconfig: Quantization configuration
        """
        super().__init__()
        
        # Convert all convolutions to QAT versions
        self.cv1 = self._convert_conv(original_csp.cv1, qconfig)
        self.cv2 = self._convert_conv(original_csp.cv2, qconfig)
        
        # Handle additional convs if present
        if hasattr(original_csp, 'cv3'):
            self.cv3 = self._convert_conv(original_csp.cv3, qconfig)
        
        # Handle bottleneck modules
        if hasattr(original_csp, 'm'):
            if isinstance(original_csp.m, nn.ModuleList):
                self.m = nn.ModuleList()
                for module in original_csp.m:
                    self.m.append(self._convert_bottleneck(module, qconfig))
            else:
                self.m = self._convert_bottleneck(original_csp.m, qconfig)
        
        # Additional attributes for QAT
        self.activation_post_process = None
        if qconfig is not None:
            self.qconfig = qconfig
            self.activation_post_process = qconfig.activation()
    
    def _convert_conv(self, conv, qconfig):
        """
        Convert convolution to QAT version.
        
        Args:
            conv: Convolution module
            qconfig: Quantization configuration
            
        Returns:
            QAT convolution
        """
        if isinstance(conv, nn.Conv2d):
            qat_conv = QATConv2d.from_float(conv)
            if qconfig is not None:
                qat_conv.qconfig = qconfig
            return qat_conv
        
        # For compound modules (Conv+BN+Act), convert each part
        if hasattr(conv, 'conv'):
            conv.conv = self._convert_conv(conv.conv, qconfig)
        
        if hasattr(conv, 'bn'):
            if isinstance(conv.bn, nn.BatchNorm2d):
                qat_bn = QATBatchNorm2d.from_float(conv.bn)
                if qconfig is not None:
                    qat_bn.qconfig = qconfig
                conv.bn = qat_bn
        
        return conv
    
    def _convert_bottleneck(self, bottleneck, qconfig):
        """
        Convert bottleneck module to QAT version.
        
        Args:
            bottleneck: Bottleneck module
            qconfig: Quantization configuration
            
        Returns:
            QAT bottleneck
        """
        # Convert all convolutions in the bottleneck
        if hasattr(bottleneck, 'cv1'):
            bottleneck.cv1 = self._convert_conv(bottleneck.cv1, qconfig)
        
        if hasattr(bottleneck, 'cv2'):
            bottleneck.cv2 = self._convert_conv(bottleneck.cv2, qconfig)
        
        # Apply QConfig to bottleneck
        if qconfig is not None:
            bottleneck.qconfig = qconfig
        
        return bottleneck
    
    def forward(self, x):
        """
        Forward pass of CSP layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # First branch
        y1 = self.cv1(x)
        
        # Second branch
        if hasattr(self, 'm'):
            if isinstance(self.m, nn.ModuleList):
                y2 = self.cv2(x)
                for module in self.m:
                    y2 = module(y2)
            else:
                y2 = self.m(self.cv2(x))
        else:
            y2 = self.cv2(x)
        
        # Concatenate branches
        out = torch.cat([y1, y2], dim=1)
        
        # Final convolution
        if hasattr(self, 'cv3'):
            out = self.cv3(out)
        
        # Apply activation post-process if available
        if self.activation_post_process is not None:
            out = self.activation_post_process(out)
        
        return out
    
    @classmethod
    def from_float(cls, module, qconfig=None):
        """
        Create QAT CSP layer from float module.
        
        Args:
            module: Float CSP layer
            qconfig: Quantization configuration
            
        Returns:
            QAT CSP layer
        """
        qat_csp = cls(module, qconfig)
        return qat_csp


class QATBottleneckCSP(nn.Module):
    """
    Quantization-aware Bottleneck CSP module for YOLOv8.
    
    This is a QAT version of the Bottleneck CSP module used in YOLOv8,
    which is a specialized form of the CSP architecture.
    """
    
    def __init__(self, original_bottleneck, qconfig=None):
        """
        Initialize QAT Bottleneck CSP module.
        
        Args:
            original_bottleneck: Original YOLOv8 Bottleneck CSP module
            qconfig: Quantization configuration
        """
        super().__init__()
        
        # Convert all convolutions to QAT versions
        self.cv1 = self._convert_conv(original_bottleneck.cv1, qconfig)
        self.cv2 = self._convert_conv(original_bottleneck.cv2, qconfig)
        self.cv3 = self._convert_conv(original_bottleneck.cv3, qconfig)
        self.cv4 = self._convert_conv(original_bottleneck.cv4, qconfig)
        
        # Handle bottleneck modules
        if hasattr(original_bottleneck, 'm'):
            if isinstance(original_bottleneck.m, nn.ModuleList):
                self.m = nn.ModuleList()
                for module in original_bottleneck.m:
                    self.m.append(self._convert_bottleneck(module, qconfig))
            else:
                self.m = self._convert_bottleneck(original_bottleneck.m, qconfig)
        
        # Convert batch norm if present
        if hasattr(original_bottleneck, 'bn'):
            self.bn = QATBatchNorm2d.from_float(original_bottleneck.bn)
            if qconfig is not None:
                self.bn.qconfig = qconfig
        
        # Copy activation function
        if hasattr(original_bottleneck, 'act'):
            if isinstance(original_bottleneck.act, nn.ReLU):
                self.act = QATReLU.from_float(original_bottleneck.act)
                if qconfig is not None:
                    self.act.qconfig = qconfig
            else:
                self.act = original_bottleneck.act
        
        # Additional attributes for QAT
        self.activation_post_process = None
        if qconfig is not None:
            self.qconfig = qconfig
            self.activation_post_process = qconfig.activation()
    
    def _convert_conv(self, conv, qconfig):
        """
        Convert convolution to QAT version.
        
        Args:
            conv: Convolution module
            qconfig: Quantization configuration
            
        Returns:
            QAT convolution
        """
        if isinstance(conv, nn.Conv2d):
            qat_conv = QATConv2d.from_float(conv)
            if qconfig is not None:
                qat_conv.qconfig = qconfig
            return qat_conv
        
        # For compound modules (Conv+BN+Act), convert each part
        if hasattr(conv, 'conv'):
            conv.conv = self._convert_conv(conv.conv, qconfig)
        
        if hasattr(conv, 'bn'):
            if isinstance(conv.bn, nn.BatchNorm2d):
                qat_bn = QATBatchNorm2d.from_float(conv.bn)
                if qconfig is not None:
                    qat_bn.qconfig = qconfig
                conv.bn = qat_bn
        
        return conv
    
    def _convert_bottleneck(self, bottleneck, qconfig):
        """
        Convert bottleneck module to QAT version.
        
        Args:
            bottleneck: Bottleneck module
            qconfig: Quantization configuration
            
        Returns:
            QAT bottleneck
        """
        # Convert all convolutions in the bottleneck
        if hasattr(bottleneck, 'cv1'):
            bottleneck.cv1 = self._convert_conv(bottleneck.cv1, qconfig)
        
        if hasattr(bottleneck, 'cv2'):
            bottleneck.cv2 = self._convert_conv(bottleneck.cv2, qconfig)
        
        # Apply QConfig to bottleneck
        if qconfig is not None:
            bottleneck.qconfig = qconfig
        
        return bottleneck
    
    def forward(self, x):
        """
        Forward pass of Bottleneck CSP module.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        y1 = self.cv3(self.cv1(x))
        
        # Handle bottleneck modules
        y2 = self.cv2(x)
        if hasattr(self, 'm'):
            if isinstance(self.m, nn.ModuleList):
                for module in self.m:
                    y2 = module(y2)
            else:
                y2 = self.m(y2)
        
        # Apply batch norm if present
        if hasattr(self, 'bn'):
            y = torch.cat([y1, y2], dim=1)
            y = self.bn(y)
        else:
            y = torch.cat([y1, y2], dim=1)
        
        # Apply activation if present
        if hasattr(self, 'act'):
            y = self.act(y)
        
        # Final convolution
        y = self.cv4(y)
        
        # Apply activation post-process if available
        if self.activation_post_process is not None:
            y = self.activation_post_process(y)
        
        return y
    
    @classmethod
    def from_float(cls, module, qconfig=None):
        """
        Create QAT Bottleneck CSP module from float module.
        
        Args:
            module: Float Bottleneck CSP module
            qconfig: Quantization configuration
            
        Returns:
            QAT Bottleneck CSP module
        """
        qat_bottleneck = cls(module, qconfig)
        return qat_bottleneck