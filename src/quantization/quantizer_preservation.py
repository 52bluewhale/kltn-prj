#!/usr/bin/env python
"""
src/quantization/quantizer_preservation.py

Quantizer Preservation Module
FIXES: Quantizer loss during YOLOv8 training by using enable/disable instead of replace/remove

CREATE THIS FILE in your src/quantization/ directory
"""

import torch
import torch.nn as nn
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

class QuantizerPreserver:
    """
    Preserves quantizers during phased training by using enable/disable approach.
    CRITICAL FIX: Never replaces or removes FakeQuantize modules.
    """
    
    def __init__(self, model):
        self.model = model
        self.weight_quantizers = []
        self.activation_quantizers = []
        self.original_forwards = {}
        self._enhance_all_quantizers()
    
    def _enhance_all_quantizers(self):
        """Enhance all quantizers with preservation capabilities."""
        logger.info("🔧 Enhancing quantizers for preservation...")
        
        for name, module in self.model.named_modules():
            # Enhance weight quantizers
            if hasattr(module, 'weight_fake_quant'):
                quantizer = module.weight_fake_quant
                if self._is_fake_quantize(quantizer):
                    self._enhance_quantizer(quantizer, f"{name}.weight")
                    self.weight_quantizers.append((name, quantizer))
            
            # Enhance activation quantizers  
            if hasattr(module, 'activation_post_process'):
                quantizer = module.activation_post_process
                if self._is_fake_quantize(quantizer):
                    self._enhance_quantizer(quantizer, f"{name}.activation")
                    self.activation_quantizers.append((name, quantizer))
        
        logger.info(f"✅ Enhanced {len(self.weight_quantizers)} weight and "
                   f"{len(self.activation_quantizers)} activation quantizers")
    
    def _is_fake_quantize(self, module):
        """Check if module is a FakeQuantize module."""
        return module is not None and 'FakeQuantize' in type(module).__name__
    
    def _enhance_quantizer(self, quantizer, name):
        """Add preservation capabilities to a quantizer."""
        # Store original forward if not already stored
        if not hasattr(quantizer, '_preservation_enabled'):
            quantizer._original_forward = quantizer.forward
            quantizer._preservation_enabled = True
            quantizer._quantizer_name = name
            
            # Create preserving forward method
            def preserving_forward(x):
                if getattr(quantizer, '_preservation_enabled', True):
                    return quantizer._original_forward(x)
                else:
                    return x  # Pass through without quantization
            
            # Replace forward method
            quantizer.forward = preserving_forward
            
            # Add control methods
            quantizer.enable_quantization = lambda: setattr(quantizer, '_preservation_enabled', True)
            quantizer.disable_quantization = lambda: setattr(quantizer, '_preservation_enabled', False)
            quantizer.is_quantization_enabled = lambda: getattr(quantizer, '_preservation_enabled', True)
    
    def set_phase_state(self, weights_enabled: bool, activations_enabled: bool):
        """
        FIXED: Set quantizer states without destroying them.
        
        Args:
            weights_enabled: Whether to enable weight quantizers
            activations_enabled: Whether to enable activation quantizers
        """
        # Control weight quantizers
        for name, quantizer in self.weight_quantizers:
            if weights_enabled:
                quantizer.enable_quantization()
            else:
                quantizer.disable_quantization()
        
        # Control activation quantizers
        for name, quantizer in self.activation_quantizers:
            if activations_enabled:
                quantizer.enable_quantization()
            else:
                quantizer.disable_quantization()
        
        # Verify state
        enabled_weights = sum(1 for _, q in self.weight_quantizers if q.is_quantization_enabled())
        enabled_activations = sum(1 for _, q in self.activation_quantizers if q.is_quantization_enabled())
        
        logger.info(f"🎯 Phase state set:")
        logger.info(f"  - Weight quantizers: {enabled_weights}/{len(self.weight_quantizers)} enabled")
        logger.info(f"  - Activation quantizers: {enabled_activations}/{len(self.activation_quantizers)} enabled")
    
    def set_phase_by_name(self, phase_name: str):
        """Set phase using your existing phase names."""
        phase_configs = {
            "phase1_weight_only": (True, False),
            "phase2_activations": (True, True),
            "phase3_full_quant": (True, True),
            "phase4_fine_tuning": (True, True)
        }
        
        if phase_name in phase_configs:
            weights_enabled, activations_enabled = phase_configs[phase_name]
            logger.info(f"🔄 Setting phase: {phase_name}")
            self.set_phase_state(weights_enabled, activations_enabled)
        else:
            logger.error(f"❌ Unknown phase: {phase_name}")
    
    def get_quantizer_stats(self):
        """Get current quantizer statistics."""
        total_fake_quants = sum(1 for n, m in self.model.named_modules() 
                               if 'FakeQuantize' in type(m).__name__)
        
        enabled_weights = sum(1 for _, q in self.weight_quantizers if q.is_quantization_enabled())
        enabled_activations = sum(1 for _, q in self.activation_quantizers if q.is_quantization_enabled())
        
        return {
            'total_fake_quantizers': total_fake_quants,
            'weight_quantizers_total': len(self.weight_quantizers),
            'weight_quantizers_enabled': enabled_weights,
            'activation_quantizers_total': len(self.activation_quantizers),
            'activation_quantizers_enabled': enabled_activations,
            'preservation_active': total_fake_quants > 0
        }
    
    def debug_quantizer_states(self):
        """Debug method to show detailed quantizer information."""
        logger.info("🔍 Debugging quantizer states:")
        
        # Check weight quantizers
        logger.info(f"Weight Quantizers ({len(self.weight_quantizers)}):")
        for name, quantizer in self.weight_quantizers[:5]:  # Show first 5
            enabled = quantizer.is_quantization_enabled()
            logger.info(f"  - {name}: {'✅ Enabled' if enabled else '❌ Disabled'}")
        
        # Check activation quantizers
        logger.info(f"Activation Quantizers ({len(self.activation_quantizers)}):")
        for name, quantizer in self.activation_quantizers[:5]:  # Show first 5
            enabled = quantizer.is_quantization_enabled()
            logger.info(f"  - {name}: {'✅ Enabled' if enabled else '❌ Disabled'}")
        
        # Overall statistics
        stats = self.get_quantizer_stats()
        logger.info(f"Overall Stats: {stats}")
    
    def verify_preservation(self):
        """Verify that quantizers are properly preserved."""
        stats = self.get_quantizer_stats()
        
        if stats['total_fake_quantizers'] == 0:
            logger.error("❌ CRITICAL: No FakeQuantize modules found!")
            return False
        
        if len(self.weight_quantizers) == 0 and len(self.activation_quantizers) == 0:
            logger.error("❌ CRITICAL: No quantizers in preservation lists!")
            return False
        
        logger.info(f"✅ Preservation verified:")
        logger.info(f"  - {stats['total_fake_quantizers']} FakeQuantize modules exist")
        logger.info(f"  - {len(self.weight_quantizers)} weight quantizers tracked")
        logger.info(f"  - {len(self.activation_quantizers)} activation quantizers tracked")
        
        return True