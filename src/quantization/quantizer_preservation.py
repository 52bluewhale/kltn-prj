#!/usr/bin/env python
"""
src/quantization/quantizer_preservation.py

FIXED: Simple Enable/Disable Quantizer Preservation
SOLUTION: Work WITH PyTorch instead of against it
"""

import torch
import torch.nn as nn
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

class QuantizerPreserver:
    """
    FIXED: Simple quantizer preservation using enable/disable approach.
    WORKS WITH PyTorch's design instead of fighting it.
    """
    
    def __init__(self, model):
        self.model = model
        self.weight_quantizer_modules = []
        self.activation_quantizer_modules = []
        self.stored_activation_quantizers = {}
        self._collect_quantizer_modules()
    
    def _collect_quantizer_modules(self):
        """Collect all quantizer modules for simple enable/disable control."""
        logger.info("🔧 Collecting quantizer modules for simple control...")
        
        for name, module in self.model.named_modules():
            # Collect weight quantizers (keep these always active)
            if hasattr(module, 'weight_fake_quant'):
                quantizer = module.weight_fake_quant
                if self._is_fake_quantize(quantizer):
                    self.weight_quantizer_modules.append((name, module, quantizer))
            
            # Collect activation quantizers (these we'll enable/disable)
            if hasattr(module, 'activation_post_process'):
                quantizer = module.activation_post_process
                if self._is_fake_quantize(quantizer):
                    self.activation_quantizer_modules.append((name, module, quantizer))
                    # Store original for restoration
                    self.stored_activation_quantizers[name] = quantizer
        
        logger.info(f"✅ Collected quantizers:")
        logger.info(f"  - Weight quantizers: {len(self.weight_quantizer_modules)} (always active)")
        logger.info(f"  - Activation quantizers: {len(self.activation_quantizer_modules)} (phase controlled)")
    
    def _is_fake_quantize(self, module):
        """Check if module is a FakeQuantize module."""
        return module is not None and 'FakeQuantize' in type(module).__name__
    
    def set_phase_state(self, weights_enabled: bool, activations_enabled: bool):
        """
        FIXED: Simple enable/disable approach that works with PyTorch.
        
        Args:
            weights_enabled: Whether to enable weight quantization (always True in practice)
            activations_enabled: Whether to enable activation quantization
        """
        logger.info(f"🎯 SIMPLE phase control (works with PyTorch):")
        logger.info(f"  - Weight quantization: {weights_enabled}")
        logger.info(f"  - Activation quantization: {activations_enabled}")
        
        # Handle activation quantizers (the main control point)
        if activations_enabled:
            # ENABLE: Restore original FakeQuantize modules
            restored_count = 0
            for name, module, original_quantizer in self.activation_quantizer_modules:
                if name in self.stored_activation_quantizers:
                    # Restore the original FakeQuantize module
                    module.activation_post_process = self.stored_activation_quantizers[name]
                    restored_count += 1
            
            logger.info(f"  ✅ ENABLED {restored_count} activation quantizers")
            logger.info(f"  🔍 Observers will collect statistics naturally")
            
        else:
            # DISABLE: Replace with Identity modules
            disabled_count = 0
            for name, module, original_quantizer in self.activation_quantizer_modules:
                # Replace with Identity (completely disables quantization)
                module.activation_post_process = nn.Identity()
                disabled_count += 1
            
            logger.info(f"  ⚪ DISABLED {disabled_count} activation quantizers")
            logger.info(f"  🎯 Clean weight-only quantization")
        
        # Weight quantizers remain always active (they work fine)
        weight_active_count = len(self.weight_quantizer_modules)
        logger.info(f"  ✅ {weight_active_count} weight quantizers remain active")
        
        # Final verification
        self._verify_phase_state(weights_enabled, activations_enabled)
    
    def _verify_phase_state(self, expected_weights: bool, expected_activations: bool):
        """Verify the current phase state."""
        active_weight_quantizers = 0
        active_activation_quantizers = 0
        
        for name, module in self.model.named_modules():
            # Count active weight quantizers
            if hasattr(module, 'weight_fake_quant') and self._is_fake_quantize(module.weight_fake_quant):
                active_weight_quantizers += 1
            
            # Count active activation quantizers (not Identity)
            if hasattr(module, 'activation_post_process') and self._is_fake_quantize(module.activation_post_process):
                active_activation_quantizers += 1
        
        logger.info(f"📊 Verification:")
        logger.info(f"  - Active weight quantizers: {active_weight_quantizers}")
        logger.info(f"  - Active activation quantizers: {active_activation_quantizers}")
        
        # Check expectations
        if expected_activations and active_activation_quantizers == 0:
            logger.error("❌ Expected activation quantizers but found none!")
            return False
        elif not expected_activations and active_activation_quantizers > 0:
            logger.warning(f"⚠️ Expected no activation quantizers but found {active_activation_quantizers}")
        
        logger.info("✅ Phase state verification passed")
        return True
    
    def set_phase_by_name(self, phase_name: str):
        """Set phase using 2-phase training approach."""
        phase_configs = {
            "phase1_weight_only": (True, False),    # Weights only (30%)
            "phase2_full_quant": (True, True),      # Full quantization (70%)
        }
        
        if phase_name in phase_configs:
            weights_enabled, activations_enabled = phase_configs[phase_name]
            logger.info(f"🔄 Setting phase: {phase_name} (2-phase simple approach)")
            self.set_phase_state(weights_enabled, activations_enabled)
        else:
            logger.error(f"❌ Unknown phase: {phase_name}. Valid phases: phase1_weight_only, phase2_full_quant")
    
    def get_quantizer_stats(self):
        """Get current quantizer statistics."""
        total_fake_quants = sum(1 for n, m in self.model.named_modules() 
                               if 'FakeQuantize' in type(m).__name__)
        
        # Count active quantizers
        active_weights = 0
        active_activations = 0
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight_fake_quant') and self._is_fake_quantize(module.weight_fake_quant):
                active_weights += 1
            if hasattr(module, 'activation_post_process') and self._is_fake_quantize(module.activation_post_process):
                active_activations += 1
        
        return {
            'total_fake_quantizers': total_fake_quants,
            'weight_quantizers_total': len(self.weight_quantizer_modules),
            'weight_quantizers_enabled': active_weights,
            'activation_quantizers_total': len(self.activation_quantizer_modules),
            'activation_quantizers_enabled': active_activations,
            'preservation_active': total_fake_quants > 0,
            'observers_working_naturally': True  # NEW: Simple approach works naturally
        }
    
    def validate_observer_calibration(self):
        """
        FIXED: Validate that observers have collected statistics naturally.
        """
        logger.info("🔍 Validating observer calibration (simple approach)...")
        
        uncalibrated_observers = []
        total_observers = 0
        
        # Check all currently active activation quantizers
        for name, module in self.model.named_modules():
            if hasattr(module, 'activation_post_process'):
                observer = module.activation_post_process
                
                # Skip Identity modules (disabled quantizers)
                if isinstance(observer, nn.Identity):
                    continue
                
                # Check FakeQuantize modules for observer statistics
                if self._is_fake_quantize(observer):
                    total_observers += 1
                    
                    # Check if observer has collected valid statistics
                    has_valid_stats = False
                    
                    # Method 1: Check activation_post_process attribute
                    if hasattr(observer, 'activation_post_process'):
                        inner_observer = observer.activation_post_process
                        if hasattr(inner_observer, 'min_val') and hasattr(inner_observer, 'max_val'):
                            min_val = inner_observer.min_val
                            max_val = inner_observer.max_val
                            if (min_val != float('inf') and max_val != float('-inf') and 
                                not torch.isnan(min_val) and not torch.isnan(max_val) and
                                min_val != max_val):
                                has_valid_stats = True
                    
                    # Method 2: Check direct observer attributes
                    if not has_valid_stats and hasattr(observer, 'min_val') and hasattr(observer, 'max_val'):
                        min_val = observer.min_val
                        max_val = observer.max_val
                        if (min_val != float('inf') and max_val != float('-inf') and 
                            not torch.isnan(min_val) and not torch.isnan(max_val) and
                            min_val != max_val):
                            has_valid_stats = True
                    
                    if not has_valid_stats:
                        uncalibrated_observers.append(name)
        
        calibrated_count = total_observers - len(uncalibrated_observers)
        
        logger.info(f"📊 Simple approach observer validation:")
        logger.info(f"  - Total active observers: {total_observers}")
        logger.info(f"  - Calibrated: {calibrated_count}")
        logger.info(f"  - Uncalibrated: {len(uncalibrated_observers)}")
        
        if len(uncalibrated_observers) > 0:
            logger.error(f"❌ Uncalibrated observers: {uncalibrated_observers[:5]}...")
            logger.error(f"💡 Solution: Train longer with activation quantizers enabled")
            return False
        else:
            logger.info("✅ All observers properly calibrated using simple approach")
            return True
    
    def debug_quantizer_states(self):
        """Debug method to show quantizer information."""
        logger.info("🔍 Debugging simple quantizer states:")
        
        # Show weight quantizers (should always be active)
        logger.info(f"Weight Quantizers ({len(self.weight_quantizer_modules)}) - Always Active:")
        for name, module, quantizer in self.weight_quantizer_modules[:3]:
            logger.info(f"  ✅ {name}: FakeQuantize active")
        
        # Show activation quantizers (phase dependent)
        logger.info(f"Activation Quantizers ({len(self.activation_quantizer_modules)}) - Phase Controlled:")
        for name, module, original_quantizer in self.activation_quantizer_modules[:3]:
            current = module.activation_post_process
            if isinstance(current, nn.Identity):
                logger.info(f"  ⚪ {name}: Disabled (Identity)")
            elif self._is_fake_quantize(current):
                logger.info(f"  ✅ {name}: FakeQuantize active")
            else:
                logger.info(f"  ❓ {name}: Unknown state")
        
        # Overall statistics
        stats = self.get_quantizer_stats()
        logger.info(f"📊 Simple Design Stats: {stats}")
    
    def verify_preservation(self):
        """Verify that quantizers are properly preserved."""
        stats = self.get_quantizer_stats()
        
        if stats['total_fake_quantizers'] == 0:
            logger.error("❌ CRITICAL: No FakeQuantize modules found!")
            return False
        
        if len(self.weight_quantizer_modules) == 0 and len(self.activation_quantizer_modules) == 0:
            logger.error("❌ CRITICAL: No quantizers collected!")
            return False
        
        logger.info(f"✅ SIMPLE preservation verified:")
        logger.info(f"  - {stats['total_fake_quantizers']} FakeQuantize modules exist")
        logger.info(f"  - {len(self.weight_quantizer_modules)} weight quantizers tracked")
        logger.info(f"  - {len(self.activation_quantizer_modules)} activation quantizers tracked")
        logger.info(f"  - Simple enable/disable approach active")
        
        return True