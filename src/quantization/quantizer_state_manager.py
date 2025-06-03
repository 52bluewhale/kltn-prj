#!/usr/bin/env python
"""
quantizer_state_manager.py

CREATE THIS FILE: src/quantization/quantizer_state_manager.py

This file fixes the quantizer reference loss issue during phase transitions.
"""

import torch
import logging
from typing import Dict, Any, Tuple
import copy
import pickle

logger = logging.getLogger(__name__)

class QuantizerStateManager:
    """
    Manages quantizer states during phased training.
    Prevents quantizer reference loss during phase transitions.
    """
    
    def __init__(self, model):
        self.model = model
        self.quantizer_registry = {}
        self.module_paths = {}
        self._build_quantizer_registry()
    
    def _build_quantizer_registry(self):
        """
        FIXED: Build a registry of all quantizers with their original instances.
        Store references to actual FakeQuantize modules, not just configs.
        """
        logger.info("🔍 Building quantizer registry...")
        
        count = 0
        for name, module in self.model.named_modules():
            # Store module path for access
            self.module_paths[name] = module
            
            quantizer_info = {
                'module_name': name,
                'module_type': type(module).__name__,
                'has_weight_quantizer': False,
                'has_activation_quantizer': False,
                'original_weight_fake_quant': None,
                'original_activation_post_process': None,
                'is_detection_head': 'detect' in name or 'model.22' in name
            }
            
            # Store original weight quantizer
            if hasattr(module, 'weight_fake_quant'):
                quantizer_info['has_weight_quantizer'] = True
                quantizer_info['original_weight_fake_quant'] = module.weight_fake_quant
                count += 1
            
            # Store original activation quantizer
            if hasattr(module, 'activation_post_process'):
                quantizer_info['has_activation_quantizer'] = True
                quantizer_info['original_activation_post_process'] = module.activation_post_process
                count += 1
            
            # Only store if module has quantizers
            if quantizer_info['has_weight_quantizer'] or quantizer_info['has_activation_quantizer']:
                self.quantizer_registry[name] = quantizer_info
        
        logger.info(f"✅ Registry built: {count} quantizers found")
        return count > 0
    
    def set_phase_state(self, phase_name: str, weights_enabled: bool = True, activations_enabled: bool = True) -> bool:
        """
        FIXED: Set quantizer states by enabling/disabling instead of replacing.
        Preserves original FakeQuantize modules by using their enable/disable methods.
        """
        logger.info(f"⚙️ Setting phase state: {phase_name}")
        
        restored_count = 0
        disabled_count = 0
        total_changes = 0
        
        for module_name, quantizer_info in self.quantizer_registry.items():
            module = self._get_module_by_path(module_name)
            if module is None:
                continue
            
            # Skip detection head if needed
            if quantizer_info['is_detection_head']:
                continue
            
            # Handle weight quantizers
            if quantizer_info['has_weight_quantizer']:
                original_weight_quant = quantizer_info['original_weight_fake_quant']
                
                if weights_enabled:
                    # RESTORE original FakeQuantize if it was replaced
                    if not self._is_fake_quantize_module(module.weight_fake_quant):
                        module.weight_fake_quant = original_weight_quant
                        restored_count += 1
                        total_changes += 1
                    # ENABLE if it's already a FakeQuantize
                    elif hasattr(module.weight_fake_quant, 'disable_fake_quant'):
                        module.weight_fake_quant.disable_fake_quant = False
                else:
                    # DISABLE FakeQuantize (but keep the module)
                    if self._is_fake_quantize_module(module.weight_fake_quant):
                        if hasattr(module.weight_fake_quant, 'disable_fake_quant'):
                            module.weight_fake_quant.disable_fake_quant = True
                            disabled_count += 1
                            total_changes += 1
            
            # Handle activation quantizers
            if quantizer_info['has_activation_quantizer']:
                original_activation_quant = quantizer_info['original_activation_post_process']
                
                if activations_enabled:
                    # RESTORE original FakeQuantize if it was replaced
                    if not self._is_fake_quantize_module(module.activation_post_process):
                        module.activation_post_process = original_activation_quant
                        restored_count += 1
                        total_changes += 1
                    # ENABLE if it's already a FakeQuantize
                    elif hasattr(module.activation_post_process, 'disable_fake_quant'):
                        module.activation_post_process.disable_fake_quant = False
                else:
                    # DISABLE FakeQuantize (but keep the module)
                    if self._is_fake_quantize_module(module.activation_post_process):
                        if hasattr(module.activation_post_process, 'disable_fake_quant'):
                            module.activation_post_process.disable_fake_quant = True
                            disabled_count += 1
                            total_changes += 1
        
        logger.info(f"✅ Phase {phase_name} configured:")
        logger.info(f"  - Restored: {restored_count} quantizers")
        logger.info(f"  - Disabled: {disabled_count} quantizers")
        logger.info(f"  - Total changes: {total_changes}")
        
        # Verify the configuration
        success = self._verify_phase_state(weights_enabled, activations_enabled)
        return success

    def _is_fake_quantize_module(self, module) -> bool:
        """Check if module is actually a FakeQuantize module."""
        if module is None:
            return False
        return 'FakeQuantize' in type(module).__name__

    def _get_module_by_path(self, path: str):
        """Get module by its path."""
        try:
            current = self.model
            for part in path.split('.'):
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = getattr(current, part)
            return current
        except (AttributeError, IndexError, KeyError):
            logger.warning(f"⚠️ Module path {path} no longer exists")
            return None

    def _has_valid_weight_quantizer(self, module):
        """Check if module has a valid (non-Identity) weight quantizer."""
        return (hasattr(module, 'weight_fake_quant') and 
                not isinstance(module.weight_fake_quant, torch.nn.Identity))
    
    def _has_valid_activation_quantizer(self, module):
        """Check if module has a valid (non-Identity) activation quantizer."""
        return (hasattr(module, 'activation_post_process') and 
                not isinstance(module.activation_post_process, torch.nn.Identity))
    
    def _reconstruct_weight_quantizer(self, module, qconfig):
        """Reconstruct weight quantizer from qconfig."""
        try:
            # Apply qconfig to module
            module.qconfig = qconfig
            
            # Create weight fake quantizer from qconfig
            if qconfig and qconfig.weight:
                module.weight_fake_quant = qconfig.weight()
                logger.debug(f"Reconstructed weight quantizer for {type(module).__name__}")
        except Exception as e:
            logger.error(f"Failed to reconstruct weight quantizer: {e}")

    def _reconstruct_activation_quantizer(self, module, qconfig):
        """Reconstruct activation quantizer from qconfig."""
        try:
            # Apply qconfig to module
            module.qconfig = qconfig
            
            # Create activation quantizer from qconfig
            if qconfig and qconfig.activation:
                module.activation_post_process = qconfig.activation()
                logger.debug(f"Reconstructed activation quantizer for {type(module).__name__}")
        except Exception as e:
            logger.error(f"Failed to reconstruct activation quantizer: {e}")
    
    def _verify_phase_state(self, expected_weights: bool, expected_activations: bool) -> bool:
        """
        FIXED: Verify the current phase state by checking both module count 
        and enabled/disabled status.
        """
        weight_total = 0
        weight_active = 0
        activation_total = 0
        activation_active = 0
        
        for module_name, quantizer_info in self.quantizer_registry.items():
            module = self._get_module_by_path(module_name)
            if module is None or quantizer_info['is_detection_head']:
                continue
            
            # Check weight quantizers
            if quantizer_info['has_weight_quantizer']:
                weight_total += 1
                if self._is_quantizer_active(module.weight_fake_quant):
                    weight_active += 1
            
            # Check activation quantizers
            if quantizer_info['has_activation_quantizer']:
                activation_total += 1
                if self._is_quantizer_active(module.activation_post_process):
                    activation_active += 1
        
        logger.info(f"📊 Verification results:")
        logger.info(f"  - Weight quantizers: {weight_active}/{weight_total} active")
        logger.info(f"  - Activation quantizers: {activation_active}/{activation_total} active")
        
        # Verify expectations
        weight_expectation_met = (expected_weights and weight_active > 0) or (not expected_weights and weight_active < weight_total)
        activation_expectation_met = (expected_activations and activation_active > 0) or (not expected_activations and activation_active < activation_total)
        
        success = weight_expectation_met and activation_expectation_met
        
        if success:
            logger.info("✅ Phase state verification PASSED")
        else:
            logger.error("❌ Phase state verification FAILED")
            if not weight_expectation_met:
                logger.error(f"   Weight expectation failed: expected {expected_weights}, got {weight_active}/{weight_total}")
            if not activation_expectation_met:
                logger.error(f"   Activation expectation failed: expected {expected_activations}, got {activation_active}/{activation_total}")
        
        return success
    
    def _is_quantizer_active(self, quantizer) -> bool:
        """
        FIXED: Check if a quantizer is active (FakeQuantize module that's enabled).
        """
        if not self._is_fake_quantize_module(quantizer):
            return False
        
        # Check if it's disabled
        if hasattr(quantizer, 'disable_fake_quant') and quantizer.disable_fake_quant:
            return False
        
        return True

    def emergency_full_reconstruction(self):
        """Emergency method to reconstruct ALL quantizers."""
        logger.info("🚨 Emergency: Full quantizer reconstruction...")
        
        # First re-prepare the entire model for QAT
        try:
            # Apply qconfigs to all modules
            for module_name, config in self.quantizer_configs.items():
                module = self._get_module_by_path(module_name)
                if module is not None and not config['is_detection_head']:
                    module.qconfig = config['qconfig']
            
            # Re-prepare model for QAT
            self.model = torch.quantization.prepare_qat(self.model, inplace=True)
            
            # Verify reconstruction
            fake_quant_count = sum(1 for n, m in self.model.named_modules() 
                                  if 'FakeQuantize' in type(m).__name__)
            
            logger.info(f"🔧 Emergency reconstruction complete: {fake_quant_count} FakeQuantize modules")
            return fake_quant_count > 0
            
        except Exception as e:
            logger.error(f"❌ Emergency reconstruction failed: {e}")
            return False
    
    def get_current_state(self) -> Dict[str, int]:
        """Get current quantizer state for debugging."""
        weight_total = 0
        weight_active = 0
        activation_total = 0
        activation_active = 0
        
        for module_name, quantizer_info in self.quantizer_registry.items():
            module = self._get_module_by_path(module_name)
            if module is None or quantizer_info['is_detection_head']:
                continue
            
            if quantizer_info['has_weight_quantizer']:
                weight_total += 1
                if self._is_quantizer_active(module.weight_fake_quant):
                    weight_active += 1
            
            if quantizer_info['has_activation_quantizer']:
                activation_total += 1
                if self._is_quantizer_active(module.activation_post_process):
                    activation_active += 1
        
        return {
            'weight_active': weight_active,
            'weight_total': weight_total,
            'activation_active': activation_active,
            'activation_total': activation_total
        }

    def count_fake_quantize_modules(self) -> int:
        """
        NEW: Count total FakeQuantize modules in the model.
        This is critical for verification that quantization is preserved.
        """
        count = 0
        for name, module in self.model.named_modules():
            if self._is_fake_quantize_module(module):
                count += 1
        return count

    def emergency_restore_all_quantizers(self) -> bool:
        """
        FIXED: Emergency restore all quantizers to their original FakeQuantize state.
        """
        logger.info("🚨 Emergency: Restoring all quantizers...")
        
        restored_count = 0
        for module_name, quantizer_info in self.quantizer_registry.items():
            module = self._get_module_by_path(module_name)
            if module is None:
                continue
            
            # Restore weight quantizer
            if quantizer_info['has_weight_quantizer']:
                original = quantizer_info['original_weight_fake_quant']
                if original is not None:
                    module.weight_fake_quant = original
                    restored_count += 1
            
            # Restore activation quantizer
            if quantizer_info['has_activation_quantizer']:
                original = quantizer_info['original_activation_post_process']
                if original is not None:
                    module.activation_post_process = original
                    restored_count += 1
        
        # Verify restoration
        fake_quant_count = self.count_fake_quantize_modules()
        
        logger.info(f"🔧 Emergency restoration complete:")
        logger.info(f"  - Restored: {restored_count} quantizers")
        logger.info(f"  - Total FakeQuantize modules: {fake_quant_count}")
        
        return fake_quant_count > 0