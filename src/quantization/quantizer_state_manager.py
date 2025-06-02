#!/usr/bin/env python
"""
quantizer_state_manager.py

CREATE THIS FILE: src/quantization/quantizer_state_manager.py

This file fixes the quantizer reference loss issue during phase transitions.
"""

import torch
import logging

logger = logging.getLogger(__name__)

class QuantizerStateManager:
    """
    Manages quantizer states during phased training.
    Prevents quantizer reference loss during phase transitions.
    """
    
    def __init__(self, model):
        self.model = model
        self.quantizer_configs = {}
        self.module_paths = {}
        self._build_quantizer_configs()
    
    def _build_quantizer_configs(self):
        """Build configuration templates for all quantizers."""
        logger.info("🔧 Building quantizer configuration templates...")
        
        from src.quantization.qconfig import get_default_qat_qconfig, get_first_layer_qconfig
        
        for name, module in self.model.named_modules():
            # Store module path for reconstruction
            self.module_paths[name] = module
            
            # Build configs for quantizable modules
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                # Determine appropriate qconfig
                if "model.0.conv" in name:
                    qconfig = get_first_layer_qconfig()
                else:
                    qconfig = get_default_qat_qconfig()
                
                self.quantizer_configs[name] = {
                    'module_type': type(module).__name__,
                    'qconfig': qconfig,
                    'should_have_weight_quant': True,
                    'should_have_activation_quant': True,
                    'is_detection_head': 'detect' in name or 'model.22' in name
                }
        
        logger.info(f"✅ Built configs for {len(self.quantizer_configs)} quantizable modules")
    
    def set_phase_state(self, phase_name, weights_enabled=True, activations_enabled=True):
        """
        Set quantizer states by RECONSTRUCTING them instead of storing references.
        """
        logger.info(f"🔄 Reconstructing quantizers for {phase_name}")
        
        changes_made = 0
        
        for module_name, config in self.quantizer_configs.items():
            module = self._get_module_by_path(module_name)
            if module is None:
                continue
            
            # Skip detection head if configured
            if config['is_detection_head']:
                continue
            
            # Determine what should be enabled for this module
            should_enable_weights = weights_enabled and config['should_have_weight_quant']
            should_enable_activations = activations_enabled and config['should_have_activation_quant']
            
            # RECONSTRUCT weight quantizer
            if should_enable_weights:
                if not self._has_valid_weight_quantizer(module):
                    self._reconstruct_weight_quantizer(module, config['qconfig'])
                    changes_made += 1
            else:
                if self._has_valid_weight_quantizer(module):
                    module.weight_fake_quant = torch.nn.Identity()
                    changes_made += 1
            
            # RECONSTRUCT activation quantizer
            if should_enable_activations:
                if not self._has_valid_activation_quantizer(module):
                    self._reconstruct_activation_quantizer(module, config['qconfig'])
                    changes_made += 1
            else:
                if self._has_valid_activation_quantizer(module):
                    module.activation_post_process = torch.nn.Identity()
                    changes_made += 1
        
        logger.info(f"✅ Phase {phase_name}: Reconstructed {changes_made} quantizers")
        
        # Verify the reconstruction
        return self._verify_phase_state(weights_enabled, activations_enabled)

    def _get_module_by_path(self, path):
        """Get module by its path, handling potential model structure changes."""
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
    
    def _verify_phase_state(self, expected_weights, expected_activations):
        """Verify the current phase state."""
        weight_active = 0
        activation_active = 0
        total_modules = 0
        
        for module_name, config in self.quantizer_configs.items():
            module = self._get_module_by_path(module_name)
            if module is None or config['is_detection_head']:
                continue
            
            total_modules += 1
            
            if self._has_valid_weight_quantizer(module):
                weight_active += 1
            
            if self._has_valid_activation_quantizer(module):
                activation_active += 1
        
        logger.info(f"📊 Verification: {weight_active}/{total_modules} weight, {activation_active}/{total_modules} activation")
        
        # Check expectations
        if expected_weights and weight_active == 0:
            logger.error("❌ Expected weight quantizers but found none!")
            return False
        
        if expected_activations and activation_active == 0:
            logger.error("❌ Expected activation quantizers but found none!")
            return False
        
        if not expected_activations and activation_active > 0:
            logger.warning(f"⚠️ Found {activation_active} activation quantizers when none expected")
        
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
    
    def get_current_state(self):
        """Get current quantizer state for debugging."""
        weight_active = 0
        activation_active = 0
        total_weight = 0
        total_activation = 0
        
        for module_name, config in self.quantizer_configs.items():
            module = self._get_module_by_path(module_name)
            if module is None or config['is_detection_head']:
                continue
            
            # Count total quantizers
            if config['should_have_weight_quant']:
                total_weight += 1
                if self._has_valid_weight_quantizer(module):
                    weight_active += 1
            
            if config['should_have_activation_quant']:
                total_activation += 1
                if self._has_valid_activation_quantizer(module):
                    activation_active += 1
        
        return {
            'weight_active': weight_active,
            'activation_active': activation_active,
            'total_weight': total_weight,
            'total_activation': total_activation
        }