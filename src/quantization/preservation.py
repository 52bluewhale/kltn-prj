#!/usr/bin/env python
"""
src/quantization/preservation.py

Create this file to handle quantization preservation during YOLOv8 training.
This fixes the critical issue where FakeQuantize modules are lost after training.
"""

import torch
import logging
import copy
from pathlib import Path

logger = logging.getLogger(__name__)

class QuantizationStateBackup:
    """
    Backup and restore quantization state to prevent loss during YOLOv8 training.
    """
    
    def __init__(self, model):
        self.model = model
        self.quantization_backup = {}
        self.qconfig_backup = {}
        self.module_registry = {}
        
    def backup_quantization_state(self):
        """Backup all quantization modules and qconfigs before training."""
        logger.info("🔒 Backing up quantization state before training...")
        
        self.quantization_backup.clear()
        self.qconfig_backup.clear()
        self.module_registry.clear()
        
        for name, module in self.model.named_modules():
            module_path = name
            
            # Backup FakeQuantize modules
            if hasattr(module, 'weight_fake_quant'):
                self.quantization_backup[f"{module_path}.weight_fake_quant"] = {
                    'module': copy.deepcopy(module.weight_fake_quant),
                    'type': 'weight_fake_quant',
                    'parent_path': module_path
                }
            
            if hasattr(module, 'activation_post_process'):
                self.quantization_backup[f"{module_path}.activation_post_process"] = {
                    'module': copy.deepcopy(module.activation_post_process),
                    'type': 'activation_post_process', 
                    'parent_path': module_path
                }
            
            # Backup qconfig
            if hasattr(module, 'qconfig') and module.qconfig is not None:
                self.qconfig_backup[module_path] = copy.deepcopy(module.qconfig)
            
            # Store module reference
            self.module_registry[module_path] = module
        
        backup_count = len(self.quantization_backup)
        qconfig_count = len(self.qconfig_backup)
        
        logger.info(f"✅ Backed up {backup_count} quantization modules and {qconfig_count} qconfigs")
        return backup_count > 0
    
    def restore_quantization_state(self):
        """Restore quantization state after training."""
        logger.info("🔓 Restoring quantization state after training...")
        
        if not self.quantization_backup:
            logger.error("❌ No quantization backup found!")
            return False
        
        restored_count = 0
        
        # First restore qconfigs
        for module_path, qconfig in self.qconfig_backup.items():
            if module_path in self.module_registry:
                module = self.module_registry[module_path]
                if hasattr(module, 'qconfig'):
                    module.qconfig = qconfig
        
        # Then restore quantization modules
        for quant_path, backup_info in self.quantization_backup.items():
            parent_path = backup_info['parent_path']
            attr_name = backup_info['type']
            backed_up_module = backup_info['module']
            
            if parent_path in self.module_registry:
                parent_module = self.module_registry[parent_path]
                
                # Restore the quantization module
                setattr(parent_module, attr_name, backed_up_module)
                restored_count += 1
        
        logger.info(f"✅ Restored {restored_count} quantization modules")
        
        # Verify restoration
        current_fake_quant = sum(1 for n, m in self.model.named_modules() 
                                if 'FakeQuantize' in type(m).__name__)
        
        logger.info(f"🔍 Post-restoration verification: {current_fake_quant} FakeQuantize modules")
        
        return restored_count > 0
    
    def get_backup_stats(self):
        """Get statistics about the backup."""
        return {
            'quantization_modules': len(self.quantization_backup),
            'qconfigs': len(self.qconfig_backup),
            'has_backup': len(self.quantization_backup) > 0
        }


class QuantizationPreservingTrainer:
    """
    Custom trainer that preserves quantization during YOLOv8 training.
    """
    
    def __init__(self, qat_model):
        self.qat_model = qat_model
        self.backup_manager = QuantizationStateBackup(qat_model.model.model)
        self.training_completed = False
        
    def train_with_quantization_preservation(self, **train_args):
        """
        Train model while preserving quantization structure.
        
        Args:
            **train_args: Arguments to pass to YOLOv8 training
            
        Returns:
            Training results
        """
        logger.info("🚀 Starting quantization-preserving training...")
        
        # Step 1: Backup quantization state
        backup_success = self.backup_manager.backup_quantization_state()
        if not backup_success:
            logger.error("❌ Failed to backup quantization state!")
            raise RuntimeError("Cannot proceed without quantization backup")
        
        # Step 2: Train with YOLOv8
        try:
            logger.info("🏋️ Running YOLOv8 training (quantization may be temporarily lost)...")
            results = self.qat_model.model.train(**train_args)
            self.training_completed = True
            logger.info("✅ YOLOv8 training completed")
            
        except Exception as e:
            logger.error(f"❌ YOLOv8 training failed: {e}")
            raise
        
        # Step 3: Restore quantization state  
        logger.info("🔄 Restoring quantization state...")
        restore_success = self.backup_manager.restore_quantization_state()
        
        if not restore_success:
            logger.error("❌ Failed to restore quantization state!")
            logger.info("🔧 Attempting emergency quantization recovery...")
            
            # Emergency recovery: re-prepare model for QAT
            success = self._emergency_recovery()
            if not success:
                raise RuntimeError("Failed to recover quantization state")
        
        # Step 4: Verify final state
        final_verification = self._verify_final_state()
        if not final_verification:
            logger.warning("⚠️ Quantization verification failed, but training completed")
        
        return results
    
    def _emergency_recovery(self):
        """Emergency recovery of quantization state."""
        logger.info("🚨 Attempting emergency quantization recovery...")
        
        try:
            # Re-prepare the model for QAT
            self.qat_model.prepare_for_qat()
            
            # Verify recovery
            fake_quant_count = sum(1 for n, m in self.qat_model.model.model.named_modules() 
                                  if 'FakeQuantize' in type(m).__name__)
            
            if fake_quant_count > 0:
                logger.info(f"✅ Emergency recovery successful: {fake_quant_count} FakeQuantize modules")
                return True
            else:
                logger.error("❌ Emergency recovery failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Emergency recovery failed: {e}")
            return False
    
    def _verify_final_state(self):
        """Verify that quantization is properly restored."""
        fake_quant_count = sum(1 for n, m in self.qat_model.model.model.named_modules() 
                              if 'FakeQuantize' in type(m).__name__)
        
        qconfig_count = sum(1 for n, m in self.qat_model.model.model.named_modules() 
                           if hasattr(m, 'qconfig') and m.qconfig is not None)
        
        logger.info(f"🔍 Final verification:")
        logger.info(f"  - FakeQuantize modules: {fake_quant_count}")
        logger.info(f"  - Modules with qconfig: {qconfig_count}")
        
        success = fake_quant_count > 0 and qconfig_count > 0
        
        if success:
            logger.info("✅ Quantization successfully preserved through training!")
        else:
            logger.error("❌ Quantization was not preserved!")
        
        return success


class QuantizationDebugger:
    """
    Helper class to debug quantization issues.
    """
    
    @staticmethod
    def analyze_model_state(model, phase_name="unknown"):
        """Analyze and log the current quantization state of the model."""
        logger.info(f"🔍 Analyzing model state for {phase_name}:")
        
        fake_quant_modules = []
        qconfig_modules = []
        
        for name, module in model.named_modules():
            # Check for FakeQuantize modules
            if 'FakeQuantize' in type(module).__name__:
                fake_quant_modules.append(name)
            
            # Check for qconfig
            if hasattr(module, 'qconfig') and module.qconfig is not None:
                qconfig_modules.append(name)
        
        logger.info(f"  📊 Statistics:")
        logger.info(f"    - FakeQuantize modules: {len(fake_quant_modules)}")
        logger.info(f"    - Modules with qconfig: {len(qconfig_modules)}")
        
        if len(fake_quant_modules) > 0:
            logger.info(f"  📝 First 5 FakeQuantize modules:")
            for i, name in enumerate(fake_quant_modules[:5]):
                logger.info(f"    {i+1}. {name}")
        
        return {
            'fake_quant_count': len(fake_quant_modules),
            'qconfig_count': len(qconfig_modules),
            'fake_quant_modules': fake_quant_modules,
            'qconfig_modules': qconfig_modules
        }
    
    @staticmethod
    def compare_states(before_state, after_state, operation_name="operation"):
        """Compare quantization states before and after an operation."""
        logger.info(f"🔄 Comparing states before/after {operation_name}:")
        
        fake_quant_diff = after_state['fake_quant_count'] - before_state['fake_quant_count']
        qconfig_diff = after_state['qconfig_count'] - before_state['qconfig_count']
        
        logger.info(f"  📊 Changes:")
        logger.info(f"    - FakeQuantize: {before_state['fake_quant_count']} → {after_state['fake_quant_count']} ({fake_quant_diff:+d})")
        logger.info(f"    - QConfig: {before_state['qconfig_count']} → {after_state['qconfig_count']} ({qconfig_diff:+d})")
        
        if fake_quant_diff < 0:
            logger.warning(f"⚠️ Lost {abs(fake_quant_diff)} FakeQuantize modules during {operation_name}")
        
        if qconfig_diff < 0:
            logger.warning(f"⚠️ Lost {abs(qconfig_diff)} qconfigs during {operation_name}")
        
        return fake_quant_diff >= 0 and qconfig_diff >= 0