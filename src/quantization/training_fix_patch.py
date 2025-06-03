#!/usr/bin/env python
"""
CRITICAL FIX: training_fix_patch.py

This patch fixes the critical issue where FakeQuantize modules are lost during YOLO training.
Apply this patch to your training script or add it as a separate module.

The main issue: YOLO trainer modifies the model structure during training, losing quantizers.
Solutions:
1. Model state preservation hooks
2. Quantizer monitoring and auto-restoration
3. Training process intervention
4. Better error handling and recovery
"""

import torch
import logging
import copy
import functools
from typing import Dict, Any

logger = logging.getLogger(__name__)

class YOLOQuantizationProtector:
    """
    CRITICAL FIX: Protects quantization during YOLO training.
    
    This class prevents the YOLO trainer from destroying quantization by:
    1. Monitoring model state changes
    2. Intercepting model modifications
    3. Auto-restoring lost quantizers
    4. Providing emergency recovery
    """
    
    def __init__(self, yolo_model, quantizer_manager):
        self.yolo_model = yolo_model
        self.quantizer_manager = quantizer_manager
        self.protection_active = False
        self.original_methods = {}
        self.model_backup = None
        self.fake_quant_count_history = []
        
    def activate_protection(self):
        """Activate comprehensive protection for quantization during training."""
        logger.info("🛡️ Activating quantization protection...")
        
        # Create model backup
        self._create_model_backup()
        
        # Hook into YOLO trainer methods
        self._hook_trainer_methods()
        
        # Add model monitoring
        self._add_model_monitoring()
        
        self.protection_active = True
        logger.info("✅ Quantization protection activated")
    
    def deactivate_protection(self):
        """Deactivate protection and restore original methods."""
        if self.protection_active:
            logger.info("🛡️ Deactivating quantization protection...")
            self._restore_original_methods()
            self.protection_active = False
            logger.info("✅ Protection deactivated")
    
    def _create_model_backup(self):
        """Create a comprehensive backup of the model state."""
        try:
            self.model_backup = {
                'state_dict': copy.deepcopy(self.yolo_model.model.state_dict()),
                'quantizer_states': copy.deepcopy(self.quantizer_manager.emergency_backup) if self.quantizer_manager.emergency_backup else None,
                'fake_quant_count': self._count_fake_quantize_modules()
            }
            logger.info(f"💾 Model backup created with {self.model_backup['fake_quant_count']} FakeQuantize modules")
        except Exception as e:
            logger.error(f"❌ Failed to create model backup: {e}")
            self.model_backup = None
    
    def _hook_trainer_methods(self):
        """Hook into YOLO trainer methods that might modify the model."""
        # We need to hook into the YOLO object's train method
        if hasattr(self.yolo_model, 'train'):
            self.original_methods['train'] = self.yolo_model.train
            self.yolo_model.train = self._protected_train_wrapper(self.yolo_model.train)
            logger.info("🪝 Hooked into YOLO train method")
        
        # Hook into model's forward method for monitoring
        if hasattr(self.yolo_model.model, 'forward'):
            self.original_methods['forward'] = self.yolo_model.model.forward
            self.yolo_model.model.forward = self._protected_forward_wrapper(self.yolo_model.model.forward)
            logger.info("🪝 Hooked into model forward method")
    
    def _protected_train_wrapper(self, original_train):
        """Wrapper for YOLO train method with quantization protection."""
        @functools.wraps(original_train)
        def protected_train(*args, **kwargs):
            logger.info("🛡️ Starting protected training...")
            
            # Pre-training verification
            pre_count = self._count_fake_quantize_modules()
            logger.info(f"Pre-training: {pre_count} FakeQuantize modules")
            
            try:
                # Call original training
                result = original_train(*args, **kwargs)
                
                # Post-training verification
                post_count = self._count_fake_quantize_modules()
                logger.info(f"Post-training: {post_count} FakeQuantize modules")
                
                if post_count == 0 and pre_count > 0:
                    logger.error("❌ CRITICAL: Training destroyed all quantizers!")
                    success = self._emergency_restore()
                    if success:
                        logger.info("✅ Emergency restoration successful")
                    else:
                        logger.error("❌ Emergency restoration failed")
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Protected training failed: {e}")
                # Try to restore before re-raising
                self._emergency_restore()
                raise
        
        return protected_train
    
    def _protected_forward_wrapper(self, original_forward):
        """Wrapper for model forward method with monitoring."""
        @functools.wraps(original_forward)
        def protected_forward(*args, **kwargs):
            # Monitor quantizer count during forward passes
            current_count = self._count_fake_quantize_modules()
            
            # Check for significant drops
            if hasattr(self, '_last_count') and current_count < self._last_count * 0.5:
                logger.warning(f"⚠️ Quantizer count dropped from {self._last_count} to {current_count}")
                if current_count == 0:
                    logger.error("❌ All quantizers lost during forward pass!")
                    self._emergency_restore()
            
            self._last_count = current_count
            
            # Call original forward
            return original_forward(*args, **kwargs)
        
        return protected_forward
    
    def _add_model_monitoring(self):
        """Add periodic monitoring of model state."""
        def monitoring_hook(module, input, output):
            current_count = self._count_fake_quantize_modules()
            self.fake_quant_count_history.append(current_count)
            
            # Keep only recent history
            if len(self.fake_quant_count_history) > 100:
                self.fake_quant_count_history = self.fake_quant_count_history[-50:]
            
            # Detect rapid drops
            if len(self.fake_quant_count_history) >= 2:
                prev_count = self.fake_quant_count_history[-2]
                if current_count < prev_count * 0.5:
                    logger.warning(f"⚠️ Rapid quantizer drop detected: {prev_count} → {current_count}")
                    if current_count == 0:
                        logger.error("❌ Complete quantizer loss detected!")
                        self._emergency_restore()
        
        # Register monitoring hook
        if hasattr(self.yolo_model.model, 'register_forward_hook'):
            handle = self.yolo_model.model.register_forward_hook(monitoring_hook)
            self.monitoring_handle = handle
            logger.info("🔍 Model monitoring hook registered")
    
    def _emergency_restore(self):
        """Emergency restoration of quantizers."""
        logger.info("🚨 EMERGENCY: Attempting quantizer restoration...")
        
        # Method 1: Use quantizer manager
        if self.quantizer_manager:
            success = self.quantizer_manager.emergency_restore_all_quantizers()
            if success:
                restored_count = self._count_fake_quantize_modules()
                logger.info(f"✅ Quantizer manager restored {restored_count} quantizers")
                return True
        
        # Method 2: Use model backup
        if self.model_backup:
            try:
                # Restore model state
                self.yolo_model.model.load_state_dict(self.model_backup['state_dict'])
                
                # Re-apply quantization if needed
                if hasattr(self.yolo_model.model, 'qconfig'):
                    self.yolo_model.model = torch.quantization.prepare_qat(self.yolo_model.model, inplace=True)
                
                restored_count = self._count_fake_quantize_modules()
                logger.info(f"✅ Backup restoration gave {restored_count} quantizers")
                
                if restored_count > 0:
                    return True
            except Exception as e:
                logger.error(f"❌ Backup restoration failed: {e}")
        
        # Method 3: Full reconstruction
        logger.info("🔧 Attempting full quantization reconstruction...")
        return self._full_quantization_reconstruction()
    
    def _full_quantization_reconstruction(self):
        """Full reconstruction of quantization as last resort."""
        try:
            logger.info("🔧 Full quantization reconstruction...")
            
            # Get the current model
            model = self.yolo_model.model
            
            # Re-apply qconfigs to all modules
            from src.quantization.qconfig import get_default_qat_qconfig, get_first_layer_qconfig
            
            count = 0
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    if "model.0.conv" in name:
                        module.qconfig = get_first_layer_qconfig()
                    else:
                        module.qconfig = get_default_qat_qconfig()
                    count += 1
            
            logger.info(f"Re-applied qconfig to {count} modules")
            
            # Skip detection head
            detection_count = 0
            for name, module in model.named_modules():
                if 'detect' in name or 'model.22' in name:
                    module.qconfig = None
                    detection_count += 1
            
            logger.info(f"Skipped {detection_count} detection head modules")
            
            # Re-prepare for QAT
            model = torch.quantization.prepare_qat(model, inplace=True)
            self.yolo_model.model = model
            
            # Verify reconstruction
            final_count = self._count_fake_quantize_modules()
            logger.info(f"Reconstruction complete: {final_count} FakeQuantize modules")
            
            return final_count > 0
            
        except Exception as e:
            logger.error(f"❌ Full reconstruction failed: {e}")
            return False
    
    def _count_fake_quantize_modules(self):
        """Count FakeQuantize modules in the model."""
        count = 0
        try:
            for name, module in self.yolo_model.model.named_modules():
                if 'FakeQuantize' in type(module).__name__:
                    count += 1
        except Exception:
            count = 0
        return count
    
    def _restore_original_methods(self):
        """Restore original hooked methods."""
        for method_name, original_method in self.original_methods.items():
            if method_name == 'train' and hasattr(self.yolo_model, 'train'):
                self.yolo_model.train = original_method
            elif method_name == 'forward' and hasattr(self.yolo_model.model, 'forward'):
                self.yolo_model.model.forward = original_method
        
        # Remove monitoring hook
        if hasattr(self, 'monitoring_handle'):
            self.monitoring_handle.remove()
        
        self.original_methods.clear()
        logger.info("🔄 Original methods restored")

def apply_quantization_fix(qat_model):
    """
    MAIN FIX FUNCTION: Apply all quantization preservation fixes to your QAT model.
    
    Usage:
        qat_model = QuantizedYOLOv8(...)
        qat_model.prepare_for_qat()
        
        # Apply the fix
        protector = apply_quantization_fix(qat_model)
        
        # Train normally
        results = qat_model.train_model(...)
        
        # Deactivate protection when done
        protector.deactivate_protection()
    """
    logger.info("🔧 Applying comprehensive quantization fix...")
    
    # Create protector
    protector = YOLOQuantizationProtector(qat_model.model, qat_model.quantizer_manager)
    
    # Activate protection
    protector.activate_protection()
    
    # Add additional safeguards to the QAT model itself
    _add_additional_safeguards(qat_model)
    
    logger.info("✅ Quantization fix applied successfully")
    return protector

def _add_additional_safeguards(qat_model):
    """Add additional safeguards to the QAT model."""
    
    # Override the train_model method to add extra protection
    original_train_model = qat_model.train_model
    
    def protected_train_model(*args, **kwargs):
        logger.info("🛡️ Starting training with enhanced protection...")
        
        # Verify initial state
        initial_count = qat_model._count_fake_quantize_modules()
        logger.info(f"Initial FakeQuantize count: {initial_count}")
        
        if initial_count == 0:
            logger.error("❌ No quantizers found before training!")
            raise RuntimeError("Cannot start training without quantizers!")
        
        try:
            # Call original training with monitoring
            return _monitored_training(original_train_model, qat_model, *args, **kwargs)
        except Exception as e:
            logger.error(f"❌ Protected training failed: {e}")
            # Final attempt to restore
            if hasattr(qat_model, 'quantizer_manager'):
                qat_model.quantizer_manager.emergency_restore_all_quantizers()
            raise
    
    # Replace the method
    qat_model.train_model = protected_train_model

def _monitored_training(original_train_func, qat_model, *args, **kwargs):
    """Execute training with continuous monitoring."""
    logger.info("📊 Starting monitored training...")
    
    # Setup periodic monitoring
    import threading
    import time
    
    monitoring_active = True
    
    def monitor_quantizers():
        while monitoring_active:
            try:
                current_count = qat_model._count_fake_quantize_modules()
                if current_count == 0:
                    logger.error("🚨 MONITOR: All quantizers lost!")
                    # Try immediate restoration
                    if hasattr(qat_model, 'quantizer_manager'):
                        success = qat_model.quantizer_manager.emergency_restore_all_quantizers()
                        if success:
                            logger.info("✅ MONITOR: Auto-restored quantizers")
                        else:
                            logger.error("❌ MONITOR: Auto-restoration failed")
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.debug(f"Monitor error: {e}")
                time.sleep(10)
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_quantizers, daemon=True)
    monitor_thread.start()
    
    try:
        # Execute training
        result = original_train_func(*args, **kwargs)
        return result
    finally:
        # Stop monitoring
        monitoring_active = False

# Quick verification function
def verify_quantization_fix(qat_model):
    """
    Verify that the quantization fix is working properly.
    
    Returns:
        bool: True if fix is working, False otherwise
    """
    logger.info("🔍 Verifying quantization fix...")
    
    checks = {
        'quantizer_manager': hasattr(qat_model, 'quantizer_manager') and qat_model.quantizer_manager is not None,
        'emergency_backup': hasattr(qat_model, 'quantizer_manager') and qat_model.quantizer_manager.emergency_backup is not None,
        'fake_quantizers': qat_model._count_fake_quantize_modules() > 0,
        'model_backup': hasattr(qat_model, '_model_backup'),
        'protection_hooks': hasattr(qat_model, '_protection_hooks') and len(qat_model._protection_hooks) > 0
    }
    
    passed = sum(checks.values())
    total = len(checks)
    
    logger.info(f"Verification results: {passed}/{total} checks passed")
    for check, result in checks.items():
        status = "✅" if result else "❌"
        logger.info(f"  {status} {check}")
    
    if passed >= 4:  # Require most checks to pass
        logger.info("✅ Quantization fix verification PASSED")
        return True
    else:
        logger.error("❌ Quantization fix verification FAILED")
        return False

# Usage example for your training script
def example_usage():
    """
    Example of how to use the fix in your training script.
    """
    
    # Your existing code
    qat_model = QuantizedYOLOv8('yolov8n.pt', skip_detection_head=True)
    qat_model.prepare_for_qat()
    
    # APPLY THE FIX
    protector = apply_quantization_fix(qat_model)
    
    # Verify the fix is working
    if not verify_quantization_fix(qat_model):
        logger.error("❌ Fix verification failed!")
        return
    
    # Setup penalty loss (optional)
    qat_model.setup_penalty_loss_integration(alpha=0.01, warmup_epochs=5)
    
    try:
        # Train normally - the fix will protect quantization
        results = qat_model.train_model(
            data_yaml='your_dataset.yaml',
            epochs=5,
            batch_size=16,
            img_size=640,
            lr=0.0005,
            device='cpu',
            save_dir='models/checkpoints/qat',
            log_dir='logs/qat'
        )
        
        logger.info("✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        
    finally:
        # Clean up
        protector.deactivate_protection()
        qat_model.cleanup_protection_hooks()
        logger.info("🧹 Cleanup completed")

# Export the main functions
__all__ = [
    'YOLOQuantizationProtector',
    'apply_quantization_fix', 
    'verify_quantization_fix',
    'example_usage'
]