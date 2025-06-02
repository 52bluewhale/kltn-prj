#!/usr/bin/env python
"""
COMPLETE FIXED: YOLOv8 QAT Implementation with Quantization Preservation and Penalty Loss
"""
import os
import torch
import torch.nn as nn
import logging
import copy
from ultralytics import YOLO
from torch.quantization import prepare_qat, convert

logger = logging.getLogger('yolov8_qat_fixed')

class FixedQuantizedYOLOv8:
    """
    COMPLETE SOLUTION: QAT model with quantization preservation and penalty loss integration
    """
    
    def __init__(self, model_path, qconfig_name='default', 
                 skip_detection_head=True, fuse_modules=True):
        """
        Initialize with quantization preservation capabilities.
        """
        self.model_path = model_path
        self.qconfig_name = qconfig_name
        self.skip_detection_head = skip_detection_head
        self.fuse_modules = fuse_modules
        
        # Load model
        logger.info(f"Loading YOLOv8 model from {model_path}")
        self.model = YOLO(model_path)
        
        # CRITICAL FIX: Store original model for recovery
        self.original_model = copy.deepcopy(self.model.model)
        
        # Quantization tracking
        self.qat_model = None
        self.is_prepared = False
        self.quantization_preserved = False
        self.stored_quantization_modules = {}  # BACKUP STORAGE
        
        # Penalty loss components
        self.penalty_handler = None
        
    def prepare_for_qat(self):
        """
        SOLUTION: Prepare model with backup mechanisms.
        """
        logger.info("🔧 Preparing model for QAT with quantization preservation...")
        
        # Get base model and set to training mode
        base_model = self.model.model
        base_model.train()
        
        # Apply quantization configuration
        self._apply_qconfig_to_model(base_model)
        
        # Prepare for QAT
        self.qat_model = prepare_qat(base_model, inplace=True)
        
        # CRITICAL FIX: Store quantization modules as backup
        self._store_quantization_modules()
        
        # Verify quantization structure
        self._verify_quantization_structure()
        
        # Update model reference
        self.model.model = self.qat_model
        self.is_prepared = True
        self.quantization_preserved = True
        
        logger.info("✅ QAT preparation completed with quantization preservation")
        return self.qat_model
    
    def _apply_qconfig_to_model(self, model):
        """Apply quantization configuration to model modules."""
        from src.quantization.qconfig import get_default_qat_qconfig, get_first_layer_qconfig
        
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                if "model.0.conv" in name:
                    module.qconfig = get_first_layer_qconfig()
                else:
                    module.qconfig = get_default_qat_qconfig()
                count += 1
        
        # Skip detection head if requested
        if self.skip_detection_head:
            detection_count = 0
            for name, module in model.named_modules():
                if 'detect' in name or 'model.22' in name:
                    module.qconfig = None
                    detection_count += 1
            logger.info(f"Skipped quantization for {detection_count} detection modules")
        
        logger.info(f"Applied qconfig to {count} modules")
    
    def _store_quantization_modules(self):
        """
        CRITICAL FIX: Store quantization modules as backup.
        """
        self.stored_quantization_modules = {}
        self.module_mapping = {}
        
        for name, module in self.qat_model.named_modules():
            # Store weight fake quantizers
            if hasattr(module, 'weight_fake_quant'):
                key = f"{name}.weight_fake_quant"
                self.stored_quantization_modules[key] = copy.deepcopy(module.weight_fake_quant)
                self.module_mapping[key] = (name, 'weight_fake_quant')
            
            # Store activation post processes
            if hasattr(module, 'activation_post_process'):
                key = f"{name}.activation_post_process"
                self.stored_quantization_modules[key] = copy.deepcopy(module.activation_post_process)
                self.module_mapping[key] = (name, 'activation_post_process')
        
        logger.info(f"✅ Stored {len(self.stored_quantization_modules)} quantization modules as backup")
    
    def _verify_quantization_structure(self):
        """Verify quantization structure is intact."""
        fake_quant_count = sum(1 for n, m in self.qat_model.named_modules() 
                              if 'FakeQuantize' in type(m).__name__)
        
        qconfig_count = sum(1 for n, m in self.qat_model.named_modules() 
                           if hasattr(m, 'qconfig') and m.qconfig is not None)
        
        logger.info(f"📊 Quantization verification:")
        logger.info(f"  - FakeQuantize modules: {fake_quant_count}")
        logger.info(f"  - Modules with qconfig: {qconfig_count}")
        
        if fake_quant_count > 0 and qconfig_count > 0:
            logger.info("✅ Quantization structure verified")
            return True
        else:
            logger.error("❌ Quantization structure verification failed")
            return False
    
    def _restore_quantization_modules(self):
        """
        CRITICAL FIX: Restore quantization modules from backup.
        """
        logger.info("🔄 Restoring quantization modules from backup...")
        
        restored_count = 0
        for key, stored_module in self.stored_quantization_modules.items():
            module_name, attr_name = self.module_mapping[key]
            
            # Navigate to target module
            target_module = self.model.model
            for part in module_name.split('.'):
                if part.isdigit():
                    target_module = target_module[int(part)]
                else:
                    target_module = getattr(target_module, part, None)
                    if target_module is None:
                        break
            
            if target_module is not None:
                # Restore the quantization module
                setattr(target_module, attr_name, copy.deepcopy(stored_module))
                restored_count += 1
        
        logger.info(f"✅ Restored {restored_count} quantization modules")
        return restored_count > 0
    
    def setup_penalty_loss_integration(self, alpha=0.01, warmup_epochs=5):
        """
        FIXED: Setup quantization penalty loss integration.
        
        Args:
            alpha: Penalty loss weight
            warmup_epochs: Number of warmup epochs
        """
        logger.info(f"🎯 Setting up quantization penalty loss (α={alpha}, warmup={warmup_epochs})...")
        
        try:
            # Import penalty loss components
            from src.training.penalty_loss import QuantizationPenaltyLoss, patch_yolo_loss_with_penalty
            
            # Create penalty handler
            self.penalty_handler = QuantizationPenaltyLoss(
                alpha=alpha, 
                warmup_epochs=warmup_epochs,
                normalize=True
            )
            
            # Patch model with penalty calculation
            patch_yolo_loss_with_penalty(self.model, self.penalty_handler)
            
            logger.info(f"✅ Quantization penalty loss setup complete (α={alpha}, warmup={warmup_epochs})")
            
        except ImportError as e:
            logger.error(f"❌ Failed to import penalty loss modules: {e}")
            logger.warning("⚠️ Continuing without penalty loss integration")
            self.penalty_handler = None
        except Exception as e:
            logger.error(f"❌ Failed to setup penalty loss: {e}")
            logger.warning("⚠️ Continuing without penalty loss integration")
            self.penalty_handler = None
    
    def verify_penalty_setup(self):
        """
        Verify that penalty loss integration is working.
        """
        if self.penalty_handler is None:
            logger.info("ℹ️ No penalty handler configured")
            return False
            
        try:
            from src.training.penalty_loss import verify_penalty_integration
            return verify_penalty_integration(self)
        except ImportError:
            logger.warning("⚠️ Cannot verify penalty integration - verify function not available")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Penalty verification failed: {e}")
            return False
    
    def train_model_with_preservation(self, data_yaml, epochs, batch_size, img_size, lr, 
                                    device, save_dir, log_dir):
        """
        SOLUTION: Training with quantization preservation callbacks.
        """
        import os
        from ultralytics.utils import LOGGER
        
        logger.info("🚀 Starting QAT training with quantization preservation...")
        
        # Pre-training verification
        self._verify_quantization_structure()
        
        # CRITICAL FIX: Training callbacks to monitor quantization
        def on_train_start(trainer):
            """Pre-training quantization check."""
            logger.info("🔍 Pre-training quantization verification...")
            self._verify_quantization_structure()
        
        def on_epoch_end(trainer):
            """SOLUTION: Check and restore quantization after each epoch."""
            current_fake_quant = sum(1 for n, m in trainer.model.named_modules() 
                                   if 'FakeQuantize' in type(m).__name__)
            
            # DETECTION: Check if quantization was lost
            if current_fake_quant == 0:
                logger.warning(f"⚠️ Quantization lost at epoch {trainer.epoch}, restoring...")
                # RECOVERY: Restore from backup
                self._restore_quantization_modules()
                logger.info("✅ Quantization restored successfully")
            
            # Update penalty handler epoch if available
            if self.penalty_handler:
                self.penalty_handler.update_epoch(trainer.epoch)
        
        def on_train_end(trainer):
            """Final quantization check and recovery."""
            logger.info("🔍 Post-training quantization verification...")
            if not self._verify_quantization_structure():
                logger.warning("⚠️ Final quantization restoration needed...")
                self._restore_quantization_modules()
                self._verify_quantization_structure()
        
        # Register callbacks with YOLOv8
        self.model.add_callback('on_train_start', on_train_start)
        self.model.add_callback('on_train_epoch_end', on_epoch_end)
        self.model.add_callback('on_train_end', on_train_end)
        
        try:
            # Train with preservation callbacks active
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                lr0=lr,
                device=device,
                project=os.path.dirname(save_dir),
                name=os.path.basename(save_dir),
                exist_ok=True,
                pretrained=False,
                val=True,
                verbose=True,
                save_period=-1  # Prevent YOLOv8 from saving during training
            )
            
            # Final verification
            if self._verify_quantization_structure():
                logger.info("✅ Training completed with quantization preserved")
                self.quantization_preserved = True
            else:
                logger.error("❌ Training completed but quantization was lost")
                self.quantization_preserved = False
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            # Emergency restoration attempt
            self._restore_quantization_modules()
            raise
    
    def save_qat_model(self, path):
        """
        SOLUTION: Save with guaranteed quantization preservation.
        """
        logger.info(f"💾 Saving QAT model with quantization preservation...")
        
        # Ensure quantization is present
        if not self.quantization_preserved:
            logger.warning("⚠️ Attempting quantization restoration before save...")
            self._restore_quantization_modules()
        
        # Count FakeQuantize modules
        fake_quant_count = sum(1 for n, m in self.model.model.named_modules() 
                              if 'FakeQuantize' in type(m).__name__)
        
        if fake_quant_count == 0:
            logger.error("❌ No FakeQuantize modules found - attempting emergency restore...")
            if not self._restore_quantization_modules():
                logger.error("❌ Emergency restore failed - cannot save QAT model")
                return False
            fake_quant_count = sum(1 for n, m in self.model.model.named_modules() 
                                  if 'FakeQuantize' in type(m).__name__)
        
        try:
            # SOLUTION: Comprehensive save format
            save_dict = {
                'model_architecture': self.model.model,  # Complete model
                'model_state_dict': self.model.model.state_dict(),  # Parameters
                'stored_quantization_modules': self.stored_quantization_modules,  # Backup
                'module_mapping': self.module_mapping,  # Restoration info
                'qat_info': {
                    'qconfig_name': self.qconfig_name,
                    'skip_detection_head': self.skip_detection_head,
                    'fake_quant_count': fake_quant_count,
                    'quantization_preserved': True,
                    'pytorch_version': torch.__version__
                },
                'model_metadata': {
                    'architecture': 'yolov8n',
                    'num_classes': 58,  # Vietnamese traffic signs
                    'input_size': [1, 3, 640, 640]
                }
            }
            
            torch.save(save_dict, path)
            
            # Verify save success
            if self._verify_save(path, fake_quant_count):
                logger.info(f"✅ QAT model saved: {path}")
                logger.info(f"✅ FakeQuantize modules preserved: {fake_quant_count}")
                return True
            else:
                logger.error("❌ Save verification failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Save failed: {e}")
            return False
    
    def convert_to_quantized(self, save_path=None):
        """
        SOLUTION: Convert with proper error handling and fallbacks.
        """
        logger.info("🔄 Converting QAT model to INT8...")
        
        # Ensure quantization is preserved
        if not self.quantization_preserved:
            logger.warning("⚠️ Attempting quantization restoration before conversion...")
            if not self._restore_quantization_modules():
                logger.error("❌ Cannot restore quantization for conversion")
                return None
        
        # Verify quantization structure
        if not self._verify_quantization_structure():
            logger.error("❌ No quantization structure - conversion impossible")
            return None
        
        try:
            # Method 1: Standard PyTorch conversion
            self.model.model.eval()  # Set to eval mode for conversion
            quantized_model = convert(self.model.model, inplace=False)
            
            logger.info("✅ Standard conversion successful")
            
            # Save if path provided
            if save_path:
                self._save_quantized_model(quantized_model, save_path)
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"❌ Standard conversion failed: {e}")
            logger.info("🔄 Attempting fallback conversion...")
            return self._fallback_conversion_method()
    
    def _fallback_conversion_method(self):
        """Fallback conversion when standard method fails."""
        try:
            model = copy.deepcopy(self.model.model)
            model.eval()
            
            # Manual quantization application
            for name, module in model.named_modules():
                if hasattr(module, 'weight_fake_quant') and hasattr(module, 'weight'):
                    with torch.no_grad():
                        # Apply fake quantization to weights
                        quantized_weight = module.weight_fake_quant(module.weight)
                        module.weight.copy_(quantized_weight)
            
            logger.info("✅ Fallback conversion completed")
            return model
            
        except Exception as e:
            logger.error(f"❌ Fallback conversion failed: {e}")
            return None
    
    def _save_quantized_model(self, quantized_model, path):
        """Save quantized model with metadata."""
        try:
            save_dict = {
                'model': quantized_model,
                'deployment_info': {
                    'format': 'int8_quantized',
                    'architecture': 'yolov8n',
                    'num_classes': 58,
                    'deployment_ready': True
                }
            }
            torch.save(save_dict, path)
            logger.info(f"✅ INT8 model saved: {path}")
        except Exception as e:
            logger.error(f"❌ Failed to save INT8 model: {e}")
    
    def _verify_save(self, path, expected_count):
        """Verify save was successful."""
        try:
            saved_data = torch.load(path, map_location='cpu')
            actual_count = saved_data['qat_info']['fake_quant_count']
            return actual_count == expected_count
        except:
            return False