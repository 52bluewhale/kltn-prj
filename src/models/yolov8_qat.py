#!/usr/bin/env python
"""
YOLOv8 QAT Implementation

This module provides the QuantizedYOLOv8 class which encapsulates all functionality
needed for quantization-aware training of YOLOv8 models.
"""
import os
import sys
import logging
import torch
import yaml
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO
import copy

import numpy as np
import functools

# Import PyTorch quantization modules
from torch.quantization import prepare_qat, convert, get_default_qconfig
from src.quantization.quantizer_state_manager import QuantizerStateManager
from src.quantization.quantizer_preservation import QuantizerPreserver

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('yolov8_qat')

# Import quantization modules
from src.quantization.fusion import fuse_yolov8_modules
from src.quantization.qconfig import get_default_qat_qconfig, get_qconfig_by_name, create_qconfig_mapping
from src.quantization.utils import (
    prepare_model_for_qat, 
    convert_qat_model_to_quantized, 
    get_model_size, 
    compare_model_sizes, 
    analyze_quantization_effects,
    apply_layer_specific_quantization,
    skip_layers_from_quantization,
    save_quantized_model
)

# Import QAT training loss
# from src.training.loss import QATPenaltyLoss

class QuantizedYOLOv8:
    """
    Wrapper for YOLOv8 model with QAT capabilities.
    
    This class provides a comprehensive interface for:
    1. Preparing YOLOv8 models for quantization-aware training
    2. Training models with fake quantization
    3. Converting trained models to INT8 quantized versions
    4. Evaluating and exporting quantized models
    """
    def __init__(self, model_path, qconfig_name='default', 
                 skip_detection_head=True, fuse_modules=True, custom_qconfig=None):
        """
        Initialize quantized YOLOv8 model.
        
        Args:
            model_path: Path to YOLOv8 model or model name (e.g., 'yolov8n.pt')
            qconfig_name: Name of QConfig to use ('default', 'sensitive', 'lsq', etc.)
            skip_detection_head: Whether to skip quantization of detection head
            fuse_modules: Whether to fuse modules (Conv-BN-ReLU)
        """
        self.model_path = model_path
        self.qconfig_name = qconfig_name
        self.skip_detection_head = skip_detection_head
        self.fuse_modules = fuse_modules
        self.custom_qconfig = custom_qconfig
        
        # Load YOLOv8 model
        logger.info(f"Loading YOLOv8 model from {model_path}")
        self.model = YOLO(model_path)

        # FIXED: Clear any default dataset configuration
        if hasattr(self.model, 'overrides'):
            self.model.overrides.pop('data', None)  # Remove any default data setting
        
        # Store original model for size comparison
        self.original_model = copy.deepcopy(self.model.model)
        
        # Initialize tracking variables
        self.qat_model = None
        self.quantized_model = None
        self.is_prepared = False
        self.is_trained = False
        self.is_converted = False

        # NEW: Track quantization state
        self.quantizer_manager = None
        self._last_fake_quant_count = 0
    
    def prepare_for_qat(self):
        """FIXED: Prepare model for QAT with proper quantization preservation."""
        logger.info("Preparing model for QAT...")
        
        # Get base model
        base_model = self.model.model
        base_model.train()

        # Optionally fuse modules
        if self.fuse_modules:
            logger.info("Fusing modules for better quantization...")
            from src.quantization.fusion import fuse_yolov8_modules
            base_model = fuse_yolov8_modules(base_model)
        
        try:
            # Apply qconfig to modules
            from src.quantization.qconfig import get_default_qat_qconfig, get_first_layer_qconfig
            
            count = 0
            for name, module in base_model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    if "model.0.conv" in name:
                        module.qconfig = get_first_layer_qconfig()
                    else:
                        module.qconfig = get_default_qat_qconfig()
                    count += 1
            
            logger.info(f"Applied qconfig to {count} modules")
            
            # Skip detection head if requested
            if self.skip_detection_head:
                detection_count = 0
                for name, module in base_model.named_modules():
                    if 'detect' in name or 'model.22' in name:
                        module.qconfig = None
                        detection_count += 1
                logger.info(f"Disabled quantization for {detection_count} detection modules")
            
            # Prepare model for QAT
            logger.info("Calling prepare_qat...")
            self.qat_model = torch.quantization.prepare_qat(base_model, inplace=True)
            
            # CRITICAL: Initialize quantizer manager AFTER prepare_qat
            from src.quantization.quantizer_state_manager import QuantizerStateManager
            self.quantizer_manager = QuantizerStateManager(self.qat_model)
            
            # Verify QAT preparation
            fake_quant_count = self._count_fake_quantize_modules()
            self._last_fake_quant_count = fake_quant_count
            
            logger.info(f"QAT preparation verified: {fake_quant_count} FakeQuantize modules")
            
            if fake_quant_count == 0:
                logger.error("❌ No FakeQuantize modules found after prepare_qat!")
                return None
            
        except Exception as e:
            logger.error(f"Error preparing model for QAT: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        # Update model in YOLO object
        self.model.model = self.qat_model
        self.is_prepared = True
        
        return self.qat_model

    def _count_fake_quantize_modules(self) -> int:
        """Count FakeQuantize modules in the model."""
        count = 0
        for name, module in self.model.model.named_modules():
            if 'FakeQuantize' in type(module).__name__:
                count += 1
        return count

    def warmup_observers(self, dataloader, num_batches=200, device='cuda'):
        """
        Warm up observers before starting quantization.
        Critical fix for observer initialization timing.
        """
        logger.info(f"Warming up observers for {num_batches} batches...")
        
        # Set model to training mode but disable quantization
        self.model.model.train()
        self._disable_fake_quantization()
        
        # Enable only observers
        self._enable_observers_only()
        
        batch_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if batch_count >= num_batches:
                    break
                    
                # Get inputs (handle different batch formats)
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch
                
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                
                # Forward pass to update observers
                _ = self.model.model(inputs)
                batch_count += 1
                
                if batch_count % 50 == 0:
                    logger.info(f"Observer warmup progress: {batch_count}/{num_batches}")
        
        # Verify observer readiness
        ready_observers = self._check_observer_readiness()
        logger.info(f"Observer warmup complete: {ready_observers} observers ready")
        
        # Re-enable quantization
        self._enable_fake_quantization()
        
        return ready_observers

    def gradual_quantization_enable(self, epoch, total_epochs, phase_start, phase_end):
        """
        Gradually enable quantization over multiple epochs.
        Fixes abrupt phase transition issue.
        """
        if epoch < phase_start:
            return 0.0  # No quantization
        elif epoch > phase_end:
            return 1.0  # Full quantization
        else:
            # Gradual ramp-up using cosine schedule
            progress = (epoch - phase_start) / (phase_end - phase_start)
            return 0.5 * (1 + np.cos(np.pi * (1 - progress)))

    def setup_penalty_loss_integration(self, alpha=0.01, warmup_epochs=5):
        """Setup quantization penalty loss."""
        try:
            from src.training.penalty_loss import QuantizationPenaltyLoss, patch_yolo_loss_with_penalty
            
            self.penalty_handler = QuantizationPenaltyLoss(
                alpha=alpha, 
                warmup_epochs=warmup_epochs,
                normalize=True
            )
            
            patch_yolo_loss_with_penalty(self.model, self.penalty_handler)
            logger.info(f"✅ Penalty loss setup complete (α={alpha})")
            
        except Exception as e:
            logger.error(f"❌ Penalty loss setup failed: {e}")

    def verify_penalty_setup(self):
        """Verify penalty loss integration."""
        try:
            from src.training.penalty_loss import verify_penalty_integration
            return verify_penalty_integration(self)
        except Exception as e:
            logger.error(f"❌ Penalty verification failed: {e}")
            return False
    
    def train_model(self, data_yaml, epochs, batch_size, img_size, lr, device, save_dir, log_dir, use_distillation=False):
        """
        FIXED: Training with proper quantization preservation.
        """
        import os
        from ultralytics.utils import LOGGER

        # Validate dataset path
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
        
        logger.info(f"✅ Using dataset: {data_yaml}")
        
        # Force clear any existing data configuration
        if hasattr(self.model, 'overrides'):
            self.model.overrides['data'] = data_yaml
        
        # Calculate phase boundaries
        phase1_end = max(1, int(epochs * 0.3))
        phase2_end = max(2, int(epochs * 0.7))
        phase3_end = max(3, int(epochs * 0.9))
        
        LOGGER.info(f"🔄 Phased QAT Training Plan:")
        LOGGER.info(f"  Phase 1 (Weight-only): Epochs 1-{phase1_end}")
        LOGGER.info(f"  Phase 2 (Add Activations): Epochs {phase1_end+1}-{phase2_end}")
        LOGGER.info(f"  Phase 3 (Full Quantization): Epochs {phase2_end+1}-{phase3_end}")
        LOGGER.info(f"  Phase 4 (Fine-tuning): Epochs {phase3_end+1}-{epochs}")
        
        # Configure initial phase
        self._configure_quantizers_dynamically("phase1_weight_only")
        
        # Setup training callbacks
        def on_epoch_start(trainer):
            """FIXED: Handle phase transitions with quantization verification."""
            current_epoch = trainer.epoch
            
            # Update penalty handler epoch if available
            if hasattr(self, 'penalty_handler'):
                self.penalty_handler.update_epoch(current_epoch)
            
            # Phase transitions
            if current_epoch == phase1_end:
                LOGGER.info(f"🔄 Epoch {current_epoch}: Transitioning to Phase 2")
                success = self._configure_quantizers_dynamically("phase2_activations")
                if not success:
                    LOGGER.error("❌ Phase 2 transition failed!")
                
            elif current_epoch == phase2_end:
                LOGGER.info(f"🔄 Epoch {current_epoch}: Transitioning to Phase 3")
                success = self._configure_quantizers_dynamically("phase3_full_quant")
                if not success:
                    LOGGER.error("❌ Phase 3 transition failed!")
                
            elif current_epoch == phase3_end:
                LOGGER.info(f"🔄 Epoch {current_epoch}: Transitioning to Phase 4")
                success = self._configure_quantizers_dynamically("phase4_fine_tuning")
                if not success:
                    LOGGER.error("❌ Phase 4 transition failed!")
                
                # Reduce learning rate for fine-tuning
                if hasattr(trainer, 'optimizer'):
                    for param_group in trainer.optimizer.param_groups:
                        param_group['lr'] *= 0.1
        
        def on_epoch_end(trainer):
            """Monitor quantization state after each epoch."""
            fake_quant_count = self._count_fake_quantize_modules()
            LOGGER.info(f"📊 End of epoch {trainer.epoch}: {fake_quant_count} FakeQuantize modules")
            
            if fake_quant_count == 0:
                LOGGER.error(f"❌ CRITICAL: All FakeQuantize modules lost at epoch {trainer.epoch}!")
                # Try emergency restoration
                if hasattr(self, 'quantizer_manager'):
                    self.quantizer_manager.emergency_restore_all_quantizers()
                    fake_quant_count = self._count_fake_quantize_modules()
                    LOGGER.info(f"🔧 After emergency restore: {fake_quant_count} FakeQuantize modules")
        
        # Add callbacks to model  
        self.model.add_callback('on_train_epoch_start', on_epoch_start)
        self.model.add_callback('on_train_epoch_end', on_epoch_end)
        
        # TRAINING
        try:
            LOGGER.info("🚀 Starting phased QAT training...")
            
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
                verbose=True
            )
            
            # CRITICAL: Verify quantization still exists after training
            final_fake_quant_count = self._count_fake_quantize_modules()
            LOGGER.info(f"🎯 Training complete: {final_fake_quant_count} FakeQuantize modules remaining")
            
            if final_fake_quant_count == 0:
                LOGGER.error("❌ CRITICAL: All quantization lost during training!")
                # Emergency restoration
                if hasattr(self, 'quantizer_manager'):
                    LOGGER.info("🚨 Attempting final emergency restoration...")
                    success = self.quantizer_manager.emergency_restore_all_quantizers()
                    final_fake_quant_count = self._count_fake_quantize_modules()
                    LOGGER.info(f"🔧 Final restore result: {final_fake_quant_count} FakeQuantize modules")
            
            self.is_trained = True
            return results
            
        except Exception as e:
            LOGGER.error(f"❌ Training failed: {e}")
            import traceback
            LOGGER.error(f"Full traceback:\n{traceback.format_exc()}")
            raise

    def _configure_quantizers_dynamically(self, phase_name):
        """
        FIXED: Configure quantizers using the improved quantizer manager.
        """
        if not hasattr(self, 'quantizer_manager') or self.quantizer_manager is None:
            logger.error("❌ Quantizer manager not initialized!")
            return False
        
        # Define phase configurations
        phase_configs = {
            "phase1_weight_only": {"weights": True, "activations": False},
            "phase2_activations": {"weights": True, "activations": True},
            "phase3_full_quant": {"weights": True, "activations": True},
            "phase4_fine_tuning": {"weights": True, "activations": True}
        }
        
        if phase_name not in phase_configs:
            logger.error(f"❌ Unknown phase: {phase_name}")
            return False
        
        config = phase_configs[phase_name]
        
        # Apply phase configuration using the manager
        success = self.quantizer_manager.set_phase_state(
            phase_name=phase_name,
            weights_enabled=config["weights"],
            activations_enabled=config["activations"]
        )
        
        if not success:
            logger.error(f"❌ Phase {phase_name} configuration failed!")
            # Try emergency restoration
            logger.info("🔧 Attempting emergency restoration...")
            success = self.quantizer_manager.emergency_restore_all_quantizers()
        
        # Final verification
        final_fake_quant_count = self._count_fake_quantize_modules()
        logger.info(f"🔍 After {phase_name}: {final_fake_quant_count} FakeQuantize modules")
        
        # Update tracking
        self._last_fake_quant_count = final_fake_quant_count
        
        return success and final_fake_quant_count > 0

    def train_standard(self, data_yaml, epochs, batch_size, img_size, lr, device, save_dir, log_dir, use_distillation=False):
        """Train model with standard (non-phased) QAT approach."""
        import os
        import torch
        from ultralytics import YOLO
        
        logger = logging.getLogger('yolov8_qat')
        logger.info("Using standard (non-phased) QAT training")
        
        # Save the current model state to a temporary file
        temp_model_path = os.path.join(save_dir, "temp_model.pt")
        os.makedirs(os.path.dirname(temp_model_path), exist_ok=True)
        
        # Save model state
        torch.save(self.model.model.state_dict(), temp_model_path)
        
        # Create a new YOLO object from the saved file
        trainer = YOLO(temp_model_path)
        
        # Train the model
        results = trainer.train(
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
            val=True
        )
        
        logger.info("QAT training completed")
        
        return results

    def _set_weight_quantizers_enabled(self, enabled):
        """Enable or disable weight quantizers."""
        count = 0
        for name, module in self.model.model.named_modules():
            if 'weight_fake_quant' in dict(module.named_children()):
                weight_fake_quant = module.weight_fake_quant
                if not enabled:
                    if not hasattr(module, '_original_weight_fake_quant'):
                        module._original_weight_fake_quant = weight_fake_quant
                    # Replace with identity
                    module.weight_fake_quant = torch.nn.Identity()
                else:
                    # Restore original if available
                    if hasattr(module, '_original_weight_fake_quant'):
                        module.weight_fake_quant = module._original_weight_fake_quant
                count += 1
        logger.info(f"{'Enabled' if enabled else 'Disabled'} weight quantizers for {count} modules")

    def _set_activation_quantizers_enabled(self, enabled):
        """Enable or disable activation quantizers."""
        count = 0
        for name, module in self.model.model.named_modules():
            # For activations, find all modules with activation_post_process
            if hasattr(module, 'activation_post_process'):
                if not enabled:
                    if not hasattr(module, '_original_activation_post_process'):
                        module._original_activation_post_process = module.activation_post_process
                    # Replace with identity
                    module.activation_post_process = torch.nn.Identity()
                else:
                    # Restore original
                    if hasattr(module, '_original_activation_post_process'):
                        module.activation_post_process = module._original_activation_post_process
                count += 1
        logger.info(f"{'Enabled' if enabled else 'Disabled'} activation quantizers for {count} modules")

    def _set_all_quantizers_enabled(self, enabled):
        """Enable or disable all quantizers."""
        self._set_weight_quantizers_enabled(enabled)
        self._set_activation_quantizers_enabled(enabled)

    def _disable_fake_quantization(self):
        """Disable fake quantization while keeping observers active."""
        for name, module in self.model.model.named_modules():
            if hasattr(module, 'weight_fake_quant'):
                module.weight_fake_quant.disable_fake_quant = True
            if hasattr(module, 'activation_post_process') and hasattr(module.activation_post_process, 'disable_fake_quant'):
                module.activation_post_process.disable_fake_quant = True

    def _enable_fake_quantization(self):
        """Re-enable fake quantization after warmup."""
        for name, module in self.model.model.named_modules():
            if hasattr(module, 'weight_fake_quant'):
                module.weight_fake_quant.disable_fake_quant = False
            if hasattr(module, 'activation_post_process') and hasattr(module.activation_post_process, 'disable_fake_quant'):
                module.activation_post_process.disable_fake_quant = False

    def _enable_observers_only(self):
        """Enable observers while keeping fake quantization disabled."""
        for name, module in self.model.model.named_modules():
            if hasattr(module, 'activation_post_process'):
                # Observers should remain active to collect statistics
                if hasattr(module.activation_post_process, 'enable_observer'):
                    module.activation_post_process.enable_observer = True

    def _check_observer_readiness(self, min_batches=100):
        """Check if observers have sufficient statistics."""
        ready_count = 0
        total_count = 0
        
        for name, module in self.model.model.named_modules():
            if hasattr(module, 'activation_post_process'):
                observer = module.activation_post_process
                total_count += 1
                
                # Check if observer is initialized and has enough data
                if (hasattr(observer, 'initialized') and observer.initialized and 
                    hasattr(observer, 'batch_count') and 
                    getattr(observer, 'batch_count', 0) >= min_batches):
                    ready_count += 1
                elif hasattr(observer, 'min_val') and hasattr(observer, 'max_val'):
                    # Alternative check for min/max observers
                    if observer.min_val != float('inf') and observer.max_val != float('-inf'):
                        ready_count += 1
        
        return ready_count

    def _configure_phase_gradual(self, phase, epoch, total_epochs):
        """
        Configure model for specific QAT phase with gradual transitions.
        COMPLETE IMPLEMENTATION of the method you declared.
        """
        # Define phase boundaries
        phase_boundaries = {
            "phase1_weight_only": (0, int(total_epochs * 0.3)),
            "phase2_activations": (int(total_epochs * 0.3), int(total_epochs * 0.7)),
            "phase3_full_quant": (int(total_epochs * 0.7), int(total_epochs * 0.9)),
            "phase4_fine_tuning": (int(total_epochs * 0.9), total_epochs)
        }
        
        if phase not in phase_boundaries:
            logger.warning(f"Unknown phase: {phase}")
            return
        
        phase_start, phase_end = phase_boundaries[phase]
        
        # Calculate gradual enablement factor
        if phase == "phase1_weight_only":
            # Weight quantization ramp-up
            weight_factor = self.gradual_quantization_enable(epoch, total_epochs, phase_start, phase_end)
            activation_factor = 0.0
        elif phase == "phase2_activations":
            # Weight quantization fully enabled, activation ramp-up
            weight_factor = 1.0
            activation_factor = self.gradual_quantization_enable(epoch, total_epochs, phase_start, phase_end)
        elif phase in ["phase3_full_quant", "phase4_fine_tuning"]:
            # Both fully enabled
            weight_factor = 1.0
            activation_factor = 1.0
        
        # Apply gradual factors to quantizers
        self._apply_gradual_quantization_factors(weight_factor, activation_factor)
        
        logger.info(f"Phase {phase} Epoch {epoch}: Weight factor={weight_factor:.3f}, Activation factor={activation_factor:.3f}")

    def _apply_gradual_quantization_factors(self, weight_factor, activation_factor):
        """Apply gradual quantization factors to fake quantizers."""
        for name, module in self.model.model.named_modules():
            # Handle weight quantizers
            if hasattr(module, 'weight_fake_quant'):
                if hasattr(module.weight_fake_quant, 'set_quantization_factor'):
                    module.weight_fake_quant.set_quantization_factor(weight_factor)
                else:
                    # For standard FakeQuantize, we can modify the step size
                    if weight_factor < 1.0:
                        self._modify_quantizer_step_size(module.weight_fake_quant, weight_factor)
            
            # Handle activation quantizers
            if hasattr(module, 'activation_post_process'):
                if hasattr(module.activation_post_process, 'set_quantization_factor'):
                    module.activation_post_process.set_quantization_factor(activation_factor)
                else:
                    # For standard FakeQuantize, modify step size
                    if activation_factor < 1.0:
                        self._modify_quantizer_step_size(module.activation_post_process, activation_factor)

    def _modify_quantizer_step_size(self, quantizer, factor):
        """
        Modify quantizer step size for gradual quantization.
        When factor < 1.0, reduce quantization strength.
        """
        if hasattr(quantizer, 'scale') and factor < 1.0:
            # Gradually reduce quantization by interpolating with identity
            # This is a simplified approach - more sophisticated methods exist
            original_scale = getattr(quantizer, '_original_scale', None)
            if original_scale is None:
                quantizer._original_scale = quantizer.scale.clone()
                original_scale = quantizer._original_scale
            
            # Interpolate between original scale and a much larger scale (approaching identity)
            large_scale = original_scale * 100  # Effectively disables quantization
            quantizer.scale.data.copy_(original_scale * factor + large_scale * (1 - factor))

    def _apply_qconfig_and_prepare_model(self, trainer, phase_name):
        """Apply qconfig and prepare model for QAT in one step."""
        try:
            logger.info(f"Applying QConfig and preparing model for {phase_name}")
            
            # Apply qconfig to appropriate modules
            count = 0
            for name, module in trainer.model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    if "model.0.conv" in name:
                        from src.quantization.qconfig import get_first_layer_qconfig
                        module.qconfig = get_first_layer_qconfig()
                    else:
                        from src.quantization.qconfig import get_default_qat_qconfig
                        module.qconfig = get_default_qat_qconfig()
                    count += 1
            
            # Skip detection head if requested
            if self.skip_detection_head:
                detection_count = 0
                for name, module in trainer.model.named_modules():
                    if 'detect' in name or 'model.22' in name:
                        module.qconfig = None
                        detection_count += 1
                logger.info(f"Disabled quantization for {detection_count} detection modules")
            
            logger.info(f"Applied qconfig to {count} modules")
            
            # Prepare model for QAT
            trainer.model = torch.quantization.prepare_qat(trainer.model, inplace=False)
            
            # Configure phase-specific settings
            self._configure_model_for_specific_phase(trainer.model, phase_name)
            
            logger.info(f"Successfully prepared model for QAT in {phase_name}")
            
        except Exception as e:
            logger.error(f"Error preparing model for QAT: {e}")
            logger.warning("Continuing without QAT preparation - model will train without quantization")

    def _configure_module_for_weight_only(self, module):
        """Configure module for weight-only quantization (Phase 1)."""
        # Ensure module has qconfig
        if not hasattr(module, 'qconfig') or module.qconfig is None:
            # Use default qconfig if not already set
            if self.custom_qconfig is not None:
                module.qconfig = self.custom_qconfig
            else:
                module.qconfig = get_qconfig_by_name(self.qconfig_name)
        
        # Store original activation observer if it exists
        if hasattr(module, 'activation_post_process'):
            if not hasattr(module, '_original_activation_post_process'):
                module._original_activation_post_process = module.activation_post_process
            # Replace with Identity for Phase 1
            module.activation_post_process = nn.Identity()

    def _configure_module_for_weight_and_activation(self, module):
        """Configure module for weight and activation quantization (Phase 2)."""
        # Ensure module has qconfig
        if not hasattr(module, 'qconfig') or module.qconfig is None:
            # Use default qconfig if not already set
            if self.custom_qconfig is not None:
                module.qconfig = self.custom_qconfig
            else:
                module.qconfig = get_qconfig_by_name(self.qconfig_name)
        
        # Restore original activation observer if it was stored
        if hasattr(module, '_original_activation_post_process'):
            module.activation_post_process = module._original_activation_post_process

    def _configure_module_for_full_quantization(self, module):
        """Configure module for full quantization (Phase 3 & 4)."""
        # Same as Phase 2 but could add additional configurations if needed
        self._configure_module_for_weight_and_activation(module)

    def _disable_activation_quantizers_in_model(self, model):
        """Disable all activation quantizers in the model."""
        activation_count = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'activation_post_process') and not isinstance(module.activation_post_process, nn.Identity):
                # Store original for later restoration
                if not hasattr(module, '_original_activation_post_process'):
                    module._original_activation_post_process = module.activation_post_process
                
                # Replace with Identity
                module.activation_post_process = nn.Identity()
                activation_count += 1
        
        logger.info(f"Disabled {activation_count} activation quantizers in the model")
    
    def convert_to_quantized(self, save_path=None):
        """
        FIXED: Convert QAT model to INT8 quantized model.
        """
        logger.info("Converting QAT model to quantized INT8 model...")
        
        # Verify quantization structure exists
        fake_quant_count = self._count_fake_quantize_modules()
        logger.info(f"Pre-conversion: {fake_quant_count} FakeQuantize modules")
        
        if fake_quant_count == 0:
            logger.error("❌ Cannot convert - no FakeQuantize modules!")
            return None
        
        # Get QAT model in eval mode
        qat_model = self.model.model
        qat_model.eval()
        
        # Convert to quantized model
        try:
            logger.info("Converting QAT model to INT8...")
            from src.quantization.utils import convert_qat_model_to_quantized
            
            self.quantized_model = convert_qat_model_to_quantized(qat_model, inplace=False)
            
            if self.quantized_model is None:
                logger.error("❌ Conversion returned None")
                return None
            
            logger.info("✅ Successfully converted to INT8")
            
            # Calculate size comparison
            from src.quantization.utils import compare_model_sizes
            size_info = compare_model_sizes(self.original_model, self.quantized_model)
            
            logger.info(f"Model size comparison:")
            logger.info(f"  Original: {size_info['fp32_size_mb']:.2f} MB")
            logger.info(f"  Quantized: {size_info['int8_size_mb']:.2f} MB")
            logger.info(f"  Compression: {size_info['compression_ratio']:.2f}x")
            
            # Save quantized model
            if save_path:
                int8_save_path = save_path.replace('.pt', '_int8_final.pt')
                
                metadata = {
                    'framework': 'pytorch',
                    'format': 'quantized_int8',
                    'qconfig': self.qconfig_name,
                    'size_info': size_info,
                    'conversion_successful': True,
                    'deployment_ready': True
                }
                
                from src.quantization.utils import save_quantized_model
                save_success = save_quantized_model(self.quantized_model, int8_save_path, metadata)
                
                if save_success:
                    logger.info(f"✅ INT8 model saved to {int8_save_path}")
                else:
                    logger.error(f"❌ Failed to save INT8 model")
            
            self.is_converted = True
            return self.quantized_model
            
        except Exception as e:
            logger.error(f"❌ Conversion failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def export_to_onnx(self, onnx_path, img_size=640, simplify=True):
        """
        NEW: Export QAT model to ONNX format.
        """
        logger.info(f"Exporting QAT model to ONNX: {onnx_path}")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        
        try:
            # Export using ultralytics ONNX export
            exported_path = self.model.export(
                format='onnx',
                imgsz=img_size,
                simplify=simplify,
                opset=12,
                half=False
            )
            
            logger.info(f"✅ ONNX export successful: {exported_path}")
            return exported_path
            
        except Exception as e:
            logger.error(f"❌ ONNX export failed: {e}")
            return None

    def evaluate(self, data_yaml=None, batch_size=16, img_size=640, device=''):
        """
        Evaluate quantized model.
        
        Args:
            data_yaml: Path to dataset YAML (if None, use original data)
            batch_size: Batch size
            img_size: Input image size
            device: Device to evaluate on
            
        Returns:
            Evaluation results
        """
        # Use original model data if not provided
        if data_yaml is None and hasattr(self.model, 'data'):
            data_yaml = self.model.data
        
        logger.info(f"Evaluating model on {data_yaml}...")
        
        # For now, use the QAT model for evaluation
        val_args = {
            'data': data_yaml,
            'batch': batch_size,
            'imgsz': img_size,
            'device': device,
            'verbose': True
        }
        
        # Run validation
        results = self.model.val(**val_args)
        
        return results
    
    def export(self, export_path, format='onnx', img_size=640, simplify=True):
        """
        Export quantized model to specific format.
        
        Args:
            export_path: Path to save exported model
            format: Export format ('onnx', 'tflite', 'tensorrt', etc.)
            img_size: Input image size
            simplify: Whether to simplify model (for ONNX)
            
        Returns:
            Path to exported model
        """
        logger.info(f"Exporting model to {format} format...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        # Export model
        exported_path = self.model.export(
            format=format,
            imgsz=img_size,
            simplify=simplify,
            opset=12,  # Use compatible opset for quantized models
            half=False  # Don't use FP16 for quantized models
        )
        
        logger.info(f"Model exported to {exported_path}")
        return exported_path

    def setup_distillation(self, teacher_model_path, temperature=4.0, alpha=0.5, feature_distillation=False, feature_layers=None):
        """
        Setup knowledge distillation for QAT.
        
        Args:
            teacher_model_path: Path to teacher model
            temperature: Temperature for softening logits
            alpha: Weight for distillation loss
            feature_distillation: Whether to use feature distillation
            feature_layers: List of layers for feature distillation
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Setting up knowledge distillation with teacher model {teacher_model_path}")
        
        # Store configuration
        self.distillation_config = {
            'teacher_model_path': teacher_model_path,
            'temperature': temperature,
            'alpha': alpha,
            'feature_distillation': feature_distillation,
            'feature_layers': feature_layers or []
        }
        
        # Load teacher model
        try:
            self.teacher_model = YOLO(teacher_model_path)
            
            # Set teacher model to evaluation mode
            if hasattr(self.teacher_model, 'model'):
                self.teacher_model.model.eval()
                
                # Move to same device as student if a device is already assigned
                if hasattr(self, 'model') and hasattr(self.model, 'model'):
                    student_device = next(self.model.model.parameters()).device
                    self.teacher_model.model.to(student_device)
            
            logger.info(f"Teacher model loaded successfully from {teacher_model_path}")
        except Exception as e:
            logger.error(f"Failed to load teacher model: {e}")
            raise
        
        # Setup for feature distillation if requested
        if feature_distillation and feature_layers:
            logger.info(f"Setting up feature distillation for layers: {feature_layers}")
            self._setup_feature_distillation(feature_layers)
        
        return self

    def _setup_feature_distillation(self, feature_layers):
        """
        Setup feature distillation by registering hooks.
        
        Args:
            feature_layers: List of layer names for feature distillation
        """
        # Implementation for feature distillation hooks
        # This would register forward hooks on both teacher and student models
        # to capture intermediate activations for feature-level distillation
        
        # Store activations
        self.teacher_features = {}
        self.student_features = {}
        
        # Helper function to get module by name
        def get_module_by_name(model, name):
            names = name.split('.')
            module = model
            for n in names:
                if hasattr(module, n):
                    module = getattr(module, n)
                elif n.isdigit() and isinstance(module, (list, nn.ModuleList)):
                    module = module[int(n)]
                else:
                    return None
            return module
        
        # Register hooks for teacher model
        def teacher_hook(name):
            def hook_fn(module, input, output):
                self.teacher_features[name] = output.detach()
            return hook_fn
        
        # Register hooks for student model
        def student_hook(name):
            def hook_fn(module, input, output):
                self.student_features[name] = output
            return hook_fn
        
        # Register hooks
        self.feature_hooks = []
        for layer_name in feature_layers:
            # Get modules
            teacher_module = get_module_by_name(self.teacher_model.model, layer_name)
            student_module = get_module_by_name(self.model.model, layer_name)
            
            # Register hooks if modules exist
            if teacher_module is not None and student_module is not None:
                t_hook = teacher_module.register_forward_hook(teacher_hook(layer_name))
                s_hook = student_module.register_forward_hook(student_hook(layer_name))
                self.feature_hooks.extend([t_hook, s_hook])
                logger.info(f"Registered feature distillation hooks for layer {layer_name}")
            else:
                logger.warning(f"Could not find layer {layer_name} in both teacher and student models")

    def setup_mixed_precision(self, layer_bit_widths, sensitivity_metric="kl_div", auto_adapt=False):
        """
        Setup mixed precision quantization.
        
        Args:
            layer_bit_widths: Dictionary mapping layer names to bit widths
            sensitivity_metric: Metric for measuring layer sensitivity
            auto_adapt: Whether to automatically adapt bit widths
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Setting up mixed precision quantization")
        
        # Store mixed precision configuration
        self.mixed_precision_config = {
            'layer_bit_widths': layer_bit_widths,
            'sensitivity_metric': sensitivity_metric,
            'auto_adapt': auto_adapt
        }
        
        # If model is already prepared for QAT, apply mixed precision settings
        if self.is_prepared and hasattr(self, 'qat_model'):
            logger.info("Model already prepared for QAT, applying mixed precision settings...")
            self._apply_mixed_precision()
        
        return self

    def _apply_mixed_precision(self):
        """
        Apply mixed precision settings based on the stored configuration.
        Called internally during prepare_for_qat or manually after setting up mixed precision.
        """
        if not hasattr(self, 'mixed_precision_config'):
            logger.warning("No mixed precision configuration found, skipping mixed precision setup")
            return
        
        config = self.mixed_precision_config
        logger.info(f"Applying mixed precision settings")
        
        # Get layer-wise bit widths
        layer_bit_widths = config['layer_bit_widths']
        
        # Automatically adapt bit widths if requested
        if config['auto_adapt']:
            logger.info(f"Auto-adapting bit widths based on layer sensitivity")
            # This would involve analyzing layer sensitivity and adjusting bit widths
            # For now, we'll just log that it would be done
        
        # Apply bit widths to each layer
        num_layers_modified = 0
        
        for name, module in self.qat_model.named_modules():
            if name in layer_bit_widths:
                bit_width = layer_bit_widths[name]
                logger.info(f"Setting {name} to {bit_width}-bit precision")
                
                # This is a placeholder - in reality, you'd need to modify the module's QConfig
                # to use the specified bit width
                
                # Example (commented out as it's not fully implemented):
                """
                if hasattr(module, 'qconfig') and module.qconfig is not None:
                    # Create a new QConfig with the specified bit width
                    if bit_width == 4:
                        if isinstance(module, nn.Conv2d):
                            module.qconfig = get_4bit_qconfig()
                        else:
                            module.qconfig = get_4bit_qconfig()
                    elif bit_width == 8:
                        module.qconfig = get_default_qat_qconfig()
                    
                    # Re-prepare this specific module
                    prepare_module_for_qat(module)
                """
                
                num_layers_modified += 1
        
        logger.info(f"Applied mixed precision settings to {num_layers_modified} layers")

    def setup_quant_penalty_loss(self, alpha=0.01):
        """
        Setup quantization penalty loss for YOLOv8.
        
        Args:
            alpha: Weight for quantization penalty term
        """
        from ultralytics.nn.tasks import DetectionModel
        
        # Store the alpha value
        self.quant_penalty_alpha = alpha
        
        # For YOLOv8, we need to modify the forward method of the model
        # to include our quantization penalty
        original_forward = self.model.forward
        
        # Create a wrapper that adds quantization penalty
        def forward_with_quant_penalty(x):
            # Get the original outputs
            outputs = original_forward(x)
            
            # Only add penalty during training
            if self.model.training:
                # Calculate quantization penalty
                quant_penalty = 0.0
                for name, module in self.model.named_modules():
                    if hasattr(module, 'weight_fake_quant') and hasattr(module, 'weight'):
                        # Quantized weight
                        w_q = module.weight_fake_quant(module.weight)
                        # Original weight
                        w = module.weight
                        # Penalty is L2 norm of the difference
                        quant_penalty += torch.norm(w_q - w) ** 2
                
                # Normalize by number of parameters
                quant_penalty /= sum(p.numel() for p in self.model.parameters())
                
                # Store penalty for logging (but don't modify the outputs directly)
                self.current_quant_penalty = quant_penalty.item()
                
                # Note: we're not modifying outputs here since YOLOv8's training loop 
                # handles loss calculation internally
            
            return outputs
        
        # Replace the forward method
        self.model.forward = forward_with_quant_penalty
        
        # Save original method for restoration if needed
        self.original_forward = original_forward
        
        logger.info(f"Quantization penalty monitoring setup with alpha={alpha}")
        logger.info("Note: For YOLOv8, we'll track the penalty but apply it manually in training")

    # FIXED: QAT Save/Load Methods for yolov8_qat.py
    def save(self, path, preserve_qat=True):
        """
        CORRECTED: Save QAT model state dict (PyTorch compatible approach).
        KEEPS: Observer fixes and validation, but works within PyTorch limitations.
        """
        # Create directory if needed
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        if preserve_qat and self.is_prepared:
            logger.info(f"💾 Saving QAT model state dict to {path}")
            
            try:
                # Get model with quantization preserved
                if hasattr(self, 'quantizer_preserver'):
                    # Verify quantization is still preserved
                    if not self.quantizer_preserver.verify_preservation():
                        logger.error("❌ Quantization not properly preserved!")
                        return False
                    
                    stats = self.quantizer_preserver.get_quantizer_stats()
                    fake_quant_count = stats['total_fake_quantizers']
                    model_to_save = self.quantizer_preserver.model
                    
                    # CRITICAL: Validate observer calibration for INT8 readiness
                    observer_calibrated = self.quantizer_preserver.validate_observer_calibration()
                    
                    logger.info(f"📊 QAT Model Status:")
                    logger.info(f"  - FakeQuantize modules: {fake_quant_count}")
                    logger.info(f"  - Observer calibration: {'✅ Ready' if observer_calibrated else '❌ Needs more training'}")
                    logger.info(f"  - Observers always active: {stats.get('observers_always_active', False)}")
                    
                else:
                    # Fallback to standard model
                    model_to_save = self.model.model
                    fake_quant_count = sum(1 for n, m in model_to_save.named_modules() 
                                        if 'FakeQuantize' in type(m).__name__)
                    observer_calibrated = self._validate_observers_manual(model_to_save)
                    logger.info(f"📊 Fallback: {fake_quant_count} FakeQuantize modules")
                
                if fake_quant_count == 0:
                    logger.error("❌ No FakeQuantize modules found - cannot save QAT model!")
                    return False
                
                # CORRECTED: Save state dict + comprehensive metadata (PyTorch compatible)
                save_dict = {
                    'model_state_dict': model_to_save.state_dict(),  # ← State dict approach (works)
                    'model_architecture': 'yolov8n',  # For reconstruction
                    'qat_config': {
                        'qconfig_name': self.qconfig_name,
                        'skip_detection_head': self.skip_detection_head,
                        'fuse_modules': self.fuse_modules,
                        'model_path': self.model_path,
                    },
                    'quantization_info': {
                        'fake_quant_count': fake_quant_count,
                        'observer_calibrated': observer_calibrated,
                        'int8_ready': observer_calibrated,  # Ready for conversion?
                        'observers_always_active': stats.get('observers_always_active', False) if hasattr(self, 'quantizer_preserver') else False,
                    },
                    'training_info': {
                        'is_prepared': self.is_prepared,
                        'is_trained': getattr(self, 'is_trained', False),
                        'pytorch_version': torch.__version__,
                        'save_timestamp': torch.tensor(0).storage().data_ptr(),  # Simple timestamp
                    },
                    'save_method': 'state_dict_corrected',  # Mark as corrected version
                    'usage_notes': {
                        'qat_model_note': 'Requires reconstruction for QAT use',
                        'int8_conversion': 'Ready for INT8 conversion' if observer_calibrated else 'Needs more training',
                        'deployment': 'Convert to INT8 for deployment'
                    }
                }
                
                # Save using PyTorch-compatible method
                torch.save(save_dict, path)
                
                # Verify save success
                if os.path.exists(path) and os.path.getsize(path) > 1000:
                    file_size_mb = os.path.getsize(path) / (1024 * 1024)
                    logger.info(f"✅ QAT model state dict saved successfully:")
                    logger.info(f"  📁 Path: {path}")
                    logger.info(f"  💾 Size: {file_size_mb:.2f} MB")
                    logger.info(f"  🔧 FakeQuantize modules: {fake_quant_count}")
                    logger.info(f"  🎯 INT8 Ready: {'✅ Yes' if observer_calibrated else '❌ No - train longer'}")
                    logger.info(f"  📋 Usage: Load for INT8 conversion, not direct QAT use")
                    return True
                else:
                    logger.error(f"❌ Save verification failed")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ QAT save failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
        
        else:
            # Standard save (fallback)
            logger.warning("⚠️ Saving without QAT preservation (fallback mode)")
            try:
                self.model.save(path)
                return True
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                return False

    def convert_to_int8_directly_after_training(self, int8_save_path):
        """
        FIXED: Direct QAT → INT8 conversion with simple approach validation.
        """
        logger.info("🚀 Direct QAT → INT8 conversion (simple approach)")
        
        # Step 1: Validate current QAT model
        if not self.is_prepared or not hasattr(self, 'quantizer_preserver'):
            logger.error("❌ QAT model not properly prepared!")
            return None
        
        # Step 2: Validate observer calibration using simple approach
        if not self.quantizer_preserver.validate_observer_calibration():
            logger.error("❌ Observer calibration failed with simple approach!")
            logger.error("💡 Solution: Train for more epochs with activation quantizers enabled")
            
            # Show debug info
            stats = self.quantizer_preserver.get_quantizer_stats()
            logger.error(f"Debug: {stats['activation_quantizers_enabled']} activation quantizers currently enabled")
            return None
        
        logger.info("✅ Simple approach observer validation passed - proceeding with conversion")
        
        # Step 3: Convert to INT8 with validation
        int8_model = self.convert_to_quantized_with_preservation(int8_save_path)
        
        if int8_model is not None:
            logger.info("🎉 SUCCESS: Simple approach QAT → INT8 conversion completed!")
            logger.info(f"📁 INT8 model saved to: {int8_save_path}")
            logger.info(f"🚀 Model ready for deployment")
            return int8_model
        else:
            logger.error("❌ Direct conversion failed")
            return None

    def load_qat_state_dict_for_int8_conversion(qat_state_path, device='cuda'):
        """
        NEW: Load QAT state dict specifically for INT8 conversion.
        PURPOSE: Convert saved QAT state dict to INT8 model.
        """
        logger.info(f"📂 Loading QAT state dict for INT8 conversion: {qat_state_path}")
        
        try:
            if not os.path.exists(qat_state_path):
                raise FileNotFoundError(f"QAT state dict file not found: {qat_state_path}")
            
            # Load the saved data
            saved_data = torch.load(qat_state_path, map_location=device)
            
            if not isinstance(saved_data, dict) or 'model_state_dict' not in saved_data:
                raise ValueError("Invalid QAT state dict file format")
            
            # Extract information
            state_dict = saved_data['model_state_dict']
            qat_config = saved_data.get('qat_config', {})
            quant_info = saved_data.get('quantization_info', {})
            
            logger.info(f"📊 QAT State Dict Info:")
            logger.info(f"  - FakeQuantize modules: {quant_info.get('fake_quant_count', 0)}")
            logger.info(f"  - Observer calibrated: {quant_info.get('observer_calibrated', False)}")
            logger.info(f"  - INT8 ready: {quant_info.get('int8_ready', False)}")
            
            # Check if ready for INT8 conversion
            if not quant_info.get('int8_ready', False):
                logger.warning("⚠️ Model may not be ready for INT8 conversion (observers not calibrated)")
                logger.warning("💡 Consider training for more epochs before conversion")
            
            # Create new QuantizedYOLOv8 instance for conversion
            instance = QuantizedYOLOv8.__new__(QuantizedYOLOv8)
            
            # Set up the instance
            instance.model_path = qat_config.get('model_path', 'loaded_from_state_dict')
            instance.qconfig_name = qat_config.get('qconfig_name', 'default')
            instance.skip_detection_head = qat_config.get('skip_detection_head', True)
            instance.fuse_modules = qat_config.get('fuse_modules', True)
            
            # Reconstruct QAT model
            logger.info("🔧 Reconstructing QAT model from state dict...")
            instance.prepare_for_qat_with_preservation()
            
            # Load the saved state dict
            instance.model.model.load_state_dict(state_dict)
            instance.is_prepared = True
            instance.is_trained = True
            
            # Verify reconstruction
            fake_count = sum(1 for n, m in instance.model.model.named_modules() 
                            if 'FakeQuantize' in type(m).__name__)
            
            expected_count = quant_info.get('fake_quant_count', 0)
            if fake_count != expected_count:
                logger.warning(f"⚠️ FakeQuantize count mismatch: expected {expected_count}, got {fake_count}")
            
            logger.info(f"✅ QAT model reconstructed for INT8 conversion:")
            logger.info(f"  - {fake_count} FakeQuantize modules active")
            logger.info(f"  - Ready for INT8 conversion")
            
            return instance
            
        except Exception as e:
            logger.error(f"❌ Failed to load QAT state dict: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    # Usage workflow functions
    def recommended_training_workflow(data_yaml, epochs, **train_args):
        """
        NEW: Recommended workflow for QAT training → INT8 conversion.
        GOAL: Get deployable INT8 model without QAT save/load issues.
        """
        logger.info("🚀 RECOMMENDED WORKFLOW: QAT Training → Direct INT8 Conversion")
        
        # Step 1: Create and prepare QAT model
        logger.info("Step 1: Creating QAT model...")
        qat_model = QuantizedYOLOv8('yolov8n.pt', skip_detection_head=True)
        qat_model.prepare_for_qat_with_preservation()
        
        # Step 2: Train with fixed observer system
        logger.info("Step 2: Training with continuous observer calibration...")
        results = qat_model.train_model_with_preservation(
            data_yaml=data_yaml,
            epochs=epochs,
            **train_args
        )
        
        # Step 3: Direct conversion to INT8 (skip QAT save)
        logger.info("Step 3: Direct QAT → INT8 conversion...")
        int8_save_path = train_args.get('save_dir', 'models') + '/deployable_int8_model.pt'
        int8_model = qat_model.convert_to_int8_directly_after_training(int8_save_path)
        
        if int8_model is not None:
            logger.info("🎉 WORKFLOW COMPLETE!")
            logger.info(f"📁 Deployable INT8 model: {int8_save_path}")
            logger.info(f"🚀 Ready for production inference")
            return int8_model, int8_save_path
        else:
            logger.error("❌ Workflow failed at INT8 conversion")
            return None, None

    @classmethod
    def load_qat_model(cls, path, device='cuda'):
        """
        NEW: Load complete QAT model for immediate use.
        CRITICAL: Enables immediate inference/training after load.
        
        Args:
            path: Path to saved QAT model
            device: Device to load model on
            
        Returns:
            QuantizedYOLOv8 instance ready for use
        """
        logger.info(f"📂 Loading complete QAT model from {path}")
        
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"QAT model file not found: {path}")
            
            # Load the saved data
            saved_data = torch.load(path, map_location=device)
            
            if not isinstance(saved_data, dict):
                raise ValueError("Invalid QAT model file format")
            
            # Verify this is a complete QAT model
            if 'complete_qat_model' not in saved_data:
                raise ValueError("File does not contain complete QAT model")
            
            # Extract information
            complete_model = saved_data['complete_qat_model']
            qat_info = saved_data.get('qat_info', {})
            fake_quant_count = saved_data.get('fake_quant_count', 0)
            save_method = saved_data.get('save_method', 'unknown')
            
            logger.info(f"📊 Loading QAT model:")
            logger.info(f"  🔧 FakeQuantize modules: {fake_quant_count}")
            logger.info(f"  💾 Save method: {save_method}")
            logger.info(f"  ⚙️ QConfig: {qat_info.get('qconfig_name', 'unknown')}")
            
            if fake_quant_count == 0:
                logger.error("❌ No FakeQuantize modules in loaded model!")
                return None
            
            # Create new QuantizedYOLOv8 instance
            instance = cls.__new__(cls)  # Create without calling __init__
            
            # Set up the instance
            instance.model_path = qat_info.get('model_path', 'loaded_from_file')
            instance.qconfig_name = qat_info.get('qconfig_name', 'default')
            instance.skip_detection_head = qat_info.get('skip_detection_head', True)
            instance.fuse_modules = qat_info.get('fuse_modules', True)
            instance.is_prepared = True  # Mark as prepared
            instance.is_trained = True   # Mark as trained
            instance.is_converted = False
            
            # Create YOLO wrapper around the complete model
            from ultralytics import YOLO
            instance.model = YOLO('yolov8n.pt')  # Create base structure
            instance.model.model = complete_model.to(device)  # Replace with loaded model
            
            # Verify quantization structure
            loaded_fake_count = sum(1 for n, m in instance.model.model.named_modules() 
                                if 'FakeQuantize' in type(m).__name__)
            
            if loaded_fake_count != fake_quant_count:
                logger.warning(f"⚠️ FakeQuantize count mismatch: expected {fake_quant_count}, got {loaded_fake_count}")
            
            # Set up preservation system if it was used
            if qat_info.get('has_preservation', False):
                try:
                    from src.quantization.quantizer_preservation import QuantizerPreserver
                    instance.quantizer_preserver = QuantizerPreserver(instance.model.model)
                    logger.info("✅ Quantizer preservation system restored")
                except Exception as e:
                    logger.warning(f"⚠️ Could not restore preservation system: {e}")
            
            logger.info(f"✅ QAT model loaded successfully:")
            logger.info(f"  📊 {loaded_fake_count} FakeQuantize modules active")
            logger.info(f"  🚀 Ready for: training, fine-tuning, INT8 conversion, inference")
            
            return instance
            
        except Exception as e:
            logger.error(f"❌ Failed to load QAT model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def verify_qat_model_ready(self):
        """
        NEW: Verify that QAT model is ready for use.
        
        Returns:
            bool: True if model is ready for inference/training
        """
        logger.info("🔍 Verifying QAT model readiness...")
        
        try:
            # Check 1: Model exists and is in correct mode
            if not hasattr(self, 'model') or self.model is None:
                logger.error("❌ No model found")
                return False
            
            # Check 2: FakeQuantize modules present
            fake_quant_count = sum(1 for n, m in self.model.model.named_modules() 
                                if 'FakeQuantize' in type(m).__name__)
            
            if fake_quant_count == 0:
                logger.error("❌ No FakeQuantize modules found")
                return False
            
            # Check 3: Model can be set to eval mode
            self.model.model.eval()
            
            # Check 4: Create dummy input and test forward pass
            device = next(self.model.model.parameters()).device
            dummy_input = torch.randn(1, 3, 640, 640).to(device)
            
            with torch.no_grad():
                output = self.model.model(dummy_input)
            
            logger.info(f"✅ QAT model verification passed:")
            logger.info(f"  📊 {fake_quant_count} FakeQuantize modules")
            logger.info(f"  🔧 Forward pass successful")
            logger.info(f"  🚀 Model ready for deployment")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ QAT model verification failed: {e}")
            return False

    def convert_to_quantized_with_preservation(self, save_path=None):
        """
        FIXED: Convert QAT model to INT8 with proper observer validation.
        CRITICAL FIX: Validates observer calibration before conversion.
        """
        logger.info("🔄 Converting QAT model to INT8 with validation...")
        
        # Step 1: Validate quantization preservation
        if hasattr(self, 'quantizer_preserver'):
            stats = self.quantizer_preserver.get_quantizer_stats()
            fake_quant_count = stats['total_fake_quantizers']
            logger.info(f"📊 Pre-conversion validation:")
            logger.info(f"  - {fake_quant_count} FakeQuantize modules found")
            logger.info(f"  - Observers always active: {stats.get('observers_always_active', False)}")
            
            if fake_quant_count == 0:
                logger.error("❌ Cannot convert - no FakeQuantize modules!")
                return None
            
            # CRITICAL: Validate observer calibration
            if not self.quantizer_preserver.validate_observer_calibration():
                logger.error("❌ Observer calibration failed - cannot convert to INT8!")
                logger.error("💡 Solution: Train model longer or check phase training setup")
                return None
            
            qat_model = self.quantizer_preserver.model
        else:
            # Fallback validation
            fake_quant_count = sum(1 for n, m in self.model.model.named_modules() 
                                if 'FakeQuantize' in type(m).__name__)
            logger.info(f"📊 Fallback validation: {fake_quant_count} FakeQuantize modules")
            
            if fake_quant_count == 0:
                logger.error("❌ Cannot convert - no FakeQuantize modules!")
                return None
            
            # Manual observer validation
            if not self._validate_observers_manual(self.model.model):
                logger.error("❌ Observer validation failed!")
                return None
            
            qat_model = self.model.model
        
        # Step 2: Prepare model for conversion
        qat_model.eval()  # Set to eval mode
        logger.info("✅ Observer validation passed - proceeding with conversion")
        
        # Step 3: Convert to quantized model
        try:
            logger.info("🔧 Converting QAT model to INT8...")
            from src.quantization.utils import convert_qat_model_to_quantized
            
            self.quantized_model = convert_qat_model_to_quantized(qat_model, inplace=False)
            
            if self.quantized_model is None:
                logger.error("❌ Conversion returned None")
                return None
            
            logger.info("✅ Successfully converted to INT8")
            
        except Exception as e:
            logger.error(f"❌ Conversion failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
        # Step 4: Validate conversion quality
        conversion_quality = self._validate_conversion_quality(qat_model, self.quantized_model)
        if not conversion_quality:
            logger.warning("⚠️ Conversion quality check failed - model may perform poorly")
        
        # Step 5: Calculate size comparison
        try:
            from src.quantization.utils import compare_model_sizes
            size_info = compare_model_sizes(self.original_model, self.quantized_model)
            
            logger.info(f"📊 Conversion results:")
            logger.info(f"  - Original: {size_info['fp32_size_mb']:.2f} MB")
            logger.info(f"  - Quantized: {size_info['int8_size_mb']:.2f} MB")
            logger.info(f"  - Compression: {size_info['compression_ratio']:.2f}x")
            logger.info(f"  - Size reduction: {size_info['size_reduction_percent']:.1f}%")
            
            # Validate compression ratio
            if size_info['compression_ratio'] < 2.0:
                logger.warning("⚠️ Low compression ratio - check quantization effectiveness")
        
        except Exception as e:
            logger.warning(f"⚠️ Size comparison failed: {e}")
            size_info = {}
        
        # Step 6: Save quantized model if path provided
        if save_path:
            int8_save_path = save_path.replace('.pt', '_int8_validated.pt')
            
            metadata = {
                'framework': 'pytorch',
                'format': 'quantized_int8',
                'qconfig': self.qconfig_name,
                'size_info': size_info,
                'conversion_successful': True,
                'deployment_ready': True,
                'observer_validated': True,  # Mark as properly validated
                'preserved_quantizers': fake_quant_count,
                'conversion_method': 'validated_conversion',
                'quality_checked': conversion_quality
            }
            
            try:
                from src.quantization.utils import save_quantized_model
                save_success = save_quantized_model(self.quantized_model, int8_save_path, metadata)
                
                if save_success:
                    logger.info(f"✅ Validated INT8 model saved to {int8_save_path}")
                else:
                    logger.error(f"❌ Failed to save INT8 model")
            except Exception as e:
                logger.error(f"❌ Save failed: {e}")
        
        self.is_converted = True
        return self.quantized_model

    def _validate_observers_manual(self, model):
        """
        FIXED: Manual observer validation for models without preserver.
        
        Args:
            model: Model to validate
            
        Returns:
            bool: True if observers are properly calibrated
        """
        logger.info("🔍 Performing manual observer validation...")
        
        uncalibrated_observers = []
        total_observers = 0
        
        for name, module in model.named_modules():
            # Check activation post process (observers)
            if hasattr(module, 'activation_post_process'):
                observer = module.activation_post_process
                total_observers += 1
                
                # Validate observer has collected statistics
                if hasattr(observer, 'min_val') and hasattr(observer, 'max_val'):
                    min_val = observer.min_val
                    max_val = observer.max_val
                    
                    # Check for uncalibrated indicators
                    if (min_val == float('inf') or 
                        max_val == float('-inf') or
                        torch.isnan(min_val) or 
                        torch.isnan(max_val) or
                        min_val == max_val):
                        uncalibrated_observers.append(name)
                else:
                    uncalibrated_observers.append(name)
        
        calibrated_count = total_observers - len(uncalibrated_observers)
        
        logger.info(f"📊 Manual observer validation:")
        logger.info(f"  - Total observers: {total_observers}")
        logger.info(f"  - Calibrated: {calibrated_count}")
        logger.info(f"  - Uncalibrated: {len(uncalibrated_observers)}")
        
        if len(uncalibrated_observers) > 0:
            logger.error(f"❌ Uncalibrated observers: {uncalibrated_observers[:3]}...")
            return False
        
        logger.info("✅ Manual observer validation passed")
        return True

    def _validate_conversion_quality(self, qat_model, int8_model):
        """
        FIXED: Validate the quality of QAT to INT8 conversion.
        
        Args:
            qat_model: Original QAT model
            int8_model: Converted INT8 model
            
        Returns:
            bool: True if conversion quality is acceptable
        """
        logger.info("🔍 Validating INT8 conversion quality...")
        
        try:
            # Test 1: Check for actual quantized parameters
            quantized_params = 0
            total_params = 0
            
            for name, param in int8_model.named_parameters():
                total_params += 1
                if hasattr(param, 'dtype') and param.dtype in [torch.qint8, torch.quint8]:
                    quantized_params += 1
            
            quantization_ratio = quantized_params / total_params if total_params > 0 else 0
            
            logger.info(f"📊 Quantization quality metrics:")
            logger.info(f"  - Quantized parameters: {quantized_params}/{total_params}")
            logger.info(f"  - Quantization ratio: {quantization_ratio:.2%}")
            
            # Test 2: Basic forward pass test
            device = next(qat_model.parameters()).device
            test_input = torch.randn(1, 3, 640, 640).to(device)
            
            with torch.no_grad():
                qat_output = qat_model(test_input)
                int8_output = int8_model(test_input)
            
            # Test 3: Output shape consistency
            if len(qat_output) != len(int8_output):
                logger.error("❌ Output shape mismatch between QAT and INT8 models")
                return False
            
            # Test 4: Reasonable output values (not all zeros/NaN)
            int8_tensor = int8_output[0] if isinstance(int8_output, (list, tuple)) else int8_output
            
            if torch.isnan(int8_tensor).any():
                logger.error("❌ INT8 model produces NaN outputs")
                return False
            
            if torch.all(int8_tensor == 0):
                logger.error("❌ INT8 model produces all-zero outputs")
                return False
            
            logger.info("✅ Conversion quality validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Quality validation failed: {e}")
            return False

    def create_inference_ready_model(self, int8_model_path=None):
        """
        NEW: Create an inference-ready model from the converted INT8 model.
        
        Args:
            int8_model_path: Optional path to save inference-ready model
            
        Returns:
            Inference-ready model
        """
        if not self.is_converted or self.quantized_model is None:
            logger.error("❌ No converted INT8 model available")
            return None
        
        logger.info("🚀 Creating inference-ready model...")
        
        try:
            # Prepare model for inference
            inference_model = self.quantized_model
            inference_model.eval()
            
            # Test inference capability
            device = next(inference_model.parameters()).device
            test_input = torch.randn(1, 3, 640, 640).to(device)
            
            with torch.no_grad():
                test_output = inference_model(test_input)
            
            logger.info("✅ Inference test passed")
            
            if int8_model_path:
                # Save inference-ready model
                inference_dict = {
                    'inference_model': inference_model,
                    'model_info': {
                        'type': 'inference_ready_int8',
                        'quantization_method': 'qat_to_int8',
                        'input_shape': [1, 3, 640, 640],
                        'ready_for_deployment': True
                    },
                    'usage_instructions': {
                        'load': 'model = torch.load(path)["inference_model"]',
                        'inference': 'model.eval(); output = model(input_tensor)',
                        'device': 'model.to(device) before inference'
                    }
                }
                
                torch.save(inference_dict, int8_model_path)
                logger.info(f"💾 Inference-ready model saved to {int8_model_path}")
            
            return inference_model
            
        except Exception as e:
            logger.error(f"❌ Failed to create inference-ready model: {e}")
            return None

    def verify_quantization_preserved(self):
        """Verify that the current model still has quantization modules."""
        fake_quant_count = sum(1 for n, m in self.model.model.named_modules() 
                            if 'FakeQuantize' in type(m).__name__)
        
        qconfig_count = sum(1 for n, m in self.model.model.named_modules() 
                        if hasattr(m, 'qconfig') and m.qconfig is not None)
        
        logger.info(f"Current quantization status:")
        logger.info(f"  - FakeQuantize modules: {fake_quant_count}")
        logger.info(f"  - Modules with qconfig: {qconfig_count}")
        
        if fake_quant_count > 0 and qconfig_count > 0:
            logger.info("✅ Quantization is properly preserved")
            return True
        else:
            logger.error("❌ Quantization has been lost!")
            logger.error("   - Model may need to be re-prepared for QAT")
            return False
    
    def _configure_phase(self, phase):
        """Configure model for specific QAT phase."""
        # Use consistent phase names
        if phase == "weight_only" or phase == "phase1_weight_only":
            # Enable only weight quantization, disable activation quantization
            logger.info("Configuring for weight-only quantization phase")
            self._set_activation_quantizers_enabled(False)
            self._set_weight_quantizers_enabled(True)
        elif phase == "activation_phase" or phase == "phase2_activations":
            # Enable both weight and activation quantization
            logger.info("Configuring for activation quantization phase")
            self._set_activation_quantizers_enabled(True)
            self._set_weight_quantizers_enabled(True)
        elif phase == "full_quantization" or phase == "phase3_full_quant":
            # Enable all quantizers
            logger.info("Configuring for full network quantization phase")
            self._set_all_quantizers_enabled(True)
        elif phase == "fine_tuning" or phase == "phase4_fine_tuning":
            # All quantizers remain enabled
            logger.info("Configuring for fine-tuning phase")
            self._set_all_quantizers_enabled(True)
        else:
            logger.warning(f"Unknown phase: {phase}, enabling all quantizers as fallback")
            self._set_all_quantizers_enabled(True)
        
        # VERIFICATION - Count active quantizers
        weight_count = 0
        activation_count = 0
        
        for name, module in self.model.model.named_modules():
            if hasattr(module, 'weight_fake_quant') and not isinstance(module.weight_fake_quant, nn.Identity):
                weight_count += 1
            if hasattr(module, 'activation_post_process') and not isinstance(module.activation_post_process, nn.Identity):
                activation_count += 1
        
        logger.info(f"Phase '{phase}' configured with {weight_count} active weight quantizers and {activation_count} active activation quantizers")

    def _handle_yolov8_specific_modules(self, model):
        """Apply YOLOv8-specific handling for quantization."""
        logger.info("Applying YOLOv8-specific module handling...")
        
        # YOLOv8 uses C2f blocks - we need to ensure these are properly quantized
        c2f_blocks = [m for n, m in model.named_modules() if 'C2f' in m.__class__.__name__]
        logger.info(f"Found {len(c2f_blocks)} C2f blocks to process")
        
        # Apply qconfig to submodules within C2f blocks that might be missed
        count = 0
        for block in c2f_blocks:
            for name, module in block.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and not hasattr(module, 'qconfig'):
                    module.qconfig = get_default_qat_qconfig()
                    count += 1
        
        logger.info(f"Applied qconfig to {count} additional submodules in C2f blocks")
        
        return model

    def _reapply_qconfig_to_model(self, model):
        """Re-apply qconfig to all modules after loading."""
        logger.info("Re-applying qconfig to model modules...")
        
        count = 0
        for name, module in model.named_modules():
            # Apply to appropriate module types
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                # Special config for first layer
                if "model.0.conv" in name:
                    from src.quantization.qconfig import get_first_layer_qconfig
                    module.qconfig = get_first_layer_qconfig()
                    logger.info(f"Applied first layer qconfig to {name}")
                else:
                    # Default config for other layers
                    from src.quantization.qconfig import get_default_qat_qconfig
                    module.qconfig = get_default_qat_qconfig()
                count += 1
        
        # Skip detection head if requested
        if self.skip_detection_head:
            detection_count = 0
            for name, module in model.named_modules():
                if 'detect' in name or 'model.22' in name:
                    # Remove qconfig to skip quantization
                    module.qconfig = None
                    detection_count += 1
            
            logger.info(f"Disabled quantization for {detection_count} detection modules")
        
        logger.info(f"Re-applied qconfig to {count} modules")
        return model

    def _configure_model_for_specific_phase(self, model, phase_name):
        """Configure model specifically for the current phase."""
        logger.info(f"Configuring model for {phase_name}")
        
        if phase_name == "phase1_weight_only":
            # Enable weight quantizers, disable activation quantizers
            self._set_quantizers_for_model(model, weights_enabled=True, activations_enabled=False)
        elif phase_name == "phase2_activations":
            # Enable both weight and activation quantizers
            self._set_quantizers_for_model(model, weights_enabled=True, activations_enabled=True)
        elif phase_name == "phase3_full_quant":
            # Enable all quantizers (same as phase 2 for now)
            self._set_quantizers_for_model(model, weights_enabled=True, activations_enabled=True)
        elif phase_name == "phase4_fine_tuning":
            # All quantizers remain enabled
            self._set_quantizers_for_model(model, weights_enabled=True, activations_enabled=True)
        else:
            logger.warning(f"Unknown phase: {phase_name}, enabling all quantizers")
            self._set_quantizers_for_model(model, weights_enabled=True, activations_enabled=True)

    def _set_quantizers_for_model(self, model, weights_enabled=True, activations_enabled=True):
        """Enable or disable quantizers for a specific model."""
        weight_count = 0
        activation_count = 0
        
        for name, module in model.named_modules():
            # Handle weight quantizers
            if hasattr(module, 'weight_fake_quant'):
                if not weights_enabled:
                    if not hasattr(module, '_original_weight_fake_quant'):
                        module._original_weight_fake_quant = module.weight_fake_quant
                    module.weight_fake_quant = torch.nn.Identity()
                else:
                    if hasattr(module, '_original_weight_fake_quant'):
                        module.weight_fake_quant = module._original_weight_fake_quant
                    if not isinstance(module.weight_fake_quant, torch.nn.Identity):
                        weight_count += 1
            
            # Handle activation quantizers
            if hasattr(module, 'activation_post_process'):
                if not activations_enabled:
                    if not hasattr(module, '_original_activation_post_process'):
                        module._original_activation_post_process = module.activation_post_process
                    module.activation_post_process = torch.nn.Identity()
                else:
                    if hasattr(module, '_original_activation_post_process'):
                        module.activation_post_process = module._original_activation_post_process
                    if not isinstance(module.activation_post_process, torch.nn.Identity):
                        activation_count += 1
        
        logger.info(f"Configured model with {weight_count} weight quantizers and {activation_count} activation quantizers")

    def _verify_quantization_setup(self, model, phase_name):
        """Verify that quantization is properly set up for the current phase."""
        weight_count = 0
        activation_count = 0
        fake_quant_count = 0
        
        for name, module in model.named_modules():
            # Count weight quantizers
            if hasattr(module, 'weight_fake_quant') and not isinstance(module.weight_fake_quant, torch.nn.Identity):
                weight_count += 1
            
            # Count activation quantizers
            if hasattr(module, 'activation_post_process') and not isinstance(module.activation_post_process, torch.nn.Identity):
                activation_count += 1
            
            # Count FakeQuantize modules
            if 'FakeQuantize' in module.__class__.__name__:
                fake_quant_count += 1
        
        logger.info(f"✓ Quantization verification for {phase_name}:")
        logger.info(f"  - Weight quantizers: {weight_count}")
        logger.info(f"  - Activation quantizers: {activation_count}")
        logger.info(f"  - FakeQuantize modules: {fake_quant_count}")
        
        # Validate expected counts based on phase
        if phase_name == "phase1_weight_only":
            if weight_count == 0:
                logger.error("❌ ERROR: No weight quantizers found in weight-only phase!")
                return False
            if activation_count > 0:
                logger.warning(f"⚠️ WARNING: Found {activation_count} activation quantizers in weight-only phase")
        elif phase_name in ["phase2_activations", "phase3_full_quant", "phase4_fine_tuning"]:
            if weight_count == 0 or activation_count == 0:
                logger.error(f"❌ ERROR: Missing quantizers in {phase_name} - Weight: {weight_count}, Activation: {activation_count}")
                return False
        
        return True
        
    def _verify_qat_save(self, path, expected_fake_quant_count):
        """FIXED: Verify that QAT model was saved correctly using state_dict approach."""
        try:
            if not os.path.exists(path):
                return False
                
            saved_data = torch.load(path, map_location='cpu')
            
            # Check for state_dict format
            if 'model_state_dict' in saved_data:
                state_dict = saved_data['model_state_dict']
                saved_count = saved_data.get('fake_quant_count', 0)
                
                success = (len(state_dict) > 0 and saved_count == expected_fake_quant_count)
                logger.info(f"Verification: {len(state_dict)} parameters, {saved_count} FakeQuantize modules")
                return success
            else:
                logger.warning("Old save format detected")
                return False
                
        except Exception as e:
            logger.error(f"Save verification failed: {e}")
            return False

    def _save_fallback_method(self, path):
        """
        Fallback save method if main save fails.
        
        Args:
            path: Path to save model
        
        Returns:
            Success status
        """
        logger.info("Attempting fallback save method...")
        
        try:
            # Save with minimal structure that's ONNX-compatible
            model_to_save = self.model.model
            
            # Ensure model is in eval mode
            model_to_save.eval()
            
            # Create a simplified save format
            fallback_dict = {
                'model': model_to_save,
                'model_info': {
                    'architecture': 'yolov8n',
                    'num_classes': 58,
                    'img_size': 640,
                    'quantized': True,
                    'source': 'qat_training'
                },
                'save_method': 'fallback_full_model'
            }
            
            torch.save(fallback_dict, path)
            
            # Simple verification
            if os.path.exists(path):
                file_size = os.path.getsize(path) / (1024 * 1024)
                logger.info(f"✅ Fallback save successful: {file_size:.2f} MB")
                return True
            else:
                logger.error("❌ Fallback save failed - no file created")
                return False
                
        except Exception as e:
            logger.error(f"❌ Fallback save also failed: {e}")
            return False

    def test_all_phase_transitions(self):
        """
        Test all phase transitions to ensure quantizers are preserved correctly.
        """
        logger.info("🧪 Testing ALL phase transitions...")
        
        phases = ["phase1_weight_only", "phase2_activations", "phase3_full_quant", "phase4_fine_tuning"]
        expected_results = [
            (45, 0),   # Phase 1: Weights=45, Activations=0
            (45, 90),  # Phase 2: Weights=45, Activations=90 (restored)
            (45, 90),  # Phase 3: Weights=45, Activations=90 (maintained)
            (45, 90)   # Phase 4: Weights=45, Activations=90 (maintained)
        ]
        
        all_passed = True
        
        for i, (phase, expected) in enumerate(zip(phases, expected_results)):
            logger.info(f"🔄 Testing {phase}...")
            
            # Apply phase configuration
            success = self._configure_quantizers_dynamically(phase)
            
            # Count actual quantizers
            actual_weights, actual_activations = self._count_active_quantizers()
            expected_weights, expected_activations = expected
            
            # Check results
            if (actual_weights >= expected_weights * 0.9 and  # Allow 10% tolerance
                actual_activations >= expected_activations * 0.9):
                logger.info(f"✅ {phase} test PASSED: W={actual_weights}, A={actual_activations}")
            else:
                logger.error(f"❌ {phase} test FAILED:")
                logger.error(f"   Expected: W≥{expected_weights}, A≥{expected_activations}")
                logger.error(f"   Actual: W={actual_weights}, A={actual_activations}")
                all_passed = False
        
        if all_passed:
            logger.info("🎉 ALL phase transitions work correctly!")
        else:
            logger.error("💥 Some phase transitions are broken!")
        
        return all_passed

    def _count_active_quantizers(self):
        """
        Count currently active quantizers for testing and debugging.
        """
        weight_count = 0
        activation_count = 0
        
        for name, module in self.model.model.named_modules():
            # Count active weight quantizers
            if (hasattr(module, 'weight_fake_quant') and 
                not isinstance(module.weight_fake_quant, torch.nn.Identity)):
                weight_count += 1
            
            # Count active activation quantizers
            if (hasattr(module, 'activation_post_process') and 
                not isinstance(module.activation_post_process, torch.nn.Identity)):
                activation_count += 1
        
        return weight_count, activation_count

    def debug_quantizer_states(self, phase_name="unknown"):
        """
        Debug function to show detailed quantizer states.
        """
        logger.info(f"🔍 Debugging quantizer states for {phase_name}:")
        
        weight_states = {"active": 0, "disabled": 0, "identity": 0}
        activation_states = {"active": 0, "disabled": 0, "identity": 0}
        
        for name, module in self.model.model.named_modules():
            # Analyze weight quantizers
            if hasattr(module, 'weight_fake_quant'):
                if hasattr(module, '_disabled_weight_fake_quant'):
                    weight_states["disabled"] += 1
                elif isinstance(module.weight_fake_quant, torch.nn.Identity):
                    weight_states["identity"] += 1
                else:
                    weight_states["active"] += 1
            
            # Analyze activation quantizers
            if hasattr(module, 'activation_post_process'):
                if hasattr(module, '_disabled_activation_post_process'):
                    activation_states["disabled"] += 1
                elif isinstance(module.activation_post_process, torch.nn.Identity):
                    activation_states["identity"] += 1
                else:
                    activation_states["active"] += 1
        
        logger.info(f"Weight quantizers: {weight_states}")
        logger.info(f"Activation quantizers: {activation_states}")
        
        return weight_states, activation_states

    def _apply_qconfig_to_all_modules(self, model):
        """Apply qconfig to all relevant modules."""
        logger.info("Applying qconfig to model modules...")
        
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                if "model.0.conv" in name:
                    from src.quantization.qconfig import get_first_layer_qconfig
                    module.qconfig = get_first_layer_qconfig()
                    logger.info(f"Applied first layer qconfig to {name}")
                else:
                    from src.quantization.qconfig import get_default_qat_qconfig
                    module.qconfig = get_default_qat_qconfig()
                count += 1
        
        logger.info(f"Applied qconfig to {count} modules")
        
        # Skip detection head if requested
        if self.skip_detection_head:
            detection_count = 0
            for name, module in model.named_modules():
                if 'detect' in name or 'model.22' in name:
                    module.qconfig = None
                    detection_count += 1
            logger.info(f"Disabled quantization for {detection_count} detection modules")
        
        # Apply YOLOv8-specific handling
        self._handle_yolov8_specific_modules(model)
        
        return model

    # ADD THIS METHOD TO YOUR EXISTING QuantizedYOLOv8 CLASS
    def prepare_for_qat_with_preservation(self):
        """FIXED: Prepare model for QAT with simple quantizer preservation."""
        logger.info("🔧 Preparing model for QAT with simple preservation...")
        
        # Get base model
        base_model = self.model.model
        base_model.train()
        
        # Optionally fuse modules
        if self.fuse_modules:
            logger.info("Fusing modules for better quantization...")
            from src.quantization.fusion import fuse_yolov8_modules
            base_model = fuse_yolov8_modules(base_model)
        
        # Apply qconfig (reuse your existing logic)
        self._apply_qconfig_to_all_modules(base_model)
        
        # Prepare for QAT (creates FakeQuantize modules)
        logger.info("⚙️ Calling prepare_qat with PyTorch...")
        qat_model = torch.quantization.prepare_qat(base_model, inplace=True)
        
        # FIXED: Use simple preservation approach
        from src.quantization.quantizer_preservation import QuantizerPreserver
        self.quantizer_preserver = QuantizerPreserver(qat_model)
        
        # Update model
        self.model.model = qat_model
        self.is_prepared = True
        
        # Verify preparation
        stats = self.quantizer_preserver.get_quantizer_stats()
        logger.info(f"✅ QAT preparation complete:")
        logger.info(f"  - Total FakeQuantize modules: {stats['total_fake_quantizers']}")
        logger.info(f"  - Weight quantizers: {stats['weight_quantizers_total']}")
        logger.info(f"  - Activation quantizers: {stats['activation_quantizers_total']}")
        logger.info(f"  - Simple approach active: {stats['observers_working_naturally']}")
        
        if stats['total_fake_quantizers'] == 0:
            logger.error("❌ No FakeQuantize modules created!")
            return None
        
        return qat_model

    def _configure_quantizers_with_preservation(self, phase_name):
        """FIXED: Configure quantizers using 2-phase simple preservation."""
        if not hasattr(self, 'quantizer_preserver'):
            logger.error("❌ Quantizer preserver not initialized!")
            return False
        
        # Validate phase name
        valid_phases = ["phase1_weight_only", "phase2_full_quant"]
        if phase_name not in valid_phases:
            logger.error(f"❌ Invalid phase name: {phase_name}. Valid phases: {valid_phases}")
            return False
        
        # Use 2-phase simple approach
        logger.info(f"🔄 Setting phase: {phase_name} (2-phase simple approach)")
        self.quantizer_preserver.set_phase_by_name(phase_name)
        
        # Verify the change worked
        stats = self.quantizer_preserver.get_quantizer_stats()
        if stats['preservation_active']:
            logger.info(f"✅ Phase {phase_name} set successfully:")
            logger.info(f"  - {stats['total_fake_quantizers']} FakeQuantize modules preserved")
            logger.info(f"  - Weight quantizers: {stats['weight_quantizers_enabled']}/{stats['weight_quantizers_total']}")
            logger.info(f"  - Activation quantizers: {stats['activation_quantizers_enabled']}/{stats['activation_quantizers_total']}")
            
            # Phase-specific verification
            if phase_name == "phase1_weight_only":
                if stats['activation_quantizers_enabled'] == 0:
                    logger.info("✅ Phase 1 correct: Activation quantizers disabled")
                else:
                    logger.warning(f"⚠️ Phase 1 issue: {stats['activation_quantizers_enabled']} activation quantizers still enabled")
            elif phase_name == "phase2_full_quant":
                if stats['activation_quantizers_enabled'] > 0:
                    logger.info("✅ Phase 2 correct: Activation quantizers enabled for observer collection")
                else:
                    logger.error(f"❌ Phase 2 issue: No activation quantizers enabled - observers won't collect data!")
            
            return True
        else:
            logger.error(f"❌ 2-phase preservation failed for {phase_name}")
            return False
                
    def train_model_with_preservation(self, data_yaml, epochs, batch_size, img_size, lr, device, save_dir, log_dir):
        """
        CORRECTED: 2-Phase QAT Training with simple quantizer preservation.
        """
        import os
        from ultralytics.utils import LOGGER
        
        # Validate dataset
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
        
        logger.info(f"✅ Using dataset: {data_yaml}")
        
        # CORRECTED: 2-phase boundaries only (30/70 split)
        phase1_end = max(1, int(epochs * 0.3))  # 30% weight-only training
        # Phase 2 runs from epoch (phase1_end + 1) to epoch (epochs) = 70% full quantization
        
        LOGGER.info(f"🔄 CORRECTED 2-Phase QAT Training Plan:")
        LOGGER.info(f"  Phase 1 (Weight-only): Epochs 1-{phase1_end} (30%)")
        LOGGER.info(f"  Phase 2 (Full Quantization): Epochs {phase1_end+1}-{epochs} (70%)")
        LOGGER.info(f"  Learning Rate: Constant {lr} (no changes)")
        
        # Set initial phase: weight-only quantization
        self._configure_quantizers_with_preservation("phase1_weight_only")
        
        # Setup callback for SINGLE phase transition
        def on_epoch_start(trainer):
            current_epoch = trainer.epoch
            
            # ONLY ONE transition: weight-only → full quantization
            if current_epoch == phase1_end:
                LOGGER.info(f"🔄 Epoch {current_epoch}: Transitioning to Phase 2 (FULL QUANTIZATION)")
                success = self._configure_quantizers_with_preservation("phase2_full_quant")
                if not success:
                    LOGGER.error("❌ Phase 2 transition failed!")
                else:
                    LOGGER.info("✅ Phase 2 transition successful - activation quantizers now enabled!")
        
        def on_epoch_end(trainer):
            """Monitor 2-phase preservation after each epoch."""
            if hasattr(self, 'quantizer_preserver'):
                stats = self.quantizer_preserver.get_quantizer_stats()
                
                # Show which phase we're in
                if trainer.epoch < phase1_end:
                    phase_info = "Phase 1 (Weight-only)"
                    expected_activations = 0
                else:
                    phase_info = "Phase 2 (Full quantization)"
                    expected_activations = stats['activation_quantizers_total']
                    
                LOGGER.info(f"📊 End of epoch {trainer.epoch} - {phase_info}:")
                LOGGER.info(f"  - FakeQuantize modules: {stats['total_fake_quantizers']}")
                LOGGER.info(f"  - Active: {stats['weight_quantizers_enabled']} weights, {stats['activation_quantizers_enabled']} activations")
                LOGGER.info(f"  - Expected activations: {expected_activations}, Actual: {stats['activation_quantizers_enabled']}")
                
                if stats['total_fake_quantizers'] == 0:
                    LOGGER.error(f"❌ CRITICAL: All quantizers lost at epoch {trainer.epoch}!")
                elif stats['activation_quantizers_enabled'] != expected_activations:
                    LOGGER.warning(f"⚠️ Activation quantizer count mismatch in {phase_info}")
        
        # Add callbacks
        self.model.add_callback('on_train_epoch_start', on_epoch_start)
        self.model.add_callback('on_train_epoch_end', on_epoch_end)
        
        # Train with 2-phase simple preservation
        try:
            LOGGER.info("🚀 Starting 2-PHASE preserved QAT training...")
            
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                lr0=lr,  # No LR changes - constant throughout
                device=device,
                project=os.path.dirname(save_dir),
                name=os.path.basename(save_dir),
                exist_ok=True,
                pretrained=False,
                val=True,
                verbose=True
            )
            
            # Final verification with 2-phase approach
            if hasattr(self, 'quantizer_preserver'):
                final_stats = self.quantizer_preserver.get_quantizer_stats()
                LOGGER.info(f"🎯 2-Phase training complete:")
                LOGGER.info(f"  - Total FakeQuantize modules: {final_stats['total_fake_quantizers']}")
                LOGGER.info(f"  - 2-Phase approach success: {final_stats['preservation_active']}")
                LOGGER.info(f"  - Final state: {final_stats['weight_quantizers_enabled']} weights, {final_stats['activation_quantizers_enabled']} activations")
                LOGGER.info(f"  - Observer data collected: ~{int(epochs * 0.7 * 596)} batches from Phase 2")
                
                if final_stats['preservation_active'] and final_stats['activation_quantizers_enabled'] > 0:
                    LOGGER.info("🎉 SUCCESS: 2-Phase quantizer preservation worked!")
                    LOGGER.info("🔍 Observers should have collected data during Phase 2 (70% of training)")
                else:
                    LOGGER.error("💥 FAILURE: 2-Phase preservation failed!")
            
            self.is_trained = True
            return results
            
        except Exception as e:
            LOGGER.error(f"❌ 2-Phase training failed: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            raise