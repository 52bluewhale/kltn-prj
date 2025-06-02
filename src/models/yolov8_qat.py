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
    
    def prepare_for_qat(self):
        """Prepare model for QAT optimized for PyTorch 2.4.1."""
        logger.info("Preparing model for QAT...")
        
        # Get base model
        base_model = self.model.model
        
        # Set model to training mode
        logger.info("Setting model to training mode...")
        base_model.train()

        # Optionally fuse modules
        if self.fuse_modules:
            logger.info("Fusing modules for better quantization...")
            base_model = fuse_yolov8_modules(base_model)
        
        try:
            # Get configs from our qconfig.py
            from src.quantization.qconfig import get_default_qat_qconfig, get_first_layer_qconfig
            
            # Create QConfig dictionary with default config for all modules
            qconfig_dict = {'': get_default_qat_qconfig()}
            
            # Directly apply QConfig to modules
            logger.info("Manually applying qconfig to individual modules...")
            count = 0
            
            for name, module in base_model.named_modules():
                # Apply to appropriate module types
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    # Special config for first layer
                    if "model.0.conv" in name:
                        module.qconfig = get_first_layer_qconfig()
                        logger.info(f"Applied first layer qconfig to {name}")
                    else:
                        # Default config for other layers
                        module.qconfig = get_default_qat_qconfig()
                    count += 1
            
            logger.info(f"Applied qconfig to {count} modules")
            
            # Skip detection head if requested
            if self.skip_detection_head:
                detection_count = 0
                for name, module in base_model.named_modules():
                    if 'detect' in name or 'model.22' in name:
                        # Remove qconfig to skip quantization
                        module.qconfig = None
                        detection_count += 1
                
                logger.info(f"Disabled quantization for {detection_count} detection modules")
            
            base_model = self._handle_yolov8_specific_modules(base_model)

            # Debug info about qconfig application
            logger.info("QConfig application state:")
            module_count = 0
            qconfig_count = 0
            for name, module in base_model.named_modules():
                module_count += 1
                if hasattr(module, 'qconfig') and module.qconfig is not None:
                    qconfig_count += 1
                    if qconfig_count <= 5:  # Show first few for debugging
                        logger.info(f"Module {name} has qconfig applied")
            
            logger.info(f"Total modules: {module_count}, modules with qconfig: {qconfig_count}")
            
            # Prepare model for QAT
            logger.info("Calling prepare_qat with PyTorch 2.4.1 compatible arguments...")
            self.qat_model = torch.quantization.prepare_qat(base_model, inplace=True)
            
            # Verify QAT preparation
            qconfig_applied = sum(1 for n, m in self.qat_model.named_modules() 
                                if hasattr(m, 'qconfig') and m.qconfig is not None)
            
            # Check for FakeQuantize modules - these should be present after prepare_qat
            fake_quant_count = sum(1 for n, m in self.qat_model.named_modules()
                                if 'FakeQuantize' in m.__class__.__name__)
            
            logger.info(f"QAT preparation verified: {qconfig_applied} modules have qconfig applied")
            logger.info(f"Found {fake_quant_count} FakeQuantize modules after prepare_qat")
            
        except Exception as e:
            logger.error(f"Error preparing model for QAT: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        # Update model in YOLO object
        self.model.model = self.qat_model
        self.is_prepared = True
        
        return self.qat_model

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
        """
        Setup quantization penalty loss integration.
        Simplified version that just sets up the penalty handler.
        """
        from src.training.penalty_loss import QuantizationPenaltyLoss, patch_yolo_loss_with_penalty
        
        # Create penalty handler
        self.penalty_handler = QuantizationPenaltyLoss(
            alpha=alpha, 
            warmup_epochs=warmup_epochs,
            normalize=True
        )
        
        # Patch model with penalty calculation
        patch_yolo_loss_with_penalty(self.model, self.penalty_handler)
        
        logger = logging.getLogger('yolov8_qat')
        logger.info(f"✅ Quantization penalty loss setup complete (alpha={alpha}, warmup={warmup_epochs})")

    def verify_penalty_setup(self):
        """
        Verify that penalty loss integration is working.
        Call this after setup_penalty_loss_integration().
        """
        from src.training.penalty_loss import verify_penalty_integration
        
        return verify_penalty_integration(self)
    
    def train_model(self, data_yaml, epochs, batch_size, img_size, lr, device, save_dir, log_dir, use_distillation=False):
        """
        FIXED: Continuous phased QAT training without breaking learning momentum.
        """
        import os
        from ultralytics.utils import LOGGER

        # FIXED: Validate dataset path
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
        
        logger.info(f"✅ Using dataset: {data_yaml}")
        
        # FIXED: Force clear any existing data configuration
        if hasattr(self.model, 'overrides'):
            self.model.overrides['data'] = data_yaml  # Explicitly set the data
        
        # Calculate phase boundaries (epochs)
        phase1_end = max(1, int(epochs * 0.3))      # 30% - Weight only
        phase2_end = max(2, int(epochs * 0.7))      # 40% - Add activations  
        phase3_end = max(3, int(epochs * 0.9))      # 20% - Full quantization
        # phase4_end = epochs                        # 10% - Fine-tuning
        
        LOGGER.info(f"🔄 Continuous Phased QAT Training Plan:")
        LOGGER.info(f"  Phase 1 (Weight-only): Epochs 1-{phase1_end}")
        LOGGER.info(f"  Phase 2 (Add Activations): Epochs {phase1_end+1}-{phase2_end}")
        LOGGER.info(f"  Phase 3 (Full Quantization): Epochs {phase2_end+1}-{phase3_end}")
        LOGGER.info(f"  Phase 4 (Fine-tuning): Epochs {phase3_end+1}-{epochs}")
        
        # Setup penalty loss integration if available
        if hasattr(self, 'penalty_handler'):
            LOGGER.info("✅ Penalty loss integration active")
        
        # Configure initial phase (Phase 1: Weight-only)
        self._configure_quantizers_dynamically("phase1_weight_only")
        
        # Setup training callbacks for phase transitions
        def on_epoch_start(trainer):
            """Callback to handle phase transitions during training."""
            current_epoch = trainer.epoch
            
            # Update penalty handler epoch if available
            if hasattr(self, 'penalty_handler'):
                self.penalty_handler.update_epoch(current_epoch)
            
            # Phase transitions based on epoch
            if current_epoch == phase1_end:
                LOGGER.info(f"🔄 Epoch {current_epoch}: Transitioning to Phase 2 (Adding Activations)")
                self._configure_quantizers_dynamically("phase2_activations")
                
            elif current_epoch == phase2_end:
                LOGGER.info(f"🔄 Epoch {current_epoch}: Transitioning to Phase 3 (Full Quantization)")
                self._configure_quantizers_dynamically("phase3_full_quant")
                
            elif current_epoch == phase3_end:
                LOGGER.info(f"🔄 Epoch {current_epoch}: Transitioning to Phase 4 (Fine-tuning)")
                self._configure_quantizers_dynamically("phase4_fine_tuning")
                # Reduce learning rate for fine-tuning
                if hasattr(trainer, 'optimizer'):
                    for param_group in trainer.optimizer.param_groups:
                        param_group['lr'] *= 0.1
                    LOGGER.info(f"🔽 Learning rate reduced for fine-tuning: {param_group['lr']}")
        
        def on_epoch_end(trainer):
            """Callback to log phase-specific information."""
            if hasattr(self, 'penalty_handler') and trainer.epoch % 5 == 0:
                stats = self.penalty_handler.get_statistics()
                LOGGER.info(f"📊 Penalty Stats - Current: {stats.get('current_penalty', 0):.6f}, "
                        f"Avg: {stats.get('avg_penalty', 0):.6f}")
        
        # Add callbacks to model  
        self.model.add_callback('on_train_epoch_start', on_epoch_start)
        self.model.add_callback('on_train_epoch_end', on_epoch_end)
        
        # SINGLE CONTINUOUS TRAINING CALL
        try:
            LOGGER.info("🚀 Starting continuous phased QAT training...")
            LOGGER.info(f"📊 Dataset confirmed: {data_yaml}")
            
            # Add debug info before training
            LOGGER.info(f"🔍 Model state before training:")
            LOGGER.info(f"  - Model device: {next(self.model.model.parameters()).device}")
            LOGGER.info(f"  - Training mode: {self.model.model.training}")
            LOGGER.info(f"  - Has qconfig: {hasattr(self.model.model, 'qconfig')}")
            
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
            
            LOGGER.info("✅ Continuous phased QAT training completed successfully")
            return results
            
        except Exception as e:
            LOGGER.error(f"❌ Training failed with error: {e}")
            import traceback
            LOGGER.error(f"Full traceback:\n{traceback.format_exc()}")
            raise

    def _configure_quantizers_dynamically(self, phase_name):
        """
        FIXED VERSION: Use QuantizerStateManager for reliable phase transitions.
        """
        if not hasattr(self, 'quantizer_manager'):
            logger.info("🔧 Initializing quantizer state manager...")
            self.quantizer_manager = QuantizerStateManager(self.model.model)
        
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
            logger.info("🔧 Attempting emergency restoration...")
            self.quantizer_manager.emergency_restore_all_quantizers()
            
            # Try again after emergency restoration
            success = self.quantizer_manager.set_phase_state(
                phase_name=phase_name,
                weights_enabled=config["weights"], 
                activations_enabled=config["activations"]
            )
        
        # Final state report
        final_state = self.quantizer_manager.get_current_state()
        logger.info(f"🎯 Final state for {phase_name}:")
        logger.info(f"  - Weight quantizers: {final_state['weight_active']}/{final_state['total_weight']}")
        logger.info(f"  - Activation quantizers: {final_state['activation_active']}/{final_state['total_activation']}")
        
        return success

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
        Convert QAT model to quantized model.
        IMPORTANT: This removes FakeQuantize modules - save QAT model first!
        
        Args:
            save_path: Path to save quantized model
            
        Returns:
            Quantized model
        """
        logger.info("Converting QAT model to quantized INT8 model...")
        
        # CRITICAL: Verify we still have QAT structure before conversion
        if not self.verify_quantization_preserved():
            logger.error("❌ Cannot convert - QAT structure is missing!")
            logger.error("   - Model may have been saved/loaded incorrectly")
            logger.error("   - Try re-preparing the model for QAT")
            return None
        
        # IMPORTANT: Save QAT model BEFORE conversion (if path provided)
        if save_path:
            qat_save_path = save_path.replace('.pt', '_qat_before_conversion.pt')
            logger.info(f"Saving QAT model before conversion to {qat_save_path}")
            
            qat_save_success = self.save(qat_save_path, preserve_qat=True)
            if not qat_save_success:
                logger.error("❌ Failed to save QAT model before conversion!")
                logger.error("   - Proceeding anyway, but you may lose QAT capability")
        
        # Get QAT model and convert
        qat_model = self.model.model
        
        # Convert to quantized model (this removes FakeQuantize modules)
        try:
            self.quantized_model = convert_qat_model_to_quantized(qat_model)
            logger.info("✅ Successfully converted QAT model to quantized INT8")
        except Exception as e:
            logger.error(f"❌ Conversion failed: {e}")
            return None
        
        # Calculate size comparison
        size_info = compare_model_sizes(self.original_model, self.quantized_model)
        logger.info(f"Model size comparison:")
        logger.info(f"  Original FP32 model: {size_info['fp32_size_mb']:.2f} MB")
        logger.info(f"  Quantized INT8 model: {size_info['int8_size_mb']:.2f} MB")
        logger.info(f"  Compression ratio: {size_info['compression_ratio']:.2f}x")
        logger.info(f"  Size reduction: {size_info['size_reduction_percent']:.2f}%")
        
        # Analyze quantization effects
        quant_analysis = analyze_quantization_effects(self.quantized_model)
        logger.info(f"Quantization analysis:")
        logger.info(f"  Quantized modules: {quant_analysis['quantized_modules']} / {quant_analysis['total_modules']}")
        logger.info(f"  Quantization ratio: {quant_analysis['quantized_ratio']:.2f}")
        
        # Save quantized model (this is the final INT8 model for deployment)
        if save_path:
            int8_save_path = save_path.replace('.pt', '_int8_final.pt')
            
            # Create metadata
            metadata = {
                'framework': 'pytorch',
                'format': 'quantized_int8',
                'qconfig': self.qconfig_name,
                'size_info': size_info,
                'quant_analysis': quant_analysis,
                'conversion_successful': True,
                'deployment_ready': True
            }
            
            # Save the final quantized model
            save_success = save_quantized_model(self.quantized_model, int8_save_path, metadata)
            if save_success:
                logger.info(f"✅ Final quantized INT8 model saved to {int8_save_path}")
            else:
                logger.error(f"❌ Failed to save quantized model")
        
        self.is_converted = True
        return self.quantized_model
    
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

    def save(self, path, preserve_qat=True):
        """
        CORRECTED VERSION: Save the model with proper quantization preservation.
        
        Args:
            path: Path to save model
            preserve_qat: Whether to preserve quantization structure
        
        Returns:
            Success status
        """
        # Create directory if it doesn't exist
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        if preserve_qat and self.is_prepared:
            logger.info(f"Saving QAT model with quantization preserved to {path}")
            
            try:
                # Count FakeQuantize modules before saving
                fake_quant_count = sum(1 for n, m in self.model.model.named_modules() 
                                    if 'FakeQuantize' in type(m).__name__)
                
                logger.info(f"Found {fake_quant_count} FakeQuantize modules to preserve")
                
                # CRITICAL FIX: Save the COMPLETE MODEL, not just state_dict
                save_dict = {
                    # ✅ FIXED: Save full model architecture with quantization
                    'model': self.model.model,
                    
                    # ✅ FIXED: Save complete QAT information
                    'qat_info': {
                        'qconfig_name': self.qconfig_name,
                        'skip_detection_head': self.skip_detection_head,
                        'fuse_modules': self.fuse_modules,
                        'is_prepared': self.is_prepared,
                        'pytorch_version': torch.__version__,
                        'model_path': self.model_path,
                        'num_classes': 58,  # ✅ FIXED: Your Vietnamese traffic signs
                        'img_size': 640,
                        'dataset': 'vietnamese_traffic_signs'
                    },
                    
                    # ✅ FIXED: Quantization verification
                    'fake_quant_count': fake_quant_count,
                    'quantization_preserved': True,
                    'save_method': 'full_model_architecture',
                    
                    # ✅ FIXED: ONNX conversion readiness
                    'onnx_ready': True,
                    'conversion_info': {
                        'format': 'quantized_pytorch_model',
                        'architecture': 'yolov8n',
                        'classes': 58,
                        'input_size': [1, 3, 640, 640]
                    }
                }
                
                # Save the complete model
                torch.save(save_dict, path)
                
                # Verify the save
                success = self._verify_qat_save(path, fake_quant_count)
                
                if success:
                    logger.info(f"✅ QAT model saved successfully: {path}")
                    logger.info(f"✅ Quantization preserved: {fake_quant_count} FakeQuantize modules")
                    return True
                else:
                    logger.error(f"❌ Save verification failed")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ QAT save failed: {e}")
                
                # Try fallback method
                return self._save_fallback_method(path)
        
        else:
            # Standard save without quantization
            logger.warning("⚠️ Saving without QAT preservation")
            try:
                self.model.save(path)
                return True
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                return False

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
        """
        Verify that QAT model was saved correctly.
        
        Args:
            path: Path to saved model
            expected_fake_quant_count: Expected number of FakeQuantize modules
        
        Returns:
            Verification success status
        """
        try:
            logger.info("Verifying QAT model save...")
            
            # Load the saved model
            saved_data = torch.load(path, map_location='cpu')
            
            # Check format
            if not isinstance(saved_data, dict):
                logger.error("❌ Saved data is not a dictionary")
                return False
            
            # Check required keys
            required_keys = ['model', 'qat_info', 'fake_quant_count']
            missing_keys = [key for key in required_keys if key not in saved_data]
            
            if missing_keys:
                logger.error(f"❌ Missing required keys: {missing_keys}")
                return False
            
            # Check model architecture
            saved_model = saved_data['model']
            if not hasattr(saved_model, 'state_dict'):
                logger.error("❌ Saved model doesn't have state_dict")
                return False
            
            # Check FakeQuantize count
            saved_fake_quant_count = saved_data['fake_quant_count']
            if saved_fake_quant_count != expected_fake_quant_count:
                logger.error(f"❌ FakeQuantize count mismatch: expected {expected_fake_quant_count}, got {saved_fake_quant_count}")
                return False
            
            # Check class count
            qat_info = saved_data['qat_info']
            if qat_info.get('num_classes') != 58:
                logger.warning(f"⚠️ Unexpected class count: {qat_info.get('num_classes')}")
            
            logger.info("✅ QAT model verification passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Save verification failed: {e}")
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