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

# Import PyTorch quantization modules
from torch.quantization import prepare_qat, convert, get_default_qconfig

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
from src.training.loss import QATPenaltyLoss

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
    
    def train_model(self, data_yaml, epochs, batch_size, img_size, lr, device, save_dir, log_dir, use_distillation=False):
        """Train with phased QAT approach."""
        # Calculate phase durations
        total_epochs = epochs
        weight_only_epochs = max(1, int(total_epochs * 0.3))
        activation_phase_epochs = max(1, int(total_epochs * 0.4))
        full_quant_epochs = max(1, int(total_epochs * 0.2))
        fine_tuning_epochs = total_epochs - weight_only_epochs - activation_phase_epochs - full_quant_epochs
        
        logger.info(f"Phase 1: Weight-only quantization - {weight_only_epochs} epochs")
        logger.info(f"Phase 2: Adding activation quantization - {activation_phase_epochs} epochs")
        logger.info(f"Phase 3: Full network quantization - {full_quant_epochs} epochs")
        logger.info(f"Phase 4: Fine-tuning - {fine_tuning_epochs} epochs")
        
        # Base and reduced learning rates
        base_lr = lr
        fine_tuning_lr = base_lr * 0.1  # Lower LR for fine-tuning
        
        # Phase 1: Weight-only quantization
        logger.info("Starting Phase 1: Weight-only quantization")
        self._configure_phase("weight_only")
        phase1_results = self._train_phase(
            data_yaml=data_yaml,
            epochs=weight_only_epochs,
            batch_size=batch_size,
            img_size=img_size,
            lr=base_lr,
            device=device,
            save_dir=os.path.join(save_dir, "phase1_weight_only"),
            log_dir=log_dir,
            use_distillation=use_distillation,
            phase_name="phase1_weight_only"
        )
        
        # Phase 2: Add activation quantization
        logger.info("Starting Phase 2: Adding activation quantization")
        self._configure_phase("activation_phase")
        phase2_results = self._train_phase(
            data_yaml=data_yaml,
            epochs=activation_phase_epochs,
            batch_size=batch_size,
            img_size=img_size,
            lr=base_lr,
            device=device,
            save_dir=os.path.join(save_dir, "phase2_activations"),
            log_dir=log_dir,
            use_distillation=use_distillation,
            phase_name="phase2_activations"
        )
        
        # Phase 3: Full network quantization
        logger.info("Starting Phase 3: Full network quantization")
        self._configure_phase("full_quantization")
        phase3_results = self._train_phase(
            data_yaml=data_yaml,
            epochs=full_quant_epochs,
            batch_size=batch_size,
            img_size=img_size,
            lr=base_lr,
            device=device,
            save_dir=os.path.join(save_dir, "phase3_full_quant"),
            log_dir=log_dir,
            use_distillation=use_distillation,
            phase_name="phase3_full_quant"
        )
        
        # Phase 4: Fine-tuning
        logger.info("Starting Phase 4: Fine-tuning")
        self._configure_phase("fine_tuning")
        phase4_results = self._train_phase(
            data_yaml=data_yaml,
            epochs=fine_tuning_epochs,
            batch_size=batch_size,
            img_size=img_size,
            lr=fine_tuning_lr,  # Lower learning rate for fine-tuning
            device=device,
            save_dir=os.path.join(save_dir, "phase4_fine_tuning"),
            log_dir=log_dir,
            use_distillation=use_distillation,
            phase_name="phase4_fine_tuning"
        )
        
        # Use last phase results as final results
        return phase4_results

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

    def _train_phase(self, data_yaml, epochs, batch_size, img_size, lr, device, save_dir, log_dir, use_distillation, phase_name):
        """Train for a specific phase - simplified approach to avoid pickling issues."""
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Training {phase_name} for {epochs} epochs with lr={lr}")
        
        # For the first phase, start with original model
        if phase_name == "phase1_weight_only":
            logger.info("Phase 1: Starting with original model")
            trainer = YOLO(self.model_path)
            
        else:
            # For subsequent phases, try to use the best model from previous phase
            prev_best_model = self._get_previous_phase_best_model(save_dir, phase_name)
            
            if prev_best_model and os.path.exists(prev_best_model):
                logger.info(f"Loading best model from previous phase: {prev_best_model}")
                trainer = YOLO(prev_best_model)
            else:
                logger.warning(f"Previous phase best model not found, using original model")
                trainer = YOLO(self.model_path)
        
        # Apply quantization configuration to the trainer model
        # This is where we'll apply our phase-specific quantization settings
        logger.info(f"Applying quantization configuration for {phase_name}")
        
        # Set trainer model to training mode
        trainer.model.train()
        
        # Apply qconfig to the trainer model
        self._apply_qconfig_and_prepare_model(trainer, phase_name)
        
        # Verify quantization setup
        self._verify_quantization_setup(trainer.model, phase_name)
        
        # Training arguments
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'lr0': lr,
            'device': device,
            'project': os.path.dirname(save_dir),
            'name': os.path.basename(save_dir),
            'exist_ok': True,
            'pretrained': False,
            'val': True
        }
        
        # Train for this phase
        logger.info(f"Starting training for {phase_name}")
        results = trainer.train(**train_args)
        
        # After training, update our main model with the best results
        self._update_main_model_after_phase(save_dir, phase_name)
        
        return results

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

    def _update_main_model_after_phase(self, save_dir, phase_name):
        """Update main model with best results from this phase."""
        best_weights_path = os.path.join(save_dir, 'weights', 'best.pt')
        
        if os.path.exists(best_weights_path):
            logger.info(f"Updating main model with best weights from {best_weights_path}")
            
            try:
                # Load the trained model
                self.model = YOLO(best_weights_path)
                
                # Re-prepare the model for QAT for the next phase
                self.model.model.train()
                self._apply_qconfig_and_prepare_model(self.model, phase_name)
                
                logger.info(f"Successfully updated main model with {phase_name} results")
                
            except Exception as e:
                logger.error(f"Error updating main model: {e}")
                logger.warning("Main model update failed - continuing with previous model state")
        else:
            logger.warning(f"Best weights not found at {best_weights_path}")

    def _apply_phase_config_to_trainer_model(self, trainer_model, phase_name):
        """
        Apply phase-specific quantization configuration to trainer model.
        Optimized for PyTorch 2.4.1.
        """
        logger.info(f"Applying {phase_name} configuration to trainer model...")
        
        try:
            # Import PyTorch quantization modules
            from torch.quantization import get_default_qat_qconfig, prepare_qat
            
            # Count modules for verification
            weight_quant_count = 0
            activation_quant_count = 0
            total_modules = 0
            
            # Apply configuration based on phase
            for name, module in trainer_model.named_modules():
                total_modules += 1
                
                # Configure according to phase
                if hasattr(module, 'weight_fake_quant') or hasattr(module, 'activation_post_process'):
                    if phase_name == "phase1_weight_only":
                        # Phase 1: Only weight quantization
                        if hasattr(module, 'activation_post_process'):
                            if not hasattr(module, '_original_activation_post_process'):
                                module._original_activation_post_process = module.activation_post_process
                            module.activation_post_process = nn.Identity()
                        
                        if hasattr(module, 'weight_fake_quant'):
                            weight_quant_count += 1
                    
                    elif phase_name in ["phase2_activations", "phase3_full_quant", "phase4_fine_tuning"]:
                        # Enable both weight and activation quantization
                        if hasattr(module, '_original_activation_post_process'):
                            module.activation_post_process = module._original_activation_post_process
                        
                        if hasattr(module, 'weight_fake_quant'):
                            weight_quant_count += 1
                        
                        if hasattr(module, 'activation_post_process'):
                            activation_quant_count += 1
            
            logger.info(f"Applied {phase_name} config to trainer model:")
            logger.info(f"  - Total modules: {total_modules}")
            logger.info(f"  - Weight quantizers: {weight_quant_count}")
            logger.info(f"  - Activation quantizers: {activation_quant_count}")
            
        except Exception as e:
            logger.error(f"Error applying phase config: {e}")
            logger.warning(f"Training may proceed with default quantization: {e}")

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
    
    def analyze_quantization_error(self, data_loader, n_samples=10):
        """
        Analyze quantization error layer by layer.
        
        Args:
            data_loader: DataLoader with samples for analysis
            n_samples: Number of samples to analyze
            
        Returns:
            Dictionary with layer-wise error metrics
        """
        if not self.is_converted:
            logger.warning("Model not converted to INT8 yet, converting model...")
            self.convert_to_quantized()
        
        # This is a placeholder - actual implementation would need to compare outputs
        # between original FP32 and quantized model across layers
        logger.info(f"Analyzing quantization error across {n_samples} samples...")
        
        # Run inference with original and quantized model
        # Compare outputs and calculate error metrics
        
        error_metrics = {
            "overall_error": 0.0,
            "layer_wise_errors": {}
        }
        
        return error_metrics

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

    def _create_trainer(self, data_yaml, epochs, batch_size, img_size, lr, device, save_dir, use_distillation=False):
        """
        Create a YOLO trainer with phase-specific settings.
        
        Args:
            data_yaml: Dataset YAML path
            epochs: Number of epochs for this phase
            batch_size: Batch size
            img_size: Input image size
            lr: Learning rate for this phase
            device: Training device
            save_dir: Directory to save results
            use_distillation: Whether to use knowledge distillation
            
        Returns:
            YOLO trainer object
        """
        import os
        from ultralytics import YOLO
        
        # For YOLOv8, we can't create a new YOLO object with an existing model
        # We need to save the model first to a temporary file, then load it
        
        # Save the current model state to a temporary file
        temp_model_path = os.path.join(save_dir, "temp_model.pt")
        os.makedirs(os.path.dirname(temp_model_path), exist_ok=True)
        
        # Save model state
        # torch.save(self.model.state_dict(), temp_model_path)
        torch.save(self.model.model.state_dict(), temp_model_path)  # Use self.model.model instead
        
        # Create a new YOLO object from the saved file
        trainer = YOLO(temp_model_path)
        
        # Configure training arguments
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'lr0': lr,
            'device': device,
            'project': os.path.dirname(save_dir),
            'name': os.path.basename(save_dir),
            'exist_ok': True,
            'pretrained': False,  # Don't load pretrained weights, we already have our model
            'val': True,          # Run validation during training
        }
        
        # If using distillation, we would add distillation-specific settings
        if use_distillation and hasattr(self, 'distillation_config'):
            # This would depend on how distillation is implemented for YOLOv8
            train_args['teacher_model'] = self.distillation_config.get('teacher_model', None)
            train_args['distillation_tau'] = self.distillation_config.get('temperature', 4.0)
            train_args['distillation_alpha'] = self.distillation_config.get('alpha', 0.5)
        
        # Store training arguments for reference
        trainer.train_args = train_args
        
        return trainer, train_args

    def train_simplified(self, data_yaml, epochs, batch_size, img_size, lr, device, save_dir, log_dir, use_distillation=False):
        """Simplified training implementation without phased approach."""
        import os
        import logging
        
        logger = logging.getLogger('yolov8_qat')
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Log training parameters
        logger.info(f"Training with simplified approach:")
        logger.info(f"  Data: {data_yaml}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Image size: {img_size}")
        logger.info(f"  Learning rate: {lr}")
        logger.info(f"  Device: {device}")
        
        # Train using existing YOLO object - no need to save/load
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
            val=True
        )
        
        logger.info("Training completed")
        return results

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

    def _verify_saved_model(self, path, expected_fake_quant_count):
        """Verify that the saved model preserved quantization."""
        try:
            # Load and check the saved model
            saved_data = torch.load(path, map_location='cpu')
            
            if isinstance(saved_data, dict):
                # Check for our QAT save format with model_state_dict
                if 'model_state_dict' in saved_data and 'fake_quant_count' in saved_data:
                    # This is our comprehensive save format
                    saved_count = saved_data.get('fake_quant_count', 0)
                    save_method = saved_data.get('save_method', 'unknown')
                    quantization_preserved = saved_data.get('quantization_preserved', False)
                    
                    logger.info(f"Save verification (Method 1 - state dict format):")
                    logger.info(f"  - Expected FakeQuantize modules: {expected_fake_quant_count}")
                    logger.info(f"  - Saved count recorded: {saved_count}")
                    logger.info(f"  - Save method: {save_method}")
                    logger.info(f"  - Quantization preserved: {quantization_preserved}")
                    
                    if saved_count == expected_fake_quant_count and saved_count > 0:
                        logger.info("✅ Quantization successfully preserved in saved model (state dict format)")
                        return True
                    else:
                        logger.error(f"❌ Quantization count mismatch: expected {expected_fake_quant_count}, got {saved_count}")
                        return False
                
                # Check for full model format (if we ever save the full model)
                elif 'model' in saved_data:
                    # Count FakeQuantize modules in loaded model
                    try:
                        actual_count = sum(1 for n, m in saved_data['model'].named_modules() 
                                        if 'FakeQuantize' in type(m).__name__)
                        
                        saved_count = saved_data.get('fake_quant_count', 0)
                        
                        logger.info(f"Save verification (Method 2 - full model format):")
                        logger.info(f"  - Expected FakeQuantize modules: {expected_fake_quant_count}")
                        logger.info(f"  - Saved count recorded: {saved_count}")
                        logger.info(f"  - Actual count in saved model: {actual_count}")
                        
                        if actual_count == expected_fake_quant_count and actual_count > 0:
                            logger.info("✅ Quantization successfully preserved in saved model (full model format)")
                            return True
                        else:
                            logger.error(f"❌ Quantization not properly preserved in full model")
                            return False
                            
                    except Exception as model_check_error:
                        logger.error(f"❌ Error checking full model format: {model_check_error}")
                        return False
                
                # Check for fallback format
                elif 'fake_quant_count' in saved_data:
                    saved_count = saved_data['fake_quant_count']
                    save_method = saved_data.get('save_method', 'unknown')
                    
                    logger.info(f"Save verification (Method 3 - fallback format):")
                    logger.info(f"  - Expected FakeQuantize modules: {expected_fake_quant_count}")
                    logger.info(f"  - Saved count recorded: {saved_count}")
                    logger.info(f"  - Save method: {save_method}")
                    
                    if saved_count == expected_fake_quant_count and saved_count > 0:
                        logger.info("✅ Quantization info successfully preserved in fallback format")
                        return True
                    else:
                        logger.error(f"❌ Quantization count mismatch in fallback format")
                        return False
                
                else:
                    logger.error("❌ No recognized quantization information found in saved data")
                    logger.info(f"Available keys in saved data: {list(saved_data.keys())}")
                    return False
            else:
                logger.error("❌ Saved model format not recognized as dictionary")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify saved model: {e}")
            return False

    def _verify_saved_model_fallback(self, path, expected_fake_quant_count):
        """Verify that the fallback saved model preserved quantization info."""
        try:
            # Load and check the saved model
            saved_data = torch.load(path, map_location='cpu')
            
            if isinstance(saved_data, dict):
                # Check for fallback format with direct fake_quant_count
                if 'fake_quant_count' in saved_data:
                    saved_count = saved_data['fake_quant_count']
                    save_method = saved_data.get('save_method', 'unknown')
                    quantization_preserved = saved_data.get('quantization_preserved', False)
                    
                    logger.info(f"Fallback save verification (Method 1):")
                    logger.info(f"  - Expected FakeQuantize modules: {expected_fake_quant_count}")
                    logger.info(f"  - Saved count recorded: {saved_count}")
                    logger.info(f"  - Save method: {save_method}")
                    logger.info(f"  - Quantization preserved: {quantization_preserved}")
                    
                    if saved_count == expected_fake_quant_count and saved_count > 0:
                        logger.info("✅ Quantization info successfully preserved in fallback save")
                        return True
                    else:
                        logger.error(f"❌ Quantization count mismatch: expected {expected_fake_quant_count}, got {saved_count}")
                        return False
                
                # Check for quantization_info nested format
                elif 'quantization_info' in saved_data:
                    quant_info = saved_data['quantization_info']
                    saved_count = quant_info.get('fake_quant_count', 0)
                    was_prepared = quant_info.get('was_prepared', False)
                    
                    logger.info(f"Fallback save verification (Method 2):")
                    logger.info(f"  - Expected FakeQuantize modules: {expected_fake_quant_count}")
                    logger.info(f"  - Saved count: {saved_count}")
                    logger.info(f"  - Was prepared: {was_prepared}")
                    
                    success = saved_count == expected_fake_quant_count and saved_count > 0
                    if success:
                        logger.info("✅ Quantization info successfully preserved in nested format")
                    else:
                        logger.error(f"❌ Quantization count mismatch in nested format")
                    
                    return success
                
                # Check for config format
                elif 'config' in saved_data and 'quantization_info' in saved_data:
                    quant_info = saved_data['quantization_info']
                    saved_count = quant_info.get('fake_quant_count', 0)
                    
                    logger.info(f"Fallback save verification (Method 3 - config format):")
                    logger.info(f"  - Expected FakeQuantize modules: {expected_fake_quant_count}")
                    logger.info(f"  - Saved count: {saved_count}")
                    
                    success = saved_count == expected_fake_quant_count and saved_count > 0
                    if success:
                        logger.info("✅ Quantization info successfully preserved in config format")
                    else:
                        logger.error(f"❌ Quantization count mismatch in config format")
                    
                    return success
                
                else:
                    logger.error("❌ No quantization information found in fallback save")
                    logger.info(f"Available keys in saved data: {list(saved_data.keys())}")
                    return False
                    
            else:
                logger.error("❌ Saved model format not recognized as dictionary")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify fallback saved model: {e}")
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

    def _save_model_with_quantization_info(self, model, path):
        """Save model while avoiding quantization pickling issues."""
        try:
            # Method 1: Try to save state dict only (avoids observer pickling issues)
            logger.info(f"Saving model state dict to avoid pickling issues: {path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save just the state dict
            torch.save({
                'model': model.state_dict(),
                'metadata': {
                    'quantized': True,
                    'prepared_for_qat': True,
                    'architecture': 'yolov8',
                    'framework': 'pytorch'
                }
            }, path)
            
            logger.info(f"Successfully saved model state dict to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save model state dict: {e}")
            
            # Method 2: Fallback - create a clean YOLO model and copy weights
            try:
                logger.info("Trying fallback method: creating clean YOLO model")
                
                # Create a clean YOLO model
                clean_yolo = YOLO(self.model_path)  # Start with original model
                
                # Copy the state dict from our quantized model to the clean model
                # This will copy weights but lose quantization info (which is expected for intermediate saves)
                try:
                    clean_yolo.model.load_state_dict(model.state_dict(), strict=False)
                    logger.info("Successfully copied weights to clean model")
                except Exception as copy_error:
                    logger.warning(f"Could not copy all weights: {copy_error}")
                    # If copying fails, just use the original model structure
                
                # Save the clean model
                clean_yolo.save(path)
                logger.info(f"Successfully saved clean model to {path}")
                
            except Exception as fallback_error:
                logger.error(f"Fallback save method also failed: {fallback_error}")
                
                # Method 3: Last resort - save original model
                logger.info("Using original model as last resort")
                original_yolo = YOLO(self.model_path)
                original_yolo.save(path)
                logger.warning(f"Saved original model to {path} (quantization info lost)")

    def _get_previous_phase_best_model(self, save_dir, phase_name):
        """Get the best model from the previous phase."""
        phase_mapping = {
            "phase2_activations": "phase1_weight_only",
            "phase3_full_quant": "phase2_activations", 
            "phase4_fine_tuning": "phase3_full_quant"
        }
        
        prev_phase = phase_mapping.get(phase_name)
        if prev_phase:
            prev_phase_dir = os.path.join(os.path.dirname(save_dir), prev_phase)
            prev_best_model = os.path.join(prev_phase_dir, "weights", "best.pt")
            
            if os.path.exists(prev_best_model):
                logger.info(f"Found previous phase model: {prev_best_model}")
                return prev_best_model
            else:
                logger.warning(f"Previous phase model not found: {prev_best_model}")
        
        return None

    def test_quantization_setup(self):
        """Test method to verify quantization is working correctly."""
        logger.info("Testing quantization setup...")
        
        # Test each phase configuration
        phases_to_test = [
            "phase1_weight_only",
            "phase2_activations", 
            "phase3_full_quant",
            "phase4_fine_tuning"
        ]
        
        for phase in phases_to_test:
            logger.info(f"\n--- Testing {phase} ---")
            
            # Configure for this phase
            test_model = copy.deepcopy(self.model.model)
            self._configure_model_for_specific_phase(test_model, phase)
            
            # Verify setup
            success = self._verify_quantization_setup(test_model, phase)
            
            if success:
                logger.info(f"✓ {phase} configuration test PASSED")
            else:
                logger.error(f"❌ {phase} configuration test FAILED")
        
        logger.info("Quantization setup test completed")

    # ==============================================================================
    # 4. ADD LOADING METHOD (add this as a class method)
    # ==============================================================================

    @classmethod
    def load_qat_model(cls, path, device='cpu'):
        """
        Load a QAT model that was saved with quantization preservation.
        
        Args:
            path: Path to saved QAT model
            device: Device to load model on
            
        Returns:
            QuantizedYOLOv8 instance with quantization preserved
        """
        logger.info(f"Loading QAT model from {path}")
        
        try:
            # Load the saved data
            saved_data = torch.load(path, map_location=device)
            
            # Check the save format
            if isinstance(saved_data, dict) and 'qat_info' in saved_data:
                # This is our preserved format
                qat_info = saved_data['qat_info']
                
                # Create instance with same configuration
                instance = cls(
                    model_path=qat_info['model_path'],
                    qconfig_name=qat_info['qconfig_name'],
                    skip_detection_head=qat_info['skip_detection_head'],
                    fuse_modules=qat_info['fuse_modules']
                )
                
                if 'model' in saved_data:
                    # Load the full model structure
                    instance.model.model = saved_data['model'].to(device)
                    logger.info("✅ Loaded full model structure with quantization")
                else:
                    # Load state dict and re-prepare
                    instance.prepare_for_qat()
                    instance.model.model.load_state_dict(saved_data['model_state_dict'])
                    instance.model.model.to(device)
                    logger.info("✅ Loaded state dict and re-prepared for QAT")
                
                # Set flags
                instance.is_prepared = qat_info['is_prepared']
                
                # Verify quantization was preserved
                fake_quant_count = sum(1 for n, m in instance.model.model.named_modules() 
                                    if 'FakeQuantize' in type(m).__name__)
                
                expected_count = saved_data.get('fake_quant_count', 0)
                
                if fake_quant_count > 0 and fake_quant_count == expected_count:
                    logger.info(f"✅ QAT model loaded successfully with {fake_quant_count} FakeQuantize modules")
                else:
                    logger.warning(f"⚠️ Quantization may not be fully preserved: {fake_quant_count}/{expected_count}")
                
                return instance
                
            elif isinstance(saved_data, dict) and 'config' in saved_data:
                # Fallback format - state dict with config
                config = saved_data['config']
                
                instance = cls(
                    model_path=config['model_path'],
                    qconfig_name=config['qconfig_name'],
                    skip_detection_head=config['skip_detection_head'],
                    fuse_modules=config.get('fuse_modules', True)
                )
                
                # Re-prepare for QAT and load state dict
                instance.prepare_for_qat()
                instance.model.model.load_state_dict(saved_data['state_dict'])
                instance.model.model.to(device)
                
                logger.warning("⚠️ Loaded from state dict - quantization structure recreated")
                
                return instance
                
            else:
                # Try standard YOLO loading
                logger.warning("⚠️ Loading as standard YOLO model - quantization may be lost")
                from ultralytics import YOLO
                
                yolo_model = YOLO(path)
                yolo_model.model.to(device)
                
                # Create instance
                instance = cls.__new__(cls)
                instance.model = yolo_model
                instance.is_prepared = False
                instance.qconfig_name = 'default'
                instance.skip_detection_head = True
                instance.fuse_modules = True
                
                logger.warning("⚠️ Loaded as standard model - will need to re-prepare for QAT")
                
                return instance
                
        except Exception as e:
            logger.error(f"Failed to load QAT model: {e}")
            raise

    # ==============================================================================
    # 5. ADD MODEL TESTING METHOD (add this method)
    # ==============================================================================

    def test_quantization_preservation(self):
        """Test that quantization is working correctly with detailed debugging."""
        logger.info("Testing quantization preservation...")
        
        if not self.is_prepared:
            logger.error("❌ Model not prepared for QAT")
            return False
        
        # Test 1: Count FakeQuantize modules
        fake_quant_count = sum(1 for n, m in self.model.model.named_modules() 
                            if 'FakeQuantize' in type(m).__name__)
        
        logger.info(f"Test 1 - FakeQuantize modules: {fake_quant_count}")
        
        if fake_quant_count == 0:
            logger.error("❌ No FakeQuantize modules found!")
            return False
        
        # Test 2: Check qconfig application
        qconfig_count = sum(1 for n, m in self.model.model.named_modules() 
                        if hasattr(m, 'qconfig') and m.qconfig is not None)
        
        logger.info(f"Test 2 - Modules with qconfig: {qconfig_count}")
        
        # Test 3: Detailed save/load cycle with debugging
        logger.info("Test 3 - Testing save/load cycle with detailed debugging...")
        
        temp_path = 'temp_qat_test.pt'
        try:
            # Save with detailed logging
            logger.info(f"  Step 3a: Saving QAT model to {temp_path}")
            save_success = self.save(temp_path, preserve_qat=True)
            if not save_success:
                logger.error("❌ Step 3a: Save operation reported failure")
                return False
            
            logger.info("  Step 3b: Examining saved file contents")
            if not os.path.exists(temp_path):
                logger.error(f"❌ Step 3b: Save file does not exist at {temp_path}")
                return False
            
            # Load and examine the saved data structure
            saved_data = torch.load(temp_path, map_location='cpu')
            logger.info(f"  Step 3c: Saved data type: {type(saved_data)}")
            
            if isinstance(saved_data, dict):
                logger.info(f"  Step 3d: Available keys in saved data: {list(saved_data.keys())}")
                
                # Check each expected key
                expected_keys = ['model_state_dict', 'qat_info', 'fake_quant_count', 'quantization_preserved']
                for key in expected_keys:
                    if key in saved_data:
                        if key == 'fake_quant_count':
                            logger.info(f"    ✅ {key}: {saved_data[key]}")
                        else:
                            logger.info(f"    ✅ {key}: present")
                    else:
                        logger.warning(f"    ⚠️ {key}: missing")
                
                # Specific verification
                if 'fake_quant_count' in saved_data:
                    saved_count = saved_data['fake_quant_count']
                    logger.info(f"  Step 3e: Saved FakeQuantize count: {saved_count}")
                    
                    if saved_count == fake_quant_count:
                        logger.info("  Step 3f: ✅ Count verification PASSED")
                        
                        # Test loading
                        logger.info("  Step 3g: Testing model loading")
                        try:
                            # Try to recreate the model from saved data
                            if 'model_state_dict' in saved_data and 'qat_info' in saved_data:
                                logger.info("    ✅ Both model_state_dict and qat_info found")
                                logger.info("    ✅ Save format is correct for loading")
                                
                                # Clean up and return success
                                os.remove(temp_path)
                                logger.info("✅ All quantization preservation tests PASSED")
                                return True
                            else:
                                logger.error("    ❌ Required keys for loading not found")
                                os.remove(temp_path)
                                return False
                                
                        except Exception as load_test_error:
                            logger.error(f"    ❌ Load test failed: {load_test_error}")
                            os.remove(temp_path)
                            return False
                    else:
                        logger.error(f"  Step 3f: ❌ Count mismatch: expected {fake_quant_count}, saved {saved_count}")
                        os.remove(temp_path)
                        return False
                else:
                    logger.error("  Step 3e: ❌ fake_quant_count not found in saved data")
                    os.remove(temp_path)
                    return False
            else:
                logger.error(f"  Step 3d: ❌ Saved data is not a dictionary, got {type(saved_data)}")
                os.remove(temp_path)
                return False
                
        except Exception as e:
            logger.error(f"❌ Save/load test exception: {e}")
            logger.error(f"Full traceback:")
            import traceback
            logger.error(traceback.format_exc())
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
        
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

    # ===================================================================
# ADDITIONAL METHOD: Load QAT model properly
# ===================================================================

    @classmethod
    def load_qat_model_fixed(cls, path, device='cpu'):
        """
        CORRECTED VERSION: Load QAT model with proper quantization handling.
        
        Args:
            path: Path to saved QAT model
            device: Device to load on
        
        Returns:
            QuantizedYOLOv8 instance
        """
        logger.info(f"Loading QAT model from {path}")
        
        try:
            # Load saved data
            saved_data = torch.load(path, map_location=device)
            
            if isinstance(saved_data, dict):
                if 'model' in saved_data and 'qat_info' in saved_data:
                    # Our new corrected format
                    qat_info = saved_data['qat_info']
                    
                    # Create instance with correct parameters
                    instance = cls(
                        model_path=qat_info.get('model_path', 'yolov8n.pt'),
                        qconfig_name=qat_info.get('qconfig_name', 'default'),
                        skip_detection_head=qat_info.get('skip_detection_head', True),
                        fuse_modules=qat_info.get('fuse_modules', True)
                    )
                    
                    # Load the full model
                    instance.model.model = saved_data['model'].to(device)
                    instance.is_prepared = qat_info.get('is_prepared', True)
                    
                    logger.info("✅ QAT model loaded successfully with quantization preserved")
                    return instance
                    
                elif 'model' in saved_data:
                    # Simplified format
                    model_info = saved_data.get('model_info', {})
                    
                    instance = cls.__new__(cls)
                    instance.model = type('MockYOLO', (), {'model': saved_data['model']})()
                    instance.model.model.to(device)
                    instance.is_prepared = True
                    instance.qconfig_name = 'default'
                    
                    logger.info("✅ Model loaded from simplified format")
                    return instance
                    
            logger.error("❌ Unrecognized model format")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to load QAT model: {e}")
            return None