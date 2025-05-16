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
        """Train for a specific phase using YOLO's built-in export/import."""
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Training {phase_name} for {epochs} epochs with lr={lr}")
        
        # For the first phase, use a clean copy of the model
        if phase_name == "phase1_weight_only":
            # Start with a clean model
            clean_model = YOLO(self.model_path)
            temp_model_path = os.path.join(save_dir, f"{phase_name}_temp.pt")
            clean_model.save(temp_model_path)
            
            # Create trainer from this saved file
            trainer = YOLO(temp_model_path)
        else:
            # For subsequent phases, use the previous phase's best model
            prev_phase_dir = None
            if phase_name == "phase2_activations":
                prev_phase_dir = os.path.join(os.path.dirname(save_dir), "phase1_weight_only")
            elif phase_name == "phase3_full_quant":
                prev_phase_dir = os.path.join(os.path.dirname(save_dir), "phase2_activations")
            elif phase_name == "phase4_fine_tuning":
                prev_phase_dir = os.path.join(os.path.dirname(save_dir), "phase3_full_quant")
            
            # Use the best model from previous phase
            prev_best_model = os.path.join(prev_phase_dir, "weights", "best.pt")
            if os.path.exists(prev_best_model):
                temp_model_path = prev_best_model
                logger.info(f"Using best model from previous phase: {temp_model_path}")
                trainer = YOLO(temp_model_path)
            else:
                # Fallback - use original model
                logger.warning(f"Previous phase best model not found, using original model")
                trainer = YOLO(self.model_path)
        
        # Apply phase-specific configuration to the new trainer model
        self._apply_phase_config_to_trainer_model(trainer.model, phase_name)
        
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
        results = trainer.train(**train_args)
        
        # After training, update our model by creating a fresh instance from the trained weights
        best_weights_path = os.path.join(save_dir, 'weights', 'best.pt')
        if os.path.exists(best_weights_path):
            logger.info(f"Loading best weights from {best_weights_path}")
            # Replace our model with the trained model
            self.model = YOLO(best_weights_path)
            # Re-apply our phase configuration
            self._configure_phase(phase_name)
        else:
            logger.warning(f"Best weights not found, using last weights")
            last_weights_path = os.path.join(save_dir, 'weights', 'last.pt')
            if os.path.exists(last_weights_path):
                self.model = YOLO(last_weights_path)
                # Re-apply our phase configuration
                self._configure_phase(phase_name)
        
        return results

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
        
        Args:
            save_path: Path to save quantized model
            
        Returns:
            Quantized model
        """
        logger.info("Converting QAT model to quantized INT8 model...")
        
        # Get QAT model
        qat_model = self.model.model
        
        # Convert to quantized model
        self.quantized_model = convert_qat_model_to_quantized(qat_model)
        
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
        
        # Save quantized model
        if save_path:
            # Create metadata
            metadata = {
                'framework': 'pytorch',
                'format': 'quantized_int8',
                'qconfig': self.qconfig_name,
                'size_info': size_info,
                'quant_analysis': quant_analysis
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save model
            save_quantized_model(self.quantized_model, save_path, metadata)
            logger.info(f"Quantized model saved to {save_path}")
        
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

    def save(self, path):
        """
        Save the model to disk.
        
        Args:
            path: Path to save model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # For YOLOv8, we need to save the original YOLO object
        try:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            
            # Fallback: try to save using torch.save
            try:
                torch.save(self.model.model, path)
                logger.info(f"Model saved to {path} using torch.save")
                return True
            except Exception as inner_e:
                logger.error(f"Error saving model with torch.save: {inner_e}")
                return False

    def _configure_phase(self, phase):
        """Configure model for specific QAT phase."""
        if phase == "weight_only":
            # Enable only weight quantization, disable activation quantization
            logger.info("Configuring for weight-only quantization phase")
            self._set_activation_quantizers_enabled(False)
            self._set_weight_quantizers_enabled(True)
        elif phase == "activation_phase":
            # Enable both weight and activation quantization
            logger.info("Configuring for activation quantization phase")
            self._set_activation_quantizers_enabled(True)
            self._set_weight_quantizers_enabled(True)
        elif phase == "full_quantization":
            # Enable all quantizers
            logger.info("Configuring for full network quantization phase")
            self._set_all_quantizers_enabled(True)
        elif phase == "fine_tuning":
            # All quantizers remain enabled
            logger.info("Configuring for fine-tuning phase")
            self._set_all_quantizers_enabled(True)
        else:
            logger.warning(f"Unknown phase: {phase}")
        
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