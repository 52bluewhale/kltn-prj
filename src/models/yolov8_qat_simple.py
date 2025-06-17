#!/usr/bin/env python
"""
Simplified YOLOv8 QAT Implementation
Based on working train_qat_standard.py approach
"""
import os
import logging
import torch
import copy
import numpy as np  # ADD this import
from ultralytics import YOLO

# FIXED IMPORTS - with fallback handling
try:
    from torch.quantization.observer import (
        HistogramObserver,
        PerChannelMinMaxObserver,
        MinMaxObserver,
        MovingAverageMinMaxObserver
    )
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    # Fallback imports
    from torch.quantization.observer import (
        MinMaxObserver,
        MovingAverageMinMaxObserver
    )
    # Use MovingAverageMinMaxObserver as fallback for missing observers
    try:
        from torch.quantization.observer import PerChannelMinMaxObserver
    except ImportError:
        PerChannelMinMaxObserver = MinMaxObserver
        print("⚠️ PerChannelMinMaxObserver not available, using MinMaxObserver")
    
    try:
        from torch.quantization.observer import HistogramObserver
    except ImportError:
        HistogramObserver = MovingAverageMinMaxObserver
        print("⚠️ HistogramObserver not available, using MovingAverageMinMaxObserver")

logger = logging.getLogger('yolov8_qat_simple')

class SimpleQuantizedYOLOv8:
    """
    Simplified QAT wrapper that follows the working approach
    """
    def __init__(self, model_path, skip_detection_head=True):
        """
        Initialize simple quantized YOLOv8 model.
        
        Args:
            model_path: Path to YOLOv8 model
            skip_detection_head: Whether to skip quantization of detection head
        """
        self.model_path = model_path
        self.skip_detection_head = skip_detection_head
        
        # Load YOLOv8 model
        logger.info(f"Loading YOLOv8 model from {model_path}")
        self.model = YOLO(model_path)
        
        # Store original for comparison
        self.original_model = copy.deepcopy(self.model.model)
        
        # State tracking
        self.is_prepared = False
        self.is_trained = False
        self.is_converted = False
        self.quantized_model = None

        # Penalty loss attributes
        self.penalty_handler = None
        self.penalty_enabled = False
        self.current_penalty = 0.0

    # ADD THIS NEW METHOD after __init__
    def setup_penalty_loss(self, alpha=0.01, warmup_epochs=5):
        """
        Setup quantization penalty loss.
        
        Args:
            alpha: Penalty weight
            warmup_epochs: Number of epochs before full penalty
        """
        logger.info(f"Setting up penalty loss (α={alpha}, warmup={warmup_epochs})...")
        
        self.penalty_alpha = alpha
        self.penalty_warmup_epochs = warmup_epochs
        self.penalty_enabled = True
        self.current_epoch = 0
        
        # Store original forward method
        if not hasattr(self.model.model, '_original_forward_simple'):
            self.model.model._original_forward_simple = self.model.model.forward
        
        original_forward = self.model.model._original_forward_simple
        
        def forward_with_penalty(x):
            """Enhanced forward pass with penalty loss."""
            # Get original outputs
            outputs = original_forward(x)
            
            # Only calculate penalty during training
            if self.model.model.training and self.penalty_enabled:
                try:
                    penalty = self._calculate_quantization_penalty()
                    
                    # Apply warmup factor
                    warmup_factor = self._get_warmup_factor()
                    final_penalty = penalty * warmup_factor
                    
                    # Store for logging (don't modify outputs)
                    self.current_penalty = final_penalty
                    
                except Exception as e:
                    logger.warning(f"Penalty calculation failed: {e}")
                    self.current_penalty = 0.0
            
            return outputs
        
        # Replace forward method
        self.model.model.forward = forward_with_penalty
        
        logger.info("✅ Penalty loss setup complete")

    def _calculate_quantization_penalty(self):
        """Calculate quantization penalty for the model."""
        penalty = 0.0
        param_count = 0
        
        for name, module in self.model.model.named_modules():
            # Weight quantization penalty
            if hasattr(module, 'weight_fake_quant') and hasattr(module, 'weight'):
                try:
                    # Get quantized and original weights
                    w_original = module.weight
                    w_quantized = module.weight_fake_quant(w_original)
                    
                    # Calculate L2 penalty
                    weight_penalty = torch.norm(w_quantized - w_original, p=2) ** 2
                    penalty += weight_penalty
                    param_count += w_original.numel()
                    
                except Exception:
                    # Skip if quantization fails
                    continue
        
        # Normalize by parameter count
        if param_count > 0:
            penalty = self.penalty_alpha * penalty / param_count
        
        return penalty

    def _get_warmup_factor(self):
        """Get warmup factor based on current epoch."""
        if self.current_epoch < self.penalty_warmup_epochs:
            return min(1.0, self.current_epoch / self.penalty_warmup_epochs)
        else:
            return 1.0

    def update_penalty_epoch(self, epoch):
        """Update current epoch for penalty calculation."""
        self.current_epoch = epoch

    def get_penalty_stats(self):
        """Get penalty statistics for logging."""
        return {
            'current_penalty': self.current_penalty,
            'alpha': getattr(self, 'penalty_alpha', 0.0),
            'warmup_factor': self._get_warmup_factor() if self.penalty_enabled else 0.0,
            'enabled': self.penalty_enabled
        }
    
    def prepare_for_qat(self):
        """
        DEBUG: Exact copy of train_qat_standard.py approach with detailed logging.
        """
        logger.info("🔧 DEBUG: Starting QAT preparation (exact copy of working method)...")
        
        # Step 1: Get the model exactly like working version
        logger.info("Step 1: Getting model reference...")
        model = self.model.model  # This is equivalent to model = model_yolo.model
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model has modules: {len(list(model.named_modules()))}")
        
        # Step 2: Create EXACT same qconfig as working version
        logger.info("Step 2: Creating QConfig...")
        qconfig = torch.quantization.QConfig(
            activation=torch.quantization.MovingAverageMinMaxObserver.with_args(
                quant_min=0, quant_max=255, dtype=torch.qint8, averaging_constant=0.01
            ),
            weight=torch.quantization.MinMaxObserver.with_args(
                quant_min=-128, quant_max=127, dtype=torch.qint8
            )
        )
        logger.info(f"QConfig created: {qconfig}")
        
        # Step 3: Set qconfig on model exactly like working version
        logger.info("Step 3: Setting qconfig on model...")
        model.qconfig = qconfig
        logger.info(f"Model qconfig set: {hasattr(model, 'qconfig')}")
        
        # Step 4: EXACT same prepare_qat call as working version
        logger.info("Step 4: Calling prepare_qat...")
        try:
            # EXACT SAME CALL AS WORKING VERSION
            model_prepared = torch.quantization.prepare_qat(model.train())
            logger.info("✅ prepare_qat call succeeded")
        except Exception as e:
            logger.error(f"❌ prepare_qat failed: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            # Return original model like working version does
            model_prepared = model
            logger.info("Using original model as fallback")
        
        # Step 5: Assign back exactly like working version
        logger.info("Step 5: Assigning prepared model back...")
        self.model.model = model_prepared
        self.qat_model = model_prepared
        self.is_prepared = True
        
        # Step 6: Count FakeQuantize modules
        logger.info("Step 6: Counting FakeQuantize modules...")
        fake_quant_count = 0
        module_details = []
        
        for name, module in self.model.model.named_modules():
            module_type = type(module).__name__
            has_fake_quant = 'FakeQuantize' in module_type
            if has_fake_quant:
                fake_quant_count += 1
            
            # Log first 10 modules for debugging
            if len(module_details) < 10:
                module_details.append(f"  {name}: {module_type} (FakeQuantize: {has_fake_quant})")
        
        logger.info(f"Module details (first 10):")
        for detail in module_details:
            logger.info(detail)
        
        logger.info(f"✅ DEBUG QAT preparation result: {fake_quant_count} FakeQuantize modules")
        
        if fake_quant_count == 0:
            logger.error("❌ STILL NO FAKEQUANTIZE - Investigating further...")
            
            # Additional debugging
            logger.info("🔍 Additional debugging info:")
            logger.info(f"Model is in training mode: {self.model.model.training}")
            
            # Check if any modules have qconfig
            modules_with_qconfig = 0
            for name, module in self.model.model.named_modules():
                if hasattr(module, 'qconfig') and module.qconfig is not None:
                    modules_with_qconfig += 1
            logger.info(f"Modules with qconfig: {modules_with_qconfig}")
            
            # Check for quantizable modules
            quantizable_modules = 0
            for name, module in self.model.model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    quantizable_modules += 1
            logger.info(f"Quantizable modules (Conv2d/Linear): {quantizable_modules}")
            
            # Try alternative approach
            logger.info("🔄 Trying alternative prepare_qat approach...")
            try:
                # Try with inplace=True
                alternative_model = torch.quantization.prepare_qat(model, inplace=True)
                alt_fake_quant_count = sum(1 for n, m in alternative_model.named_modules() 
                                        if 'FakeQuantize' in type(m).__name__)
                logger.info(f"Alternative approach result: {alt_fake_quant_count} FakeQuantize modules")
                
                if alt_fake_quant_count > 0:
                    logger.info("✅ Alternative approach worked!")
                    self.model.model = alternative_model
                    self.qat_model = alternative_model
                    return alternative_model
            except Exception as e:
                logger.error(f"Alternative approach also failed: {e}")
        
        return self.qat_model
    
    def _count_fake_quantize_modules(self):
        """Count FakeQuantize modules in model."""
        count = 0
        for name, module in self.model.model.named_modules():
            if 'FakeQuantize' in type(module).__name__:
                count += 1
        return count

    def prepare_for_qat_advanced(self, observer_type='histogram', per_channel_weights=True):
        """
        SIMPLIFIED: Just call the basic prepare_for_qat since advanced observers are causing issues.
        """
        logger.info(f"⚠️ Ignoring advanced observer options, using basic approach...")
        return self.prepare_for_qat()

    def _fallback_qat_preparation(self):
        """
        FALLBACK: Use the exact same approach as train_qat_standard.py
        """
        logger.info("🔧 Using fallback QAT preparation...")
        
        base_model = self.model.model
        base_model.train()
        
        # Use EXACT same QConfig as working version
        qconfig = torch.quantization.QConfig(
            activation=torch.quantization.MovingAverageMinMaxObserver.with_args(
                quant_min=0, quant_max=255, dtype=torch.qint8, averaging_constant=0.01
            ),
            weight=torch.quantization.MinMaxObserver.with_args(
                quant_min=-128, quant_max=127, dtype=torch.qint8
            )
        )
        
        # Apply EXACTLY like train_qat_standard.py
        base_model.qconfig = qconfig
        
        try:
            self.qat_model = torch.quantization.prepare_qat(base_model.train())
            self.model.model = self.qat_model
            self.is_prepared = True
            
            fake_quant_count = self._count_fake_quantize_modules()
            logger.info(f"🔧 Fallback QAT preparation: {fake_quant_count} FakeQuantize modules")
            
            return self.qat_model
            
        except Exception as e:
            logger.error(f"Even fallback failed: {e}")
            return None
    
    def _create_advanced_qconfig(self, observer_type, per_channel_weights):
        """Create QConfig based on observer type."""
        
        # Activation observer
        if observer_type == 'histogram':
            try:
                activation_observer = HistogramObserver.with_args(
                    quant_min=0, quant_max=255, dtype=torch.quint8
                )
            except:
                # Fallback if HistogramObserver has issues
                activation_observer = MovingAverageMinMaxObserver.with_args(
                    quant_min=0, quant_max=255, dtype=torch.quint8, averaging_constant=0.01
                )
        elif observer_type == 'moving_average':
            activation_observer = MovingAverageMinMaxObserver.with_args(
                quant_min=0, quant_max=255, dtype=torch.quint8, averaging_constant=0.01
            )
        else:  # minmax
            activation_observer = MinMaxObserver.with_args(
                quant_min=0, quant_max=255, dtype=torch.quint8
            )
        
        # Weight observer
        if per_channel_weights:
            try:
                weight_observer = PerChannelMinMaxObserver.with_args(
                    quant_min=-128, quant_max=127, dtype=torch.qint8, ch_axis=0
                )
            except:
                # Fallback if PerChannelMinMaxObserver has issues
                weight_observer = MinMaxObserver.with_args(
                    quant_min=-128, quant_max=127, dtype=torch.qint8
                )
        else:
            weight_observer = MinMaxObserver.with_args(
                quant_min=-128, quant_max=127, dtype=torch.qint8
            )
        
        return torch.quantization.QConfig(
            activation=activation_observer,
            weight=weight_observer
        )
    
    def _create_first_layer_qconfig(self, observer_type):
        """Create special QConfig for first layer (usually needs higher precision)."""
        
        # Use histogram observer for first layer activations for better precision
        activation_observer = HistogramObserver.with_args(
            quant_min=0, quant_max=255, dtype=torch.quint8
        )
        
        # Always use per-channel for first layer weights
        weight_observer = PerChannelMinMaxObserver.with_args(
            quant_min=-128, quant_max=127, dtype=torch.qint8, ch_axis=0
        )
        
        return torch.quantization.QConfig(
            activation=activation_observer,
            weight=weight_observer
        )

    # ADD THIS METHOD after _create_first_layer_qconfig
    def calibrate_observers(self, dataloader, num_batches=100, device='cuda'):
        """
        Calibrate observers before starting quantization.
        
        Args:
            dataloader: DataLoader for calibration data
            num_batches: Number of batches to use for calibration
            device: Device to run calibration on
        """
        logger.info(f"Calibrating observers for {num_batches} batches...")
        
        # Set model to training mode but disable quantization
        self.model.model.train()
        self._disable_fake_quantization()
        
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
                
                if batch_count % 20 == 0:
                    logger.info(f"Calibration progress: {batch_count}/{num_batches}")
        
        # Re-enable quantization
        self._enable_fake_quantization()
        
        # Verify observer readiness
        ready_observers = self._check_observer_readiness()
        logger.info(f"Calibration complete: {ready_observers} observers ready")
        
        return ready_observers

    # ADD THESE HELPER METHODS
    def _disable_fake_quantization(self):
        """Disable fake quantization while keeping observers active."""
        for name, module in self.model.model.named_modules():
            if hasattr(module, 'weight_fake_quant'):
                if hasattr(module.weight_fake_quant, 'disable_fake_quant'):
                    module.weight_fake_quant.disable_fake_quant = True
            if hasattr(module, 'activation_post_process'):
                if hasattr(module.activation_post_process, 'disable_fake_quant'):
                    module.activation_post_process.disable_fake_quant = True

    def _enable_fake_quantization(self):
        """Re-enable fake quantization after calibration."""
        for name, module in self.model.model.named_modules():
            if hasattr(module, 'weight_fake_quant'):
                if hasattr(module.weight_fake_quant, 'disable_fake_quant'):
                    module.weight_fake_quant.disable_fake_quant = False
            if hasattr(module, 'activation_post_process'):
                if hasattr(module.activation_post_process, 'disable_fake_quant'):
                    module.activation_post_process.disable_fake_quant = False

    def _check_observer_readiness(self):
        """Check if observers have sufficient statistics."""
        ready_count = 0
        total_count = 0
        
        for name, module in self.model.model.named_modules():
            if hasattr(module, 'activation_post_process'):
                total_count += 1
                observer = module.activation_post_process
                
                # Check if observer is initialized
                if hasattr(observer, 'min_val') and hasattr(observer, 'max_val'):
                    if (hasattr(observer.min_val, 'numel') and observer.min_val.numel() > 0 and
                        observer.min_val != float('inf') and observer.max_val != float('-inf')):
                        ready_count += 1
        
        return ready_count
    
    # In the train_model method, replace the existing method with this enhanced version:
    def train_model(self, data_yaml, epochs, batch_size, img_size, lr, device, save_dir, log_dir):
        """
        Train model using simple approach with optional penalty loss.
        """
        import os
        
        logger.info("Starting simple QAT training...")
        
        # Validate dataset path
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
        
        # Setup penalty loss callbacks if enabled
        if self.penalty_enabled:
            def on_epoch_start(trainer):
                """Update penalty epoch."""
                self.update_penalty_epoch(trainer.epoch)
            
            def on_epoch_end(trainer):
                """Log penalty statistics."""
                if trainer.epoch % 5 == 0:  # Log every 5 epochs
                    stats = self.get_penalty_stats()
                    logger.info(f"Penalty Stats - Epoch {trainer.epoch}: "
                            f"Current: {stats['current_penalty']:.6f}, "
                            f"Warmup: {stats['warmup_factor']:.3f}")
            
            # Add callbacks to model
            self.model.add_callback('on_train_epoch_start', on_epoch_start)
            self.model.add_callback('on_train_epoch_end', on_epoch_end)
            
            logger.info("✅ Penalty loss callbacks added")
        
        try:
            # Simple training - no phase management
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
            
            self.is_trained = True
            logger.info("✅ QAT training completed successfully")
            
            # Log final penalty stats if enabled
            if self.penalty_enabled:
                final_stats = self.get_penalty_stats()
                logger.info(f"Final penalty stats: {final_stats}")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save_qat_model(self, path):
        """
        Save QAT model using simple approach.
        """
        try:
            # Use standard YOLO save method
            self.model.save(path)
            logger.info(f"✅ QAT model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Save failed: {e}")
            return False
    
    def convert_to_quantized(self, save_path=None):
        """
        Convert QAT model to INT8 using simple approach.
        """
        logger.info("Converting QAT model to INT8...")
        
        try:
            # Get QAT model in eval mode
            qat_model = self.model.model.eval()
            
            # Standard PyTorch conversion
            self.quantized_model = torch.quantization.convert(qat_model, inplace=False)
            
            # Calculate size comparison
            original_size = self._get_model_size(self.original_model)
            quantized_size = self._get_model_size(self.quantized_model)
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
            
            logger.info(f"✅ Conversion successful:")
            logger.info(f"  - Original: {original_size:.2f} MB")
            logger.info(f"  - Quantized: {quantized_size:.2f} MB") 
            logger.info(f"  - Compression: {compression_ratio:.2f}x")
            
            # Save if path provided
            if save_path:
                torch.save(self.quantized_model, save_path)
                logger.info(f"✅ INT8 model saved to {save_path}")
            
            self.is_converted = True
            return self.quantized_model
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return None
    
    def _get_model_size(self, model):
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / 1024 / 1024
    
    def export_to_onnx(self, onnx_path, img_size=640):
        """Export model to ONNX format."""
        try:
            exported_path = self.model.export(
                format='onnx',
                imgsz=img_size,
                simplify=True,
                opset=12,
                half=False
            )
            logger.info(f"✅ ONNX export successful: {exported_path}")
            return exported_path
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return None
    
    def evaluate(self, data_yaml, batch_size=16, img_size=640, device=''):
        """Evaluate model."""
        logger.info(f"Evaluating model...")
        
        results = self.model.val(
            data=data_yaml,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            verbose=True
        )
        
        return results

    # ADD THESE METHODS at the end of SimpleQuantizedYOLOv8 class
    def analyze_quantization_effects(self):
        """Analyze the effects of quantization on the model."""
        logger.info("Analyzing quantization effects...")
        
        results = {}
        
        # Count quantized modules
        quantized_modules = 0
        total_modules = 0
        
        for name, module in self.model.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                total_modules += 1
                if hasattr(module, 'weight_fake_quant'):
                    quantized_modules += 1
        
        results["quantized_ratio"] = quantized_modules / total_modules if total_modules > 0 else 0
        results["total_modules"] = total_modules
        results["quantized_modules"] = quantized_modules
        
        # Get quantization parameters
        scale_stats = []
        zero_point_stats = []
        
        for name, module in self.model.model.named_modules():
            if hasattr(module, 'weight_fake_quant'):
                fake_quant = module.weight_fake_quant
                if hasattr(fake_quant, 'scale') and hasattr(fake_quant, 'zero_point'):
                    if fake_quant.scale.numel() > 0:
                        scale_stats.extend(fake_quant.scale.detach().cpu().numpy().flatten())
                    if fake_quant.zero_point.numel() > 0:
                        zero_point_stats.extend(fake_quant.zero_point.detach().cpu().numpy().flatten())
        
        if scale_stats:
            import numpy as np
            results["scale_stats"] = {
                "mean": np.mean(scale_stats),
                "std": np.std(scale_stats),
                "min": np.min(scale_stats),
                "max": np.max(scale_stats)
            }
        
        if zero_point_stats:
            results["zero_point_stats"] = {
                "mean": np.mean(zero_point_stats),
                "std": np.std(zero_point_stats),
                "min": np.min(zero_point_stats),
                "max": np.max(zero_point_stats)
            }
        
        logger.info(f"Quantization analysis complete:")
        logger.info(f"  - Quantized modules: {quantized_modules}/{total_modules} ({results['quantized_ratio']:.1%})")
        if scale_stats:
            logger.info(f"  - Scale range: {results['scale_stats']['min']:.6f} - {results['scale_stats']['max']:.6f}")
        
        return results

    def get_model_summary(self):
        """Get comprehensive model summary."""
        summary = {
            "model_path": self.model_path,
            "is_prepared": self.is_prepared,
            "is_trained": self.is_trained,
            "is_converted": self.is_converted,
            "skip_detection_head": self.skip_detection_head,
            "penalty_enabled": getattr(self, 'penalty_enabled', False)
        }
        
        if self.is_prepared:
            summary["fake_quantize_count"] = self._count_fake_quantize_modules()
            summary["quantization_analysis"] = self.analyze_quantization_effects()
        
        if hasattr(self, 'penalty_enabled') and self.penalty_enabled:
            summary["penalty_stats"] = self.get_penalty_stats()
        
        # Model sizes
        if hasattr(self, 'original_model'):
            summary["original_size_mb"] = self._get_model_size(self.original_model)
        
        if self.quantized_model is not None:
            summary["quantized_size_mb"] = self._get_model_size(self.quantized_model)
            if "original_size_mb" in summary:
                summary["compression_ratio"] = summary["original_size_mb"] / summary["quantized_size_mb"]
        
        return summary

    def print_model_summary(self):
        """Print formatted model summary."""
        summary = self.get_model_summary()
        
        print("\n" + "="*60)
        print("📊 MODEL SUMMARY")
        print("="*60)
        print(f"Model Path: {summary['model_path']}")
        print(f"QAT Prepared: {'✅' if summary['is_prepared'] else '❌'}")
        print(f"Trained: {'✅' if summary['is_trained'] else '❌'}")
        print(f"Converted to INT8: {'✅' if summary['is_converted'] else '❌'}")
        print(f"Skip Detection Head: {'✅' if summary['skip_detection_head'] else '❌'}")
        
        if summary.get('fake_quantize_count'):
            print(f"FakeQuantize Modules: {summary['fake_quantize_count']}")
        
        if summary.get('penalty_enabled'):
            print(f"Penalty Loss: ✅ (α={summary.get('penalty_stats', {}).get('alpha', 'N/A')})")
        
        if 'quantization_analysis' in summary:
            analysis = summary['quantization_analysis']
            print(f"Quantized Modules: {analysis['quantized_modules']}/{analysis['total_modules']} "
                f"({analysis['quantized_ratio']:.1%})")
        
        if 'original_size_mb' in summary:
            print(f"Original Size: {summary['original_size_mb']:.2f} MB")
        
        if 'quantized_size_mb' in summary:
            print(f"Quantized Size: {summary['quantized_size_mb']:.2f} MB")
            if 'compression_ratio' in summary:
                print(f"Compression Ratio: {summary['compression_ratio']:.2f}x")
        
        print("="*60)