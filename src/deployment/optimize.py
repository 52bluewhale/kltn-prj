"""
Optimization utilities for YOLOv8 QAT models deployment.

This module provides functions for optimizing models for deployment,
including pruning, fusion, and format conversion.
"""

import torch
import numpy as np
import logging
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import copy

# Setup logging
logger = logging.getLogger(__name__)

def optimize_model_for_deployment(
    model: torch.nn.Module,
    target_format: str = "onnx",
    optimize: bool = True,
    quantized: bool = True,
    config_path: Optional[str] = None
) -> torch.nn.Module:
    """
    Optimize model for deployment with specified target format.
    
    Args:
        model: Model to optimize
        target_format: Target deployment format
        optimize: Whether to apply optimizations
        quantized: Whether the model is already quantized
        config_path: Path to configuration file
        
    Returns:
        Optimized model
    """
    # Make a copy of the model to avoid modifying the original
    model_copy = copy.deepcopy(model)
    
    # Set model to evaluation mode
    model_copy.eval()
    
    # Load configuration if provided
    config = {}
    if config_path:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            logger.warning(f"Configuration file {config_path} not found. Using default settings.")
    
    # Apply optimizations if requested
    if optimize:
        # Fuse model operations for better performance
        model_copy = fuse_model_for_deployment(model_copy, config.get("optimization", {}))
        
        # Prune model if enabled in config
        if config.get("optimization", {}).get("pruning", {}).get("enabled", False):
            model_copy = prune_model(
                model_copy, 
                config.get("optimization", {}).get("pruning", {})
            )
    
    return model_copy


def prune_model(
    model: torch.nn.Module,
    pruning_config: Dict[str, Any]
) -> torch.nn.Module:
    """
    Prune model to reduce size and improve performance.
    
    Args:
        model: Model to prune
        pruning_config: Pruning configuration
        
    Returns:
        Pruned model
    """
    try:
        import torch.nn.utils.prune as prune
        
        # Get pruning parameters
        method = pruning_config.get("method", "l1_unstructured")
        amount = pruning_config.get("amount", 0.3)
        
        # Get layers to prune (default: Conv2d and Linear)
        layer_types = []
        if pruning_config.get("prune_conv", True):
            layer_types.append(torch.nn.Conv2d)
        if pruning_config.get("prune_linear", True):
            layer_types.append(torch.nn.Linear)
        
        # Skip certain layers specified in config
        skip_layers = pruning_config.get("skip_layers", [])
        
        # Apply pruning to each layer
        for name, module in model.named_modules():
            # Skip if in exclusion list
            if any(skip in name for skip in skip_layers):
                continue
            
            # Apply pruning to supported layer types
            if isinstance(module, tuple(layer_types)):
                if method == "l1_unstructured":
                    prune.l1_unstructured(module, name="weight", amount=amount)
                elif method == "random_unstructured":
                    prune.random_unstructured(module, name="weight", amount=amount)
                elif method == "ln_structured":
                    dim = pruning_config.get("dim", 0)
                    n = pruning_config.get("n", 2)
                    prune.ln_structured(module, name="weight", amount=amount, n=n, dim=dim)
        
        # Make pruning permanent
        for name, module in model.named_modules():
            if isinstance(module, tuple(layer_types)):
                try:
                    prune.remove(module, "weight")
                except:
                    # weight might not have been pruned
                    pass
        
        logger.info(f"Model pruned with method {method} and amount {amount}")
        return model
    
    except ImportError:
        logger.warning("torch.nn.utils.prune not available. Skipping pruning.")
        return model
    except Exception as e:
        logger.error(f"Error during pruning: {e}")
        return model


def fuse_model_for_deployment(
    model: torch.nn.Module,
    optimization_config: Dict[str, Any]
) -> torch.nn.Module:
    """
    Fuse operations in the model for better performance.
    
    Args:
        model: Model to fuse
        optimization_config: Optimization configuration
        
    Returns:
        Fused model
    """
    # Check if fusion is enabled
    if not optimization_config.get("fuse", True):
        return model
    
    try:
        # Try to use model's own fusion method first
        if hasattr(model, "fuse"):
            logger.info("Using model's built-in fusion method")
            model.fuse()
            return model
        
        # For YOLOv8 models, try to use our custom fusion methods
        try:
            from ..models.model_transforms import fuse_yolov8_model_modules
            logger.info("Using custom YOLOv8 fusion method")
            return fuse_yolov8_model_modules(model)
        except ImportError:
            pass
        
        # Fall back to PyTorch's fusion utilities
        from torch.quantization.fuse_modules import fuse_modules
        
        # Define fusion patterns
        fusion_patterns = [
            ['conv', 'bn'],
            ['conv', 'bn', 'relu'],
            ['conv', 'relu'],
        ]
        
        # Find modules to fuse
        modules_to_fuse = []
        
        # Find potential Conv+BN+ReLU/ReLU6 patterns
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Sequential):
                # Check if the sequential module contains a fusion pattern
                if len(module) >= 2:
                    if isinstance(module[0], torch.nn.Conv2d):
                        if isinstance(module[1], torch.nn.BatchNorm2d):
                            if len(module) >= 3 and isinstance(module[2], torch.nn.ReLU):
                                modules_to_fuse.append([f"{name}.0", f"{name}.1", f"{name}.2"])
                            else:
                                modules_to_fuse.append([f"{name}.0", f"{name}.1"])
                        elif isinstance(module[1], torch.nn.ReLU):
                            modules_to_fuse.append([f"{name}.0", f"{name}.1"])
        
        # Fuse modules
        if modules_to_fuse:
            logger.info(f"Fusing {len(modules_to_fuse)} module patterns")
            model = fuse_modules(model, modules_to_fuse, inplace=True)
        
        return model
    
    except Exception as e:
        logger.error(f"Error during fusion: {e}")
        return model


def quantize_for_deployment(
    model: torch.nn.Module,
    calibration_data: Optional[List[torch.Tensor]] = None,
    quantization_config: Dict[str, Any] = None
) -> torch.nn.Module:
    """
    Quantize model for deployment if needed.
    
    Args:
        model: Model to quantize
        calibration_data: Optional calibration data for PTQ
        quantization_config: Quantization configuration
        
    Returns:
        Quantized model
    """
    # Check if model is already quantized
    # (This is a simplified check, might need enhancement for
    # more robust detection of quantized models)
    if hasattr(model, "_is_quantized") and model._is_quantized:
        logger.info("Model is already quantized")
        return model
    
    # Set default quantization config
    if quantization_config is None:
        quantization_config = {
            "backend": "qnnpack",  # or "fbgemm" for x86, "qnnpack" for ARM
            "dtype": "qint8",
            "method": "static"
        }
    
    try:
        # Check if model is already prepared for QAT
        is_qat_ready = any(hasattr(m, "weight_fake_quant") for m in model.modules())
        
        if is_qat_ready:
            # Convert QAT model to fully quantized model
            model.eval()
            from torch.quantization import convert
            quantized_model = convert(model)
            logger.info("Converted QAT model to quantized model")
            quantized_model._is_quantized = True
            return quantized_model
        else:
            # For PTQ, use static quantization method
            if calibration_data:
                from torch.quantization import (
                    get_default_qconfig, prepare, convert, quantize_static
                )
                
                # Set backend
                backend = quantization_config.get("backend", "qnnpack")
                torch.backends.quantized.engine = backend
                
                # Create qconfig
                qconfig = get_default_qconfig(backend)
                model.qconfig = qconfig
                
                # Prepare model for quantization
                prepared_model = prepare(model)
                
                # Calibrate with calibration data
                with torch.no_grad():
                    for x in calibration_data:
                        prepared_model(x)
                
                # Convert to quantized model
                quantized_model = convert(prepared_model)
                quantized_model._is_quantized = True
                
                logger.info(f"Applied PTQ with {backend} backend using {len(calibration_data)} calibration samples")
                return quantized_model
            else:
                logger.warning("No calibration data provided for PTQ. Skipping quantization.")
                return model
    
    except Exception as e:
        logger.error(f"Error during quantization: {e}")
        return model


def convert_to_target_format(
    model: torch.nn.Module,
    output_path: str,
    target_format: str = "onnx",
    config_path: Optional[str] = None,
    input_shape: Optional[Tuple[int, ...]] = None
) -> str:
    """
    Convert model to target deployment format.
    
    Args:
        model: Model to convert
        output_path: Path to save converted model
        target_format: Target format (onnx, tensorrt, openvino, tflite, etc.)
        config_path: Path to export configuration
        input_shape: Optional input shape (default: (1, 3, 640, 640))
        
    Returns:
        Path to converted model
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load configuration if provided
    config = {}
    if config_path:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            logger.warning(f"Configuration file {config_path} not found. Using default settings.")
    
    # Set default input shape if not provided
    if input_shape is None:
        input_shape = (1, 3, 640, 640)
    
    # Set model to evaluation mode
    model.eval()
    
    # Convert to target format
    if target_format.lower() == "onnx":
        return _convert_to_onnx(model, output_path, input_shape, config.get("onnx", {}))
    elif target_format.lower() == "tensorrt":
        return _convert_to_tensorrt(model, output_path, input_shape, config.get("tensorrt", {}))
    elif target_format.lower() == "openvino":
        return _convert_to_openvino(model, output_path, input_shape, config.get("openvino", {}))
    elif target_format.lower() == "tflite":
        return _convert_to_tflite(model, output_path, input_shape, config.get("tflite", {}))
    elif target_format.lower() == "coreml":
        return _convert_to_coreml(model, output_path, input_shape, config.get("coreml", {}))
    else:
        raise ValueError(f"Unsupported target format: {target_format}")


def _convert_to_onnx(model, output_path, input_shape, config):
    """Convert model to ONNX format."""
    try:
        import torch.onnx
        
        # Set ONNX export parameters
        opset_version = config.get("opset", 13)
        dynamic = config.get("dynamic", True)
        simplify = config.get("simplify", True)
        
        # Create dummy input
        x = torch.randn(input_shape, requires_grad=False)
        
        # Set output names
        output_names = config.get("output_names", ["output0"])
        
        # Export model to ONNX
        torch.onnx.export(
            model,
            x,
            output_path,
            verbose=False,
            opset_version=opset_version,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes={"images": {0: "batch_size"}, "output0": {0: "batch_size"}} if dynamic else None
        )
        
        # Simplify ONNX model if requested
        if simplify:
            try:
                import onnx
                import onnxsim
                
                # Load exported model
                onnx_model = onnx.load(output_path)
                
                # Simplify model
                simplified_model, check = onnxsim.simplify(onnx_model)
                
                if check:
                    # Save simplified model
                    onnx.save(simplified_model, output_path)
                    logger.info("ONNX model simplified successfully")
                else:
                    logger.warning("ONNX model simplification failed")
            except (ImportError, Exception) as e:
                logger.warning(f"ONNX simplification failed: {e}")
        
        logger.info(f"Model exported to ONNX format at {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error during ONNX export: {e}")
        return None


def _convert_to_tensorrt(model, output_path, input_shape, config):
    """Convert model to TensorRT format."""
    try:
        # First export to ONNX
        onnx_path = output_path.replace(".engine", ".onnx")
        onnx_path = _convert_to_onnx(model, onnx_path, input_shape, config)
        
        if not onnx_path:
            logger.error("Failed to export to ONNX, cannot proceed with TensorRT conversion")
            return None
        
        # Convert ONNX to TensorRT
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            # Create TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Create builder and network
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            
            # Create ONNX parser
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX file
            with open(onnx_path, 'rb') as f:
                parser.parse(f.read())
            
            # Create builder config
            config = builder.create_builder_config()
            
            # Set TensorRT parameters
            workspace_size = config.get("workspace", 4) * 1 << 30  # Convert to bytes
            config.max_workspace_size = workspace_size
            
            # Set precision
            if config.get("fp16", False):
                config.set_flag(trt.BuilderFlag.FP16)
            
            if config.get("int8", False):
                config.set_flag(trt.BuilderFlag.INT8)
                
                # For INT8, calibrator is needed but not implemented here
                # This would require a calibration dataset and implementation
                # of a Calibrator class
            
            # Build and save engine
            engine = builder.build_engine(network, config)
            
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"Model exported to TensorRT format at {output_path}")
            return output_path
        
        except ImportError:
            logger.error("TensorRT or PyCUDA not found. Cannot convert to TensorRT format.")
            return None
    
    except Exception as e:
        logger.error(f"Error during TensorRT export: {e}")
        return None


def _convert_to_openvino(model, output_path, input_shape, config):
    """Convert model to OpenVINO format."""
    try:
        # First export to ONNX
        onnx_path = output_path.replace(".xml", ".onnx")
        onnx_path = _convert_to_onnx(model, onnx_path, input_shape, config)
        
        if not onnx_path:
            logger.error("Failed to export to ONNX, cannot proceed with OpenVINO conversion")
            return None
        
        # Convert ONNX to OpenVINO IR
        try:
            import subprocess
            
            # Get OpenVINO installation directory
            openvino_install_dir = os.environ.get("INTEL_OPENVINO_DIR", "")
            if not openvino_install_dir:
                logger.warning("INTEL_OPENVINO_DIR environment variable not set. Trying to use openvino package...")
                
                try:
                    import openvino as ov
                    from openvino.tools import mo
                    
                    # Use OpenVINO Python API for conversion
                    model_xml_path = output_path
                    ov_model = mo.convert_model(
                        onnx_path, 
                        compress_to_fp16=config.get("half_precision", False)
                    )
                    
                    # Save the converted model
                    ov.save_model(ov_model, model_xml_path)
                    logger.info(f"Model exported to OpenVINO format at {model_xml_path}")
                    return model_xml_path
                
                except ImportError:
                    logger.error("OpenVINO Python API not found. Cannot convert to OpenVINO format.")
                    return None
            
            # Use model optimizer script
            mo_script = os.path.join(openvino_install_dir, "tools", "mo", "mo.py")
            if not os.path.exists(mo_script):
                logger.error(f"OpenVINO Model Optimizer not found at {mo_script}")
                return None
            
            # Prepare command line arguments
            cmd = [
                "python", mo_script,
                "--input_model", onnx_path,
                "--output_dir", os.path.dirname(output_path),
                "--model_name", os.path.splitext(os.path.basename(output_path))[0]
            ]
            
            # Add additional parameters from config
            if config.get("data_type", "FP32") == "FP16":
                cmd.append("--compress_to_fp16")
            
            # Execute model optimizer
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"OpenVINO conversion failed: {result.stderr}")
                return None
            
            logger.info(f"Model exported to OpenVINO format at {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error during OpenVINO conversion: {e}")
            return None
    
    except Exception as e:
        logger.error(f"Error during OpenVINO export: {e}")
        return None


def _convert_to_tflite(model, output_path, input_shape, config):
    """Convert model to TFLite format."""
    try:
        # First export to ONNX
        onnx_path = output_path.replace(".tflite", ".onnx")
        onnx_path = _convert_to_onnx(model, onnx_path, input_shape, config)
        
        if not onnx_path:
            logger.error("Failed to export to ONNX, cannot proceed with TFLite conversion")
            return None
        
        # Convert ONNX to TFLite
        try:
            import tf2onnx
            import tensorflow as tf
            
            # Convert ONNX to TensorFlow SavedModel
            saved_model_dir = os.path.join(os.path.dirname(output_path), "saved_model")
            cmd = ["python", "-m", "tf2onnx.convert", 
                   "--onnx", onnx_path, 
                   "--output", saved_model_dir, 
                   "--target", "tf"]
            
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"TensorFlow conversion failed: {result.stderr}")
                return None
            
            # Convert SavedModel to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            
            # Set optimization options
            if config.get("optimize", True):
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Set quantization options
            if config.get("quantize", False):
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                
                # For quantization, a representative dataset would be needed
                # This is a simplified example and doesn't include it
            
            # Convert model
            tflite_model = converter.convert()
            
            # Save model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Model exported to TFLite format at {output_path}")
            return output_path
        
        except ImportError:
            logger.error("TensorFlow or tf2onnx not found. Cannot convert to TFLite format.")
            return None
    
    except Exception as e:
        logger.error(f"Error during TFLite export: {e}")
        return None


def _convert_to_coreml(model, output_path, input_shape, config):
    """Convert model to Core ML format."""
    try:
        # First export to ONNX
        onnx_path = output_path.replace(".mlmodel", ".onnx")
        onnx_path = _convert_to_onnx(model, onnx_path, input_shape, config)
        
        if not onnx_path:
            logger.error("Failed to export to ONNX, cannot proceed with Core ML conversion")
            return None
        
        # Convert ONNX to Core ML
        try:
            import onnx
            import coremltools as ct
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Specify input shape
            input_shape_dict = {"images": input_shape}
            
            # Convert to Core ML
            mlmodel = ct.converters.onnx.convert(
                model=onnx_model,
                minimum_ios_deployment_target=config.get("minimum_ios_version", "13"),
                source='onnx',
                inputs=input_shape_dict if config.get("specify_input_shape", True) else None
            )
            
            # Set model metadata
            mlmodel.author = config.get("author", "YOLOv8 QAT")
            mlmodel.license = config.get("license", "Unknown")
            mlmodel.short_description = config.get("description", "YOLOv8 object detection model")
            
            # Set user-defined metadata
            user_metadata = config.get("metadata", {})
            for key, value in user_metadata.items():
                mlmodel.user_defined_metadata[key] = str(value)
            
            # Save the model
            mlmodel.save(output_path)
            
            logger.info(f"Model exported to Core ML format at {output_path}")
            return output_path
        
        except ImportError:
            logger.error("CoreMLTools or ONNX not found. Cannot convert to Core ML format.")
            return None
    
    except Exception as e:
        logger.error(f"Error during Core ML export: {e}")
        return None