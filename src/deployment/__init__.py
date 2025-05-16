"""
Deployment utilities for YOLOv8 QAT models.

This module provides functions for deploying, optimizing, and running inference
with quantized YOLOv8 models in various deployment targets.
"""

from .inference import (
    create_inference_engine,
    run_inference,
    run_batch_inference,
    get_inference_profile,
    load_model_for_inference
)

from .optimize import (
    optimize_model_for_deployment,
    prune_model,
    fuse_model_for_deployment,
    quantize_for_deployment,
    convert_to_target_format
)

from .benchmark import (
    benchmark_inference,
    benchmark_memory_usage,
    benchmark_precision,
    run_deployment_benchmark,
    compare_backends
)

# Main API functions
def prepare_model_for_deployment(model, target_format="onnx", optimize=True, quantized=True, config_path=None):
    """
    Prepare model for deployment with specified optimization options.
    
    Args:
        model: Model to prepare for deployment
        target_format: Target deployment format (onnx, tensorrt, openvino, etc.)
        optimize: Whether to apply optimizations
        quantized: Whether the model is already quantized
        config_path: Path to configuration file
        
    Returns:
        Prepared model ready for deployment
    """
    from .optimize import optimize_model_for_deployment
    return optimize_model_for_deployment(
        model=model,
        target_format=target_format,
        optimize=optimize,
        quantized=quantized,
        config_path=config_path
    )

def deploy_model(model, output_path, target_format="onnx", config_path=None):
    """
    Export model to target deployment format.
    
    Args:
        model: Model to deploy
        output_path: Path to save deployed model
        target_format: Target deployment format
        config_path: Path to export configuration
        
    Returns:
        Path to deployed model
    """
    from .optimize import convert_to_target_format
    return convert_to_target_format(
        model=model,
        output_path=output_path,
        target_format=target_format,
        config_path=config_path
    )

def create_deployer(model, backend="onnx"):
    """
    Create a deployer for the specified backend.
    
    Args:
        model: Model or path to model file
        backend: Inference backend to use
        
    Returns:
        Deployment engine for inference
    """
    from .inference import create_inference_engine
    return create_inference_engine(model, backend=backend)