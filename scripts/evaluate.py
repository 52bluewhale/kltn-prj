#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for YOLOv8 models (FP32 and INT8).

This script evaluates YOLOv8 models on validation/test datasets,
providing comprehensive metrics for both floating-point and quantized models.
It also supports comparison between both model types to analyze the
impact of quantization on model performance.
"""

import os
import sys
import argparse
import yaml
import logging
import torch
import time
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path to allow imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from src.models import (
    create_yolov8_model,
    convert_yolov8_to_quantized
)
from src.data_utils import (
    create_dataloader,
    get_dataset_from_yaml
)
from src.evaluation import (
    evaluate_model,
    compare_fp32_int8_models,
    generate_evaluation_report
)
from src.deployment import (
    create_inference_engine,
    benchmark_inference,
    measure_layer_wise_quantization_error
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('evaluate')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8 models (FP32 and INT8)')
    
    # Dataset settings
    parser.add_argument('--data', type=str, default='dataset/vietnam-traffic-sign-detection/dataset.yaml',
                        help='path to dataset configuration yaml file')
    
    # Model settings
    parser.add_argument('--model', type=str, default=None,
                        help='path to model file (can be FP32 or INT8)')
    parser.add_argument('--fp32-model', type=str, default=None,
                        help='path to FP32 model file (for comparison)')
    parser.add_argument('--int8-model', type=str, default=None,
                        help='path to INT8 model file (for comparison)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='input image size')
    
    # Evaluation settings
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size for evaluation')
    parser.add_argument('--workers', type=int, default=8,
                        help='number of data loading workers')
    parser.add_argument('--device', type=str, default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='confidence threshold for detection')
    parser.add_argument('--iou-thres', type=float, default=0.5,
                        help='NMS IoU threshold')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='number of samples to evaluate (default: all)')
    
    # Comparison settings
    parser.add_argument('--compare', action='store_true',
                        help='compare FP32 and INT8 models')
    parser.add_argument('--benchmark', action='store_true',
                        help='benchmark model inference speed')
    parser.add_argument('--error-analysis', action='store_true',
                        help='analyze quantization error')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='logs/evaluation',
                        help='path to save evaluation results')
    parser.add_argument('--save-plots', action='store_true',
                        help='save visualization plots')
    parser.add_argument('--num-vis', type=int, default=5,
                        help='number of images to visualize')
    
    return parser.parse_args()


def load_model(model_path, device, is_quantized=False):
    """
    Load model from file.
    
    Args:
        model_path: Path to model file
        device: Device to load model on
        is_quantized: Whether the model is quantized
    
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model from checkpoint
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                # Need to initialize model first
                # Try to determine model name and number of classes
                model_name = 'yolov8n'  # Default
                num_classes = 80  # Default
                
                # Check if metadata exists
                if 'metadata' in checkpoint:
                    metadata = checkpoint['metadata']
                    if 'model_name' in metadata:
                        model_name = metadata['model_name']
                    if 'num_classes' in metadata:
                        num_classes = metadata['num_classes']
                
                # Create model
                model = create_yolov8_model(
                    model_name=model_name,
                    num_classes=num_classes,
                    pretrained=False
                )
                
                # Load state dict
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Try to load as a state dict directly
                model = checkpoint
        else:
            model = checkpoint
        
        # Convert to quantized model if specified
        if is_quantized and not _is_already_quantized(model):
            logger.info("Converting model to quantized format")
            model = convert_yolov8_to_quantized(model)
        
        # Move model to device
        model = model.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def _is_already_quantized(model):
    """
    Check if model is already quantized.
    
    Args:
        model: Model to check
        
    Returns:
        True if model is quantized, False otherwise
    """
    # Check if any module has quantized parameters
    for module in model.modules():
        if hasattr(module, '_is_quantized') and module._is_quantized:
            return True
        if hasattr(module, 'weight_fake_quant') or hasattr(module, 'activation_post_process'):
            # This is likely a QAT model, not a fully quantized model
            return False
    
    # Check for quantized modules patterns
    quantized_module_types = [
        'QuantizedConv2d',
        'QuantizedLinear',
        'QuantizedBatchNorm2d'
    ]
    
    for name, module in model.named_modules():
        module_type = module.__class__.__name__
        if any(qtype in module_type for qtype in quantized_module_types):
            return True
    
    return False


def create_dataloader_from_args(args, mode='val'):
    """
    Create dataloader from arguments.
    
    Args:
        args: Command line arguments
        mode: Dataset mode ('val' or 'test')
        
    Returns:
        DataLoader object
    """
    logger.info(f"Creating {mode} dataloader from {args.data}")
    
    dataloader, dataset = create_dataloader(
        dataset_yaml=args.data,
        batch_size=args.batch_size,
        img_size=args.img_size,
        augment=False,
        shuffle=False,
        workers=args.workers,
        mode=mode
    )
    
    # If num_samples is specified, limit the dataset
    if args.num_samples is not None and args.num_samples < len(dataset):
        logger.info(f"Limiting evaluation to {args.num_samples} samples")
        
        # Create subset
        from torch.utils.data import Subset
        indices = torch.randperm(len(dataset))[:args.num_samples]
        dataset = Subset(dataset, indices)
        
        # Create new dataloader with subset
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=getattr(dataset, 'collate_fn', None)
        )
    
    return dataloader


def evaluate_single_model(model, dataloader, args):
    """
    Evaluate a single model.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        args: Command line arguments
        
    Returns:
        Evaluation metrics
    """
    logger.info("Evaluating model...")
    
    # Create device
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Perform evaluation
    start_time = time.time()
    
    metrics = evaluate_model(
        model=model, 
        dataloader=dataloader,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
        device=device
    )
    
    evaluation_time = time.time() - start_time
    logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
    
    # Log results
    logger.info("Evaluation results:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            logger.info(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    
    # Add benchmark if requested
    if args.benchmark:
        logger.info("Benchmarking model inference speed...")
        benchmark_results = benchmark_inference(
            model=model,
            input_shape=(1, 3, args.img_size, args.img_size),
            num_runs=100,
            device=device
        )
        
        # Add benchmark results to metrics
        metrics['benchmark'] = benchmark_results
        
        # Log benchmark results
        logger.info(f"Inference time: {benchmark_results['mean_inference_time']*1000:.2f} ms")
        logger.info(f"Frames per second: {benchmark_results['fps']:.2f}")
    
    return metrics


def compare_models(fp32_model, int8_model, dataloader, args):
    """
    Compare FP32 and INT8 models.
    
    Args:
        fp32_model: FP32 model
        int8_model: INT8 model
        dataloader: DataLoader for evaluation
        args: Command line arguments
        
    Returns:
        Comparison results
    """
    logger.info("Comparing FP32 and INT8 models...")
    
    # Create device
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define metrics to compute
    metrics = ['accuracy', 'latency']
    if args.error_analysis:
        metrics.append('output_error')
        metrics.append('layer_outputs')
    
    # Perform comparison
    start_time = time.time()
    
    results = compare_fp32_int8_models(
        fp32_model=fp32_model,
        int8_model=int8_model,
        dataloader=dataloader,
        metrics=metrics,
        device=device,
        num_samples=args.num_samples
    )
    
    comparison_time = time.time() - start_time
    logger.info(f"Comparison completed in {comparison_time:.2f} seconds")
    
    # Log comparison results
    if 'accuracy_comparison' in results:
        acc_comp = results['accuracy_comparison']
        logger.info(f"FP32 mAP@.5: {acc_comp.get('fp32_map50', 0):.4f}")
        logger.info(f"INT8 mAP@.5: {acc_comp.get('int8_map50', 0):.4f}")
        logger.info(f"Absolute change: {acc_comp.get('absolute_change', 0):.4f}")
        logger.info(f"Relative change: {acc_comp.get('relative_change', 0)*100:.2f}%")
    
    if 'latency_comparison' in results:
        lat_comp = results['latency_comparison']
        logger.info(f"FP32 inference time: {lat_comp.get('fp32_time', 0)*1000:.2f} ms")
        logger.info(f"INT8 inference time: {lat_comp.get('int8_time', 0)*1000:.2f} ms")
        logger.info(f"Speedup: {lat_comp.get('speedup', 0):.2f}x")
        logger.info(f"FP32 FPS: {lat_comp.get('fp32_fps', 0):.2f}")
        logger.info(f"INT8 FPS: {lat_comp.get('int8_fps', 0):.2f}")
    
    # Perform error analysis if requested
    if args.error_analysis and 'layer_comparison' in results:
        # Sort layers by error
        layer_errors = [(name, data['mean_error']) for name, data in results['layer_comparison'].items()]
        layer_errors.sort(key=lambda x: x[1], reverse=True)
        
        # Log top 10 layers with highest error
        logger.info("Top 10 layers with highest quantization error:")
        for i, (name, error) in enumerate(layer_errors[:10]):
            logger.info(f"{i+1}. {name}: {error:.6f}")
    
    return results


def analyze_quantization_error(fp32_model, int8_model, args):
    """
    Analyze quantization error between FP32 and INT8 models.
    
    Args:
        fp32_model: FP32 model
        int8_model: INT8 model
        args: Command line arguments
        
    Returns:
        Error analysis results
    """
    logger.info("Analyzing quantization error...")
    
    # Create device
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test input
    test_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    
    # Measure layer-wise quantization error
    error_results = measure_layer_wise_quantization_error(fp32_model, int8_model, test_input)
    
    # Sort layers by error
    sorted_errors = sorted(error_results.items(), key=lambda x: x[1]['abs_error'], reverse=True)
    
    # Log top 10 layers with highest error
    logger.info("Top 10 layers with highest quantization error:")
    for i, (name, error) in enumerate(sorted_errors[:10]):
        logger.info(f"{i+1}. {name}: abs_error={error['abs_error']:.6f}, rel_error={error['rel_error']:.6f}")
    
    return error_results


def save_results(results, args, name_suffix=""):
    """
    Save evaluation results to file.
    
    Args:
        results: Evaluation results
        args: Command line arguments
        name_suffix: Suffix for filename
        
    Returns:
        Path to saved results
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create results filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(args.output_dir, f"evaluation_results_{timestamp}{name_suffix}.json")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        else:
            return obj
    
    # Convert results to JSON-serializable format
    serializable_results = convert_for_json(results)
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # If save_plots is True and comparison results, generate visual report
    if args.save_plots and 'accuracy_comparison' in results:
        report_dir = os.path.join(args.output_dir, f"report_{timestamp}")
        report_path = generate_evaluation_report(results, output_path=report_dir)
        logger.info(f"Evaluation report generated at {report_path}")
    
    return results_file


def main():
    """Main evaluation function"""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create device
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create evaluation dataloader
    dataloader = create_dataloader_from_args(args, mode='val')
    
    # Determine which model(s) to evaluate
    if args.compare:
        # Both FP32 and INT8 models need to be provided for comparison
        if not args.fp32_model or not args.int8_model:
            raise ValueError("Both --fp32-model and --int8-model must be provided for comparison")
        
        # Load FP32 model
        fp32_model = load_model(args.fp32_model, device, is_quantized=False)
        
        # Load INT8 model
        int8_model = load_model(args.int8_model, device, is_quantized=True)
        
        # Compare models
        comparison_results = compare_models(fp32_model, int8_model, dataloader, args)
        
        # Save comparison results
        save_results(comparison_results, args, name_suffix="_comparison")
        
        # Perform error analysis if requested
        if args.error_analysis:
            error_results = analyze_quantization_error(fp32_model, int8_model, args)
            save_results(error_results, args, name_suffix="_error_analysis")
    else:
        # Single model evaluation
        if not args.model:
            raise ValueError("--model must be provided for single model evaluation")
        
        # Determine if model is quantized based on filename
        is_quantized = "int8" in args.model.lower() or "quantized" in args.model.lower()
        
        # Load model
        model = load_model(args.model, device, is_quantized=is_quantized)
        
        # Evaluate model
        eval_results = evaluate_single_model(model, dataloader, args)
        
        # Save evaluation results
        model_type = "int8" if is_quantized else "fp32"
        save_results(eval_results, args, name_suffix=f"_{model_type}")
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error during evaluation: {e}")
        sys.exit(1)