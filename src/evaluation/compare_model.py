"""
Utilities for comparing FP32 and INT8 models.

This module provides functions for comparing the performance, accuracy, and 
outputs of floating-point and quantized models.
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from tqdm import tqdm
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Setup logging
logger = logging.getLogger(__name__)

def compare_fp32_int8_models(
    fp32_model: torch.nn.Module,
    int8_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    metrics: Optional[List[str]] = None,
    device: str = "cuda",
    num_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compare FP32 and INT8 models on various metrics.
    
    Args:
        fp32_model: Floating point model
        int8_model: Quantized model
        dataloader: DataLoader with evaluation dataset
        metrics: List of metrics to compute (default: ['accuracy', 'latency', 'output_error'])
        device: Device to run evaluation on
        num_samples: Optional number of samples to use for evaluation
        
    Returns:
        Dictionary with comparison results
    """
    if metrics is None:
        metrics = ['accuracy', 'latency', 'output_error']
    
    # Set models to evaluation mode
    fp32_model.eval()
    int8_model.eval()
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    fp32_model.to(device)
    int8_model.to(device)
    
    # Initialize results
    results = {
        'accuracy_comparison': {},
        'latency_comparison': {},
        'output_comparison': {},
        'layer_comparison': {}
    }
    
    # Storage for predictions and targets
    fp32_predictions = []
    int8_predictions = []
    targets = []
    
    # Storage for FP32 and INT8 outputs
    layer_outputs_fp32 = defaultdict(list)
    layer_outputs_int8 = defaultdict(list)
    
    # Register hooks to collect layer outputs
    fp32_hooks = []
    int8_hooks = []
    
    if 'layer_outputs' in metrics:
        # Create hooks to capture layer outputs
        def create_hook(layer_name, output_dict):
            def hook(module, input, output):
                # Store a copy of the output
                if isinstance(output, torch.Tensor):
                    output_dict[layer_name].append(output.detach().cpu())
                elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                    output_dict[layer_name].append(output[0].detach().cpu())
            return hook
        
        # Register hooks for important layers
        for name, module in fp32_model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
                fp32_hooks.append(module.register_forward_hook(create_hook(name, layer_outputs_fp32)))
        
        for name, module in int8_model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
                int8_hooks.append(module.register_forward_hook(create_hook(name, layer_outputs_int8)))
    
    # Track inference times
    fp32_times = []
    int8_times = []
    
    # Evaluation loop
    with torch.no_grad():
        # Limit samples if requested
        sample_loader = dataloader
        if num_samples is not None and num_samples < len(dataloader.dataset):
            indices = torch.randperm(len(dataloader.dataset))[:num_samples]
            subset = torch.utils.data.Subset(dataloader.dataset, indices)
            sample_loader = torch.utils.data.DataLoader(
                subset, batch_size=dataloader.batch_size, 
                shuffle=False, num_workers=dataloader.num_workers
            )
        
        for batch_idx, batch in enumerate(tqdm(sample_loader, desc="Comparing models")):
            # Process batch
            images, target = None, None
            
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                images, target = batch[0], batch[1]
            elif isinstance(batch, dict) and 'image' in batch and 'target' in batch:
                images, target = batch['image'], batch['target']
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")
            
            # Move to device
            images = images.to(device)
            if isinstance(target, torch.Tensor):
                target = target.to(device)
            elif isinstance(target, (tuple, list)):
                target = [t.to(device) if isinstance(t, torch.Tensor) else t for t in target]
            
            # Store targets
            targets.append(target)
            
            # Run FP32 model
            start_time = time.time()
            fp32_output = fp32_model(images)
            fp32_time = time.time() - start_time
            fp32_times.append(fp32_time)
            
            # Store FP32 predictions
            fp32_predictions.append(fp32_output)
            
            # Run INT8 model
            start_time = time.time()
            int8_output = int8_model(images)
            int8_time = time.time() - start_time
            int8_times.append(int8_time)
            
            # Store INT8 predictions
            int8_predictions.append(int8_output)
    
    # Remove hooks
    for hook in fp32_hooks:
        hook.remove()
    
    for hook in int8_hooks:
        hook.remove()
    
    # Calculate metrics
    if 'accuracy' in metrics:
        # Calculate accuracy metrics using predictions
        from ..metrics import compute_map, calculate_mean_average_precision
        
        # Determine number of classes from dataloader
        num_classes = dataloader.dataset.num_classes if hasattr(dataloader.dataset, 'num_classes') else 10
        
        # Calculate mAP for FP32 model
        fp32_map = compute_map(fp32_predictions, targets, num_classes)
        
        # Calculate mAP for INT8 model
        int8_map = compute_map(int8_predictions, targets, num_classes)
        
        # Calculate accuracy change
        map50_fp32 = fp32_map.get('mAP@.5', 0)
        map50_int8 = int8_map.get('mAP@.5', 0)
        absolute_change = map50_int8 - map50_fp32
        relative_change = absolute_change / map50_fp32 if map50_fp32 > 0 else 0
        
        # Store results
        results['accuracy_comparison'] = {
            'fp32_map': fp32_map,
            'int8_map': int8_map,
            'fp32_map50': map50_fp32,
            'int8_map50': map50_int8,
            'absolute_change': absolute_change,
            'relative_change': relative_change
        }
    
    if 'latency' in metrics:
        # Calculate latency metrics
        fp32_mean_time = np.mean(fp32_times)
        int8_mean_time = np.mean(int8_times)
        fp32_fps = 1.0 / fp32_mean_time
        int8_fps = 1.0 / int8_mean_time
        speedup = fp32_mean_time / int8_mean_time if int8_mean_time > 0 else 0
        
        # Store results
        results['latency_comparison'] = {
            'fp32_time': fp32_mean_time,
            'int8_time': int8_mean_time,
            'fp32_fps': fp32_fps,
            'int8_fps': int8_fps,
            'speedup': speedup,
            'fp32_times': fp32_times,
            'int8_times': int8_times
        }
    
    if 'output_error' in metrics:
        # Calculate output error metrics
        # Compare outputs from the first batch as example
        if len(fp32_predictions) > 0 and len(int8_predictions) > 0:
            if isinstance(fp32_predictions[0], torch.Tensor) and isinstance(int8_predictions[0], torch.Tensor):
                output_errors = []
                
                for fp32_pred, int8_pred in zip(fp32_predictions, int8_predictions):
                    # Handle different output formats
                    if isinstance(fp32_pred, (tuple, list)) and isinstance(int8_pred, (tuple, list)):
                        # Multiple outputs (e.g., boxes, scores, classes)
                        multi_errors = []
                        for fp32_out, int8_out in zip(fp32_pred, int8_pred):
                            if isinstance(fp32_out, torch.Tensor) and isinstance(int8_out, torch.Tensor):
                                # Calculate error
                                error = compute_output_error(fp32_out, int8_out)
                                multi_errors.append(error)
                        
                        if multi_errors:
                            # Average errors across multiple outputs
                            output_errors.append(np.mean(multi_errors))
                    elif isinstance(fp32_pred, torch.Tensor) and isinstance(int8_pred, torch.Tensor):
                        # Single output tensor
                        error = compute_output_error(fp32_pred, int8_pred)
                        output_errors.append(error)
                
                if output_errors:
                    # Calculate statistics of errors
                    mean_error = np.mean(output_errors)
                    max_error = np.max(output_errors)
                    std_error = np.std(output_errors)
                    
                    # Store results
                    results['output_comparison'] = {
                        'mean_error': mean_error,
                        'max_error': max_error,
                        'std_error': std_error,
                        'error_distribution': output_errors
                    }
    
    if 'layer_outputs' in metrics:
        # Calculate layer-wise differences
        common_layers = set(layer_outputs_fp32.keys()).intersection(set(layer_outputs_int8.keys()))
        layer_errors = {}
        
        for layer_name in common_layers:
            # Get outputs
            fp32_outputs = layer_outputs_fp32[layer_name]
            int8_outputs = layer_outputs_int8[layer_name]
            
            # Calculate errors for each batch
            errors = []
            for fp32_out, int8_out in zip(fp32_outputs, int8_outputs):
                error = compute_output_error(fp32_out, int8_out)
                errors.append(error)
            
            # Calculate statistics
            mean_error = np.mean(errors) if errors else 0
            max_error = np.max(errors) if errors else 0
            
            # Store results
            layer_errors[layer_name] = {
                'mean_error': mean_error,
                'max_error': max_error
            }
        
        # Store results
        results['layer_comparison'] = layer_errors
    
    return results

def compare_model_outputs(
    fp32_model: torch.nn.Module,
    int8_model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Compare outputs between FP32 and INT8 models for a single input.
    
    Args:
        fp32_model: Floating point model
        int8_model: Quantized model
        input_tensor: Input tensor
        device: Device to run inference on
        
    Returns:
        Dictionary with output comparison results
    """
    # Set models to evaluation mode
    fp32_model.eval()
    int8_model.eval()
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    fp32_model.to(device)
    int8_model.to(device)
    
    # Move input to device
    input_tensor = input_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        # FP32 model
        start_time = time.time()
        fp32_output = fp32_model(input_tensor)
        fp32_time = time.time() - start_time
        
        # INT8 model
        start_time = time.time()
        int8_output = int8_model(input_tensor)
        int8_time = time.time() - start_time
    
    # Calculate output error
    if isinstance(fp32_output, torch.Tensor) and isinstance(int8_output, torch.Tensor):
        # Single output tensor
        error = compute_output_error(fp32_output, int8_output)
    elif isinstance(fp32_output, (tuple, list)) and isinstance(int8_output, (tuple, list)):
        # Multiple outputs
        errors = []
        for fp32_out, int8_out in zip(fp32_output, int8_output):
            if isinstance(fp32_out, torch.Tensor) and isinstance(int8_out, torch.Tensor):
                errors.append(compute_output_error(fp32_out, int8_out))
        
        error = np.mean(errors) if errors else 0
    else:
        error = 0
    
    # Return results
    return {
        'fp32_output': fp32_output,
        'int8_output': int8_output,
        'output_error': error,
        'fp32_time': fp32_time,
        'int8_time': int8_time,
        'speedup': fp32_time / int8_time if int8_time > 0 else 0
    }

def compute_output_error(
    fp32_output: torch.Tensor,
    int8_output: torch.Tensor,
    error_type: str = "mse"
) -> float:
    """
    Compute error between FP32 and INT8 outputs.
    
    Args:
        fp32_output: Output tensor from FP32 model
        int8_output: Output tensor from INT8 model
        error_type: Type of error metric ('mse', 'mae', 'relative')
        
    Returns:
        Error value
    """
    # Ensure tensors are on same device and same dtype
    if fp32_output.device != int8_output.device:
        int8_output = int8_output.to(fp32_output.device)
    
    # Convert to float for consistent comparison
    fp32_output = fp32_output.float()
    int8_output = int8_output.float()
    
    # Compute error based on type
    if error_type == "mse":
        # Mean squared error
        return torch.mean((fp32_output - int8_output) ** 2).item()
    elif error_type == "mae":
        # Mean absolute error
        return torch.mean(torch.abs(fp32_output - int8_output)).item()
    elif error_type == "relative":
        # Relative error
        abs_diff = torch.abs(fp32_output - int8_output)
        abs_fp32 = torch.abs(fp32_output) + 1e-8  # Avoid division by zero
        return torch.mean(abs_diff / abs_fp32).item()
    else:
        raise ValueError(f"Unsupported error type: {error_type}")

def compare_layer_outputs(
    fp32_model: torch.nn.Module,
    int8_model: torch.nn.Module,
    input_tensor: torch.Tensor,
    layer_names: Optional[List[str]] = None,
    device: str = "cuda"
) -> Dict[str, Dict[str, float]]:
    """
    Compare outputs of specific layers between FP32 and INT8 models.
    
    Args:
        fp32_model: Floating point model
        int8_model: Quantized model
        input_tensor: Input tensor
        layer_names: Optional list of layer names to compare (all if None)
        device: Device to run inference on
        
    Returns:
        Dictionary mapping layer names to error metrics
    """
    # Set models to evaluation mode
    fp32_model.eval()
    int8_model.eval()
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    fp32_model.to(device)
    int8_model.to(device)
    
    # Move input to device
    input_tensor = input_tensor.to(device)
    
    # Storage for layer outputs
    fp32_outputs = {}
    int8_outputs = {}
    
    # Hooks to capture outputs
    fp32_hooks = []
    int8_hooks = []
    
    # Create hook function
    def create_hook(layer_name, output_dict):
        def hook(module, input, output):
            # Store output tensor
            if isinstance(output, torch.Tensor):
                output_dict[layer_name] = output.detach()
            elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                output_dict[layer_name] = output[0].detach()
        return hook
    
    # Register hooks for specified layers
    if layer_names is None:
        # Compare all layers
        for name, module in fp32_model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
                fp32_hooks.append(module.register_forward_hook(create_hook(name, fp32_outputs)))
        
        for name, module in int8_model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
                int8_hooks.append(module.register_forward_hook(create_hook(name, int8_outputs)))
    else:
        # Compare only specified layers
        for name, module in fp32_model.named_modules():
            if name in layer_names:
                fp32_hooks.append(module.register_forward_hook(create_hook(name, fp32_outputs)))
        
        for name, module in int8_model.named_modules():
            if name in layer_names:
                int8_hooks.append(module.register_forward_hook(create_hook(name, int8_outputs)))
    
    # Run inference
    with torch.no_grad():
        fp32_model(input_tensor)
        int8_model(input_tensor)
    
    # Remove hooks
    for hook in fp32_hooks:
        hook.remove()
    
    for hook in int8_hooks:
        hook.remove()
    
    # Calculate errors for each layer
    results = {}
    common_layers = set(fp32_outputs.keys()).intersection(set(int8_outputs.keys()))
    
    for layer_name in common_layers:
        fp32_output = fp32_outputs[layer_name]
        int8_output = int8_outputs[layer_name]
        
        # Calculate errors
        mse = compute_output_error(fp32_output, int8_output, "mse")
        mae = compute_output_error(fp32_output, int8_output, "mae")
        rel_error = compute_output_error(fp32_output, int8_output, "relative")
        
        # Store results
        results[layer_name] = {
            'mse': mse,
            'mae': mae,
            'relative_error': rel_error
        }
    
    return results

def export_comparison_report(
    comparison_results: Dict[str, Any],
    output_path: str = "./comparison_report",
    include_plots: bool = True
) -> str:
    """
    Export comparison results to a report.
    
    Args:
        comparison_results: Results from compare_fp32_int8_models
        output_path: Path to save the report
        include_plots: Whether to include plots in the report
        
    Returns:
        Path to the exported report
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Extract results
    accuracy_results = comparison_results.get('accuracy_comparison', {})
    latency_results = comparison_results.get('latency_comparison', {})
    output_results = comparison_results.get('output_comparison', {})
    layer_results = comparison_results.get('layer_comparison', {})
    
    # Generate plots if requested
    if include_plots:
        # Plot accuracy comparison
        if accuracy_results:
            # Extract class-wise AP@.5
            fp32_map = accuracy_results.get('fp32_map', {})
            int8_map = accuracy_results.get('int8_map', {})
            
            # Find class-specific AP values
            class_aps_fp32 = {}
            class_aps_int8 = {}
            
            for k, v in fp32_map.items():
                if k.startswith('AP@.5_class'):
                    class_idx = k.replace('AP@.5_class', '')
                    class_aps_fp32[f'Class {class_idx}'] = v
            
            for k, v in int8_map.items():
                if k.startswith('AP@.5_class'):
                    class_idx = k.replace('AP@.5_class', '')
                    class_aps_int8[f'Class {class_idx}'] = v
            
            # Plot class-wise AP comparison
            if class_aps_fp32 and class_aps_int8:
                plt.figure(figsize=(12, 6))
                
                classes = list(class_aps_fp32.keys())
                fp32_values = [class_aps_fp32.get(c, 0) for c in classes]
                int8_values = [class_aps_int8.get(c, 0) for c in classes]
                
                x = np.arange(len(classes))
                width = 0.35
                
                plt.bar(x - width/2, fp32_values, width, label='FP32')
                plt.bar(x + width/2, int8_values, width, label='INT8')
                
                plt.xlabel('Class')
                plt.ylabel('AP@.5')
                plt.title('AP@.5 Comparison by Class')
                plt.xticks(x, classes, rotation=45, ha='right')
                plt.legend()
                plt.tight_layout()
                
                ap_plot_path = os.path.join(output_path, 'ap_comparison.png')
                plt.savefig(ap_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        # Plot latency comparison
        if latency_results:
            fp32_times = latency_results.get('fp32_times', [])
            int8_times = latency_results.get('int8_times', [])
            
            if fp32_times and int8_times:
                plt.figure(figsize=(12, 6))
                
                plt.hist(np.array(fp32_times) * 1000, bins=30, alpha=0.7, label='FP32')
                plt.hist(np.array(int8_times) * 1000, bins=30, alpha=0.7, label='INT8')
                
                plt.xlabel('Inference Time (ms)')
                plt.ylabel('Frequency')
                plt.title('Inference Time Distribution')
                plt.legend()
                plt.tight_layout()
                
                latency_plot_path = os.path.join(output_path, 'latency_comparison.png')
                plt.savefig(latency_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        # Plot layer error comparison
        if layer_results:
            layer_names = list(layer_results.keys())
            mean_errors = [layer_results[name].get('mean_error', 0) for name in layer_names]
            
            # Sort by error magnitude
            sorted_indices = np.argsort(mean_errors)[::-1]  # Descending order
            sorted_layers = [layer_names[i] for i in sorted_indices]
            sorted_errors = [mean_errors[i] for i in sorted_indices]
            
            # Plot top N layers with highest error
            top_n = min(20, len(sorted_layers))
            
            plt.figure(figsize=(12, 8))
            plt.barh(np.arange(top_n), sorted_errors[:top_n], align='center')
            plt.yticks(np.arange(top_n), [l.split('.')[-1] for l in sorted_layers[:top_n]])
            plt.xlabel('Mean Error')
            plt.title('Layer-wise Quantization Error (Top 20)')
            plt.tight_layout()
            
            layer_plot_path = os.path.join(output_path, 'layer_error_comparison.png')
            plt.savefig(layer_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create report document
    report_path = os.path.join(output_path, 'comparison_report.json')
    
    # Ensure serializable results
    serializable_results = {}
    
    # Process accuracy results
    if accuracy_results:
        serializable_results['accuracy'] = {
            'fp32_map50': float(accuracy_results.get('fp32_map50', 0)),
            'int8_map50': float(accuracy_results.get('int8_map50', 0)),
            'absolute_change': float(accuracy_results.get('absolute_change', 0)),
            'relative_change': float(accuracy_results.get('relative_change', 0))
        }
    
    # Process latency results
    if latency_results:
        serializable_results['latency'] = {
            'fp32_time_ms': float(latency_results.get('fp32_time', 0) * 1000),
            'int8_time_ms': float(latency_results.get('int8_time', 0) * 1000),
            'fp32_fps': float(latency_results.get('fp32_fps', 0)),
            'int8_fps': float(latency_results.get('int8_fps', 0)),
            'speedup': float(latency_results.get('speedup', 0))
        }
    
    # Process output error results
    if output_results:
        serializable_results['output_error'] = {
            'mean_error': float(output_results.get('mean_error', 0)),
            'max_error': float(output_results.get('max_error', 0)),
            'std_error': float(output_results.get('std_error', 0))
        }
    
    # Process layer comparison results
    if layer_results:
        serializable_results['layer_errors'] = {}
        for layer_name, errors in layer_results.items():
            serializable_results['layer_errors'][layer_name] = {
                'mean_error': float(errors.get('mean_error', 0)),
                'max_error': float(errors.get('max_error', 0))
            }
    
    # Save report
    with open(report_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Comparison report saved to {report_path}")
    
    # Create CSV report for layer errors
    if layer_results:
        csv_path = os.path.join(output_path, 'layer_errors.csv')
        
        # Convert to DataFrame
        layer_data = []
        for layer_name, errors in layer_results.items():
            layer_data.append({
                'layer_name': layer_name,
                'mean_error': errors.get('mean_error', 0),
                'max_error': errors.get('max_error', 0)
            })
        
        # Create DataFrame and sort by mean error
        df = pd.DataFrame(layer_data)
        df = df.sort_values('mean_error', ascending=False)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Layer errors saved to {csv_path}")
    
    return report_path