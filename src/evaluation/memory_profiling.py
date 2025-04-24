# Analyzes memory usage
"""
Memory profiling utilities for YOLOv8 QAT evaluation.

This module provides functions for analyzing model memory usage,
comparing memory requirements, and profiling activation memory.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict

# Setup logging
logger = logging.getLogger(__name__)

def measure_model_size(
    model: torch.nn.Module,
    detailed: bool = True
) -> Dict[str, Any]:
    """
    Measure model size in memory.
    
    Args:
        model: Model to measure
        detailed: Whether to return detailed size breakdown by layer
        
    Returns:
        Dictionary with model size information
    """
    # Calculate total size
    total_size = 0
    layer_sizes = {}
    
    for name, param in model.named_parameters():
        param_size = param.nelement() * param.element_size()
        total_size += param_size
        
        if detailed:
            layer_sizes[name] = param_size
    
    # Add buffer sizes (e.g., running means and variances in BatchNorm)
    for name, buffer in model.named_buffers():
        buffer_size = buffer.nelement() * buffer.element_size()
        total_size += buffer_size
        
        if detailed:
            layer_sizes[name] = buffer_size
    
    # Convert to megabytes
    total_size_mb = total_size / (1024 * 1024)
    
    # Calculate parameter count
    param_count = sum(p.numel() for p in model.parameters())
    
    # Prepare results
    results = {
        'size_bytes': total_size,
        'size_mb': total_size_mb,
        'param_count': param_count,
        'param_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
        'buffer_size_mb': sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 * 1024)
    }
    
    if detailed:
        # Convert byte sizes to MB and sort by size
        layer_sizes_mb = {name: size / (1024 * 1024) for name, size in layer_sizes.items()}
        
        # Group by layer type
        layer_type_sizes = defaultdict(float)
        
        for name, size in layer_sizes_mb.items():
            # Extract layer type from name
            parts = name.split('.')
            layer_type = 'unknown'
            
            for part in parts:
                if any(t in part for t in ['conv', 'bn', 'linear', 'fc']):
                    layer_type = part
                    break
            
            layer_type_sizes[layer_type] += size
        
        results['layer_sizes_mb'] = layer_sizes_mb
        results['layer_type_sizes_mb'] = dict(layer_type_sizes)
    
    return results

def measure_memory_usage(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    device: str = "cuda",
    export_results: bool = True,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Measure memory usage during model inference.
    
    Args:
        model: Model to measure
        input_shape: Shape of input tensor
        device: Device to run measurement on
        export_results: Whether to export results to file
        output_path: Path to save results
        
    Returns:
        Dictionary with memory usage information
    """
    # Set output path
    if output_path is None:
        output_path = "./memory_profile"
    
    # Create output directory if exporting results
    if export_results:
        os.makedirs(output_path, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(device)
    
    # Get model size
    model_size = measure_model_size(model, detailed=True)
    
    # Initialize results
    results = {
        'model_size_mb': model_size['size_mb'],
        'param_count': model_size['param_count'],
        'param_size_mb': model_size['param_size_mb'],
        'buffer_size_mb': model_size['buffer_size_mb'],
        'layer_type_sizes_mb': model_size.get('layer_type_sizes_mb', {})
    }
    
    # Measure peak memory usage during inference
    if device.type == 'cuda' and torch.cuda.is_available():
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats(device)
        
        # Generate random input
        input_tensor = torch.rand(*input_shape, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_tensor)
        
        # Reset stats after warmup
        torch.cuda.reset_peak_memory_stats(device)
        
        # Run inference and measure peak memory
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Get peak memory usage
        peak_memory = torch.cuda.max_memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        
        # Convert to MB
        peak_memory_mb = peak_memory / (1024 * 1024)
        reserved_memory_mb = reserved_memory / (1024 * 1024)
        
        # Add to results
        results['peak_memory_mb'] = peak_memory_mb
        results['reserved_memory_mb'] = reserved_memory_mb
        results['activation_memory_mb'] = peak_memory_mb - model_size['size_mb']
    
    # Export results if requested
    if export_results:
        # Export to JSON
        json_path = os.path.join(output_path, 'memory_usage.json')
        
        # Create serializable version
        serializable_results = {
            'model_size_mb': float(results['model_size_mb']),
            'param_count': int(results['param_count']),
            'param_size_mb': float(results['param_size_mb']),
            'buffer_size_mb': float(results['buffer_size_mb']),
        }
        
        # Add layer type sizes
        if 'layer_type_sizes_mb' in results:
            serializable_results['layer_type_sizes_mb'] = {
                k: float(v) for k, v in results['layer_type_sizes_mb'].items()
            }
        
        # Add peak memory if available
        if 'peak_memory_mb' in results:
            serializable_results['peak_memory_mb'] = float(results['peak_memory_mb'])
            serializable_results['reserved_memory_mb'] = float(results['reserved_memory_mb'])
            serializable_results['activation_memory_mb'] = float(results['activation_memory_mb'])
        
        # Save to JSON
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Memory usage results saved to {json_path}")
        
        # Create bar chart of layer type sizes
        if 'layer_type_sizes_mb' in results:
            plt.figure(figsize=(10, 6))
            
            layer_types = list(results['layer_type_sizes_mb'].keys())
            sizes = [results['layer_type_sizes_mb'][t] for t in layer_types]
            
            # Sort by size
            sorted_indices = np.argsort(sizes)[::-1]  # Descending
            layer_types = [layer_types[i] for i in sorted_indices]
            sizes = [sizes[i] for i in sorted_indices]
            
            plt.bar(layer_types, sizes)
            plt.xlabel('Layer Type')
            plt.ylabel('Size (MB)')
            plt.title('Memory Usage by Layer Type')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            layer_sizes_plot_path = os.path.join(output_path, 'layer_type_sizes.png')
            plt.savefig(layer_sizes_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create pie chart of memory breakdown
        plt.figure(figsize=(10, 8))
        
        labels = ['Parameters', 'Buffers']
        sizes = [results['param_size_mb'], results['buffer_size_mb']]
        
        if 'activation_memory_mb' in results:
            labels.append('Activations')
            sizes.append(results['activation_memory_mb'])
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Memory Usage Breakdown')
        
        memory_breakdown_plot_path = os.path.join(output_path, 'memory_breakdown.png')
        plt.savefig(memory_breakdown_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

def profile_activation_memory(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    device: str = "cuda",
    export_results: bool = True,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Profile activation memory usage for each layer.
    
    Args:
        model: Model to profile
        input_shape: Shape of input tensor
        device: Device to run profiling on
        export_results: Whether to export results to file
        output_path: Path to save results
        
    Returns:
        Dictionary with activation memory usage by layer
    """
    # Set output path
    if output_path is None:
        output_path = "./activation_profile"
    
    # Create output directory if exporting results
    if export_results:
        os.makedirs(output_path, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(device)
    
    # Storage for output sizes
    output_sizes = {}
    
    # Register hooks to capture output sizes
    handles = []
    
    def output_hook(name):
        def hook(module, input, output):
            # Calculate output size
            if isinstance(output, torch.Tensor):
                size_bytes = output.nelement() * output.element_size()
                output_sizes[name] = size_bytes
            elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                size_bytes = sum(o.nelement() * o.element_size() for o in output if isinstance(o, torch.Tensor))
                output_sizes[name] = size_bytes
        
        return hook
    
    # Register hooks for all modules
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d, torch.nn.MaxPool2d)):
            handles.append(module.register_forward_hook(output_hook(name)))
    
    # Generate random input
    input_tensor = torch.rand(*input_shape, device=device)
    
    # Run inference
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Convert to MB and create layer profiles
    layer_profiles = []
    
    for name, size_bytes in output_sizes.items():
        size_mb = size_bytes / (1024 * 1024)
        
        # Get layer type
        layer_type = 'unknown'
        for part in name.split('.'):
            if any(t in part for t in ['conv', 'bn', 'linear', 'fc', 'pool']):
                layer_type = part
                break
        
        layer_profiles.append({
            'name': name,
            'type': layer_type,
            'activation_size_bytes': size_bytes,
            'activation_size_mb': size_mb
        })
    
    # Sort by activation size
    layer_profiles.sort(key=lambda x: x['activation_size_bytes'], reverse=True)
    
    # Calculate total activation memory
    total_activation_bytes = sum(p['activation_size_bytes'] for p in layer_profiles)
    total_activation_mb = total_activation_bytes / (1024 * 1024)
    
    # Prepare results
    results = {
        'layer_profiles': layer_profiles,
        'total_activation_bytes': total_activation_bytes,
        'total_activation_mb': total_activation_mb
    }
    
    # Export results if requested
    if export_results:
        # Export to CSV
        csv_path = os.path.join(output_path, 'activation_memory.csv')
        
        # Create DataFrame
        df = pd.DataFrame(layer_profiles)
        df = df.sort_values('activation_size_mb', ascending=False)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Activation memory profile saved to {csv_path}")
        
        # Create bar chart of top N layers with highest activation memory
        top_n = min(20, len(layer_profiles))
        
        plt.figure(figsize=(12, 8))
        
        top_layers = df.head(top_n)
        plt.barh(top_layers['name'], top_layers['activation_size_mb'])
        plt.xlabel('Activation Memory (MB)')
        plt.title(f'Top {top_n} Layers by Activation Memory')
        plt.tight_layout()
        
        activation_plot_path = os.path.join(output_path, 'top_activation_memory.png')
        plt.savefig(activation_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create pie chart of activation memory by layer type
        # Group by layer type
        layer_type_sizes = df.groupby('type')['activation_size_mb'].sum().reset_index()
        layer_type_sizes = layer_type_sizes.sort_values('activation_size_mb', ascending=False)
        
        plt.figure(figsize=(10, 8))
        
        plt.pie(layer_type_sizes['activation_size_mb'], labels=layer_type_sizes['type'], 
                autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Activation Memory by Layer Type')
        
        type_plot_path = os.path.join(output_path, 'activation_by_type.png')
        plt.savefig(type_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

def compare_memory_requirements(
    fp32_model: torch.nn.Module,
    int8_model: torch.nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    device: str = "cuda",
    export_results: bool = True,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare memory requirements of FP32 and INT8 models.
    
    Args:
        fp32_model: Floating point model
        int8_model: Quantized model
        input_shape: Shape of input tensor
        device: Device to run comparison on
        export_results: Whether to export results to file
        output_path: Path to save results
        
    Returns:
        Dictionary with memory requirement comparison
    """
    # Set output path
    if output_path is None:
        output_path = "./memory_comparison"
    
    # Create output directory if exporting results
    if export_results:
        os.makedirs(output_path, exist_ok=True)
    
    # Measure memory usage of FP32 model
    fp32_memory = measure_memory_usage(
        model=fp32_model,
        input_shape=input_shape,
        device=device,
        export_results=False
    )
    
    # Measure memory usage of INT8 model
    int8_memory = measure_memory_usage(
        model=int8_model,
        input_shape=input_shape,
        device=device,
        export_results=False
    )
    
    # Calculate memory savings
    model_size_reduction = fp32_memory['model_size_mb'] - int8_memory['model_size_mb']
    model_size_reduction_percent = 100 * model_size_reduction / fp32_memory['model_size_mb'] if fp32_memory['model_size_mb'] > 0 else 0
    
    # Calculate memory savings for activations if available
    activation_reduction = 0
    activation_reduction_percent = 0
    
    if 'activation_memory_mb' in fp32_memory and 'activation_memory_mb' in int8_memory:
        activation_reduction = fp32_memory['activation_memory_mb'] - int8_memory['activation_memory_mb']
        activation_reduction_percent = 100 * activation_reduction / fp32_memory['activation_memory_mb'] if fp32_memory['activation_memory_mb'] > 0 else 0
    
    # Calculate total memory savings if peak memory is available
    total_reduction = 0
    total_reduction_percent = 0
    
    if 'peak_memory_mb' in fp32_memory and 'peak_memory_mb' in int8_memory:
        total_reduction = fp32_memory['peak_memory_mb'] - int8_memory['peak_memory_mb']
        total_reduction_percent = 100 * total_reduction / fp32_memory['peak_memory_mb'] if fp32_memory['peak_memory_mb'] > 0 else 0
    
    # Prepare results
    results = {
        'fp32_model': {
            'model_size_mb': fp32_memory['model_size_mb'],
            'param_count': fp32_memory['param_count']
        },
        'int8_model': {
            'model_size_mb': int8_memory['model_size_mb'],
            'param_count': int8_memory['param_count']
        },
        'model_size_reduction_mb': model_size_reduction,
        'model_size_reduction_percent': model_size_reduction_percent,
        'compression_ratio': fp32_memory['model_size_mb'] / int8_memory['model_size_mb'] if int8_memory['model_size_mb'] > 0 else 0
    }
    
    # Add activation memory if available
    if 'activation_memory_mb' in fp32_memory and 'activation_memory_mb' in int8_memory:
        results['fp32_model']['activation_memory_mb'] = fp32_memory['activation_memory_mb']
        results['int8_model']['activation_memory_mb'] = int8_memory['activation_memory_mb']
        results['activation_reduction_mb'] = activation_reduction
        results['activation_reduction_percent'] = activation_reduction_percent
    
    # Add peak memory if available
    if 'peak_memory_mb' in fp32_memory and 'peak_memory_mb' in int8_memory:
        results['fp32_model']['peak_memory_mb'] = fp32_memory['peak_memory_mb']
        results['int8_model']['peak_memory_mb'] = int8_memory['peak_memory_mb']
        results['total_reduction_mb'] = total_reduction
        results['total_reduction_percent'] = total_reduction_percent
    
    # Export results if requested
    if export_results:
        # Export to JSON
        json_path = os.path.join(output_path, 'memory_comparison.json')
        
        # Create serializable version
        serializable_results = {
            'fp32_model': {
                'model_size_mb': float(results['fp32_model']['model_size_mb']),
                'param_count': int(results['fp32_model']['param_count'])
            },
            'int8_model': {
                'model_size_mb': float(results['int8_model']['model_size_mb']),
                'param_count': int(results['int8_model']['param_count'])
            },
            'model_size_reduction_mb': float(results['model_size_reduction_mb']),
            'model_size_reduction_percent': float(results['model_size_reduction_percent']),
            'compression_ratio': float(results['compression_ratio'])
        }
        
        # Add activation memory if available
        if 'activation_memory_mb' in results['fp32_model'] and 'activation_memory_mb' in results['int8_model']:
            serializable_results['fp32_model']['activation_memory_mb'] = float(results['fp32_model']['activation_memory_mb'])
            serializable_results['int8_model']['activation_memory_mb'] = float(results['int8_model']['activation_memory_mb'])
            serializable_results['activation_reduction_mb'] = float(results['activation_reduction_mb'])
            serializable_results['activation_reduction_percent'] = float(results['activation_reduction_percent'])
        
        # Add peak memory if available
        if 'peak_memory_mb' in results['fp32_model'] and 'peak_memory_mb' in results['int8_model']:
            serializable_results['fp32_model']['peak_memory_mb'] = float(results['fp32_model']['peak_memory_mb'])
            serializable_results['int8_model']['peak_memory_mb'] = float(results['int8_model']['peak_memory_mb'])
            serializable_results['total_reduction_mb'] = float(results['total_reduction_mb'])
            serializable_results['total_reduction_percent'] = float(results['total_reduction_percent'])
        
        # Save to JSON
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Memory comparison results saved to {json_path}")
        
        # Create bar chart comparing model sizes
        plt.figure(figsize=(10, 6))
        
        models = ['FP32', 'INT8']
        sizes = [results['fp32_model']['model_size_mb'], results['int8_model']['model_size_mb']]
        
        plt.bar(models, sizes, color=['blue', 'green'])
        plt.ylabel('Model Size (MB)')
        plt.title('Model Size Comparison')
        
        # Add text annotations
        reduction_text = f"Reduction: {model_size_reduction:.2f} MB ({model_size_reduction_percent:.2f}%)"
        compression_text = f"Compression Ratio: {results['compression_ratio']:.2f}x"
        
        plt.figtext(0.5, 0.01, reduction_text + '\n' + compression_text, 
                   ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        model_size_plot_path = os.path.join(output_path, 'model_size_comparison.png')
        plt.savefig(model_size_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create bar chart comparing peak memory if available
        if 'peak_memory_mb' in results['fp32_model'] and 'peak_memory_mb' in results['int8_model']:
            plt.figure(figsize=(10, 6))
            
            sizes = [results['fp32_model']['peak_memory_mb'], results['int8_model']['peak_memory_mb']]
            
            plt.bar(models, sizes, color=['blue', 'green'])
            plt.ylabel('Peak Memory Usage (MB)')
            plt.title('Peak Memory Usage Comparison')
            
            # Add text annotations
            reduction_text = f"Reduction: {total_reduction:.2f} MB ({total_reduction_percent:.2f}%)"
            
            plt.figtext(0.5, 0.01, reduction_text, 
                       ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            peak_memory_plot_path = os.path.join(output_path, 'peak_memory_comparison.png')
            plt.savefig(peak_memory_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create stacked bar chart showing memory breakdown
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        categories = ['Parameters', 'Buffers']
        fp32_sizes = [
            results['fp32_model']['param_size_mb'] if 'param_size_mb' in fp32_memory else 0,
            results['fp32_model']['buffer_size_mb'] if 'buffer_size_mb' in fp32_memory else 0
        ]
        
        int8_sizes = [
            results['int8_model']['param_size_mb'] if 'param_size_mb' in int8_memory else 0,
            results['int8_model']['buffer_size_mb'] if 'buffer_size_mb' in int8_memory else 0
        ]
        
        # Add activation memory if available
        if 'activation_memory_mb' in results['fp32_model'] and 'activation_memory_mb' in results['int8_model']:
            categories.append('Activations')
            fp32_sizes.append(results['fp32_model']['activation_memory_mb'])
            int8_sizes.append(results['int8_model']['activation_memory_mb'])
        
        # Create stacked bar chart
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bottom_fp32 = 0
        bottom_int8 = 0
        
        for i, (category, fp32_size, int8_size) in enumerate(zip(categories, fp32_sizes, int8_sizes)):
            ax.bar(x[0], fp32_size, width, bottom=bottom_fp32, label=f"{category} (FP32)" if i == 0 else None)
            ax.bar(x[1], int8_size, width, bottom=bottom_int8, label=f"{category} (INT8)" if i == 0 else None)
            
            # Update bottom position
            bottom_fp32 += fp32_size
            bottom_int8 += int8_size
        
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Usage Breakdown')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        
        breakdown_plot_path = os.path.join(output_path, 'memory_breakdown_comparison.png')
        plt.savefig(breakdown_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

def export_memory_profile(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
    report_format: str = "markdown"
) -> str:
    """
    Export memory profile results to a report.
    
    Args:
        results: Memory profiling results
        output_path: Path to save report
        report_format: Format of the report ('markdown', 'html', 'json')
        
    Returns:
        Path to the exported report
    """
    # Set output path
    if output_path is None:
        output_path = "./memory_profile"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize report content
    report_content = ""
    
    if report_format == "markdown":
        # Create Markdown report
        report_content += "# Memory Profile Report\n\n"
        
        # Check if this is a comparison between FP32 and INT8
        is_comparison = 'fp32_model' in results and 'int8_model' in results
        
        if is_comparison:
            # Model comparison section
            report_content += "## Model Comparison\n\n"
            
            report_content += "| Metric | FP32 Model | INT8 Model | Reduction | Reduction (%) |\n"
            report_content += "|--------|-----------|-----------|-----------|---------------|\n"
            
            # Model size
            fp32_size = results['fp32_model'].get('model_size_mb', 0)
            int8_size = results['int8_model'].get('model_size_mb', 0)
            size_reduction = results.get('model_size_reduction_mb', fp32_size - int8_size)
            size_reduction_percent = results.get('model_size_reduction_percent', 0)
            
            report_content += f"| Model Size | {fp32_size:.2f} MB | {int8_size:.2f} MB | {size_reduction:.2f} MB | {size_reduction_percent:.2f}% |\n"
            
            # Parameter count
            fp32_params = results['fp32_model'].get('param_count', 0)
            int8_params = results['int8_model'].get('param_count', 0)
            
            report_content += f"| Parameters | {fp32_params:,} | {int8_params:,} | - | - |\n"
            
            # Compression ratio
            compression_ratio = results.get('compression_ratio', 0)
            report_content += f"| Compression Ratio | - | - | {compression_ratio:.2f}x | - |\n"
            
            # Add peak memory if available
            if 'peak_memory_mb' in results['fp32_model'] and 'peak_memory_mb' in results['int8_model']:
                fp32_peak = results['fp32_model'].get('peak_memory_mb', 0)
                int8_peak = results['int8_model'].get('peak_memory_mb', 0)
                peak_reduction = results.get('total_reduction_mb', fp32_peak - int8_peak)
                peak_reduction_percent = results.get('total_reduction_percent', 0)
                
                report_content += f"| Peak Memory | {fp32_peak:.2f} MB | {int8_peak:.2f} MB | {peak_reduction:.2f} MB | {peak_reduction_percent:.2f}% |\n"
            
            # Add activation memory if available
            if 'activation_memory_mb' in results['fp32_model'] and 'activation_memory_mb' in results['int8_model']:
                fp32_act = results['fp32_model'].get('activation_memory_mb', 0)
                int8_act = results['int8_model'].get('activation_memory_mb', 0)
                act_reduction = results.get('activation_reduction_mb', fp32_act - int8_act)
                act_reduction_percent = results.get('activation_reduction_percent', 0)
                
                report_content += f"| Activation Memory | {fp32_act:.2f} MB | {int8_act:.2f} MB | {act_reduction:.2f} MB | {act_reduction_percent:.2f}% |\n"
            
            report_content += "\n"
            
            # Add visualization references
            report_content += "## Visualizations\n\n"
            report_content += "### Model Size Comparison\n\n"
            report_content += "![Model Size Comparison](model_size_comparison.png)\n\n"
            
            if 'peak_memory_mb' in results['fp32_model'] and 'peak_memory_mb' in results['int8_model']:
                report_content += "### Peak Memory Usage Comparison\n\n"
                report_content += "![Peak Memory Usage Comparison](peak_memory_comparison.png)\n\n"
            
            report_content += "### Memory Usage Breakdown\n\n"
            report_content += "![Memory Usage Breakdown](memory_breakdown_comparison.png)\n\n"
        
        else:
            # Single model profile
            report_content += "## Model Profile\n\n"
            
            report_content += "| Metric | Value |\n"
            report_content += "|--------|-------|\n"
            
            # Model size
            model_size = results.get('model_size_mb', 0)
            report_content += f"| Model Size | {model_size:.2f} MB |\n"
            
            # Parameter count
            param_count = results.get('param_count', 0)
            report_content += f"| Parameters | {param_count:,} |\n"
            
            # Parameter size
            param_size = results.get('param_size_mb', 0)
            report_content += f"| Parameter Size | {param_size:.2f} MB |\n"
            
            # Buffer size
            buffer_size = results.get('buffer_size_mb', 0)
            report_content += f"| Buffer Size | {buffer_size:.2f} MB |\n"
            
            # Add peak memory if available
            if 'peak_memory_mb' in results:
                peak_memory = results.get('peak_memory_mb', 0)
                report_content += f"| Peak Memory | {peak_memory:.2f} MB |\n"
            
            # Add activation memory if available
            if 'activation_memory_mb' in results:
                activation_memory = results.get('activation_memory_mb', 0)
                report_content += f"| Activation Memory | {activation_memory:.2f} MB |\n"
            
            report_content += "\n"
            
            # Add layer type sizes if available
            if 'layer_type_sizes_mb' in results:
                report_content += "## Memory Usage by Layer Type\n\n"
                
                report_content += "| Layer Type | Size (MB) |\n"
                report_content += "|-----------|----------|\n"
                
                layer_type_sizes = results['layer_type_sizes_mb']
                for layer_type, size in sorted(layer_type_sizes.items(), key=lambda x: x[1], reverse=True):
                    report_content += f"| {layer_type} | {size:.2f} |\n"
                
                report_content += "\n"
            
            # Add visualization references
            report_content += "## Visualizations\n\n"
            
            if 'layer_type_sizes_mb' in results:
                report_content += "### Memory Usage by Layer Type\n\n"
                report_content += "![Memory Usage by Layer Type](layer_type_sizes.png)\n\n"
            
            report_content += "### Memory Usage Breakdown\n\n"
            report_content += "![Memory Usage Breakdown](memory_breakdown.png)\n\n"
            
            # Add activation profile if available
            if 'layer_profiles' in results:
                report_content += "## Top Layers by Activation Memory\n\n"
                
                report_content += "| Layer | Type | Activation Size (MB) |\n"
                report_content += "|-------|------|---------------------|\n"
                
                # Sort by activation size and get top 10
                sorted_profiles = sorted(results['layer_profiles'], key=lambda x: x['activation_size_mb'], reverse=True)
                top_profiles = sorted_profiles[:10]
                
                for profile in top_profiles:
                    report_content += f"| {profile['name']} | {profile['type']} | {profile['activation_size_mb']:.2f} |\n"
                
                report_content += "\n"
                
                # Add visualization references
                report_content += "### Top Layers by Activation Memory\n\n"
                report_content += "![Top Layers by Activation Memory](top_activation_memory.png)\n\n"
                
                report_content += "### Activation Memory by Layer Type\n\n"
                report_content += "![Activation Memory by Layer Type](activation_by_type.png)\n\n"
        
        # Write report to file
        report_path = os.path.join(output_path, "memory_profile.md")
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Memory profile report saved to {report_path}")
        
        # Try to convert to HTML if requested
        if report_format == "html":
            try:
                import markdown
                html_content = markdown.markdown(report_content, extensions=['tables'])
                
                html_path = os.path.join(output_path, "memory_profile.html")
                with open(html_path, 'w') as f:
                    f.write(f"<!DOCTYPE html>\n<html>\n<head>\n")
                    f.write(f"<title>Memory Profile Report</title>\n")
                    f.write(f"<style>\n")
                    f.write(f"body {{ font-family: Arial, sans-serif; margin: 20px; }}\n")
                    f.write(f"table {{ border-collapse: collapse; width: 100%; }}\n")
                    f.write(f"th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}\n")
                    f.write(f"th {{ background-color: #f2f2f2; }}\n")
                    f.write(f"img {{ max-width: 100%; }}\n")
                    f.write(f"</style>\n</head>\n<body>\n")
                    f.write(html_content)
                    f.write(f"\n</body>\n</html>")
                
                logger.info(f"HTML report saved to {html_path}")
                return html_path
            except ImportError:
                logger.warning("markdown module not found, falling back to markdown report")
        
        return report_path
    
    elif report_format == "json":
        # Create JSON report
        json_path = os.path.join(output_path, "memory_profile.json")
        
        # Save results as JSON
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        
        logger.info(f"JSON report saved to {json_path}")
        return json_path
    
    else:
        raise ValueError(f"Unsupported report format: {report_format}")