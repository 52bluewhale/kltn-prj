# Measures inference speed
"""
Latency testing utilities for YOLOv8 QAT evaluation.

This module provides functions for measuring inference speed and 
benchmarking models on different hardware.
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from tqdm import tqdm
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

# Setup logging
logger = logging.getLogger(__name__)

def measure_inference_time(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Measure inference time of a model.
    
    Args:
        model: Model to benchmark
        input_shape: Shape of input tensor
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs
        device: Device to run inference on
        
    Returns:
        Dictionary with inference time statistics
    """
    # Set model to evaluation mode
    model.eval()
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(device)
    
    # Generate random input
    input_tensor = torch.rand(*input_shape).to(device)
    
    # Warm up the GPU if using CUDA
    logger.info(f"Performing {warmup_runs} warmup runs...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Measure inference time
    logger.info(f"Measuring inference time over {num_runs} runs...")
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            # Synchronize CUDA operations before timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(input_tensor)
            
            # Synchronize CUDA operations after timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    # Calculate statistics
    mean_time = np.mean(times)
    median_time = np.median(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    p95_time = np.percentile(times, 95)
    
    # Calculate frames per second
    fps = 1.0 / mean_time
    
    # Return results
    return {
        'mean_inference_time': mean_time,
        'median_inference_time': median_time,
        'std_inference_time': std_time,
        'min_inference_time': min_time,
        'max_inference_time': max_time,
        'p95_inference_time': p95_time,
        'fps': fps,
        'all_times': times
    }

def benchmark_model(
    model: torch.nn.Module,
    input_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]] = (1, 3, 640, 640),
    num_runs: int = 100,
    device: str = "cuda",
    measure_memory: bool = True,
    export_results: bool = True,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Benchmark model inference performance.
    
    Args:
        model: Model to benchmark
        input_shape: Shape of input tensor or list of shapes for benchmarking different sizes
        num_runs: Number of inference runs
        device: Device to run inference on
        measure_memory: Whether to measure memory usage
        export_results: Whether to export results to file
        output_path: Path to save results
        
    Returns:
        Dictionary with benchmark results
    """
    # Set output path
    if output_path is None:
        output_path = "./benchmark_results"
    
    # Create output directory if exporting results
    if export_results:
        os.makedirs(output_path, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(device)
    
    # Initialize results
    results = {
        'model_name': model.__class__.__name__,
        'device': str(device),
        'input_shapes': [],
        'latency': [],
        'throughput': []
    }
    
    # Add model complexity metrics
    if hasattr(model, 'named_parameters'):
        num_params = sum(p.numel() for p in model.parameters())
        results['num_parameters'] = num_params
    
    # Add device information
    if device.type == 'cuda' and torch.cuda.is_available():
        results['gpu_name'] = torch.cuda.get_device_name(device)
        results['cuda_version'] = torch.version.cuda
    
    # Convert single input shape to list
    if isinstance(input_shape, tuple):
        input_shapes = [input_shape]
    else:
        input_shapes = input_shape
    
    # Benchmark each input shape
    for shape in input_shapes:
        # Generate random input
        input_tensor = torch.rand(*shape).to(device)
        
        # Measure inference time
        latency_results = measure_inference_time(
            model=model,
            input_shape=shape,
            num_runs=num_runs,
            device=device
        )
        
        # Append results
        results['input_shapes'].append(shape)
        results['latency'].append(latency_results)
        results['throughput'].append(latency_results['fps'])
    
    # Measure memory usage if requested
    if measure_memory:
        memory_results = {}
        
        # Use largest input shape for memory measurement
        largest_shape = max(input_shapes, key=lambda s: np.prod(s))
        input_tensor = torch.rand(*largest_shape).to(device)
        
        if device.type == 'cuda' and torch.cuda.is_available():
            # Measure CUDA memory usage
            torch.cuda.reset_peak_memory_stats(device)
            
            # Run inference to measure peak memory
            with torch.no_grad():
                _ = model(input_tensor)
            
            # Get peak memory usage
            memory_results['peak_memory_mb'] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            memory_results['reserved_memory_mb'] = torch.cuda.memory_reserved(device) / (1024 * 1024)
        
        # Estimate model size
        model_size_bytes = 0
        for param in model.parameters():
            model_size_bytes += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            model_size_bytes += buffer.nelement() * buffer.element_size()
        
        memory_results['model_size_mb'] = model_size_bytes / (1024 * 1024)
        
        # Add memory results
        results['memory'] = memory_results
    
    # Export results if requested
    if export_results:
        # Export to JSON
        json_path = os.path.join(output_path, 'benchmark_results.json')
        
        # Create serializable copy
        serializable_results = {
            'model_name': results['model_name'],
            'device': results['device'],
            'num_parameters': results.get('num_parameters', 0),
            'gpu_name': results.get('gpu_name', 'N/A'),
            'cuda_version': results.get('cuda_version', 'N/A'),
            'input_shapes': [list(shape) for shape in results['input_shapes']],
            'throughput': results['throughput'],
            'latency': []
        }
        
        # Process latency results to make them serializable
        for latency in results['latency']:
            serializable_latency = {
                'mean_inference_time': float(latency['mean_inference_time']),
                'median_inference_time': float(latency['median_inference_time']),
                'std_inference_time': float(latency['std_inference_time']),
                'min_inference_time': float(latency['min_inference_time']),
                'max_inference_time': float(latency['max_inference_time']),
                'p95_inference_time': float(latency['p95_inference_time']),
                'fps': float(latency['fps'])
            }
            serializable_results['latency'].append(serializable_latency)
        
        # Add memory results if available
        if 'memory' in results:
            serializable_results['memory'] = {
                'model_size_mb': float(results['memory'].get('model_size_mb', 0)),
                'peak_memory_mb': float(results['memory'].get('peak_memory_mb', 0)),
                'reserved_memory_mb': float(results['memory'].get('reserved_memory_mb', 0))
            }
        
        # Save to JSON
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {json_path}")
        
        # Create plots
        # Latency plot
        plt.figure(figsize=(10, 6))
        
        shape_labels = [f"{shape[0]}x{shape[2]}x{shape[3]}" for shape in results['input_shapes']]
        mean_times = [lat['mean_inference_time'] * 1000 for lat in results['latency']]  # Convert to ms
        
        plt.bar(shape_labels, mean_times)
        plt.ylabel('Inference Time (ms)')
        plt.xlabel('Input Shape')
        plt.title('Inference Time by Input Shape')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        latency_plot_path = os.path.join(output_path, 'latency_plot.png')
        plt.savefig(latency_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # FPS plot
        plt.figure(figsize=(10, 6))
        
        fps_values = [lat['fps'] for lat in results['latency']]
        
        plt.bar(shape_labels, fps_values)
        plt.ylabel('Frames Per Second (FPS)')
        plt.xlabel('Input Shape')
        plt.title('Throughput by Input Shape')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        fps_plot_path = os.path.join(output_path, 'throughput_plot.png')
        plt.savefig(fps_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

def profile_model_layers(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    device: str = "cuda",
    export_results: bool = True,
    output_path: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Profile execution time of model layers.
    
    Args:
        model: Model to profile
        input_shape: Shape of input tensor
        device: Device to run profiling on
        export_results: Whether to export results to file
        output_path: Path to save results
        
    Returns:
        Dictionary with profiling results
    """
    # Set output path
    if output_path is None:
        output_path = "./profile_results"
    
    # Create output directory if exporting results
    if export_results:
        os.makedirs(output_path, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(device)
    
    # Generate random input
    input_tensor = torch.rand(*input_shape).to(device)
    
    # Storage for layer times
    layer_times = {}
    layer_types = {}
    
    # Hooks to measure execution time
    handles = []
    
    def create_hook(name):
        def hook(module, input, output):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            # Recompute forward pass
            with torch.no_grad():
                _ = module(*input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            # Store execution time
            if name not in layer_times:
                layer_times[name] = []
            
            layer_times[name].append(end_time - start_time)
            layer_types[name] = module.__class__.__name__
        
        return hook
    
    # Register hooks for all modules
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.MaxPool2d, nn.AvgPool2d)):
            handles.append(module.register_forward_hook(create_hook(name)))
    
    # Run inference multiple times to get more stable measurements
    num_runs = 10
    logger.info(f"Profiling model layers over {num_runs} runs...")
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Calculate average time for each layer
    layer_profiles = []
    
    for name, times in layer_times.items():
        avg_time = np.mean(times)
        total_time = np.sum(times)
        
        # Compute parameter count for the layer
        layer = dict(model.named_modules())[name]
        num_params = sum(p.numel() for p in layer.parameters())
        
        layer_profiles.append({
            'name': name,
            'type': layer_types[name],
            'avg_time': avg_time,
            'total_time': total_time,
            'parameters': num_params
        })
    
    # Sort by total execution time
    layer_profiles.sort(key=lambda x: x['total_time'], reverse=True)
    
    # Export results if requested
    if export_results:
        # Export to CSV
        csv_path = os.path.join(output_path, 'layer_profile.csv')
        
        # Create DataFrame
        df = pd.DataFrame(layer_profiles)
        df['avg_time_ms'] = df['avg_time'] * 1000  # Convert to ms
        df['total_time_ms'] = df['total_time'] * 1000  # Convert to ms
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Layer profile saved to {csv_path}")
        
        # Create bar chart of top N slowest layers
        top_n = min(20, len(layer_profiles))
        
        plt.figure(figsize=(12, 8))
        
        top_layers = df.sort_values('total_time_ms', ascending=False).head(top_n)
        plt.barh(top_layers['name'], top_layers['total_time_ms'])
        plt.xlabel('Total Execution Time (ms)')
        plt.title(f'Top {top_n} Slowest Layers')
        plt.tight_layout()
        
        profile_plot_path = os.path.join(output_path, 'layer_profile_plot.png')
        plt.savefig(profile_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return {
        'layer_profiles': layer_profiles,
        'total_time': sum(profile['total_time'] for profile in layer_profiles)
    }

def measure_throughput(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    batch_sizes: List[int] = [1, 2, 4, 8, 16],
    device: str = "cuda",
    duration: float = 5.0,
    export_results: bool = True,
    output_path: Optional[str] = None
) -> Dict[str, List[Dict[str, float]]]:
    """
    Measure model throughput across different batch sizes.
    
    Args:
        model: Model to benchmark
        input_shape: Shape of input tensor (excluding batch dimension)
        batch_sizes: List of batch sizes to test
        device: Device to run benchmarking on
        duration: Duration in seconds for each batch size test
        export_results: Whether to export results to file
        output_path: Path to save results
        
    Returns:
        Dictionary with throughput results
    """
    # Set output path
    if output_path is None:
        output_path = "./throughput_results"
    
    # Create output directory if exporting results
    if export_results:
        os.makedirs(output_path, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(device)
    
    # Initialize results
    results = []
    
    # Test each batch size
    for batch_size in batch_sizes:
        logger.info(f"Measuring throughput with batch size {batch_size}...")
        
        # Create input with the current batch size
        batch_shape = (batch_size,) + input_shape[1:]  # Replace batch dimension
        input_tensor = torch.rand(*batch_shape).to(device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Measure throughput
        num_iterations = 0
        start_time = time.time()
        
        with torch.no_grad():
            while time.time() - start_time < duration:
                _ = model(input_tensor)
                num_iterations += 1
                
                # Ensure we don't run indefinitely if something goes wrong
                if num_iterations > 10000:
                    break
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate throughput
        samples_per_second = (num_iterations * batch_size) / elapsed_time
        batches_per_second = num_iterations / elapsed_time
        
        # Calculate latency
        latency_per_batch = elapsed_time / num_iterations
        latency_per_sample = elapsed_time / (num_iterations * batch_size)
        
        # Store results
        results.append({
            'batch_size': batch_size,
            'samples_per_second': samples_per_second,
            'batches_per_second': batches_per_second,
            'latency_per_batch': latency_per_batch,
            'latency_per_sample': latency_per_sample,
            'iterations': num_iterations
        })
    
    # Export results if requested
    if export_results:
        # Export to CSV
        csv_path = os.path.join(output_path, 'throughput_results.csv')
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Add ms versions of latency
        df['latency_per_batch_ms'] = df['latency_per_batch'] * 1000
        df['latency_per_sample_ms'] = df['latency_per_sample'] * 1000
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Throughput results saved to {csv_path}")
        
        # Create throughput plot
        plt.figure(figsize=(12, 6))
        
        plt.plot(df['batch_size'], df['samples_per_second'], marker='o', linewidth=2)
        plt.xlabel('Batch Size')
        plt.ylabel('Samples per Second')
        plt.title('Throughput vs Batch Size')
        plt.grid(True)
        
        throughput_plot_path = os.path.join(output_path, 'throughput_plot.png')
        plt.savefig(throughput_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create latency plot
        plt.figure(figsize=(12, 6))
        
        plt.plot(df['batch_size'], df['latency_per_batch_ms'], marker='o', linewidth=2, label='Per Batch')
        plt.plot(df['batch_size'], df['latency_per_sample_ms'], marker='s', linewidth=2, label='Per Sample')
        plt.xlabel('Batch Size')
        plt.ylabel('Latency (ms)')
        plt.title('Latency vs Batch Size')
        plt.legend()
        plt.grid(True)
        
        latency_plot_path = os.path.join(output_path, 'latency_plot.png')
        plt.savefig(latency_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return {'throughput_results': results}

def export_benchmark_results(
    results: Dict[str, Any],
    output_path: str = "./benchmark_results",
    report_format: str = "markdown"
) -> str:
    """
    Export benchmark results to a report.
    
    Args:
        results: Benchmark results
        output_path: Path to save report
        report_format: Format of the report ('markdown', 'html', 'json')
        
    Returns:
        Path to the exported report
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize report content
    report_content = ""
    
    if report_format == "markdown":
        # Create Markdown report
        report_content += "# YOLOv8 Model Benchmark Report\n\n"
        
        # Model information
        report_content += "## Model Information\n\n"
        report_content += f"- Model name: {results.get('model_name', 'N/A')}\n"
        report_content += f"- Parameters: {results.get('num_parameters', 0):,}\n"
        report_content += f"- Device: {results.get('device', 'N/A')}\n"
        
        if 'gpu_name' in results:
            report_content += f"- GPU: {results.get('gpu_name', 'N/A')}\n"
        
        report_content += "\n"
        
        # Latency results
        if 'latency' in results and len(results['latency']) > 0:
            report_content += "## Latency Results\n\n"
            report_content += "| Input Shape | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | P95 (ms) | FPS |\n"
            report_content += "|------------|-----------|-------------|---------|---------|---------|-------|\n"
            
            for i, shape in enumerate(results.get('input_shapes', [])):
                latency = results['latency'][i]
                shape_str = f"{shape[0]}x{shape[2]}x{shape[3]}"
                
                report_content += f"| {shape_str} | "
                report_content += f"{latency['mean_inference_time']*1000:.2f} | "
                report_content += f"{latency['median_inference_time']*1000:.2f} | "
                report_content += f"{latency['min_inference_time']*1000:.2f} | "
                report_content += f"{latency['max_inference_time']*1000:.2f} | "
                report_content += f"{latency['p95_inference_time']*1000:.2f} | "
                report_content += f"{latency['fps']:.2f} |\n"
            
            report_content += "\n"
        
        # Memory usage
        if 'memory' in results:
            report_content += "## Memory Usage\n\n"
            report_content += f"- Model size: {results['memory'].get('model_size_mb', 0):.2f} MB\n"
            
            if 'peak_memory_mb' in results['memory']:
                report_content += f"- Peak GPU memory: {results['memory'].get('peak_memory_mb', 0):.2f} MB\n"
            
            if 'reserved_memory_mb' in results['memory']:
                report_content += f"- Reserved GPU memory: {results['memory'].get('reserved_memory_mb', 0):.2f} MB\n"
            
            report_content += "\n"
        
        # Throughput results
        if 'throughput_results' in results:
            report_content += "## Throughput Results\n\n"
            report_content += "| Batch Size | Samples/s | Batches/s | Latency/Batch (ms) | Latency/Sample (ms) |\n"
            report_content += "|------------|-----------|-----------|-------------------|------------------|\n"
            
            for result in results['throughput_results']:
                report_content += f"| {result['batch_size']} | "
                report_content += f"{result['samples_per_second']:.2f} | "
                report_content += f"{result['batches_per_second']:.2f} | "
                report_content += f"{result['latency_per_batch']*1000:.2f} | "
                report_content += f"{result['latency_per_sample']*1000:.2f} |\n"
            
            report_content += "\n"
        
        # Layer profiling
        if 'layer_profiles' in results:
            report_content += "## Layer Profiling\n\n"
            report_content += "Top 10 slowest layers:\n\n"
            report_content += "| Layer | Type | Time (ms) | Parameters |\n"
            report_content += "|------|------|-----------|------------|\n"
            
            # Sort by total time and show top 10
            sorted_layers = sorted(results['layer_profiles'], key=lambda x: x['total_time'], reverse=True)[:10]
            
            for layer in sorted_layers:
                report_content += f"| {layer['name']} | "
                report_content += f"{layer['type']} | "
                report_content += f"{layer['total_time']*1000:.2f} | "
                report_content += f"{layer['parameters']:,} |\n"
            
            report_content += "\n"
        
        # Write to file
        report_path = os.path.join(output_path, "benchmark_report.md")
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Markdown report saved to {report_path}")
        
        # Try to convert to HTML if requested
        if report_format == "html":
            try:
                import markdown
                html_content = markdown.markdown(report_content, extensions=['tables'])
                
                html_path = os.path.join(output_path, "benchmark_report.html")
                with open(html_path, 'w') as f:
                    f.write(f"<!DOCTYPE html>\n<html>\n<head>\n")
                    f.write(f"<title>YOLOv8 Model Benchmark Report</title>\n")
                    f.write(f"<style>\n")
                    f.write(f"body {{ font-family: Arial, sans-serif; margin: 20px; }}\n")
                    f.write(f"table {{ border-collapse: collapse; width: 100%; }}\n")
                    f.write(f"th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}\n")
                    f.write(f"th {{ background-color: #f2f2f2; }}\n")
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
        json_path = os.path.join(output_path, "benchmark_report.json")
        
        # Save results as JSON
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"JSON report saved to {json_path}")
        return json_path
    
    else:
        raise ValueError(f"Unsupported report format: {report_format}")