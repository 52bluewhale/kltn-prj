#!/usr/bin/env python3
# evaluate_new.py - Script to evaluate and compare FP32 and QAT models

import os
import time
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm

# Try to import ultralytics for YOLO models
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: Ultralytics not available. Please install with: pip install ultralytics")
    print("Attempting to use alternative methods for evaluation...")


def find_file(file_path, search_in_parent=True, max_depth=3):
    """Try to find a file with flexible path handling"""
    # First, try the exact path
    if os.path.isfile(file_path):
        return file_path
    
    # Try with absolute path from current directory
    abs_path = os.path.abspath(file_path)
    if os.path.isfile(abs_path):
        return abs_path
    
    # Try removing the kltn-prj prefix if it exists
    if "kltn-prj" in file_path:
        no_prefix = file_path.replace("kltn-prj/", "")
        if os.path.isfile(no_prefix):
            return no_prefix
    
    # Try searching in parent directories
    if search_in_parent:
        parts = Path(file_path).parts
        filename = parts[-1]
        
        # Search current directory and parent directories
        current_dir = Path.cwd()
        for _ in range(max_depth):
            # Look for the file recursively in this directory
            for path in current_dir.rglob(filename):
                if path.is_file():
                    return str(path)
            
            # Move up one directory
            parent_dir = current_dir.parent
            if parent_dir == current_dir:  # Reached root
                break
            current_dir = parent_dir
    
    return None


def load_yaml(yaml_file):
    """Load YAML configuration file"""
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


def evaluate_model(model_path, dataset_yaml, device='cpu', quantized=False, model_type='yolo'):
    """
    Evaluate model performance based on model type
    
    Parameters:
    - model_path: Path to the model file
    - dataset_yaml: Path to dataset YAML file
    - device: Device to use for evaluation ('cpu' or 'cuda')
    - quantized: Whether the model is quantized
    - model_type: Type of model ('yolo' or 'other')
    
    Returns:
    - Dictionary of metrics
    """
    print(f"Evaluating {'quantized' if quantized else 'fp32'} model: {model_path}")
    
    # Find model file
    found_model_path = find_file(model_path)
    if not found_model_path:
        print(f"ERROR: Model file not found: {model_path}")
        print(f"Current directory: {os.getcwd()}")
        
        # Try to suggest possible locations
        print("Checking for model files in common locations...")
        possible_locations = [
            "models/checkpoints/fp32/weights",
            "models/checkpoints/qat",
            "model/checkpoint/fp32/weights",
            "model/checkpoint/qat",
            "kltn-prj/models/checkpoints/fp32/weights",
            "kltn-prj/models/checkpoints/qat",
            "kltn-prj/model/checkpoint/fp32/weights",
            "kltn-prj/model/checkpoint/qat",
        ]
        
        for location in possible_locations:
            if os.path.exists(location):
                print(f"Directory exists: {location}")
                print(f"Files in {location}: {os.listdir(location)}")
        
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_path = found_model_path
    print(f"Using model file: {model_path}")
    
    # Find dataset YAML
    found_dataset_yaml = find_file(dataset_yaml)
    if not found_dataset_yaml:
        print(f"ERROR: Dataset YAML not found: {dataset_yaml}")
        
        # Try to suggest possible locations
        print("Checking for dataset.yaml in common locations...")
        possible_locations = [
            "dataset",
            "dataset/vietnam-traffic-sign-detection",
            "datasets",
            "kltn-prj/dataset",
            "kltn-prj/dataset/vietnam-traffic-sign-detection",
            "kltn-prj/datasets",
        ]
        
        for location in possible_locations:
            if os.path.exists(location):
                yaml_files = [f for f in os.listdir(location) if f.endswith('.yaml')]
                if yaml_files:
                    print(f"Found YAML files in {location}: {yaml_files}")
        
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    
    dataset_yaml = found_dataset_yaml
    print(f"Using dataset YAML: {dataset_yaml}")
    
    # Load and evaluate model based on type
    if model_type == 'yolo' and ULTRALYTICS_AVAILABLE:
        return evaluate_yolo_model(model_path, dataset_yaml, device, quantized)
    else:
        return evaluate_generic_model(model_path, dataset_yaml, device, quantized)


def evaluate_yolo_model(model_path, dataset_yaml, device='cpu', quantized=False):
    """Evaluate a YOLO model using Ultralytics"""
    metrics = {}
    
    # Load model
    try:
        print(f"Loading YOLO model: {model_path}")
        model = YOLO(model_path)
        print(f"Successfully loaded model")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return evaluate_generic_model(model_path, dataset_yaml, device, quantized)
    
    # Run validation
    print(f"Running validation on dataset: {dataset_yaml}")
    start_val_time = time.time()
    try:
        results = model.val(data=dataset_yaml)
        end_val_time = time.time()
        metrics['val_time'] = end_val_time - start_val_time
        
        # Extract metrics
        if hasattr(results, 'box'):
            box_metrics = results.box
            metrics_mapping = {
                'map': 'mAP',        # mAP@0.5:0.95
                'map50': 'mAP50',    # mAP@0.5
                'map75': 'mAP75',    # mAP@0.75
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1',
            }
            
            for attr, metric_name in metrics_mapping.items():
                if hasattr(box_metrics, attr):
                    value = getattr(box_metrics, attr)
                    metrics[metric_name] = float(value)
    except Exception as e:
        print(f"Error during validation: {e}")
        metrics['val_time'] = time.time() - start_val_time
    
    # Benchmark inference time
    print("Benchmarking inference time...")
    
    # Use a real image if available from the dataset config
    test_img = None
    try:
        dataset_config = load_yaml(dataset_yaml)
        test_path = dataset_config.get('test', dataset_config.get('val', None))
        if test_path:
            # Find first image in test path
            from glob import glob
            img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            for ext in img_extensions:
                img_files = glob(os.path.join(test_path, ext))
                if img_files:
                    test_img = img_files[0]
                    break
    except Exception as e:
        print(f"Could not find test image: {e}")
    
    # Fall back to dummy input if no test image
    try:
        if test_img:
            # Warm up
            for _ in range(5):
                _ = model.predict(test_img, verbose=False)
            
            # Benchmark
            num_runs = 20
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            for _ in range(num_runs):
                _ = model.predict(test_img, verbose=False)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
        else:
            # Use dummy input
            dummy_input = torch.randn(1, 3, 640, 640).to(device)
            
            # Warm up
            for _ in range(5):
                _ = model.predict(dummy_input, verbose=False)
            
            # Benchmark
            num_runs = 20
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            for _ in range(num_runs):
                _ = model.predict(dummy_input, verbose=False)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
        
        metrics['inference_time'] = (end_time - start_time) / num_runs
    except Exception as e:
        print(f"Error during inference benchmarking: {e}")
        metrics['inference_time'] = float('inf')
    
    print(f"Finished evaluating model")
    return metrics


def evaluate_generic_model(model_path, dataset_yaml, device='cpu', quantized=False):
    """
    Evaluate a generic PyTorch model (fallback when YOLO is not available)
    This provides basic metrics like model size and inference time
    """
    metrics = {}
    
    # Try to load the model
    try:
        if quantized:
            # For quantized models, try TorchScript
            print(f"Loading quantized model: {model_path}")
            model = torch.jit.load(model_path, map_location=device)
        else:
            # For regular models
            print(f"Loading model: {model_path}")
            model = torch.load(model_path, map_location=device)
            # Handle checkpoint format
            if isinstance(model, dict) and 'model' in model:
                model = model['model']
            elif isinstance(model, dict) and 'state_dict' in model:
                # Create empty model and load state dict
                print("Warning: Only state_dict found, model architecture might be missing")
                model = model['state_dict']
        
        print(f"Successfully loaded model")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Cannot evaluate the model. Please ensure the model file is valid.")
        # Return empty metrics to avoid failing the comparison
        metrics['inference_time'] = float('inf')
        return metrics
    
    # Put model in evaluation mode if it's a proper PyTorch module
    if hasattr(model, 'eval'):
        model.eval()
    
    # Try to infer input shape from model architecture
    input_shape = [1, 3, 640, 640]  # Default YOLO shape
    
    # Benchmark inference time
    print("Benchmarking inference time...")
    try:
        dummy_input = torch.randn(*input_shape).to(device)
        
        # Check if model is callable
        if callable(model):
            # Warm up
            for _ in range(5):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # Benchmark
            num_runs = 20
            torch.cuda.synchronize() if device == 'cuda' else None
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(dummy_input)
            torch.cuda.synchronize() if device == 'cuda' else None
            end_time = time.time()
            
            metrics['inference_time'] = (end_time - start_time) / num_runs
        else:
            print("Model is not callable, cannot benchmark inference time")
            metrics['inference_time'] = float('inf')
    except Exception as e:
        print(f"Error during inference: {e}")
        metrics['inference_time'] = float('inf')
    
    # Try to extract other metrics if model has them
    if hasattr(model, 'metrics'):
        try:
            model_metrics = model.metrics
            for key, value in model_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
        except:
            pass
    
    print(f"Finished evaluating model")
    return metrics


def compare_models(fp32_metrics, qat_metrics):
    """Compare metrics between FP32 and QAT models"""
    comparison = {}
    
    # Get all metrics from both models
    all_metrics = set(fp32_metrics.keys()) | set(qat_metrics.keys())
    
    # Compare each metric
    for metric in all_metrics:
        if metric in fp32_metrics and metric in qat_metrics:
            fp32_value = fp32_metrics[metric]
            qat_value = qat_metrics[metric]
            
            if isinstance(fp32_value, (int, float)) and isinstance(qat_value, (int, float)):
                abs_diff = qat_value - fp32_value
                rel_diff = abs_diff / fp32_value if fp32_value != 0 else float('inf')
                
                # Determine which model is better (higher is better except for time metrics)
                if metric in ['inference_time', 'val_time']:
                    better = 'qat' if qat_value < fp32_value else 'fp32' if fp32_value < qat_value else 'equal'
                else:
                    better = 'qat' if qat_value > fp32_value else 'fp32' if fp32_value > qat_value else 'equal'
                
                # Create comparison entry
                comparison[metric] = {
                    'fp32': fp32_value,
                    'qat': qat_value,
                    'abs_diff': abs_diff,
                    'rel_diff': rel_diff,
                    'better': better
                }
                
                # Add speedup for time metrics
                if metric in ['inference_time', 'val_time']:
                    speedup = fp32_value / qat_value if qat_value > 0 else float('inf')
                    comparison[metric]['speedup'] = speedup
        
        # Handle metrics present in only one model
        elif metric in fp32_metrics:
            comparison[metric] = {
                'fp32': fp32_metrics[metric],
                'qat': None,
                'abs_diff': None,
                'rel_diff': None,
                'better': 'fp32'
            }
        elif metric in qat_metrics:
            comparison[metric] = {
                'fp32': None,
                'qat': qat_metrics[metric],
                'abs_diff': None,
                'rel_diff': None,
                'better': 'qat'
            }
    
    return comparison


def visualize_comparison(comparison, fp32_path, qat_path, output_dir='./results'):
    """Visualize comparison results and save plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Categorize metrics
    perf_metrics = []
    time_metrics = []
    
    for metric in comparison.keys():
        if metric in ['inference_time', 'val_time']:
            time_metrics.append(metric)
        else:
            perf_metrics.append(metric)
    
    # Plot performance metrics if available
    if perf_metrics:
        fig, axes = plt.subplots(
            nrows=(len(perf_metrics) + 1) // 2, 
            ncols=2 if len(perf_metrics) > 1 else 1,
            figsize=(12, 4 * ((len(perf_metrics) + 1) // 2))
        )
        
        # Handle case with single metric
        if len(perf_metrics) == 1:
            axes = [axes]
        # Handle case with multiple metrics
        elif len(perf_metrics) > 1:
            axes = axes.flatten()
        
        for i, metric in enumerate(perf_metrics):
            if i < len(axes):
                ax = axes[i]
                data = comparison[metric]
                
                # Skip if either value is None
                if data['fp32'] is None or data['qat'] is None:
                    ax.text(0.5, 0.5, f"{metric}: Incomplete data",
                            ha='center', va='center', fontsize=12)
                    continue
                
                bars = ax.bar(['FP32', 'QAT'], [data['fp32'], data['qat']])
                
                # Color bars based on which is better
                if data['better'] == 'fp32':
                    bars[0].set_color('green')
                    bars[1].set_color('blue')
                elif data['better'] == 'qat':
                    bars[0].set_color('blue')
                    bars[1].set_color('green')
                
                ax.set_title(f"{metric}")
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                
                # Annotate relative difference if available
                if data['rel_diff'] is not None:
                    rel_diff_pct = data['rel_diff'] * 100
                    ax.annotate(f"{rel_diff_pct:+.2f}%",
                                xy=(1, data['qat']),
                                xytext=(0, 10 if rel_diff_pct >= 0 else -15),
                                textcoords='offset points',
                                ha='center')
        
        # Hide unused subplots
        for i in range(len(perf_metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        perf_plot_path = os.path.join(output_dir, 'performance_metrics.png')
        plt.savefig(perf_plot_path)
        print(f"Saved performance metrics plot to {perf_plot_path}")
        plt.close()
    
    # Plot time metrics if available
    if time_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        labels = []
        fp32_times = []
        qat_times = []
        
        for metric in time_metrics:
            data = comparison[metric]
            if data['fp32'] is not None and data['qat'] is not None:
                labels.append(metric.replace('_', ' ').title())
                fp32_times.append(data['fp32'] * 1000)  # Convert to ms
                qat_times.append(data['qat'] * 1000)    # Convert to ms
        
        if labels:
            x = np.arange(len(labels))
            width = 0.35
            
            rects1 = ax.bar(x - width/2, fp32_times, width, label='FP32')
            rects2 = ax.bar(x + width/2, qat_times, width, label='QAT')
            
            ax.set_ylabel('Time (ms)')
            ax.set_title('Time Metrics Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            
            # Add speedup annotations
            for i, metric in enumerate(time_metrics):
                if i < len(labels):
                    data = comparison[metric]
                    if 'speedup' in data and data['speedup'] is not None:
                        speedup = data['speedup']
                        ax.annotate(f"{speedup:.2f}x faster",
                                   xy=(i, min(fp32_times[i], qat_times[i])),
                                   xytext=(0, -20),
                                   textcoords='offset points',
                                   ha='center')
            
            plt.tight_layout()
            time_plot_path = os.path.join(output_dir, 'time_metrics.png')
            plt.savefig(time_plot_path)
            print(f"Saved time metrics plot to {time_plot_path}")
            plt.close()
    
    # Plot model size comparison
    try:
        fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)  # MB
        qat_size = os.path.getsize(qat_path) / (1024 * 1024)    # MB
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(['FP32', 'QAT'], [fp32_size, qat_size])
        
        # Color the smaller model in green
        if fp32_size > qat_size:
            bars[1].set_color('green')
        else:
            bars[0].set_color('green')
        
        plt.title('Model Size Comparison')
        plt.ylabel('Size (MB)')
        plt.grid(True, alpha=0.3)
        
        # Annotate size reduction
        size_reduction = (fp32_size - qat_size) / fp32_size * 100
        plt.annotate(f"{size_reduction:.2f}% reduction",
                    xy=(1, qat_size),
                    xytext=(0, 10 if size_reduction > 0 else -15),
                    textcoords='offset points',
                    ha='center')
        
        size_plot_path = os.path.join(output_dir, 'model_size_comparison.png')
        plt.savefig(size_plot_path)
        print(f"Saved model size comparison to {size_plot_path}")
        plt.close()
    except Exception as e:
        print(f"Error creating model size comparison: {e}")


def save_summary(comparison, fp32_path, qat_path, output_dir='./results'):
    """Save evaluation summary to a text file"""
    summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("Models:\n")
        f.write(f"  FP32: {fp32_path}\n")
        f.write(f"  QAT:  {qat_path}\n\n")
        
        # Model size comparison
        try:
            fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)  # MB
            qat_size = os.path.getsize(qat_path) / (1024 * 1024)    # MB
            size_reduction = (fp32_size - qat_size) / fp32_size * 100
            
            f.write("Model Size:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  FP32: {fp32_size:.2f} MB\n")
            f.write(f"  QAT:  {qat_size:.2f} MB\n")
            f.write(f"  Reduction: {size_reduction:.2f}%\n\n")
        except Exception as e:
            f.write(f"Model Size: Error calculating - {str(e)}\n\n")
        
        # Categorize metrics
        perf_metrics = []
        time_metrics = []
        
        for metric in comparison.keys():
            if metric in ['inference_time', 'val_time']:
                time_metrics.append(metric)
            else:
                perf_metrics.append(metric)
        
        # Performance metrics
        if perf_metrics:
            f.write("Performance Metrics:\n")
            f.write("-" * 40 + "\n")
            for metric in perf_metrics:
                data = comparison[metric]
                
                f.write(f"  {metric}:\n")
                
                if data['fp32'] is not None:
                    f.write(f"    FP32: {data['fp32']:.6f}\n")
                else:
                    f.write(f"    FP32: N/A\n")
                
                if data['qat'] is not None:
                    f.write(f"    QAT:  {data['qat']:.6f}\n")
                else:
                    f.write(f"    QAT:  N/A\n")
                
                if data['rel_diff'] is not None:
                    diff = data['rel_diff'] * 100
                    f.write(f"    Diff: {diff:+.2f}%\n")
                
                f.write(f"    Better: {data['better'].upper()}\n\n")
        
        # Time metrics
        if time_metrics:
            f.write("Time Metrics:\n")
            f.write("-" * 40 + "\n")
            for metric in time_metrics:
                data = comparison[metric]
                
                f.write(f"  {metric}:\n")
                
                if data['fp32'] is not None:
                    fp32_time = data['fp32'] * 1000  # ms
                    f.write(f"    FP32: {fp32_time:.2f} ms\n")
                else:
                    f.write(f"    FP32: N/A\n")
                
                if data['qat'] is not None:
                    qat_time = data['qat'] * 1000  # ms
                    f.write(f"    QAT:  {qat_time:.2f} ms\n")
                else:
                    f.write(f"    QAT:  N/A\n")
                
                if 'speedup' in data and data['speedup'] is not None:
                    f.write(f"    Speedup: {data['speedup']:.2f}x\n")
                
                f.write(f"    Better: {data['better'].upper()}\n\n")
        
        # Overall Summary
        f.write("SUMMARY:\n")
        f.write("-" * 40 + "\n")
        
        # Count metrics where each model performs better
        better_count = {'fp32': 0, 'qat': 0, 'equal': 0}
        for metric, data in comparison.items():
            if data['better'] in better_count:
                better_count[data['better']] += 1
        
        f.write(f"  FP32 better in: {better_count['fp32']} metrics\n")
        f.write(f"  QAT better in: {better_count['qat']} metrics\n")
        f.write(f"  Equal performance in: {better_count['equal']} metrics\n\n")
        
        # Check if QAT is faster
        faster = None
        if 'inference_time' in comparison:
            data = comparison['inference_time']
            if data['fp32'] is not None and data['qat'] is not None:
                if data['qat'] < data['fp32']:
                    speedup = data['fp32'] / data['qat']
                    faster = f"QAT model is {speedup:.2f}x faster for inference"
                elif data['fp32'] < data['qat']:
                    slowdown = data['qat'] / data['fp32'] 
                    faster = f"QAT model is {slowdown:.2f}x slower for inference"
        
        # Check accuracy drop
        accuracy_drop = None
        if 'mAP' in comparison:
            data = comparison['mAP']
            if data['fp32'] is not None and data['qat'] is not None:
                drop = (data['fp32'] - data['qat']) / data['fp32'] * 100
                if drop > 0:
                    accuracy_drop = f"QAT model has {drop:.2f}% mAP drop"
                else:
                    accuracy_drop = f"QAT model has {-drop:.2f}% mAP improvement"
        
        # Final recommendation
        f.write("Recommendation:\n")
        try:
            # Calculate speed vs accuracy tradeoff
            has_speed_advantage = False
            has_size_advantage = False
            has_accuracy_disadvantage = False
            
            if faster and "faster" in faster:
                has_speed_advantage = True
                f.write(f"  {faster}\n")
            elif faster:
                f.write(f"  {faster}\n")
                
            if 'mAP' in comparison or 'precision' in comparison:
                metric_name = 'mAP' if 'mAP' in comparison else 'precision'
                data = comparison[metric_name]
                if data['fp32'] is not None and data['qat'] is not None:
                    drop = (data['fp32'] - data['qat']) / data['fp32'] * 100
                    if drop > 1.0:  # More than 1% drop
                        has_accuracy_disadvantage = True
                        f.write(f"  QAT model has {drop:.2f}% {metric_name} drop\n")
                    elif drop > 0:
                        f.write(f"  QAT model has negligible {drop:.2f}% {metric_name} drop\n")
                    else:
                        f.write(f"  QAT model has {-drop:.2f}% {metric_name} improvement\n")
            
            try:
                if size_reduction > 20:  # More than 20% size reduction
                    has_size_advantage = True
                    f.write(f"  QAT model is {size_reduction:.2f}% smaller\n")
                elif size_reduction > 0:
                    f.write(f"  QAT model is {size_reduction:.2f}% smaller\n")
            except:
                pass
            
            # Make recommendation
            f.write("\n  Final recommendation: ")
            if has_speed_advantage and has_size_advantage and not has_accuracy_disadvantage:
                f.write("Use QAT model - Significant speed and size improvement with minimal accuracy impact.\n")
            elif has_speed_advantage and not has_accuracy_disadvantage:
                f.write("Use QAT model - Faster inference with minimal accuracy impact.\n")
            elif has_size_advantage and not has_accuracy_disadvantage:
                f.write("Use QAT model - Smaller size with minimal accuracy impact.\n")
            elif has_accuracy_disadvantage and (has_speed_advantage or has_size_advantage):
                f.write("Consider tradeoff - QAT model has speed/size advantages but with some accuracy loss.\n")
            else:
                f.write("Use FP32 model - No significant benefits observed from quantization.\n")
                
        except Exception as e:
            f.write(f"  Could not determine recommendation: {str(e)}\n")
    
    print(f"Saved evaluation summary to {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare FP32 and QAT models')
    parser.add_argument('--fp32', type=str, default='kltn-prj/models/checkpoints/fp32/weights/best.pt',
                        help='Path to FP32 model')
    parser.add_argument('--qat', type=str, default='kltn-prj/models/checkpoints/qat/quantized_model.pt',
                        help='Path to QAT (quantized) model')
    parser.add_argument('--dataset', type=str, default='kltn-prj/datasets/vietnam-traffic-sign-detection/dataset.yaml',
                        help='Path to dataset YAML file')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use for evaluation')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--model-type', type=str, default='yolo', choices=['yolo', 'other'],
                        help='Type of model to evaluate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print(f"Starting evaluation of models:")
    print(f"  FP32: {args.fp32}")
    print(f"  QAT:  {args.qat}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Device: {args.device}")
    print("="*80)
    
    # Enhanced error handling to avoid crashing on one model failure
    try:
        # Evaluate FP32 model
        fp32_metrics = evaluate_model(
            args.fp32, 
            args.dataset, 
            device=args.device, 
            quantized=False,
            model_type=args.model_type
        )
    except Exception as e:
        print(f"Error evaluating FP32 model: {e}")
        fp32_metrics = {'error': str(e)}
    
    try:
        # Evaluate QAT model
        qat_metrics = evaluate_model(
            args.qat, 
            args.dataset, 
            device=args.device, 
            quantized=True,
            model_type=args.model_type
        )
    except Exception as e:
        print(f"Error evaluating QAT model: {e}")
        qat_metrics = {'error': str(e)}
    
    # Check if we have enough metrics to compare
    if (set(fp32_metrics.keys()) - {'error'}) and (set(qat_metrics.keys()) - {'error'}):
        # Filter out error keys
        if 'error' in fp32_metrics:
            del fp32_metrics['error']
        if 'error' in qat_metrics:
            del qat_metrics['error']
            
        # Compare models
        comparison = compare_models(fp32_metrics, qat_metrics)
        
        # Visualize comparison
        visualize_comparison(comparison, args.fp32, args.qat, output_dir=args.output_dir)
        
        # Save summary
        summary_path = save_summary(comparison, args.fp32, args.qat, output_dir=args.output_dir)
        
        print("="*80)
        print(f"Evaluation complete! Results saved to: {args.output_dir}")
        print(f"Summary: {summary_path}")
        print("="*80)
    else:
        print("="*80)
        print("Evaluation failed - not enough metrics to compare models")
        print("="*80)


if __name__ == "__main__":
    main()