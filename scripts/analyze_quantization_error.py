#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze layer-wise quantization error in YOLOv8 models.

This script analyzes the quantization error between floating-point tensors and their
quantized counterparts across different layers of a YOLOv8 model. It helps identify
which layers are most affected by quantization, providing insights for potential
layer-specific optimization strategies.
"""

import os
import sys
import argparse
import yaml
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# Add project root to path to allow imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from src.models import create_yolov8_model, prepare_model_for_qat
from src.data_utils import create_dataloader, get_dataset_from_yaml
from src.evaluation import measure_layer_wise_quantization_error
from src.quantization import load_quantization_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('analyze_quantization_error')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze quantization error in YOLOv8 model")
    
    # Model settings
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to QAT or INT8 model checkpoint')
    parser.add_argument('--fp32-model', type=str, default=None,
                        help='Path to reference FP32 model checkpoint (if analyzing INT8 model)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')
    
    # Dataset settings
    parser.add_argument('--data', type=str, default='dataset/vietnam-traffic-sign-detection/dataset.yaml',
                        help='Path to dataset YAML')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    
    # Analysis settings
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to analyze')
    parser.add_argument('--error-threshold', type=float, default=0.01,
                        help='Error threshold for highlighting problematic layers')
    parser.add_argument('--detailed', action='store_true',
                        help='Generate detailed per-channel analysis')
    parser.add_argument('--device', type=str, default='',
                        help='Device to run analysis on (cuda or cpu)')
    
    # Output settings
    parser.add_argument('--output', type=str, default='logs/quantization_analysis',
                        help='Output directory')
    
    return parser.parse_args()


def load_model(model_path, device, is_qat=True):
    """
    Load model from file path.
    
    Args:
        model_path: Path to model file
        device: Device to load model on
        is_qat: Whether the model is a QAT model
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                # This requires knowing the model architecture
                # Try to get it from metadata if available
                if 'metadata' in checkpoint and 'model_name' in checkpoint['metadata']:
                    model_name = checkpoint['metadata']['model_name']
                    num_classes = checkpoint['metadata'].get('num_classes', 80)
                else:
                    # Default to yolov8n with COCO classes if not specified
                    logger.warning("Model architecture not specified, using default yolov8n")
                    model_name = 'yolov8n'
                    num_classes = 80
                
                # Create model instance
                model = create_yolov8_model(model_name, num_classes=num_classes, pretrained=False)
                
                # Load state dict
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Assume it's a direct model
                model = checkpoint
        else:
            # Direct model object
            model = checkpoint
        
        # Move model to device
        model = model.to(device)
        
        # Set to evaluation mode
        model.eval()
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def register_hooks(model):
    """
    Register hooks to capture pre and post quantization activations.
    
    Args:
        model: Model to attach hooks to
    
    Returns:
        Dictionary of activations and list of handles
    """
    activations = {}
    handles = []
    
    def _make_pre_hook(name):
        def hook(module, input):
            if input and isinstance(input[0], torch.Tensor):
                activations[f"{name}_pre"] = input[0].detach()
            return input
        return hook
    
    def _make_post_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[f"{name}_post"] = output.detach()
            return output
        return hook
    
    # Register hooks for fake quantize modules and relevant quantizable operations
    for name, module in model.named_modules():
        # Look for quantization-specific modules
        if 'fake_quantize' in name or 'observer' in name:
            logger.debug(f"Registering hooks for {name}")
            handles.append(module.register_forward_pre_hook(_make_pre_hook(name)))
            handles.append(module.register_forward_hook(_make_post_hook(name)))
        
        # Also track certain layer types if they might be quantized
        if any(typ in type(module).__name__.lower() for typ in ['conv', 'linear', 'bn']):
            # Check if this module has quantization attributes
            if hasattr(module, 'weight_fake_quant') or hasattr(module, 'activation_post_process'):
                layer_name = name
                handles.append(module.register_forward_pre_hook(_make_pre_hook(layer_name)))
                handles.append(module.register_forward_hook(_make_post_hook(layer_name)))
    
    return activations, handles


def calculate_errors(activations):
    """
    Calculate quantization errors from pre and post quantization activations.
    
    Args:
        activations: Dictionary of activations
        
    Returns:
        Dictionary of error metrics per layer
    """
    errors = {}
    
    # Find pairs of pre/post quantization activations
    for name in list(activations.keys()):
        if name.endswith('_pre'):
            base_name = name[:-4]  # Remove '_pre' suffix
            post_name = base_name + '_post'
            
            if post_name in activations:
                # Get activations
                pre_tensor = activations[name].float()
                post_tensor = activations[post_name].float()
                
                # Check tensor shapes match
                if pre_tensor.shape != post_tensor.shape:
                    logger.warning(f"Shape mismatch for {base_name}: {pre_tensor.shape} vs {post_tensor.shape}")
                    continue
                
                # Calculate error metrics
                abs_diff = torch.abs(pre_tensor - post_tensor)
                mse = torch.mean((pre_tensor - post_tensor) ** 2).item()
                mae = torch.mean(abs_diff).item()
                
                # Avoid division by zero
                if torch.max(torch.abs(pre_tensor)) > 1e-10:
                    rel_error = torch.mean(abs_diff / (torch.abs(pre_tensor) + 1e-10)).item()
                else:
                    rel_error = 0.0
                
                # Maximum absolute error
                max_abs_error = torch.max(abs_diff).item()
                
                # Calculate per-channel errors (for Conv layers)
                per_channel_errors = None
                if len(pre_tensor.shape) == 4:  # Conv layer (B, C, H, W)
                    # Calculate per-channel MSE
                    channel_mse = torch.mean((pre_tensor - post_tensor) ** 2, dim=(0, 2, 3))
                    per_channel_errors = channel_mse.cpu().numpy()
                
                # Store error metrics
                errors[base_name] = {
                    'mse': mse,
                    'mae': mae,
                    'rel_error': rel_error,
                    'max_abs_error': max_abs_error,
                    'per_channel_errors': per_channel_errors
                }
    
    return errors


def visualize_errors(errors, output_dir, error_threshold=0.01, detailed=False):
    """
    Visualize quantization errors.
    
    Args:
        errors: Dictionary of error metrics per layer
        output_dir: Directory to save visualizations
        error_threshold: Threshold for highlighting problematic layers
        detailed: Whether to generate detailed per-channel analysis
        
    Returns:
        Path to error summary file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert error dict to sorted list
    error_items = [(name, data) for name, data in errors.items()]
    
    # Sort by relative error (most severe first)
    error_items.sort(key=lambda x: x[1]['rel_error'], reverse=True)
    
    # Create summary dataframe
    import pandas as pd
    df = pd.DataFrame([
        {
            'layer': name,
            'relative_error': data['rel_error'],
            'mse': data['mse'],
            'max_abs_error': data['max_abs_error'],
            'problematic': data['rel_error'] > error_threshold
        }
        for name, data in error_items
    ])
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "quantization_error_summary.csv")
    df.to_csv(csv_path, index=False)
    
    # Create bar plot of relative errors
    plt.figure(figsize=(12, 8))
    bars = plt.barh(
        df['layer'][:20],  # Top 20 layers
        df['relative_error'][:20],
        color=[('red' if x > error_threshold else 'blue') for x in df['relative_error'][:20]]
    )
    plt.xlabel('Relative Error')
    plt.ylabel('Layer')
    plt.title('Top 20 Layers by Quantization Error')
    plt.axvline(x=error_threshold, color='red', linestyle='--', label=f'Threshold ({error_threshold})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quantization_error_top_layers.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create histogram of errors
    plt.figure(figsize=(10, 6))
    plt.hist(df['relative_error'], bins=30, alpha=0.7)
    plt.xlabel('Relative Error')
    plt.ylabel('Count')
    plt.title('Distribution of Quantization Errors Across Layers')
    plt.axvline(x=error_threshold, color='red', linestyle='--', label=f'Threshold ({error_threshold})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quantization_error_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate detailed per-channel analysis if requested
    if detailed:
        # Filter layers with per-channel data
        per_channel_layers = [(name, data) for name, data in error_items 
                              if data['per_channel_errors'] is not None]
        
        # Create detailed directory
        detailed_dir = os.path.join(output_dir, "detailed")
        os.makedirs(detailed_dir, exist_ok=True)
        
        # Generate per-channel visualizations for top most problematic layers
        for name, data in per_channel_layers[:10]:  # Top 10 problematic layers
            plt.figure(figsize=(12, 6))
            channel_errors = data['per_channel_errors']
            plt.bar(range(len(channel_errors)), channel_errors)
            plt.xlabel('Channel Index')
            plt.ylabel('MSE')
            plt.title(f'Per-Channel Quantization Error: {name}')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(detailed_dir, f"{name.replace('.', '_')}_channel_errors.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create error summary text file
    summary_path = os.path.join(output_dir, "quantization_error_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("Quantization Error Analysis Summary\n")
        f.write("==================================\n\n")
        
        # Overall statistics
        f.write(f"Total layers analyzed: {len(error_items)}\n")
        f.write(f"Problematic layers (error > {error_threshold}): "
                f"{sum(1 for _, data in error_items if data['rel_error'] > error_threshold)}\n\n")
        
        f.write("Top 10 Most Problematic Layers:\n")
        f.write("---------------------------------\n")
        
        for i, (name, data) in enumerate(error_items[:10]):
            f.write(f"{i+1}. {name}:\n")
            f.write(f"   Relative Error: {data['rel_error']:.6f}\n")
            f.write(f"   MSE: {data['mse']:.6f}\n")
            f.write(f"   Max Absolute Error: {data['max_abs_error']:.6f}\n")
            f.write("\n")
    
    logger.info(f"Error summary saved to {summary_path}")
    return summary_path


def analyze_single_model(model, dataloader, args):
    """
    Analyze quantization error in a single QAT model.
    
    Args:
        model: QAT model to analyze
        dataloader: DataLoader for analysis
        args: Command line arguments
        
    Returns:
        Dictionary of error metrics
    """
    logger.info("Analyzing quantization error in model...")
    
    # Register hooks to capture activations
    activations, handles = register_hooks(model)
    
    # Process samples
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)
    
    # Process a few batches to get activations
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            if batch_idx >= args.num_samples:
                break
            
            # Extract images
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                images = batch[0]
            elif isinstance(batch, dict) and 'img' in batch:
                images = batch['img']
            elif isinstance(batch, dict) and 'image' in batch:
                images = batch['image']
            else:
                logger.warning(f"Unsupported batch format: {type(batch)}")
                continue
            
            # Move to device
            images = images.to(device)
            
            # Run forward pass to collect activations
            _ = model(images)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Calculate errors from collected activations
    errors = calculate_errors(activations)
    
    # Visualize errors
    visualize_errors(
        errors, 
        args.output, 
        error_threshold=args.error_threshold,
        detailed=args.detailed
    )
    
    logger.info(f"Analysis completed with {len(errors)} quantized layers analyzed")
    return errors


def analyze_model_pair(fp32_model, int8_model, dataloader, args):
    """
    Analyze quantization error between FP32 and INT8 models.
    
    Args:
        fp32_model: FP32 reference model
        int8_model: INT8 quantized model
        dataloader: DataLoader for analysis
        args: Command line arguments
        
    Returns:
        Dictionary of error metrics
    """
    logger.info("Analyzing quantization error between FP32 and INT8 models...")
    
    # Measure layer-wise quantization error
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    fp32_model = fp32_model.to(device)
    int8_model = int8_model.to(device)
    
    # Create test inputs from dataloader
    test_inputs = []
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_samples:
            break
        
        # Extract images
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            images = batch[0]
        elif isinstance(batch, dict) and 'img' in batch:
            images = batch['img']
        elif isinstance(batch, dict) and 'image' in batch:
            images = batch['image']
        else:
            logger.warning(f"Unsupported batch format: {type(batch)}")
            continue
        
        # Add to test inputs
        test_inputs.append(images.to(device))
    
    # Combine results across multiple inputs
    all_errors = {}
    
    # Process each test input
    for idx, test_input in enumerate(tqdm(test_inputs, desc="Processing samples")):
        # Get layer-wise error for this input
        errors = measure_layer_wise_quantization_error(fp32_model, int8_model, test_input)
        
        # Merge with overall results
        for layer_name, error_metrics in errors.items():
            if layer_name not in all_errors:
                all_errors[layer_name] = {
                    'abs_error': [],
                    'rel_error': [],
                    'mse': []
                }
            
            all_errors[layer_name]['abs_error'].append(error_metrics['abs_error'])
            all_errors[layer_name]['rel_error'].append(error_metrics['rel_error'])
            
            # Add MSE if available
            if 'mse' in error_metrics:
                all_errors[layer_name]['mse'].append(error_metrics['mse'])
    
    # Calculate average errors across inputs
    avg_errors = {}
    for layer_name, error_lists in all_errors.items():
        avg_errors[layer_name] = {
            'abs_error': np.mean(error_lists['abs_error']),
            'rel_error': np.mean(error_lists['rel_error']),
            'mse': np.mean(error_lists['mse']) if error_lists['mse'] else 0.0
        }
    
    # Format for visualization
    vis_errors = {}
    for layer_name, error_metrics in avg_errors.items():
        vis_errors[layer_name] = {
            'mse': error_metrics.get('mse', 0.0),
            'rel_error': error_metrics['rel_error'],
            'max_abs_error': error_metrics['abs_error'],
            'per_channel_errors': None  # Per-channel data not available in this analysis
        }
    
    # Visualize errors
    visualize_errors(
        vis_errors, 
        args.output, 
        error_threshold=args.error_threshold,
        detailed=False  # Detailed analysis not available for model pair
    )
    
    logger.info(f"Analysis completed with {len(avg_errors)} layers compared")
    return avg_errors


def create_analysis_dataloader(args):
    """
    Create dataloader for analysis.
    
    Args:
        args: Command line arguments
        
    Returns:
        DataLoader for analysis
    """
    logger.info(f"Creating dataloader from {args.data}")
    
    # Create dataloader with small batch size for analysis
    dataloader, _ = create_dataloader(
        dataset_yaml=args.data,
        batch_size=args.batch_size,
        img_size=args.img_size,
        augment=False,
        shuffle=True,  # Shuffle to get diverse samples
        workers=min(4, os.cpu_count() or 1),  # Use fewer workers for analysis
        mode='val'  # Use validation set for analysis
    )
    
    logger.info(f"Created dataloader with {len(dataloader)} batches")
    return dataloader


def main():
    """Main analysis function"""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    
    # Create dataloader
    dataloader = create_analysis_dataloader(args)
    
    # Determine analysis type
    if args.fp32_model:
        # Compare FP32 and INT8 models
        logger.info("Analyzing quantization error between FP32 and INT8 models")
        
        # Load FP32 model
        fp32_model = load_model(args.fp32_model, device, is_qat=False)
        
        # Load INT8 model
        int8_model = load_model(args.model, device, is_qat=False)
        
        # Analyze models
        errors = analyze_model_pair(fp32_model, int8_model, dataloader, args)
    else:
        # Analyze single QAT model
        logger.info("Analyzing quantization error in QAT model")
        
        # Load model
        model = load_model(args.model, device, is_qat=True)
        
        # Analyze model
        errors = analyze_single_model(model, dataloader, args)
    
    logger.info(f"Analysis complete. Results saved to {args.output}")
    
    # Return most problematic layers
    return sorted([(name, data['rel_error']) for name, data in errors.items()], 
                 key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error during quantization error analysis: {e}")
        sys.exit(1)