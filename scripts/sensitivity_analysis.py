# Identifies layers most sensitive to quantization
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sensitivity analysis for YOLOv8 quantization.

This script performs sensitivity analysis to identify which layers are most
sensitive to quantization in a YOLOv8 model. It helps prioritize layers that 
need special handling during quantization-aware training, such as using 
different quantization schemes or precision levels.
"""

import os
import sys
import argparse
import yaml
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

# Add project root to path to allow imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from src.models import create_yolov8_model, prepare_model_for_qat
from src.data_utils import create_dataloader, get_dataset_from_yaml
from src.evaluation import evaluate_model, measure_quantization_error
from src.quantization import load_quantization_config
from src.quantization.qconfig import get_qconfig_by_name

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sensitivity_analysis')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Perform sensitivity analysis for YOLOv8 quantization")
    
    # Model settings
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to FP32 model file to analyze')
    parser.add_argument('--img-size', type=int, default=640, 
                        help='Input image size')
    
    # Dataset settings
    parser.add_argument('--data', type=str, default='dataset/vietnam-traffic-sign-detection/dataset.yaml',
                        help='Path to dataset configuration yaml file')
    parser.add_argument('--batch-size', type=int, default=1, 
                        help='Batch size for evaluation')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of samples to use for sensitivity analysis')
    
    # Analysis settings
    parser.add_argument('--target-layers', type=str, default=None,
                        help='Comma-separated list of layer patterns to analyze (regex supported)')
    parser.add_argument('--qconfig', type=str, default='default',
                        help='Base quantization configuration to use')
    parser.add_argument('--device', type=str, default='',
                        help='Device to run analysis on')
    parser.add_argument('--metric', type=str, default='map',
                        choices=['map', 'loss', 'error'],
                        help='Metric to use for sensitivity measurement')
    parser.add_argument('--quant-type', type=str, default='per_tensor',
                        choices=['per_tensor', 'per_channel', 'symmetric', 'asymmetric'],
                        help='Type of quantization to test')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='logs/sensitivity_analysis',
                        help='Directory to save results')
    parser.add_argument('--recommend-config', action='store_true',
                        help='Generate recommended QAT configuration file')
    
    return parser.parse_args()


def load_model(model_path, device):
    """
    Load model from file.
    
    Args:
        model_path: Path to model file
        device: Device to load model on
        
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
                
                # Try to extract from metadata if available
                if 'metadata' in checkpoint:
                    metadata = checkpoint['metadata']
                    if 'model_name' in metadata:
                        model_name = metadata['model_name']
                    if 'num_classes' in metadata:
                        num_classes = metadata['num_classes']
                
                # Create model with detected architecture
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
        
        # Move model to device
        model = model.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def get_target_layers(model, patterns=None):
    """
    Get layers to analyze based on patterns.
    
    Args:
        model: Model to analyze
        patterns: List of regex patterns to match layer names
        
    Returns:
        List of (name, module) tuples for target layers
    """
    import re
    
    # Default patterns if none provided
    if patterns is None:
        patterns = [
            r'.*\.conv\d+$',  # Convolution layers
            r'.*\.linear$',   # Linear layers
            r'.*\.bn\d+$',    # Batch normalization layers
            r'.*\.cv\d+$',    # YOLOv8 specific conv layers
            r'.*\.m\.\d+$'    # YOLOv8 bottleneck layers
        ]
    elif isinstance(patterns, str):
        patterns = patterns.split(',')
    
    # Find all layers matching the patterns
    target_layers = []
    
    for name, module in model.named_modules():
        if any(re.match(pattern, name) for pattern in patterns):
            # Only add modules that contain parameters
            if any(p.requires_grad for p in module.parameters()):
                # Skip certain layer types that shouldn't be quantized
                if not any(t in type(module).__name__.lower() for t in 
                          ['relu', 'sigmoid', 'tanh', 'pool', 'upsample']):
                    target_layers.append((name, module))
    
    logger.info(f"Found {len(target_layers)} target layers to analyze")
    return target_layers


def prepare_layer_for_quantization(model, layer_name, qconfig):
    """
    Prepare a specific layer for quantization.
    
    Args:
        model: Model to modify
        layer_name: Name of layer to quantize
        qconfig: Quantization configuration to apply
        
    Returns:
        Prepared model
    """
    # Make a deep copy of the model
    model_copy = type(model)()
    model_copy.load_state_dict(model.state_dict())
    model_copy.eval()
    
    # Avoid modifying the input model
    model = model_copy
    
    # First, make sure no layers are prepared for quantization
    for name, module in model.named_modules():
        if hasattr(module, 'qconfig'):
            module.qconfig = None
    
    # Then, prepare only the target layer
    for name, module in model.named_modules():
        if name == layer_name:
            logger.debug(f"Applying qconfig to layer: {name}")
            module.qconfig = qconfig
            
            # For Conv and Linear layers, ensure all parent modules also have qconfig
            # to support proper quantization
            parent_name = '.'.join(name.split('.')[:-1])
            while parent_name:
                parent = dict(model.named_modules()).get(parent_name)
                if parent is not None and hasattr(parent, 'qconfig'):
                    parent.qconfig = qconfig
                parent_name = '.'.join(parent_name.split('.')[:-1])
    
    # Prepare model with PyTorch's quantization utilities
    try:
        from torch.quantization import prepare_qat
        prepared_model = prepare_qat(model, inplace=True)
        return prepared_model
    except Exception as e:
        logger.error(f"Error preparing layer {layer_name} for quantization: {e}")
        return model


def evaluate_layer_sensitivity(model, layer_name, layer, dataloader, qconfig, args):
    """
    Evaluate sensitivity of a layer to quantization.
    
    Args:
        model: Base model
        layer_name: Name of layer to evaluate
        layer: Module to evaluate
        dataloader: DataLoader for evaluation
        qconfig: Quantization configuration to apply
        args: Command line arguments
        
    Returns:
        Sensitivity score and metrics
    """
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set model to eval mode
    model.eval()
    
    # Get baseline performance
    logger.debug(f"Measuring baseline performance for {layer_name}")
    
    # Create a model copy for baseline
    baseline_model = type(model)()
    baseline_model.load_state_dict(model.state_dict())
    baseline_model.to(device)
    baseline_model.eval()
    
    # Get baseline metric
    baseline_metrics = evaluate_model(
        model=baseline_model,
        dataloader=dataloader,
        conf_threshold=0.25,
        iou_threshold=0.5,
        device=device
    )
    
    # Get baseline score based on selected metric
    if args.metric == 'map':
        baseline_score = baseline_metrics.get('mAP@.5', 0)
    elif args.metric == 'loss':
        baseline_score = baseline_metrics.get('val_loss', 0)
    else:  # 'error'
        # Just use a perfect score as baseline
        baseline_score = 0
    
    # Prepare the single layer for quantization
    logger.debug(f"Preparing layer {layer_name} for quantization with {args.quant_type}")
    quantized_model = prepare_layer_for_quantization(model, layer_name, qconfig)
    quantized_model.to(device)
    quantized_model.eval()
    
    # Create a test input for error measurement
    test_batch = next(iter(dataloader))
    if isinstance(test_batch, (tuple, list)) and len(test_batch) >= 1:
        test_input = test_batch[0]
    elif isinstance(test_batch, dict) and 'img' in test_batch:
        test_input = test_batch['img']
    elif isinstance(test_batch, dict) and 'image' in test_batch:
        test_input = test_batch['image']
    else:
        logger.warning(f"Unexpected batch format: {type(test_batch)}")
        # Create a dummy input as fallback
        test_input = torch.randn(1, 3, args.img_size, args.img_size)
    
    test_input = test_input.to(device)
    
    # Measure quantization error for this layer
    if args.metric == 'error':
        # Get activations from both models to compare
        original_output = None
        quantized_output = None
        
        def _get_original_activation(module, input, output):
            nonlocal original_output
            if isinstance(output, torch.Tensor):
                original_output = output.detach()
            
        def _get_quantized_activation(module, input, output):
            nonlocal quantized_output
            if isinstance(output, torch.Tensor):
                quantized_output = output.detach()
        
        # Register hooks to get outputs
        original_layer = dict(baseline_model.named_modules()).get(layer_name)
        quantized_layer = dict(quantized_model.named_modules()).get(layer_name)
        
        if original_layer is not None and quantized_layer is not None:
            hook1 = original_layer.register_forward_hook(_get_original_activation)
            hook2 = quantized_layer.register_forward_hook(_get_quantized_activation)
            
            # Run forward passes
            with torch.no_grad():
                _ = baseline_model(test_input)
                _ = quantized_model(test_input)
            
            # Remove hooks
            hook1.remove()
            hook2.remove()
            
            # Calculate error if outputs are available
            if original_output is not None and quantized_output is not None:
                # Calculate relative error
                abs_diff = torch.abs(original_output - quantized_output)
                abs_original = torch.abs(original_output) + torch.finfo(torch.float32).eps
                rel_error = torch.mean((abs_diff / abs_original)).item()
                
                # Calculate MSE
                mse = torch.mean((original_output - quantized_output) ** 2).item()
                
                quantized_score = rel_error  # Use relative error as score
            else:
                logger.warning(f"Could not get activations for layer {layer_name}")
                quantized_score = 0
                mse = 0
        else:
            logger.warning(f"Layer {layer_name} not found in one of the models")
            quantized_score = 0
            mse = 0
    else:
        # Evaluate quantized model performance
        logger.debug(f"Measuring quantized performance for {layer_name}")
        
        quantized_metrics = evaluate_model(
            model=quantized_model,
            dataloader=dataloader,
            conf_threshold=0.25,
            iou_threshold=0.5,
            device=device
        )
        
        # Get quantized score based on selected metric
        if args.metric == 'map':
            quantized_score = quantized_metrics.get('mAP@.5', 0)
        else:  # 'loss'
            quantized_score = quantized_metrics.get('val_loss', 0)
        
    # Calculate sensitivity
    if args.metric == 'map':
        # For mAP, sensitivity is the decrease in mAP
        sensitivity = baseline_score - quantized_score
    elif args.metric == 'loss':
        # For loss, sensitivity is the increase in loss
        sensitivity = quantized_score - baseline_score
    else:  # 'error'
        # For error, sensitivity is directly the relative error
        sensitivity = quantized_score
    
    # Clean up to prevent CUDA OOM
    del quantized_model
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"Layer {layer_name} sensitivity: {sensitivity:.6f}")
    
    # Return sensitivity and additional metrics
    result = {
        'layer': layer_name,
        'layer_type': type(layer).__name__,
        'sensitivity': sensitivity,
        'baseline_score': baseline_score,
        'quantized_score': quantized_score
    }
    
    # Add MSE if calculated
    if args.metric == 'error':
        result['mse'] = mse
    
    return result


def analyze_sensitivity(model, dataloader, args):
    """
    Analyze sensitivity of layers to quantization.
    
    Args:
        model: Model to analyze
        dataloader: DataLoader for evaluation
        args: Command line arguments
        
    Returns:
        Sensitivity results
    """
    # Get target layers to analyze
    target_layers = get_target_layers(model, args.target_layers)
    
    # Get quantization configuration
    qconfig = get_qconfig_by_name(args.qconfig)
    
    # Customize qconfig based on quantization type
    if args.quant_type == 'per_channel':
        # Override with per-channel quantization
        from torch.quantization import default_per_channel_qconfig
        qconfig = default_per_channel_qconfig
    elif args.quant_type == 'symmetric':
        # Use symmetric quantization
        from src.quantization.qconfig import create_qconfig
        from src.quantization.observers import MinMaxObserver
        qconfig = create_qconfig(
            weight_qscheme=torch.per_tensor_symmetric,
            activation_qscheme=torch.per_tensor_symmetric
        )
    elif args.quant_type == 'asymmetric':
        # Use asymmetric quantization
        from src.quantization.qconfig import create_qconfig
        from src.quantization.observers import MinMaxObserver
        qconfig = create_qconfig(
            weight_qscheme=torch.per_tensor_affine,
            activation_qscheme=torch.per_tensor_affine
        )
    
    # Sample a subset of layers if there are too many
    max_layers = args.num_samples if args.num_samples > 0 else len(target_layers)
    if len(target_layers) > max_layers:
        logger.info(f"Sampling {max_layers} layers out of {len(target_layers)}")
        import random
        random.shuffle(target_layers)
        target_layers = target_layers[:max_layers]
    
    # Analyze each layer
    results = []
    
    for i, (layer_name, layer) in enumerate(tqdm(target_layers, desc="Analyzing layers")):
        try:
            # Evaluate layer sensitivity
            layer_result = evaluate_layer_sensitivity(
                model=model,
                layer_name=layer_name,
                layer=layer,
                dataloader=dataloader,
                qconfig=qconfig,
                args=args
            )
            
            # Add layer result
            results.append(layer_result)
            
            # Log progress
            logger.debug(f"Processed {i+1}/{len(target_layers)} layers")
            
        except Exception as e:
            logger.error(f"Error analyzing layer {layer_name}: {e}")
    
    # Sort results by sensitivity
    results.sort(key=lambda x: x['sensitivity'], reverse=True)
    
    # Return results
    return results


def visualize_results(results, output_dir):
    """
    Visualize sensitivity analysis results.
    
    Args:
        results: Sensitivity analysis results
        output_dir: Directory to save visualizations
        
    Returns:
        Path to saved visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "sensitivity_analysis.csv")
    df.to_csv(csv_path, index=False)
    
    # Create bar plot of top 20 most sensitive layers
    plt.figure(figsize=(12, 8))
    top_n = min(20, len(results))
    
    # Get top N layers
    top_layers = df.head(top_n)
    
    # Create bar chart
    bars = plt.barh(
        top_layers['layer'],
        top_layers['sensitivity'],
        color='blue'
    )
    
    # Add labels
    plt.xlabel('Sensitivity Score')
    plt.ylabel('Layer')
    plt.title(f'Top {top_n} Layers Most Sensitive to Quantization')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "sensitivity_top_layers.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create grouped bar chart by layer type
    plt.figure(figsize=(12, 8))
    
    # Group by layer type and calculate mean sensitivity
    layer_type_sensitivity = df.groupby('layer_type')['sensitivity'].mean().reset_index()
    layer_type_sensitivity = layer_type_sensitivity.sort_values('sensitivity', ascending=False)
    
    # Create bar chart
    plt.bar(
        layer_type_sensitivity['layer_type'],
        layer_type_sensitivity['sensitivity'],
        color='green'
    )
    
    # Add labels
    plt.xlabel('Layer Type')
    plt.ylabel('Average Sensitivity')
    plt.title('Sensitivity to Quantization by Layer Type')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot
    type_plot_path = os.path.join(output_dir, "sensitivity_by_layer_type.png")
    plt.savefig(type_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create histogram of sensitivity scores
    plt.figure(figsize=(10, 6))
    
    plt.hist(df['sensitivity'], bins=20, color='purple', alpha=0.7)
    plt.xlabel('Sensitivity Score')
    plt.ylabel('Number of Layers')
    plt.title('Distribution of Quantization Sensitivity Across Layers')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot
    hist_path = os.path.join(output_dir, "sensitivity_distribution.png")
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary report
    summary_path = os.path.join(output_dir, "sensitivity_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("Quantization Sensitivity Analysis Summary\n")
        f.write("========================================\n\n")
        
        # Overall statistics
        f.write(f"Total layers analyzed: {len(results)}\n")
        f.write(f"Average sensitivity: {df['sensitivity'].mean():.6f}\n")
        f.write(f"Maximum sensitivity: {df['sensitivity'].max():.6f}\n\n")
        
        f.write("Top 10 Most Sensitive Layers:\n")
        f.write("-----------------------------\n")
        
        for i, row in df.head(10).iterrows():
            f.write(f"{i+1}. {row['layer']} ({row['layer_type']}):\n")
            f.write(f"   Sensitivity: {row['sensitivity']:.6f}\n")
            f.write(f"   Baseline Score: {row['baseline_score']:.6f}\n")
            f.write(f"   Quantized Score: {row['quantized_score']:.6f}\n")
            f.write("\n")
        
        f.write("Layer Types Ranked by Average Sensitivity:\n")
        f.write("-----------------------------------------\n")
        
        for i, (index, row) in enumerate(layer_type_sensitivity.iterrows()):
            f.write(f"{i+1}. {row['layer_type']}: {row['sensitivity']:.6f}\n")
    
    logger.info(f"Results saved to {output_dir}")
    return summary_path


def generate_recommended_config(results, args):
    """
    Generate recommended QAT configuration based on sensitivity analysis.
    
    Args:
        results: Sensitivity analysis results
        args: Command line arguments
        
    Returns:
        Path to saved configuration file
    """
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Define sensitivity thresholds
    high_threshold = df['sensitivity'].quantile(0.9)  # Top 10%
    medium_threshold = df['sensitivity'].quantile(0.75)  # Top 25%
    
    # Categorize layers by sensitivity
    high_sensitivity = df[df['sensitivity'] >= high_threshold]
    medium_sensitivity = df[(df['sensitivity'] >= medium_threshold) & (df['sensitivity'] < high_threshold)]
    
    # Create configuration
    config = {
        "quantization": {
            "default_qconfig": "default",
            "skip_layers": [],
            "layer_configs": []
        }
    }
    
    # Add high sensitivity layers with specialized configuration
    for _, row in high_sensitivity.iterrows():
        layer_name = row['layer']
        layer_type = row['layer_type']
        
        # Escape special characters for regex pattern
        escaped_name = layer_name.replace('.', r'\.')
        
        if 'conv' in layer_type.lower():
            # For conv layers, use per-channel quantization
            config["quantization"]["layer_configs"].append({
                "pattern": escaped_name,
                "qconfig": "sensitive"
            })
        else:
            # For other layers, use higher precision
            config["quantization"]["layer_configs"].append({
                "pattern": escaped_name,
                "qconfig": "sensitive"
            })
    
    # Add medium sensitivity layers
    for _, row in medium_sensitivity.iterrows():
        layer_name = row['layer']
        
        # Escape special characters for regex pattern
        escaped_name = layer_name.replace('.', r'\.')
        
        config["quantization"]["layer_configs"].append({
            "pattern": escaped_name,
            "qconfig": "default"
        })
    
    # Add special handling for first and last layers
    config["quantization"]["layer_configs"].append({
        "pattern": r"model\.0\.conv",
        "qconfig": "first_layer"
    })
    
    config["quantization"]["layer_configs"].append({
        "pattern": r"model\.\d+\.detect",
        "qconfig": "sensitive"
    })
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "recommended_qat_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Recommended QAT configuration saved to {config_path}")
    return config_path


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, device)
    
    # Create dataloader for evaluation
    dataloader, _ = create_dataloader(
        dataset_yaml=args.data,
        batch_size=args.batch_size,
        img_size=args.img_size,
        augment=False,
        shuffle=True,  # Shuffle to get diverse samples
        workers=4,
        mode='val'  # Use validation set
    )
    
    # Analyze sensitivity
    start_time = time.time()
    results = analyze_sensitivity(model, dataloader, args)
    analysis_time = time.time() - start_time
    logger.info(f"Sensitivity analysis completed in {analysis_time:.2f} seconds")
    
    # Visualize results
    visualize_results(results, args.output_dir)
    
    # Generate recommended configuration if requested
    if args.recommend_config:
        config_path = generate_recommended_config(results, args)
        logger.info(f"Recommended QAT configuration saved to {config_path}")
    
    logger.info("Sensitivity analysis completed successfully")
    
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error during sensitivity analysis: {e}")
        sys.exit(1)