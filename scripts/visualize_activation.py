#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize activations of YOLOv8 models.

This script captures and visualizes activations from different layers
of YOLOv8 models, allowing comparison between FP32 and quantized versions
to understand the effects of quantization.
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import seaborn as sns

# Add project root to path to allow imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from src.models.yolov8_qat import YOLOv8QAT
from ultralytics.data import build_dataloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('visualize_activation')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Visualize layer activations in YOLOv8 models")
    
    # Model settings
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Path to a single model checkpoint")
    group.add_argument("--compare", action="store_true", help="Compare FP32 and INT8 models")
    
    parser.add_argument("--fp32-model", type=str, help="Path to FP32 model checkpoint (for comparison)")
    parser.add_argument("--int8-model", type=str, help="Path to quantized INT8 model checkpoint (for comparison)")
    
    # Data settings
    parser.add_argument("--data", type=str, default="data/vietnam-traffic-sign-detection/dataset.yaml", 
                        help="Dataset YAML")
    parser.add_argument("--image", type=str, default=None, 
                        help="Path to a single image for visualization")
    parser.add_argument("--batch-size", type=int, default=1, 
                        help="Batch size")
    
    # Visualization settings
    parser.add_argument("--layers", type=str, default=None, 
                        help="Comma-separated list of layers to visualize")
    parser.add_argument("--max-images", type=int, default=5, 
                        help="Maximum number of images to process")
    parser.add_argument("--max-features", type=int, default=16, 
                        help="Maximum number of features to visualize per layer")
    parser.add_argument("--show-layer-types", action="store_true", 
                        help="Show available layer types in the model")
    
    # Output settings
    parser.add_argument("--output", type=str, default="logs/activations", 
                        help="Output directory")
    
    return parser.parse_args()

def load_model(model_path, device="cuda"):
    """
    Load model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    
    try:
        if "int8" in model_path.lower() or "quantized" in model_path.lower():
            # Load quantized model
            model = YOLOv8QAT(model_path)
        else:
            # Load standard model
            checkpoint = torch.load(model_path, map_location="cpu")
            
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    model = checkpoint["model"]
                elif "model_state_dict" in checkpoint:
                    from src.models import create_yolov8_model
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
                    
                    model = create_yolov8_model(
                        model_name=model_name,
                        num_classes=num_classes,
                        pretrained=False
                    )
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model = checkpoint
            else:
                model = checkpoint
        
        # Set model to evaluation mode
        model.eval()
        
        # Move to device
        device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        model = model.to(device)
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def register_activation_hooks(model, target_layers=None):
    """
    Register hooks to capture activations from model layers.
    
    Args:
        model: Model to register hooks on
        target_layers: Optional list of layer names to capture (all conv/linear layers if None)
    
    Returns:
        Dictionary of hooks and dictionary to store activations
    """
    activations = {}
    hooks = []
    
    # Function to capture activations
    def hook_fn(name):
        def hook(module, input, output):
            # Store output tensor
            activations[name] = output.detach()
        return hook
    
    # Register hooks for target layers
    if target_layers:
        for name, module in model.named_modules():
            if any(layer in name for layer in target_layers):
                hooks.append(module.register_forward_hook(hook_fn(name)))
                logger.info(f"Registered hook for layer: {name}")
    else:
        # Default: register for all conv and linear layers
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
                logger.info(f"Registered hook for layer: {name}")
    
    return hooks, activations

def visualize_activation_maps(activation, layer_name, output_dir, img_idx=0, max_features=16):
    """
    Visualize activation maps for a layer.
    
    Args:
        activation: Activation tensor
        layer_name: Name of the layer
        output_dir: Directory to save visualizations
        img_idx: Index of the image in the batch
        max_features: Maximum number of features to visualize
    
    Returns:
        Path to saved visualization
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle different activation shapes
    if len(activation.shape) == 4:  # Conv layer: [batch_size, channels, height, width]
        # Get activation for the specified image
        act = activation[img_idx].cpu().numpy()
        num_features = min(act.shape[0], max_features)
        
        # Create plot grid
        rows = int(np.ceil(np.sqrt(num_features)))
        cols = int(np.ceil(num_features / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        if rows * cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each feature map
        for i in range(num_features):
            feature_map = act[i]
            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f"Channel {i}")
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_features, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"Activation Maps for {layer_name}")
        plt.tight_layout()
        
        # Save figure
        safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
        output_path = os.path.join(output_dir, f"{safe_layer_name}_activation_maps.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    elif len(activation.shape) == 2:  # Linear layer: [batch_size, features]
        # Get activation for the specified image
        act = activation[img_idx].cpu().numpy()
        
        # Create bar plot for activation values
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(act)), act)
        plt.title(f"Activation Values for {layer_name}")
        plt.xlabel("Feature Index")
        plt.ylabel("Activation Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
        output_path = os.path.join(output_dir, f"{safe_layer_name}_activation_values.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    else:
        logger.warning(f"Unsupported activation shape: {activation.shape} for layer {layer_name}")
        return None

def visualize_activation_distributions(activations, layer_name, output_dir, title=None):
    """
    Visualize activation value distributions for a layer.
    
    Args:
        activations: Activation tensor
        layer_name: Name of the layer
        output_dir: Directory to save visualizations
        title: Optional title for the plot
    
    Returns:
        Path to saved visualization
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten activations to 1D array
    act_flat = activations.flatten().cpu().numpy()
    
    # Create distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(act_flat, bins=50, kde=True)
    
    if title:
        plt.title(title)
    else:
        plt.title(f"Activation Distribution for {layer_name}")
    
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add summary statistics
    stats_text = f"Mean: {np.mean(act_flat):.4f}\nStd: {np.std(act_flat):.4f}\n"
    stats_text += f"Min: {np.min(act_flat):.4f}\nMax: {np.max(act_flat):.4f}"
    plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
    output_path = os.path.join(output_dir, f"{safe_layer_name}_distribution.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def compare_activation_distributions(fp32_activation, int8_activation, layer_name, output_dir):
    """
    Compare activation distributions between FP32 and INT8 models.
    
    Args:
        fp32_activation: Activation tensor from FP32 model
        int8_activation: Activation tensor from INT8 model
        layer_name: Name of the layer
        output_dir: Directory to save visualizations
    
    Returns:
        Path to saved visualization
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten activations to 1D arrays
    fp32_flat = fp32_activation.flatten().cpu().numpy()
    int8_flat = int8_activation.flatten().cpu().numpy()
    
    # Create distribution plot
    plt.figure(figsize=(12, 7))
    
    sns.histplot(fp32_flat, bins=50, kde=True, color='blue', alpha=0.6, label='FP32')
    sns.histplot(int8_flat, bins=50, kde=True, color='red', alpha=0.6, label='INT8')
    
    plt.title(f"Activation Distribution Comparison for {layer_name}")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add summary statistics
    fp32_stats = f"FP32 - Mean: {np.mean(fp32_flat):.4f}, Std: {np.std(fp32_flat):.4f}"
    int8_stats = f"INT8 - Mean: {np.mean(int8_flat):.4f}, Std: {np.std(int8_flat):.4f}"
    
    # Calculate mean and std differences
    mean_diff = np.abs(np.mean(fp32_flat) - np.mean(int8_flat))
    std_diff = np.abs(np.std(fp32_flat) - np.std(int8_flat))
    diff_stats = f"Mean Diff: {mean_diff:.4f}, Std Diff: {std_diff:.4f}"
    
    plt.figtext(0.02, 0.02, fp32_stats + '\n' + int8_stats + '\n' + diff_stats, 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
    output_path = os.path.join(output_dir, f"{safe_layer_name}_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def visualize_quantization_error(fp32_activation, int8_activation, layer_name, output_dir):
    """
    Visualize quantization error between FP32 and INT8 activations.
    
    Args:
        fp32_activation: Activation tensor from FP32 model
        int8_activation: Activation tensor from INT8 model
        layer_name: Name of the layer
        output_dir: Directory to save visualizations
    
    Returns:
        Path to saved visualization
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate absolute error
    error = torch.abs(fp32_activation - int8_activation).cpu().numpy()
    rel_error = torch.abs((fp32_activation - int8_activation) / 
                         (torch.abs(fp32_activation) + 1e-8)).cpu().numpy()
    
    # Create error visualization based on activation shape
    if len(error.shape) == 4:  # Conv layer [B, C, H, W]
        # Get error for first image
        err_img = error[0]
        rel_err_img = rel_error[0]
        
        # Calculate channel-wise error
        channel_errors = np.mean(err_img, axis=(1, 2))
        channel_rel_errors = np.mean(rel_err_img, axis=(1, 2))
        
        # Create heatmap of error
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot channel-wise absolute error
        axes[0, 0].bar(range(len(channel_errors)), channel_errors)
        axes[0, 0].set_title("Channel-wise Absolute Error")
        axes[0, 0].set_xlabel("Channel Index")
        axes[0, 0].set_ylabel("Mean Absolute Error")
        axes[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot channel-wise relative error
        axes[0, 1].bar(range(len(channel_rel_errors)), channel_rel_errors)
        axes[0, 1].set_title("Channel-wise Relative Error")
        axes[0, 1].set_xlabel("Channel Index")
        axes[0, 1].set_ylabel("Mean Relative Error")
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # Plot absolute error distribution
        sns.histplot(error.flatten(), bins=50, kde=True, ax=axes[1, 0])
        axes[1, 0].set_title("Absolute Error Distribution")
        axes[1, 0].set_xlabel("Absolute Error")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot relative error distribution
        sns.histplot(rel_error.flatten(), bins=50, kde=True, ax=axes[1, 1])
        axes[1, 1].set_title("Relative Error Distribution")
        axes[1, 1].set_xlabel("Relative Error")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        plt.suptitle(f"Quantization Error for {layer_name}")
        plt.tight_layout()
        
        # Save figure
        safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
        output_path = os.path.join(output_dir, f"{safe_layer_name}_quantization_error.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    elif len(error.shape) == 2:  # Linear layer [B, F]
        # Get error for first image
        err_vec = error[0]
        rel_err_vec = rel_error[0]
        
        # Create error visualization
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot absolute error
        axes[0].bar(range(len(err_vec)), err_vec)
        axes[0].set_title("Absolute Error")
        axes[0].set_xlabel("Feature Index")
        axes[0].set_ylabel("Absolute Error")
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot relative error
        axes[1].bar(range(len(rel_err_vec)), rel_err_vec)
        axes[1].set_title("Relative Error")
        axes[1].set_xlabel("Feature Index")
        axes[1].set_ylabel("Relative Error")
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        plt.suptitle(f"Quantization Error for {layer_name}")
        plt.tight_layout()
        
        # Add summary statistics
        mean_error = np.mean(err_vec)
        max_error = np.max(err_vec)
        mean_rel_error = np.mean(rel_err_vec)
        stats_text = f"Mean Abs Error: {mean_error:.4f}\nMax Abs Error: {max_error:.4f}\nMean Rel Error: {mean_rel_error:.4f}"
        plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # Save figure
        safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
        output_path = os.path.join(output_dir, f"{safe_layer_name}_quantization_error.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    else:
        logger.warning(f"Unsupported error shape: {error.shape} for layer {layer_name}")
        return None

def load_image(image_path, input_size=(640, 640)):
    """
    Load and preprocess an image.
    
    Args:
        image_path: Path to image file
        input_size: Size to resize the image to (width, height)
    
    Returns:
        Preprocessed image
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img = cv2.resize(img, input_size)
    
    # Normalize and convert to tensor
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = torch.from_numpy(img)
    
    return img

def generate_summary_report(visualizations, output_dir):
    """
    Generate a summary report of all visualizations.
    
    Args:
        visualizations: Dictionary of visualization paths
        output_dir: Directory to save the report
    
    Returns:
        Path to summary report
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Activation Visualization Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            h2 { color: #555; margin-top: 30px; }
            .visualization { margin-bottom: 30px; }
            img { max-width: 100%; border: 1px solid #ddd; }
            .description { margin-top: 10px; color: #666; }
        </style>
    </head>
    <body>
        <h1>Activation Visualization Report</h1>
    """
    
    # Group visualizations by layer
    layer_visualizations = {}
    
    for vis_type, paths in visualizations.items():
        for layer_name, path in paths.items():
            if layer_name not in layer_visualizations:
                layer_visualizations[layer_name] = {}
            
            layer_visualizations[layer_name][vis_type] = path
    
    # Add each layer's visualizations to the report
    for layer_name, vis_dict in layer_visualizations.items():
        html_content += f"<h2>Layer: {layer_name}</h2>\n"
        
        for vis_type, path in vis_dict.items():
            rel_path = os.path.relpath(path, output_dir)
            html_content += f"""
            <div class="visualization">
                <h3>{vis_type.replace('_', ' ').title()}</h3>
                <img src="{rel_path}" alt="{vis_type} for {layer_name}">
                <div class="description">
                    {vis_type.replace('_', ' ').title()} visualization for layer {layer_name}
                </div>
            </div>
            """
    
    # Close HTML content
    html_content += """
    </body>
    </html>
    """
    
    # Write report to file
    report_path = os.path.join(output_dir, "activation_report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Summary report generated at {report_path}")
    return report_path

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load models
    fp32_model = None
    int8_model = None
    
    if args.compare:
        # Load both models for comparison
        if not args.fp32_model or not args.int8_model:
            logger.error("Both --fp32-model and --int8-model must be provided for comparison")
            return 1
        
        fp32_model = load_model(args.fp32_model, device=device)
        int8_model = load_model(args.int8_model, device=device)
    else:
        # Load single model
        model_path = args.model
        if not model_path:
            logger.error("No model path provided. Use --model or --compare with --fp32-model and --int8-model")
            return 1
        
        fp32_model = load_model(model_path, device=device)
    
    # Print layer types if requested
    if args.show_layer_types:
        layer_types = {}
        for name, module in fp32_model.named_modules():
            module_type = str(type(module).__name__)
            if module_type not in layer_types:
                layer_types[module_type] = []
            layer_types[module_type].append(name)
        
        print("\nAvailable layer types and examples:")
        for layer_type, names in layer_types.items():
            print(f"\n{layer_type}:")
            for name in names[:3]:  # Show only first 3 examples
                print(f"  - {name}")
            if len(names) > 3:
                print(f"  - ... ({len(names)-3} more)")
        
        print("\nSpecify layers to visualize with --layers")
        return 0
    
    # Parse target layers
    target_layers = None
    if args.layers:
        target_layers = args.layers.split(',')
        logger.info(f"Targeting specific layers: {target_layers}")
    
    # Register hooks for FP32 model
    fp32_hooks, fp32_activations = register_activation_hooks(fp32_model, target_layers)
    
    # Register hooks for INT8 model if comparing
    int8_hooks = []
    int8_activations = {}
    if args.compare:
        int8_hooks, int8_activations = register_activation_hooks(int8_model, target_layers)
    
    # Process images
    if args.image:
        # Single image mode
        img = load_image(args.image)
        img = img.to(device)
        
        # Process through FP32 model
        with torch.no_grad():
            _ = fp32_model(img)
        
        # Process through INT8 model if comparing
        if args.compare:
            with torch.no_grad():
                _ = int8_model(img)
        
        # Image source for report
        image_source = f"Single image: {args.image}"
    else:
        # Dataset mode
        # Create dataloader
        val_loader = build_dataloader(args.data, batch_size=args.batch_size, mode="val")
        
        # Process images from dataloader
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= args.max_images:
                break
            
            # Extract images
            images = batch["img"]
            images = images.to(device)
            
            # Process through FP32 model
            with torch.no_grad():
                _ = fp32_model(images)
            
            # Process through INT8 model if comparing
            if args.compare:
                with torch.no_grad():
                    _ = int8_model(images)
        
        # Image source for report
        image_source = f"Dataset: {args.data}, first {min(args.max_images, len(val_loader))} images"
    
    # Remove hooks
    for hook in fp32_hooks:
        hook.remove()
    
    for hook in int8_hooks:
        hook.remove()
    
    # Create visualizations
    all_visualizations = {
        'activation_maps': {},
        'distributions': {},
        'comparisons': {},
        'quantization_error': {}
    }
    
    # Process FP32 activations
    for layer_name, activation in fp32_activations.items():
        # Skip if activation is empty or invalid
        if activation is None or activation.numel() == 0:
            continue
        
        # Create layer output directory
        layer_dir = os.path.join(args.output, "layers", layer_name.replace('.', '_').replace('/', '_'))
        os.makedirs(layer_dir, exist_ok=True)
        
        # Visualize activation maps
        map_path = visualize_activation_maps(
            activation, layer_name, layer_dir, 
            img_idx=0, max_features=args.max_features
        )
        if map_path:
            all_visualizations['activation_maps'][layer_name] = map_path
        
        # Visualize activation distribution
        dist_path = visualize_activation_distributions(
            activation, layer_name, layer_dir, 
            title=f"FP32 Activation Distribution for {layer_name}"
        )
        if dist_path:
            all_visualizations['distributions'][layer_name] = dist_path
        
        # If comparing models, create comparison visualizations
        if args.compare and layer_name in int8_activations:
            int8_activation = int8_activations[layer_name]
            
            # Compare distributions
            comp_path = compare_activation_distributions(
                activation, int8_activation, layer_name, layer_dir
            )
            if comp_path:
                all_visualizations['comparisons'][layer_name] = comp_path
            
            # Visualize quantization error
            error_path = visualize_quantization_error(
                activation, int8_activation, layer_name, layer_dir
            )
            if error_path:
                all_visualizations['quantization_error'][layer_name] = error_path
    
    # Generate summary report
    report_path = generate_summary_report(all_visualizations, args.output)
    
    logger.info(f"Visualization complete. Report available at {report_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())