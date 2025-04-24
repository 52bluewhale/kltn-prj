"""
Results visualization utilities for YOLOv8 QAT evaluation.

This module provides functions for visualizing evaluation results,
including detection visualization, plots for precision-recall curves,
confusion matrices, and activation distributions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
from pathlib import Path
import time
import json

# Setup logging
logger = logging.getLogger(__name__)

def plot_precision_recall_curve(
    precisions: List[float],
    recalls: List[float],
    classes: Optional[List[str]] = None,
    title: str = "Precision-Recall Curve",
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        precisions: List of precision values
        recalls: List of recall values
        classes: Optional list of class names
        title: Plot title
        output_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if classes is None:
        # Single PR curve
        ax.plot(recalls, precisions, marker='o', markersize=3, linewidth=2)
        
        # Calculate AP as area under curve
        ap = np.trapz(precisions, recalls)
        ax.set_title(f"{title} (AP: {ap:.4f})")
    else:
        # Multiple PR curves, one per class
        for i, (prec, rec) in enumerate(zip(precisions, recalls)):
            class_name = classes[i] if i < len(classes) else f"Class {i}"
            ax.plot(rec, prec, marker='o', markersize=3, linewidth=2, label=class_name)
            
        ax.set_title(title)
        ax.legend(loc="best")
    
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    classes: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix as numpy array
        classes: Optional list of class names
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        output_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    if normalize:
        # Normalize confusion matrix
        row_sums = confusion_matrix.sum(axis=1)
        confusion_matrix = confusion_matrix / row_sums[:, np.newaxis]
        confusion_matrix = np.nan_to_num(confusion_matrix)  # Replace NaNs with zeros
    
    # Create colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white", "#4363d8", "#3cb44b"], N=256)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot confusion matrix
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized frequency" if normalize else "Count", rotation=-90, va="bottom")
    
    # Set labels
    num_classes = confusion_matrix.shape[0]
    if classes is None:
        classes = [f"Class {i}" for i in range(num_classes)]
    
    # Add labels and ticks
    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    
    # Add text annotations inside cells
    thresh = confusion_matrix.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f"{confusion_matrix[i, j]:.2f}" if normalize else f"{int(confusion_matrix[i, j])}",
                    ha="center", va="center", 
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    
    # Add labels and title
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    
    fig.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_detections(
    image: Union[np.ndarray, torch.Tensor, str],
    predictions: Union[np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    confidence_threshold: float = 0.5,
    color_mapping: Optional[Dict[int, Tuple[int, int, int]]] = None,
    title: str = "Detection Results",
    output_path: Optional[str] = None,
    show_scores: bool = True
) -> np.ndarray:
    """
    Visualize object detection results on an image.
    
    Args:
        image: Input image as numpy array, tensor, or path to image file
        predictions: Prediction tensor [num_preds, 6] (x1, y1, x2, y2, conf, class_id)
        class_names: Optional list of class names
        confidence_threshold: Confidence threshold for visualizing predictions
        color_mapping: Optional mapping from class indices to BGR colors
        title: Plot title
        output_path: Optional path to save the visualization
        show_scores: Whether to show confidence scores
        
    Returns:
        Visualization as numpy array (BGR format)
    """
    # Load image if it's a path
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert tensor to numpy array
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        
        # Handle different tensor formats
        if image.ndim == 4:  # batch, channels, height, width
            image = image[0]  # Take first image in batch
        
        if image.shape[0] == 3:  # channels, height, width
            image = np.transpose(image, (1, 2, 0))  # -> height, width, channels
        
        # Normalize if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Make a copy to avoid modifying original
    vis_image = image.copy()
    
    # Convert predictions to numpy array
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    # Filter predictions by confidence
    if predictions.shape[1] >= 6:  # x1, y1, x2, y2, conf, class
        predictions = predictions[predictions[:, 4] >= confidence_threshold]
    
    # Create default color mapping if not provided
    if color_mapping is None:
        # Generate distinct colors
        num_classes = len(np.unique(predictions[:, 5])) if len(predictions) > 0 else 10
        color_mapping = {}
        for i in range(num_classes):
            # HSV color space for more distinct colors
            hue = i / num_classes
            rgb = plt.cm.hsv(hue)[:3]  # Convert HSV to RGB
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # Convert RGB to BGR
            color_mapping[i] = bgr
    
    # Default class names if not provided
    if class_names is None:
        max_class = int(max(predictions[:, 5])) if len(predictions) > 0 else 9
        class_names = [f"Class {i}" for i in range(max_class + 1)]
    
    # Draw predictions
    for pred in predictions:
        x1, y1, x2, y2 = int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])
        conf = pred[4]
        class_id = int(pred[5])
        
        # Get color and class name
        color = color_mapping.get(class_id, (0, 255, 0))  # Default to green
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{class_name} {conf:.2f}" if show_scores else class_name
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_image, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add title
    (title_width, title_height), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    title_x = (vis_image.shape[1] - title_width) // 2
    cv2.putText(vis_image, title, (title_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Save visualization if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return vis_image

def visualize_activation_distributions(
    fp32_activations: Dict[str, torch.Tensor],
    int8_activations: Dict[str, torch.Tensor],
    layers_to_plot: Optional[List[str]] = None,
    num_bins: int = 100,
    title: str = "Activation Distributions",
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize and compare activation distributions between FP32 and INT8 models.
    
    Args:
        fp32_activations: Dictionary mapping layer names to FP32 activations
        int8_activations: Dictionary mapping layer names to INT8 activations
        layers_to_plot: Optional list of layer names to plot (plots all if None)
        num_bins: Number of histogram bins
        title: Plot title
        output_path: Optional path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    # Select layers to plot
    if layers_to_plot is None:
        # Plot common layers
        common_layers = set(fp32_activations.keys()).intersection(set(int8_activations.keys()))
        layers_to_plot = list(common_layers)
        
        # Limit to at most 9 layers for readability
        if len(layers_to_plot) > 9:
            layers_to_plot = layers_to_plot[:9]
    
    # Determine subplot grid size
    n = len(layers_to_plot)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    
    # Flatten axes for easier indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot activations for each layer
    for i, layer_name in enumerate(layers_to_plot):
        ax = axes[i]
        
        if layer_name in fp32_activations and layer_name in int8_activations:
            # Get activations
            fp32_act = fp32_activations[layer_name].flatten().cpu().numpy()
            int8_act = int8_activations[layer_name].flatten().cpu().numpy()
            
            # Plot histograms
            ax.hist(fp32_act, bins=num_bins, alpha=0.7, label="FP32", color="blue")
            ax.hist(int8_act, bins=num_bins, alpha=0.7, label="INT8", color="red")
            
            # Add legend and labels
            ax.legend()
            ax.set_title(layer_name)
            ax.set_xlabel("Activation Value")
            ax.set_ylabel("Frequency")
            
            # Add statistics
            fp32_mean = np.mean(fp32_act)
            fp32_std = np.std(fp32_act)
            int8_mean = np.mean(int8_act)
            int8_std = np.std(int8_act)
            
            stats_text = f"FP32: μ={fp32_mean:.4f}, σ={fp32_std:.4f}\nINT8: μ={int8_mean:.4f}, σ={int8_std:.4f}"
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(len(layers_to_plot), len(axes)):
        fig.delaxes(axes[i])
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def compare_detection_results(
    image: Union[np.ndarray, torch.Tensor, str],
    fp32_predictions: Union[np.ndarray, torch.Tensor],
    int8_predictions: Union[np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    confidence_threshold: float = 0.5,
    title: str = "FP32 vs INT8 Detection Comparison",
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Compare detection results between FP32 and INT8 models.
    
    Args:
        image: Input image as numpy array, tensor, or path to image file
        fp32_predictions: Prediction tensor from FP32 model
        int8_predictions: Prediction tensor from INT8 model
        class_names: Optional list of class names
        confidence_threshold: Confidence threshold for visualizing predictions
        title: Plot title
        output_path: Optional path to save the visualization
        
    Returns:
        Visualization as numpy array (BGR format)
    """
    # Visualize FP32 detections
    fp32_vis = visualize_detections(
        image=image,
        predictions=fp32_predictions,
        class_names=class_names,
        confidence_threshold=confidence_threshold,
        title="FP32 Model",
        show_scores=True,
    )
    
    # Visualize INT8 detections
    int8_vis = visualize_detections(
        image=image,
        predictions=int8_predictions,
        class_names=class_names,
        confidence_threshold=confidence_threshold,
        title="INT8 Model",
        show_scores=True,
    )
    
    # Create comparison image
    height, width = fp32_vis.shape[:2]
    comparison = np.zeros((height, width * 2, 3), dtype=np.uint8)
    
    # Add visualizations side by side
    comparison[:, :width] = fp32_vis
    comparison[:, width:] = int8_vis
    
    # Add dividing line
    cv2.line(comparison, (width, 0), (width, height), (255, 255, 255), 2)
    
    # Add title
    (title_width, title_height), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    title_x = (comparison.shape[1] - title_width) // 2
    cv2.putText(comparison, title, (title_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Save comparison if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    return comparison

def generate_evaluation_report(
    evaluation_results: Dict[str, Any],
    output_path: str = "./evaluation_report",
    include_plots: bool = True,
    include_images: bool = True
) -> str:
    """
    Generate comprehensive evaluation report from results.
    
    Args:
        evaluation_results: Dictionary with evaluation results
        output_path: Path to save the report
        include_plots: Whether to include plots in report
        include_images: Whether to include detection images in report
        
    Returns:
        Path to the generated report
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize report parts
    report_parts = []
    
    # Add report header
    report_parts.append(f"# YOLOv8 QAT Evaluation Report\n")
    report_parts.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Model information section
    if 'model_info' in evaluation_results:
        report_parts.append("## Model Information\n\n")
        model_info = evaluation_results['model_info']
        
        if 'fp32_model' in model_info:
            report_parts.append(f"### FP32 Model\n")
            fp32_info = model_info['fp32_model']
            report_parts.append(f"- Model name: {fp32_info.get('name', 'N/A')}\n")
            report_parts.append(f"- Model size: {fp32_info.get('size_mb', 0):.2f} MB\n")
            report_parts.append(f"- Parameters: {fp32_info.get('num_parameters', 0):,}\n\n")
        
        if 'int8_model' in model_info:
            report_parts.append(f"### INT8 Model\n")
            int8_info = model_info['int8_model']
            report_parts.append(f"- Model name: {int8_info.get('name', 'N/A')}\n")
            report_parts.append(f"- Model size: {int8_info.get('size_mb', 0):.2f} MB\n")
            report_parts.append(f"- Parameters: {int8_info.get('num_parameters', 0):,}\n")
            report_parts.append(f"- Compression ratio: {model_info.get('compression_ratio', 0):.2f}x\n\n")
    
    # Performance metrics section
    report_parts.append("## Performance Metrics\n\n")
    
    # Add mAP results
    if 'mAP' in evaluation_results:
        report_parts.append("### Mean Average Precision (mAP)\n\n")
        map_results = evaluation_results['mAP']
        
        if isinstance(map_results, dict):
            # Format mAP results as a table
            report_parts.append("| Metric | Value |\n")
            report_parts.append("|--------|-------|\n")
            
            for metric, value in map_results.items():
                if not metric.startswith('AP@') and isinstance(value, (int, float)):
                    report_parts.append(f"| {metric} | {value:.4f} |\n")
            
            report_parts.append("\n")
            
            # Class-specific AP
            class_aps = {k: v for k, v in map_results.items() if k.startswith('AP@')}
            if class_aps:
                report_parts.append("#### Class-wise AP@.5\n\n")
                report_parts.append("| Class | AP@.5 |\n")
                report_parts.append("|-------|-------|\n")
                
                for class_name, ap in class_aps.items():
                    report_parts.append(f"| {class_name.replace('AP@.5_class', 'Class ')} | {ap:.4f} |\n")
                
                report_parts.append("\n")
    
    # Add precision-recall results
    if 'precision_recall' in evaluation_results:
        report_parts.append("### Precision, Recall, and F1 Score\n\n")
        pr_results = evaluation_results['precision_recall']
        
        if isinstance(pr_results, dict):
            # Format PR results as a table
            report_parts.append("| Class | Precision | Recall | F1 Score | TP | FP | FN |\n")
            report_parts.append("|-------|-----------|--------|----------|----|----|----|\n")
            
            for class_name, metrics in pr_results.items():
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1 = metrics.get('f1_score', 0)
                tp = metrics.get('true_positives', 0)
                fp = metrics.get('false_positives', 0)
                fn = metrics.get('false_negatives', 0)
                
                report_parts.append(f"| {class_name} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {tp} | {fp} | {fn} |\n")
            
            report_parts.append("\n")
    
    # Add latency results
    if 'latency' in evaluation_results:
        report_parts.append("### Inference Performance\n\n")
        latency = evaluation_results['latency']
        
        if isinstance(latency, dict):
            # Format latency results as a table
            report_parts.append("| Metric | Value |\n")
            report_parts.append("|--------|-------|\n")
            
            mean_time = latency.get('mean_inference_time', 0)
            report_parts.append(f"| Mean inference time | {mean_time*1000:.2f} ms |\n")
            
            median_time = latency.get('median_inference_time', 0)
            report_parts.append(f"| Median inference time | {median_time*1000:.2f} ms |\n")
            
            std_time = latency.get('std_inference_time', 0)
            report_parts.append(f"| Std. dev. of inference time | {std_time*1000:.2f} ms |\n")
            
            fps = latency.get('fps', 0)
            report_parts.append(f"| Frames per second (FPS) | {fps:.2f} |\n")
            
            report_parts.append("\n")
    
    # Model comparison section
    if 'comparison' in evaluation_results:
        report_parts.append("## Model Comparison (FP32 vs INT8)\n\n")
        comp_results = evaluation_results['comparison']
        
        if 'accuracy_change' in comp_results:
            acc_change = comp_results['accuracy_change']
            report_parts.append(f"### Accuracy Change\n\n")
            report_parts.append(f"- FP32 mAP@.5: {acc_change.get('fp32_map50', 0):.4f}\n")
            report_parts.append(f"- INT8 mAP@.5: {acc_change.get('int8_map50', 0):.4f}\n")
            report_parts.append(f"- Absolute change: {acc_change.get('absolute_change', 0):.4f}\n")
            report_parts.append(f"- Relative change: {acc_change.get('relative_change', 0)*100:.2f}%\n\n")
        
        if 'speed_comparison' in comp_results:
            speed_comp = comp_results['speed_comparison']
            report_parts.append(f"### Speed Comparison\n\n")
            report_parts.append(f"- FP32 inference time: {speed_comp.get('fp32_time', 0)*1000:.2f} ms\n")
            report_parts.append(f"- INT8 inference time: {speed_comp.get('int8_time', 0)*1000:.2f} ms\n")
            report_parts.append(f"- Speedup: {speed_comp.get('speedup', 0):.2f}x\n\n")
        
        if 'memory_comparison' in comp_results:
            mem_comp = comp_results['memory_comparison']
            report_parts.append(f"### Memory Usage Comparison\n\n")
            report_parts.append(f"- FP32 model size: {mem_comp.get('fp32_size_mb', 0):.2f} MB\n")
            report_parts.append(f"- INT8 model size: {mem_comp.get('int8_size_mb', 0):.2f} MB\n")
            report_parts.append(f"- Size reduction: {mem_comp.get('size_reduction_percent', 0):.2f}%\n\n")
    
    # Generate plots if requested
    if include_plots and 'precision_recall' in evaluation_results:
        report_parts.append("## Visualizations\n\n")
        
        # Precision-Recall curves
        pr_data = evaluation_results['precision_recall']
        if isinstance(pr_data, dict) and len(pr_data) > 0:
            # Extract precision and recall values for each class
            precisions = []
            recalls = []
            class_names = []
            
            for class_name, metrics in pr_data.items():
                if 'precision' in metrics and 'recall' in metrics:
                    precisions.append(metrics['precision'])
                    recalls.append(metrics['recall'])
                    class_names.append(class_name)
            
            # Generate PR curve plot
            if precisions and recalls:
                pr_curve_path = os.path.join(output_path, "precision_recall_curve.png")
                plot_precision_recall_curve(precisions, recalls, class_names, 
                                           "Precision-Recall Curves by Class", pr_curve_path)
                report_parts.append(f"### Precision-Recall Curves\n\n")
                report_parts.append(f"![Precision-Recall Curves](precision_recall_curve.png)\n\n")
    
    # Generate confusion matrix if available
    if include_plots and 'confusion_matrix' in evaluation_results:
        cm = evaluation_results['confusion_matrix']
        if isinstance(cm, np.ndarray):
            cm_path = os.path.join(output_path, "confusion_matrix.png")
            plot_confusion_matrix(cm, normalize=True, title="Normalized Confusion Matrix", 
                                 output_path=cm_path)
            report_parts.append(f"### Confusion Matrix\n\n")
            report_parts.append(f"![Confusion Matrix](confusion_matrix.png)\n\n")
    
    # Include example detections if available
    if include_images and 'detection_examples' in evaluation_results:
        examples = evaluation_results['detection_examples']
        if isinstance(examples, list) and len(examples) > 0:
            report_parts.append(f"### Detection Examples\n\n")
            
            for i, example in enumerate(examples):
                if 'fp32_image' in example and 'int8_image' in example:
                    comparison_path = os.path.join(output_path, f"detection_comparison_{i}.png")
                    
                    # Create comparison image
                    comparison = np.hstack([example['fp32_image'], example['int8_image']])
                    cv2.imwrite(comparison_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
                    
                    report_parts.append(f"#### Example {i+1}\n\n")
                    report_parts.append(f"![Detection Comparison {i+1}](detection_comparison_{i}.png)\n\n")
                    
                    # Add metrics for this example if available
                    if 'metrics' in example:
                        metrics = example['metrics']
                        report_parts.append("| Metric | FP32 | INT8 |\n")
                        report_parts.append("|--------|------|------|\n")
                        
                        for metric_name, values in metrics.items():
                            fp32_val = values.get('fp32', 'N/A')
                            int8_val = values.get('int8', 'N/A')
                            report_parts.append(f"| {metric_name} | {fp32_val} | {int8_val} |\n")
                        
                        report_parts.append("\n")
    
    # Join all report parts
    report_content = "".join(report_parts)
    
    # Write report to file
    report_path = os.path.join(output_path, "report.md")
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    # Convert to HTML if possible
    try:
        import markdown
        html_content = markdown.markdown(report_content, extensions=['tables'])
        
        html_path = os.path.join(output_path, "report.html")
        with open(html_path, 'w') as f:
            f.write(f"<!DOCTYPE html>\n<html>\n<head>\n")
            f.write(f"<title>YOLOv8 QAT Evaluation Report</title>\n")
            f.write(f"<style>\n")
            f.write(f"body {{ font-family: Arial, sans-serif; margin: 20px; }}\n")
            f.write(f"table {{ border-collapse: collapse; width: 100%; }}\n")
            f.write(f"th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}\n")
            f.write(f"th {{ background-color: #f2f2f2; }}\n")
            f.write(f"img {{ max-width: 100%; }}\n")
            f.write(f"</style>\n</head>\n<body>\n")
            f.write(html_content)
            f.write(f"\n</body>\n</html>")
        
        logger.info(f"HTML report generated at {html_path}")
    except ImportError:
        logger.info("Markdown module not found, skipping HTML report generation")
    
    logger.info(f"Evaluation report generated at {report_path}")
    
    return report_path