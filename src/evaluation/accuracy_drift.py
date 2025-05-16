# Tracks accuracy changes during quantization
"""
Accuracy drift analysis utilities for YOLOv8 QAT evaluation.

This module provides functions for analyzing how quantization affects model accuracy,
identifying critical layers, and tracking accuracy changes across different classes.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import json

# Setup logging
logger = logging.getLogger(__name__)

def track_accuracy_change(
    fp32_model: torch.nn.Module,
    int8_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int = 10,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Track accuracy changes between FP32 and INT8 models.
    
    Args:
        fp32_model: Floating point model
        int8_model: Quantized model
        dataloader: DataLoader with evaluation dataset
        num_classes: Number of classes
        confidence_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for matching predictions to targets
        device: Device to run evaluation on
        
    Returns:
        Dictionary with accuracy change metrics
    """
    # Set models to evaluation mode
    fp32_model.eval()
    int8_model.eval()
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    fp32_model.to(device)
    int8_model.to(device)
    
    # Initialize metrics
    class_correct_fp32 = np.zeros(num_classes)
    class_correct_int8 = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    # Initialize confusion matrices
    fp32_confusion = np.zeros((num_classes, num_classes))
    int8_confusion = np.zeros((num_classes, num_classes))
    
    # Evaluation loop
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Tracking accuracy")):
            # Process batch
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                images, targets = batch[0], batch[1]
            else:
                # For custom dataset formats
                images, targets = batch['image'], batch['target']
            
            # Move to device
            images = images.to(device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device)
            elif isinstance(targets, (tuple, list)):
                targets = [t.to(device) if isinstance(t, torch.Tensor) else t for t in targets]
            
            # Run models
            fp32_preds = fp32_model(images)
            int8_preds = int8_model(images)
            
            # Process predictions and targets for each image in batch
            for i in range(len(images)):
                # Extract predictions and targets for this image
                fp32_pred = fp32_preds[i] if isinstance(fp32_preds, list) else fp32_preds[i:i+1]
                int8_pred = int8_preds[i] if isinstance(int8_preds, list) else int8_preds[i:i+1]
                target = targets[i] if isinstance(targets, list) else targets[i:i+1]
                
                # Filter predictions by confidence
                if hasattr(fp32_pred, 'shape') and fp32_pred.shape[0] > 0 and fp32_pred.shape[1] >= 6:
                    fp32_pred = fp32_pred[fp32_pred[:, 4] >= confidence_threshold]
                
                if hasattr(int8_pred, 'shape') and int8_pred.shape[0] > 0 and int8_pred.shape[1] >= 6:
                    int8_pred = int8_pred[int8_pred[:, 4] >= confidence_threshold]
                
                # Process each ground truth object
                if hasattr(target, 'shape') and target.shape[0] > 0:
                    for j in range(target.shape[0]):
                        gt_class = int(target[j, 0].item())
                        gt_box = target[j, 1:5] if target.shape[1] >= 5 else None
                        
                        # Skip if class is out of range
                        if gt_class >= num_classes:
                            continue
                        
                        class_total[gt_class] += 1
                        
                        # Check FP32 predictions
                        if hasattr(fp32_pred, 'shape') and fp32_pred.shape[0] > 0 and gt_box is not None:
                            # Calculate IoU with all predictions
                            ious = box_iou(gt_box.unsqueeze(0), fp32_pred[:, :4])
                            
                            if ious.shape[1] > 0:
                                # Find best match
                                best_iou, best_idx = ious.max(1)
                                
                                if best_iou >= iou_threshold:
                                    pred_class = int(fp32_pred[best_idx, 5].item())
                                    
                                    # Update confusion matrix
                                    fp32_confusion[gt_class, pred_class] += 1
                                    
                                    # Check if class prediction is correct
                                    if pred_class == gt_class:
                                        class_correct_fp32[gt_class] += 1
                        
                        # Check INT8 predictions
                        if hasattr(int8_pred, 'shape') and int8_pred.shape[0] > 0 and gt_box is not None:
                            # Calculate IoU with all predictions
                            ious = box_iou(gt_box.unsqueeze(0), int8_pred[:, :4])
                            
                            if ious.shape[1] > 0:
                                # Find best match
                                best_iou, best_idx = ious.max(1)
                                
                                if best_iou >= iou_threshold:
                                    pred_class = int(int8_pred[best_idx, 5].item())
                                    
                                    # Update confusion matrix
                                    int8_confusion[gt_class, pred_class] += 1
                                    
                                    # Check if class prediction is correct
                                    if pred_class == gt_class:
                                        class_correct_int8[gt_class] += 1
    
    # Calculate class-wise accuracy
    class_accuracy_fp32 = np.zeros(num_classes)
    class_accuracy_int8 = np.zeros(num_classes)
    
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy_fp32[i] = class_correct_fp32[i] / class_total[i]
            class_accuracy_int8[i] = class_correct_int8[i] / class_total[i]
    
    # Calculate overall accuracy
    overall_accuracy_fp32 = class_correct_fp32.sum() / class_total.sum() if class_total.sum() > 0 else 0
    overall_accuracy_int8 = class_correct_int8.sum() / class_total.sum() if class_total.sum() > 0 else 0
    
    # Calculate absolute and relative changes
    absolute_change = class_accuracy_int8 - class_accuracy_fp32
    relative_change = np.zeros(num_classes)
    
    for i in range(num_classes):
        if class_accuracy_fp32[i] > 0:
            relative_change[i] = absolute_change[i] / class_accuracy_fp32[i]
    
    # Calculate normalized confusion matrices
    fp32_confusion_norm = np.zeros_like(fp32_confusion)
    int8_confusion_norm = np.zeros_like(int8_confusion)
    
    for i in range(num_classes):
        row_sum = fp32_confusion[i].sum()
        if row_sum > 0:
            fp32_confusion_norm[i] = fp32_confusion[i] / row_sum
        
        row_sum = int8_confusion[i].sum()
        if row_sum > 0:
            int8_confusion_norm[i] = int8_confusion[i] / row_sum
    
    # Return results
    return {
        'class_accuracy_fp32': class_accuracy_fp32,
        'class_accuracy_int8': class_accuracy_int8,
        'class_total': class_total,
        'overall_accuracy_fp32': overall_accuracy_fp32,
        'overall_accuracy_int8': overall_accuracy_int8,
        'absolute_change': absolute_change,
        'relative_change': relative_change,
        'fp32_confusion': fp32_confusion,
        'int8_confusion': int8_confusion,
        'fp32_confusion_norm': fp32_confusion_norm,
        'int8_confusion_norm': int8_confusion_norm
    }

def box_iou(box1, box2):
    """
    Calculate IoU between two sets of boxes.
    
    Args:
        box1: First set of boxes (N, 4)
        box2: Second set of boxes (M, 4)
        
    Returns:
        IoU tensor (N, M)
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Calculate intersection area
    left_top = torch.max(box1[:, None, :2], box2[:, :2])
    right_bottom = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    # Calculate union area
    union = area1[:, None] + area2 - inter
    
    # Calculate IoU
    iou = inter / (union + 1e-6)
    
    return iou

def analyze_quantization_impact(
    fp32_model: torch.nn.Module,
    int8_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int = 10,
    device: str = "cuda",
    export_results: bool = True,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze the impact of quantization on model accuracy.
    
    Args:
        fp32_model: Floating point model
        int8_model: Quantized model
        dataloader: DataLoader with evaluation dataset
        num_classes: Number of classes
        device: Device to run evaluation on
        export_results: Whether to export results to file
        output_path: Path to save results
        
    Returns:
        Dictionary with impact analysis results
    """
    # Set output path
    if output_path is None:
        output_path = "./quantization_impact"
    
    # Create output directory if exporting results
    if export_results:
        os.makedirs(output_path, exist_ok=True)
    
    # Track accuracy change
    accuracy_results = track_accuracy_change(
        fp32_model=fp32_model,
        int8_model=int8_model,
        dataloader=dataloader,
        num_classes=num_classes,
        device=device
    )
    
    # Calculate additional metrics
    overall_absolute_change = accuracy_results['overall_accuracy_int8'] - accuracy_results['overall_accuracy_fp32']
    overall_relative_change = overall_absolute_change / accuracy_results['overall_accuracy_fp32'] if accuracy_results['overall_accuracy_fp32'] > 0 else 0
    
    # Find classes with largest accuracy drop
    class_changes = [(i, accuracy_results['absolute_change'][i]) for i in range(num_classes)]
    class_changes.sort(key=lambda x: x[1])  # Sort by change (ascending, so most negative first)
    
    most_affected_classes = class_changes[:3]  # Top 3 most negatively affected
    least_affected_classes = class_changes[-3:]  # Top 3 least affected (or most positively affected)
    
    # Prepare results
    results = {
        'overall_accuracy_fp32': accuracy_results['overall_accuracy_fp32'],
        'overall_accuracy_int8': accuracy_results['overall_accuracy_int8'],
        'overall_absolute_change': overall_absolute_change,
        'overall_relative_change': overall_relative_change,
        'class_metrics': {
            'class_accuracy_fp32': accuracy_results['class_accuracy_fp32'].tolist(),
            'class_accuracy_int8': accuracy_results['class_accuracy_int8'].tolist(),
            'class_absolute_change': accuracy_results['absolute_change'].tolist(),
            'class_relative_change': accuracy_results['relative_change'].tolist(),
            'class_total': accuracy_results['class_total'].tolist()
        },
        'most_affected_classes': most_affected_classes,
        'least_affected_classes': least_affected_classes
    }
    
    # Export results if requested
    if export_results:
        # Export to JSON
        json_path = os.path.join(output_path, 'quantization_impact.json')
        
        # Ensure serializable
        serializable_results = {
            'overall_accuracy_fp32': float(results['overall_accuracy_fp32']),
            'overall_accuracy_int8': float(results['overall_accuracy_int8']),
            'overall_absolute_change': float(results['overall_absolute_change']),
            'overall_relative_change': float(results['overall_relative_change']),
            'class_metrics': {
                'class_accuracy_fp32': [float(x) for x in results['class_metrics']['class_accuracy_fp32']],
                'class_accuracy_int8': [float(x) for x in results['class_metrics']['class_accuracy_int8']],
                'class_absolute_change': [float(x) for x in results['class_metrics']['class_absolute_change']],
                'class_relative_change': [float(x) for x in results['class_metrics']['class_relative_change']],
                'class_total': [int(x) for x in results['class_metrics']['class_total']]
            },
            'most_affected_classes': [[int(c[0]), float(c[1])] for c in results['most_affected_classes']],
            'least_affected_classes': [[int(c[0]), float(c[1])] for c in results['least_affected_classes']]
        }
        
        # Save to JSON
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Quantization impact results saved to {json_path}")
        
        # Create accuracy comparison plot
        plt.figure(figsize=(12, 6))
        
        # Show side by side bar chart of FP32 vs INT8 accuracy by class
        classes = list(range(num_classes))
        fp32_values = accuracy_results['class_accuracy_fp32']
        int8_values = accuracy_results['class_accuracy_int8']
        
        x = np.arange(len(classes))
        width = 0.35
        
        plt.bar(x - width/2, fp32_values, width, label='FP32')
        plt.bar(x + width/2, int8_values, width, label='INT8')
        
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Class-wise Accuracy Comparison')
        plt.xticks(x, classes)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        accuracy_plot_path = os.path.join(output_path, 'accuracy_comparison.png')
        plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create accuracy change plot
        plt.figure(figsize=(12, 6))
        
        # Plot absolute change
        plt.bar(x, accuracy_results['absolute_change'], color=['red' if c < 0 else 'green' for c in accuracy_results['absolute_change']])
        
        plt.xlabel('Class')
        plt.ylabel('Absolute Accuracy Change')
        plt.title('Impact of Quantization on Class Accuracy')
        plt.xticks(x, classes)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add annotations for overall change
        plt.figtext(0.5, 0.01, f'Overall Accuracy Change: {overall_absolute_change:.4f} ({overall_relative_change*100:.2f}%)', 
                   ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        change_plot_path = os.path.join(output_path, 'accuracy_change.png')
        plt.savefig(change_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create confusion matrix plots
        plt.figure(figsize=(12, 10))
        
        # Plot FP32 confusion matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(accuracy_results['fp32_confusion_norm'], annot=True, fmt='.2f', cmap='Blues')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('FP32 Model Confusion Matrix')
        
        # Plot INT8 confusion matrix
        plt.subplot(1, 2, 2)
        sns.heatmap(accuracy_results['int8_confusion_norm'], annot=True, fmt='.2f', cmap='Blues')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('INT8 Model Confusion Matrix')
        
        plt.tight_layout()
        
        # Save plot
        confusion_plot_path = os.path.join(output_path, 'confusion_matrices.png')
        plt.savefig(confusion_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

def identify_critical_layers(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_patterns: Optional[List[str]] = None,
    num_classes: int = 10,
    device: str = "cuda",
    export_results: bool = True,
    output_path: Optional[str] = None
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Identify layers critical for maintaining accuracy after quantization.
    
    This function performs a sensitivity analysis by simulating quantization on 
    individual layers and measuring the impact on accuracy.
    
    Args:
        model: Model to analyze
        dataloader: DataLoader with evaluation dataset
        layer_patterns: Optional list of layer name patterns to test
        num_classes: Number of classes
        device: Device to run evaluation on
        export_results: Whether to export results to file
        output_path: Path to save results
        
    Returns:
        Dictionary with critical layers and their impact scores
    """
    # Set output path
    if output_path is None:
        output_path = "./critical_layers"
    
    # Create output directory if exporting results
    if export_results:
        os.makedirs(output_path, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(device)
    
    # Get baseline accuracy
    from .metrics import compute_evaluation_metrics
    
    baseline_metrics = compute_evaluation_metrics(
        model=model,
        dataloader=dataloader,
        metrics=['accuracy'],
        device=device
    )
    
    baseline_accuracy = baseline_metrics.get('accuracy', 0)
    
    # Find all layers matching patterns
    if layer_patterns is None:
        # Default patterns for common layer types
        layer_patterns = [
            r'.*\.conv\d+$',
            r'.*\.bn\d+$',
            r'.*\.linear$',
            r'.*\.fc$'
        ]
    
    # Collect layers to test
    import re
    layers_to_test = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            # Check if layer matches any pattern
            for pattern in layer_patterns:
                if re.match(pattern, name):
                    layers_to_test.append((name, module))
                    break
    
    logger.info(f"Found {len(layers_to_test)} layers to test for criticality")
    
    # Simulate quantization on each layer and measure accuracy impact
    layer_impacts = []
    
    for name, module in tqdm(layers_to_test, desc="Testing layer criticality"):
        # Create a temporary copy of the model
        import copy
        temp_model = copy.deepcopy(model)
        
        # Find corresponding module in the copy
        modules_dict = dict(temp_model.named_modules())
        if name in modules_dict:
            module_copy = modules_dict[name]
            
            # Apply simulated quantization to the module
            _apply_simulated_quantization(module_copy)
            
            # Measure accuracy with this layer quantized
            metrics = compute_evaluation_metrics(
                model=temp_model,
                dataloader=dataloader,
                metrics=['accuracy'],
                device=device
            )
            
            # Calculate accuracy change
            quantized_accuracy = metrics.get('accuracy', 0)
            accuracy_change = baseline_accuracy - quantized_accuracy
            
            # Store impact
            layer_impacts.append((name, accuracy_change))
            
            # Clean up
            del temp_model
    
    # Sort layers by impact (higher means more critical)
    layer_impacts.sort(key=lambda x: x[1], reverse=True)
    
    # Categorize layers
    high_impact = [(name, impact) for name, impact in layer_impacts if impact > 0.05]
    medium_impact = [(name, impact) for name, impact in layer_impacts if 0.01 <= impact <= 0.05]
    low_impact = [(name, impact) for name, impact in layer_impacts if impact < 0.01]
    
    results = {
        'high_impact_layers': high_impact,
        'medium_impact_layers': medium_impact,
        'low_impact_layers': low_impact,
        'all_layers': layer_impacts
    }
    
    # Export results if requested
    if export_results:
        # Export to CSV
        csv_path = os.path.join(output_path, 'critical_layers.csv')
        
        # Create DataFrame
        df = pd.DataFrame(layer_impacts, columns=['layer_name', 'accuracy_impact'])
        df = df.sort_values('accuracy_impact', ascending=False)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Critical layer analysis saved to {csv_path}")
        
        # Create bar chart of top N most critical layers
        top_n = min(20, len(layer_impacts))
        
        plt.figure(figsize=(12, 8))
        
        top_layers = df.head(top_n)
        plt.barh(top_layers['layer_name'], top_layers['accuracy_impact'])
        plt.xlabel('Accuracy Impact')
        plt.title(f'Top {top_n} Most Critical Layers')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        critical_plot_path = os.path.join(output_path, 'critical_layers.png')
        plt.savefig(critical_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

def measure_drift_per_class(
    fp32_model: torch.nn.Module,
    int8_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: Optional[List[str]] = None,
    confidence_threshold: float = 0.5,
    num_classes: int = 10,
    device: str = "cuda",
    export_results: bool = True,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Measure accuracy drift per class due to quantization.
    
    Args:
        fp32_model: Floating point model
        int8_model: Quantized model
        dataloader: DataLoader with evaluation dataset
        class_names: Optional list of class names
        confidence_threshold: Confidence threshold for predictions
        num_classes: Number of classes
        device: Device to run evaluation on
        export_results: Whether to export results to file
        output_path: Path to save results
        
    Returns:
        Dictionary with per-class drift metrics
    """
    # Set output path
    if output_path is None:
        output_path = "./class_drift"
    
    # Create output directory if exporting results
    if export_results:
        os.makedirs(output_path, exist_ok=True)
    
    # Set models to evaluation mode
    fp32_model.eval()
    int8_model.eval()
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    fp32_model.to(device)
    int8_model.to(device)
    
    # Initialize metrics
    class_metrics = defaultdict(lambda: {
        'fp32_correct': 0,
        'int8_correct': 0,
        'fp32_predictions': 0,
        'int8_predictions': 0,
        'total_instances': 0,
        'fp32_only_correct': 0,
        'int8_only_correct': 0,
        'both_correct': 0,
        'both_incorrect': 0
    })
    
    # Use default class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Evaluation loop
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Measuring per-class drift")):
            # Process batch
            images, targets = None, None
            
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                images, targets = batch[0], batch[1]
            elif isinstance(batch, dict) and 'image' in batch and 'target' in batch:
                images, targets = batch['image'], batch['target']
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")
            
            # Move to device
            images = images.to(device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(device)
            elif isinstance(targets, (tuple, list)):
                targets = [t.to(device) if isinstance(t, torch.Tensor) else t for t in targets]
            
            # Run models
            fp32_preds = fp32_model(images)
            int8_preds = int8_model(images)
            
            # Process each image in batch
            for i in range(images.shape[0]):
                # Get predictions and targets for this image
                fp32_pred = fp32_preds[i] if isinstance(fp32_preds, list) else fp32_preds[i:i+1]
                int8_pred = int8_preds[i] if isinstance(int8_preds, list) else int8_preds[i:i+1]
                target = targets[i] if isinstance(targets, list) else targets[i:i+1]
                
                # Process each object in the image
                if hasattr(target, 'shape') and target.shape[0] > 0:
                    for j in range(target.shape[0]):
                        gt_class = int(target[j, 0].item())
                        gt_box = target[j, 1:5] if target.shape[1] >= 5 else None
                        
                        # Skip if class is out of range
                        if gt_class >= num_classes:
                            continue
                        
                        class_name = class_names[gt_class] if gt_class < len(class_names) else f"Class {gt_class}"
                        class_metrics[class_name]['total_instances'] += 1
                        
                        # Check FP32 prediction
                        fp32_correct = False
                        if hasattr(fp32_pred, 'shape') and fp32_pred.shape[0] > 0 and gt_box is not None:
                            # Filter by confidence
                            conf_mask = fp32_pred[:, 4] >= confidence_threshold
                            fp32_filtered = fp32_pred[conf_mask]
                            
                            class_metrics[class_name]['fp32_predictions'] += len(fp32_filtered)
                            
                            if len(fp32_filtered) > 0:
                                # Calculate IoU with all predictions
                                ious = box_iou(gt_box.unsqueeze(0), fp32_filtered[:, :4])
                                
                                # Find best match
                                best_iou, best_idx = ious.max(1)
                                
                                if best_iou >= 0.5:  # Use fixed IoU threshold
                                    pred_class = int(fp32_filtered[best_idx, 5].item())
                                    
                                    if pred_class == gt_class:
                                        class_metrics[class_name]['fp32_correct'] += 1
                                        fp32_correct = True
                        
                        # Check INT8 prediction
                        int8_correct = False
                        if hasattr(int8_pred, 'shape') and int8_pred.shape[0] > 0 and gt_box is not None:
                            # Filter by confidence
                            conf_mask = int8_pred[:, 4] >= confidence_threshold
                            int8_filtered = int8_pred[conf_mask]
                            
                            class_metrics[class_name]['int8_predictions'] += len(int8_filtered)
                            
                            if len(int8_filtered) > 0:
                                # Calculate IoU with all predictions
                                ious = box_iou(gt_box.unsqueeze(0), int8_filtered[:, :4])
                                
                                # Find best match
                                best_iou, best_idx = ious.max(1)
                                
                                if best_iou >= 0.5:  # Use fixed IoU threshold
                                    pred_class = int(int8_filtered[best_idx, 5].item())
                                    
                                    if pred_class == gt_class:
                                        class_metrics[class_name]['int8_correct'] += 1
                                        int8_correct = True
                        
                        # Update combined metrics
                        if fp32_correct and int8_correct:
                            class_metrics[class_name]['both_correct'] += 1
                        elif fp32_correct and not int8_correct:
                            class_metrics[class_name]['fp32_only_correct'] += 1
                        elif not fp32_correct and int8_correct:
                            class_metrics[class_name]['int8_only_correct'] += 1
                        else:
                            class_metrics[class_name]['both_incorrect'] += 1
    
    # Calculate derived metrics
    for class_name, metrics in class_metrics.items():
        # Calculate accuracy
        if metrics['total_instances'] > 0:
            metrics['fp32_accuracy'] = metrics['fp32_correct'] / metrics['total_instances']
            metrics['int8_accuracy'] = metrics['int8_correct'] / metrics['total_instances']
            metrics['absolute_change'] = metrics['int8_accuracy'] - metrics['fp32_accuracy']
            metrics['relative_change'] = metrics['absolute_change'] / metrics['fp32_accuracy'] if metrics['fp32_accuracy'] > 0 else 0
        else:
            metrics['fp32_accuracy'] = 0
            metrics['int8_accuracy'] = 0
            metrics['absolute_change'] = 0
            metrics['relative_change'] = 0
        
        # Calculate precision
        if metrics['fp32_predictions'] > 0:
            metrics['fp32_precision'] = metrics['fp32_correct'] / metrics['fp32_predictions']
        else:
            metrics['fp32_precision'] = 0
            
        if metrics['int8_predictions'] > 0:
            metrics['int8_precision'] = metrics['int8_correct'] / metrics['int8_predictions']
        else:
            metrics['int8_precision'] = 0
    
    # Convert defaultdict to regular dict for better serialization
    results = {
        'class_metrics': dict(class_metrics),
        'class_names': class_names[:num_classes]
    }
    
    # Export results if requested
    if export_results:
        # Export to JSON
        json_path = os.path.join(output_path, 'class_drift.json')
        
        # Create serializable version
        serializable_results = {
            'class_metrics': {},
            'class_names': results['class_names']
        }
        
        for class_name, metrics in results['class_metrics'].items():
            serializable_results['class_metrics'][class_name] = {
                k: float(v) if isinstance(v, (float, np.float32, np.float64)) else int(v)
                for k, v in metrics.items()
            }
        
        # Save to JSON
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Class drift results saved to {json_path}")
        
        # Create accuracy comparison plot
        plt.figure(figsize=(14, 8))
        
        # Extract data for plotting
        classes = list(results['class_metrics'].keys())
        fp32_accuracy = [results['class_metrics'][c]['fp32_accuracy'] for c in classes]
        int8_accuracy = [results['class_metrics'][c]['int8_accuracy'] for c in classes]
        
        # Sort by FP32 accuracy
        sorted_indices = np.argsort(fp32_accuracy)[::-1]  # Descending order
        classes = [classes[i] for i in sorted_indices]
        fp32_accuracy = [fp32_accuracy[i] for i in sorted_indices]
        int8_accuracy = [int8_accuracy[i] for i in sorted_indices]
        
        # Plot
        x = np.arange(len(classes))
        width = 0.35
        
        plt.bar(x - width/2, fp32_accuracy, width, label='FP32')
        plt.bar(x + width/2, int8_accuracy, width, label='INT8')
        
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison by Class')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        accuracy_plot_path = os.path.join(output_path, 'class_accuracy_comparison.png')
        plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create accuracy change plot
        plt.figure(figsize=(14, 8))
        
        # Extract data for plotting
        absolute_changes = [results['class_metrics'][c]['absolute_change'] for c in classes]
        
        # Plot
        plt.bar(x, absolute_changes, color=['red' if c < 0 else 'green' for c in absolute_changes])
        
        plt.xlabel('Class')
        plt.ylabel('Absolute Accuracy Change')
        plt.title('Accuracy Change Due to Quantization')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        change_plot_path = os.path.join(output_path, 'class_accuracy_change.png')
        plt.savefig(change_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create "agreement" plot (both correct, FP32 only, INT8 only, both wrong)
        plt.figure(figsize=(14, 8))
        
        # Extract data for plotting
        both_correct = [results['class_metrics'][c]['both_correct'] for c in classes]
        fp32_only = [results['class_metrics'][c]['fp32_only_correct'] for c in classes]
        int8_only = [results['class_metrics'][c]['int8_only_correct'] for c in classes]
        both_wrong = [results['class_metrics'][c]['both_incorrect'] for c in classes]
        
        # Convert to percentages
        totals = np.array([results['class_metrics'][c]['total_instances'] for c in classes])
        
        # Avoid division by zero
        totals = np.maximum(totals, 1)
        
        both_correct_pct = 100 * np.array(both_correct) / totals
        fp32_only_pct = 100 * np.array(fp32_only) / totals
        int8_only_pct = 100 * np.array(int8_only) / totals
        both_wrong_pct = 100 * np.array(both_wrong) / totals
        
        # Create stacked bar chart
        plt.bar(x, both_correct_pct, label='Both Correct', color='green')
        plt.bar(x, fp32_only_pct, bottom=both_correct_pct, label='FP32 Only', color='orange')
        plt.bar(x, int8_only_pct, bottom=both_correct_pct+fp32_only_pct, label='INT8 Only', color='blue')
        plt.bar(x, both_wrong_pct, bottom=both_correct_pct+fp32_only_pct+int8_only_pct, label='Both Wrong', color='red')
        
        plt.xlabel('Class')
        plt.ylabel('Percentage')
        plt.title('Model Agreement by Class')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        agreement_plot_path = os.path.join(output_path, 'model_agreement.png')
        plt.savefig(agreement_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

def plot_accuracy_drift(
    results: Dict[str, Any],
    output_path: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate and save plots of accuracy drift.
    
    Args:
        results: Results from analyze_quantization_impact or measure_drift_per_class
        output_path: Path to save plots
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    # Set output path
    if output_path is None:
        output_path = "./accuracy_plots"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize paths dictionary
    plot_paths = {}
    
    # Check if results are from analyze_quantization_impact
    if 'class_metrics' in results and isinstance(results['class_metrics'], dict):
        # Extract class metrics
        class_metrics = results['class_metrics']
        
        if 'class_accuracy_fp32' in class_metrics and 'class_accuracy_int8' in class_metrics:
            # Create accuracy comparison plot
            plt.figure(figsize=(12, 6))
            
            # Extract data
            class_accuracy_fp32 = np.array(class_metrics['class_accuracy_fp32'])
            class_accuracy_int8 = np.array(class_metrics['class_accuracy_int8'])
            class_absolute_change = np.array(class_metrics['class_absolute_change'])
            
            # Create indices for classes
            num_classes = len(class_accuracy_fp32)
            classes = list(range(num_classes))
            
            # Create side-by-side bar chart
            x = np.arange(num_classes)
            width = 0.35
            
            plt.bar(x - width/2, class_accuracy_fp32, width, label='FP32')
            plt.bar(x + width/2, class_accuracy_int8, width, label='INT8')
            
            plt.xlabel('Class')
            plt.ylabel('Accuracy')
            plt.title('Class-wise Accuracy Comparison')
            plt.xticks(x, classes)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save plot
            accuracy_plot_path = os.path.join(output_path, 'accuracy_comparison.png')
            plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_paths['accuracy_comparison'] = accuracy_plot_path
            
            # Create accuracy change plot
            plt.figure(figsize=(12, 6))
            
            plt.bar(x, class_absolute_change, color=['red' if c < 0 else 'green' for c in class_absolute_change])
            
            plt.xlabel('Class')
            plt.ylabel('Absolute Accuracy Change')
            plt.title('Impact of Quantization on Class Accuracy')
            plt.xticks(x, classes)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add annotations for overall change
            overall_absolute_change = results.get('overall_absolute_change', 0)
            overall_relative_change = results.get('overall_relative_change', 0)
            
            plt.figtext(0.5, 0.01, f'Overall Accuracy Change: {overall_absolute_change:.4f} ({overall_relative_change*100:.2f}%)', 
                       ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save plot
            change_plot_path = os.path.join(output_path, 'accuracy_change.png')
            plt.savefig(change_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_paths['accuracy_change'] = change_plot_path
    
    # Check if results are from measure_drift_per_class
    elif 'class_metrics' in results and isinstance(results['class_metrics'], dict) and 'class_names' in results:
        # Extract class metrics and names
        class_metrics = results['class_metrics']
        class_names = results['class_names']
        
        # Create list of classes
        classes = list(class_metrics.keys())
        
        # Extract accuracy values
        fp32_accuracy = [class_metrics[c]['fp32_accuracy'] for c in classes]
        int8_accuracy = [class_metrics[c]['int8_accuracy'] for c in classes]
        absolute_changes = [class_metrics[c]['absolute_change'] for c in classes]
        
        # Sort by FP32 accuracy
        sorted_indices = np.argsort(fp32_accuracy)[::-1]  # Descending order
        classes = [classes[i] for i in sorted_indices]
        fp32_accuracy = [fp32_accuracy[i] for i in sorted_indices]
        int8_accuracy = [int8_accuracy[i] for i in sorted_indices]
        absolute_changes = [absolute_changes[i] for i in sorted_indices]
        
        # Create accuracy comparison plot
        plt.figure(figsize=(14, 8))
        
        x = np.arange(len(classes))
        width = 0.35
        
        plt.bar(x - width/2, fp32_accuracy, width, label='FP32')
        plt.bar(x + width/2, int8_accuracy, width, label='INT8')
        
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison by Class')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        accuracy_plot_path = os.path.join(output_path, 'class_accuracy_comparison.png')
        plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths['class_accuracy_comparison'] = accuracy_plot_path
        
        # Create accuracy change plot
        plt.figure(figsize=(14, 8))
        
        plt.bar(x, absolute_changes, color=['red' if c < 0 else 'green' for c in absolute_changes])
        
        plt.xlabel('Class')
        plt.ylabel('Absolute Accuracy Change')
        plt.title('Accuracy Change Due to Quantization')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        change_plot_path = os.path.join(output_path, 'class_accuracy_change.png')
        plt.savefig(change_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths['class_accuracy_change'] = change_plot_path
        
        # Extract agreement data
        both_correct = [class_metrics[c]['both_correct'] for c in classes]
        fp32_only = [class_metrics[c]['fp32_only_correct'] for c in classes]
        int8_only = [class_metrics[c]['int8_only_correct'] for c in classes]
        both_wrong = [class_metrics[c]['both_incorrect'] for c in classes]
        
        # Convert to percentages
        totals = np.array([class_metrics[c]['total_instances'] for c in classes])
        
        # Avoid division by zero
        totals = np.maximum(totals, 1)
        
        both_correct_pct = 100 * np.array(both_correct) / totals
        fp32_only_pct = 100 * np.array(fp32_only) / totals
        int8_only_pct = 100 * np.array(int8_only) / totals
        both_wrong_pct = 100 * np.array(both_wrong) / totals
        
        # Create agreement plot
        plt.figure(figsize=(14, 8))
        
        plt.bar(x, both_correct_pct, label='Both Correct', color='green')
        plt.bar(x, fp32_only_pct, bottom=both_correct_pct, label='FP32 Only', color='orange')
        plt.bar(x, int8_only_pct, bottom=both_correct_pct+fp32_only_pct, label='INT8 Only', color='blue')
        plt.bar(x, both_wrong_pct, bottom=both_correct_pct+fp32_only_pct+int8_only_pct, label='Both Wrong', color='red')
        
        plt.xlabel('Class')
        plt.ylabel('Percentage')
        plt.title('Model Agreement by Class')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        agreement_plot_path = os.path.join(output_path, 'model_agreement.png')
        plt.savefig(agreement_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths['model_agreement'] = agreement_plot_path
    
    return plot_paths

def _apply_simulated_quantization(module):
    """
    Apply simulated int8 quantization to a module.
    
    This is a helper function for identify_critical_layers.
    
    Args:
        module: Module to apply simulated quantization to
    """
    # Apply simulated 8-bit quantization to weights
    if hasattr(module, 'weight') and module.weight is not None:
        # Calculate range
        w_min = module.weight.min()
        w_max = module.weight.max()
        
        # Calculate scale and zero point
        scale = (w_max - w_min) / 255
        zero_point = -128 - w_min / scale
        
        # Quantize weights (simulate int8 quantization)
        w_quant = torch.clamp(torch.round(module.weight / scale + zero_point), -128, 127)
        
        # Dequantize (simulate conversion back to float32)
        w_dequant = (w_quant - zero_point) * scale
        
        # Replace weights with quantized-dequantized version
        module.weight.data = w_dequant