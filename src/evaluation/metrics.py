"""
Metrics calculation for YOLOv8 QAT evaluation.

This module provides functions for calculating various performance metrics
for object detection models, with special focus on quantization effects.
"""

import torch
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import time
from collections import defaultdict

# Setup logging
logger = logging.getLogger(__name__)

def compute_map(
    predictions: List[torch.Tensor], 
    targets: List[torch.Tensor], 
    num_classes: int,
    iou_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
) -> Dict[str, float]:
    """
    Compute Mean Average Precision (mAP) for object detection.
    
    Args:
        predictions: List of prediction tensors [batch_size, num_preds, 6] (x1, y1, x2, y2, conf, class_id)
        targets: List of target tensors [batch_size, num_targets, 5] (class_id, x1, y1, x2, y2)
        num_classes: Number of classes
        iou_thresholds: List of IoU thresholds for mAP calculation
        
    Returns:
        Dictionary with mAP values at different IoU thresholds, plus mAP@.5 and mAP@.5:.95
    """
    # Initialize accumulators for each class and IoU threshold
    stats = {}
    ap_class = []
    
    # Process each class
    for class_id in range(num_classes):
        # Extract predictions and targets for this class
        class_preds = [p[p[:, 5] == class_id] for p in predictions]
        class_targets = [t[t[:, 0] == class_id] for t in targets]
        
        # Compute AP for each IoU threshold
        aps = []
        for iou_threshold in iou_thresholds:
            ap = calculate_average_precision(
                class_preds, class_targets, iou_threshold=iou_threshold
            )
            aps.append(ap)
        
        # Store results
        ap_class.append(aps)
    
    # Calculate mAP
    ap_class = np.array(ap_class)
    
    # Overall mAP
    stats["mAP@.5:.95"] = ap_class[:, 0:10].mean()
    stats["mAP@.5"] = ap_class[:, 0].mean()
    
    # Class-wise mAP
    for i, c in enumerate(range(num_classes)):
        stats[f"AP@.5_class{c}"] = ap_class[i, 0]
    
    return stats

def calculate_average_precision(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate Average Precision at a specific IoU threshold.
    
    Args:
        predictions: List of prediction tensors [N, 6] (x1, y1, x2, y2, conf, class_id)
        targets: List of target tensors [M, 5] (class_id, x1, y1, x2, y2)
        iou_threshold: IoU threshold for considering a prediction correct
        
    Returns:
        Average precision at the specified IoU threshold
    """
    # Combine predictions across all images
    all_preds = []
    all_targets = []
    
    for batch_idx, (preds, tgts) in enumerate(zip(predictions, targets)):
        if len(preds) == 0:
            continue
            
        # Add batch index to predictions
        preds_with_img = torch.cat([torch.full((preds.shape[0], 1), batch_idx, device=preds.device), preds], dim=1)
        all_preds.append(preds_with_img)
        
        # Add batch index to targets
        tgts_with_img = torch.cat([torch.full((tgts.shape[0], 1), batch_idx, device=tgts.device), tgts], dim=1)
        all_targets.append(tgts_with_img)
    
    if not all_preds:
        return 0.0
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0) if all_targets else torch.zeros((0, 6))
    
    # Sort predictions by confidence
    all_preds = all_preds[all_preds[:, 5].argsort(descending=True)]
    
    # Calculate precision-recall curve
    tp = torch.zeros(len(all_preds))
    fp = torch.zeros(len(all_preds))
    
    # Track which targets have been detected
    target_detected = torch.zeros(len(all_targets))
    
    # For each prediction
    for i, pred in enumerate(all_preds):
        # Get targets for this image
        img_idx = pred[0].long()
        img_targets = all_targets[all_targets[:, 0] == img_idx]
        
        if len(img_targets) == 0:
            fp[i] = 1
            continue
        
        # Calculate IoU with all targets
        ious = box_iou(pred[1:5].unsqueeze(0), img_targets[:, 2:6])
        max_iou, max_idx = ious.max(dim=1)
        
        # Check if detection is correct
        if max_iou >= iou_threshold and not target_detected[max_idx]:
            tp[i] = 1
            target_detected[max_idx] = 1
        else:
            fp[i] = 1
    
    # Calculate precision and recall
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    recalls = tp_cumsum / (len(all_targets) + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # Add start and end points for PR curve
    precisions = torch.cat([torch.tensor([1.0]), precisions])
    recalls = torch.cat([torch.tensor([0.0]), recalls])
    
    # Calculate AP using precision-recall curve (area under curve)
    ap = torch.trapz(precisions, recalls)
    
    return ap.item()

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

def compute_precision_recall(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    num_classes: int,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Compute precision and recall for each class.
    
    Args:
        predictions: List of prediction tensors [batch_size, num_preds, 6] (x1, y1, x2, y2, conf, class_id)
        targets: List of target tensors [batch_size, num_targets, 5] (class_id, x1, y1, x2, y2)
        num_classes: Number of classes
        confidence_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for matching predictions to targets
        
    Returns:
        Dictionary with precision and recall for each class
    """
    results = {}
    
    for class_id in range(num_classes):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for batch_idx, (preds, tgts) in enumerate(zip(predictions, targets)):
            # Filter predictions by confidence and class
            class_preds = preds[(preds[:, 4] >= confidence_threshold) & (preds[:, 5] == class_id)]
            
            # Get targets for this class
            class_tgts = tgts[tgts[:, 0] == class_id]
            
            # Track which targets have been detected
            detected_tgts = torch.zeros(len(class_tgts))
            
            # For each prediction
            for pred in class_preds:
                if len(class_tgts) == 0:
                    false_positives += 1
                    continue
                
                # Calculate IoU with all targets
                ious = box_iou(pred[:4].unsqueeze(0), class_tgts[:, 1:5])
                max_iou, max_idx = ious.max(dim=1)
                
                # Check if detection is correct
                if max_iou >= iou_threshold and not detected_tgts[max_idx]:
                    true_positives += 1
                    detected_tgts[max_idx] = 1
                else:
                    false_positives += 1
            
            # Count undetected targets as false negatives
            false_negatives += (detected_tgts == 0).sum().item()
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        results[f"class_{class_id}"] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    return results

def compute_confusion_matrix(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    num_classes: int,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5
) -> np.ndarray:
    """
    Compute confusion matrix for multi-class object detection.
    
    Args:
        predictions: List of prediction tensors [batch_size, num_preds, 6] (x1, y1, x2, y2, conf, class_id)
        targets: List of target tensors [batch_size, num_targets, 5] (class_id, x1, y1, x2, y2)
        num_classes: Number of classes
        confidence_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for matching predictions to targets
        
    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    for batch_idx, (preds, tgts) in enumerate(zip(predictions, targets)):
        # Filter predictions by confidence
        preds = preds[preds[:, 4] >= confidence_threshold]
        
        if len(preds) == 0 or len(tgts) == 0:
            continue
        
        # For each ground truth box
        for tgt in tgts:
            tgt_class = int(tgt[0].item())
            tgt_box = tgt[1:5].unsqueeze(0)
            
            # Calculate IoU with all predictions
            ious = box_iou(tgt_box, preds[:, :4])
            max_iou, max_idx = ious.max(dim=1)
            
            # If IoU is above threshold, add to confusion matrix
            if max_iou >= iou_threshold:
                pred_class = int(preds[max_idx, 5].item())
                confusion_matrix[tgt_class, pred_class] += 1
            else:
                # No match found, count as missed detection
                confusion_matrix[tgt_class, -1] += 1
    
    return confusion_matrix

def compute_f1_score(
    precision: float,
    recall: float
) -> float:
    """
    Compute F1 score from precision and recall.
    
    Args:
        precision: Precision value
        recall: Recall value
        
    Returns:
        F1 score
    """
    return 2 * (precision * recall) / (precision + recall + 1e-6)

def calculate_accuracy(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate accuracy for object detection.
    
    Args:
        predictions: List of prediction tensors
        targets: List of target tensors
        confidence_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for matching predictions to targets
        
    Returns:
        Accuracy as a float
    """
    total_gt = sum(len(t) for t in targets)
    if total_gt == 0:
        return 0.0
    
    true_positives = 0
    
    for batch_idx, (preds, tgts) in enumerate(zip(predictions, targets)):
        # Filter predictions by confidence
        preds = preds[preds[:, 4] >= confidence_threshold]
        
        if len(preds) == 0:
            continue
        
        # Track which targets have been detected
        detected_tgts = torch.zeros(len(tgts))
        
        # For each prediction
        for pred in preds:
            if len(tgts) == 0:
                continue
            
            pred_class = pred[5].long()
            pred_box = pred[:4].unsqueeze(0)
            
            # Get targets with the same class
            same_class_tgts = tgts[tgts[:, 0] == pred_class]
            
            if len(same_class_tgts) == 0:
                continue
            
            # Calculate IoU with targets of the same class
            ious = box_iou(pred_box, same_class_tgts[:, 1:5])
            max_iou, max_idx = ious.max(dim=1)
            
            # Check if detection is correct
            if max_iou >= iou_threshold:
                # Get global index in all targets
                global_idx = torch.where(tgts[:, 0] == pred_class)[0][max_idx]
                
                if not detected_tgts[global_idx]:
                    true_positives += 1
                    detected_tgts[global_idx] = 1
    
    # Calculate accuracy
    accuracy = true_positives / total_gt
    
    return accuracy

def calculate_mean_average_precision(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    num_classes: int
) -> float:
    """
    Calculate mean Average Precision (mAP) for object detection.
    
    Args:
        predictions: List of prediction tensors
        targets: List of target tensors
        num_classes: Number of classes
        
    Returns:
        mAP as a float
    """
    # Calculate mAP using compute_map function
    map_results = compute_map(predictions, targets, num_classes)
    
    return map_results["mAP@.5:.95"]

def compute_evaluation_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    metrics: Optional[List[str]] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Compute evaluation metrics for a model on a given dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation dataset
        metrics: List of metrics to compute (default: ['map', 'latency'])
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation results
    """
    if metrics is None:
        metrics = ['map', 'latency']
    
    # Set model to evaluation mode
    model.eval()
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model.to(device)
    
    # Initialize results
    results = {}
    
    # Initialize lists for storing predictions and targets
    all_predictions = []
    all_targets = []
    
    # Track inference time
    inference_times = []
    
    # Evaluation loop
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Process batch
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                images, targets = batch[0], batch[1]
            else:
                # For custom dataset formats
                images, targets = batch['image'], batch['target']
            
            # Move to device
            images = images.to(device)
            targets = [t.to(device) for t in targets] if isinstance(targets, list) else targets.to(device)
            
            # Measure inference time
            start_time = time.time()
            predictions = model(images)
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            # Store predictions and targets
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    # Calculate requested metrics
    num_classes = dataloader.dataset.num_classes if hasattr(dataloader.dataset, 'num_classes') else 10
    
    if 'map' in metrics:
        # Calculate mAP
        results['mAP'] = compute_map(all_predictions, all_targets, num_classes)
    
    if 'precision_recall' in metrics:
        # Calculate precision and recall
        results['precision_recall'] = compute_precision_recall(all_predictions, all_targets, num_classes)
    
    if 'confusion_matrix' in metrics:
        # Calculate confusion matrix
        results['confusion_matrix'] = compute_confusion_matrix(all_predictions, all_targets, num_classes)
    
    if 'accuracy' in metrics:
        # Calculate accuracy
        results['accuracy'] = calculate_accuracy(all_predictions, all_targets)
    
    if 'latency' in metrics:
        # Calculate latency
        results['latency'] = {
            'mean_inference_time': np.mean(inference_times),
            'median_inference_time': np.median(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'fps': 1.0 / np.mean(inference_times)
        }
    
    return results

def preprocess_yolo_predictions(
    predictions: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    Preprocess YOLOv8 predictions to standardized format.
    
    Args:
        predictions: Raw YOLOv8 predictions
        
    Returns:
        Processed predictions in format [batch_size, num_preds, 6] (x1, y1, x2, y2, conf, class_id)
    """
    processed = []
    
    for pred in predictions:
        if isinstance(pred, (list, tuple)):
            # Handle different YOLOv8 output formats
            if len(pred) == 2:  # YOLOv8 format: [boxes, scores]
                boxes, scores = pred
                processed_pred = torch.cat([boxes, scores.unsqueeze(-1)], dim=-1)
            elif len(pred) == 3:  # YOLOv8 format: [boxes, scores, class_ids]
                boxes, scores, class_ids = pred
                processed_pred = torch.cat([boxes, scores.unsqueeze(-1), class_ids.unsqueeze(-1)], dim=-1)
            else:
                processed_pred = pred
        else:
            # Already in correct format
            processed_pred = pred
        
        processed.append(processed_pred)
    
    return processed

def preprocess_yolo_targets(
    targets: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    Preprocess YOLOv8 targets to standardized format.
    
    Args:
        targets: Raw YOLOv8 targets
        
    Returns:
        Processed targets in format [batch_size, num_targets, 5] (class_id, x1, y1, x2, y2)
    """
    processed = []
    
    for tgt in targets:
        if isinstance(tgt, dict):
            # Handle different YOLOv8 target formats
            if 'boxes' in tgt and 'labels' in tgt:
                boxes = tgt['boxes']
                labels = tgt['labels']
                processed_tgt = torch.cat([labels.unsqueeze(-1), boxes], dim=-1)
            else:
                processed_tgt = tgt
        else:
            # Already in correct format
            processed_tgt = tgt
        
        processed.append(processed_tgt)
    
    return processed