"""
Benchmarking utilities for YOLOv8 QAT models.

This module provides functions for benchmarking model performance
across different backends and configurations.
"""

import torch
import numpy as np
import time
import logging
import os
import json
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

def benchmark_inference(
    model_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    backend: str = "pytorch",
    device: str = "cuda",
    num_runs: int = 100,
    warmup_runs: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark inference performance of a model.
    
    Args:
        model_path: Path to model file
        input_shape: Shape of input tensor
        backend: Inference backend
        device: Device to run inference on
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs
        **kwargs: Additional arguments for specific backends
        
    Returns:
        Dictionary with benchmark results
    """
    # Import locally to avoid circular imports
    from .inference import create_inference_engine
    
    # Create inference engine
    engine = create_inference_engine(model_path, backend, device, **kwargs)
    
    # Create dummy input
    if backend == "pytorch":
        input_tensor = torch.rand(*input_shape)
        if device == "cuda":
            input_tensor = input_tensor.cuda()
        original_size = (input_shape[2], input_shape[3])
        input_data = (input_tensor, original_size)
    else:
        # Create numpy array for other backends
        input_tensor = np.random.rand(*input_shape).astype(np.float32)
        original_size = (input_shape[2], input_shape[3])
        input_data = engine.preprocess(input_tensor)
    
    # Warmup runs
    logger.info(f"Performing {warmup_runs} warmup runs...")
    for _ in range(warmup_runs):
        _, _ = engine.run_inference(input_data)
    
    # Benchmark runs
    logger.info(f"Measuring inference time over {num_runs} runs...")
    inference_times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        _, _ = engine.run_inference(input_data)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
    
    # Calculate statistics
    np_times = np.array(inference_times)
    mean_time = np.mean(np_times)
    median_time = np.median(np_times)
    std_time = np.std(np_times)
    min_time = np.min(np_times)
    max_time = np.max(np_times)
    p95_time = np.percentile(np_times, 95)
    
    # Calculate FPS
    fps = 1.0 / mean_time
    
    return {
        "backend": backend,
        "device": device,
        "mean_time": mean_time,
        "median_time": median_time,
        "std_time": std_time,
        "min_time": min_time,
        "max_time": max_time,
        "p95_time": p95_time,
        "fps": fps,
        "batch_size": input_shape[0],
        "all_times": inference_times
    }


def benchmark_memory_usage(
    model_path: str,
    backend: str = "pytorch",
    device: str = "cuda",
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark memory usage of a model.
    
    Args:
        model_path: Path to model file
        backend: Inference backend
        device: Device to run inference on
        input_shape: Shape of input tensor
        **kwargs: Additional arguments for specific backends
        
    Returns:
        Dictionary with memory usage results
    """
    # Import locally to avoid circular imports
    from .inference import create_inference_engine
    
    # Create inference engine
    engine = create_inference_engine(model_path, backend, device, **kwargs)
    
    # Create dummy input
    if backend == "pytorch":
        input_tensor = torch.rand(*input_shape)
        if device == "cuda":
            input_tensor = input_tensor.cuda()
        original_size = (input_shape[2], input_shape[3])
        input_data = (input_tensor, original_size)
    else:
        # Create numpy array for other backends
        input_tensor = np.random.rand(*input_shape).astype(np.float32)
        original_size = (input_shape[2], input_shape[3])
        input_data = engine.preprocess(input_tensor)
    
    # Measure baseline memory usage
    baseline_memory = None
    peak_memory = None
    
    if device == "cuda" and torch.cuda.is_available():
        # Reset peak stats
        torch.cuda.reset_peak_memory_stats()
        
        # Get baseline memory usage
        baseline_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        
        # Run inference
        _, _ = engine.run_inference(input_data)
        
        # Get peak memory usage
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    
    # Estimate model size
    model_size = None
    if backend == "pytorch" and hasattr(engine, 'model'):
        model_size = 0
        for param in engine.model.parameters():
            model_size += param.nelement() * param.element_size()
        for buffer in engine.model.buffers():
            model_size += buffer.nelement() * buffer.element_size()
        model_size = model_size / (1024 * 1024)  # MB
    elif os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    
    return {
        "backend": backend,
        "device": device,
        "model_size_mb": model_size,
        "baseline_memory_mb": baseline_memory,
        "peak_memory_mb": peak_memory,
        "memory_overhead_mb": peak_memory - baseline_memory if peak_memory is not None and baseline_memory is not None else None
    }


def benchmark_precision(
    model_path: str,
    reference_model_path: str,
    test_dataset: Optional[Any] = None,
    test_images: Optional[List[str]] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    backend: str = "pytorch",
    device: str = "cuda",
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark precision of a model against a reference model.
    
    Args:
        model_path: Path to model file
        reference_model_path: Path to reference model file
        test_dataset: Optional test dataset
        test_images: Optional list of test image paths
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        backend: Inference backend
        device: Device to run inference on
        **kwargs: Additional arguments for specific backends
        
    Returns:
        Dictionary with precision results
    """
    from .inference import create_inference_engine
    
    # Create inference engines for both models
    engine = create_inference_engine(model_path, backend, device, **kwargs)
    ref_engine = create_inference_engine(reference_model_path, "pytorch", device, **kwargs)
    
    # Initialize metrics
    class_correct = 0
    class_total = 0
    bbox_correct = 0
    bbox_total = 0
    
    # Process test data
    if test_dataset is not None:
        # Use test dataset
        results = []
        for batch in test_dataset:
            if isinstance(batch, dict):
                images = batch.get("images", batch.get("image", None))
                targets = batch.get("targets", batch.get("target", None))
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, targets = batch[0], batch[1]
            else:
                logger.warning(f"Unsupported batch format: {type(batch)}")
                continue
            
            # Process each image in batch
            for i in range(len(images)):
                image = images[i]
                target = targets[i] if isinstance(targets, (list, tuple)) else targets
                
                # Preprocess image
                input_data = engine.preprocess(image)
                
                # Run inference
                detections, _ = engine.run_inference(input_data)
                
                # Run reference model
                ref_input_data = ref_engine.preprocess(image)
                ref_detections, _ = ref_engine.run_inference(ref_input_data)
                
                # Compare detections
                compare_result = _compare_detections(
                    detections, ref_detections, target, 
                    conf_threshold, iou_threshold
                )
                
                class_correct += compare_result["class_correct"]
                class_total += compare_result["class_total"]
                bbox_correct += compare_result["bbox_correct"]
                bbox_total += compare_result["bbox_total"]
                
                results.append(compare_result)
    
    elif test_images is not None:
        # Use test images
        results = []
        for image_path in test_images:
            # Preprocess image
            input_data = engine.preprocess(image_path)
            
            # Run inference
            detections, _ = engine.run_inference(input_data)
            
            # Run reference model
            ref_input_data = ref_engine.preprocess(image_path)
            ref_detections, _ = ref_engine.run_inference(ref_input_data)
            
            # Compare detections (without ground truth)
            compare_result = _compare_detections_no_gt(
                detections, ref_detections, conf_threshold, iou_threshold
            )
            
            results.append(compare_result)
    
    else:
        logger.error("No test data provided for precision benchmark")
        return {}
    
    # Calculate overall metrics
    class_accuracy = class_correct / class_total if class_total > 0 else 0
    bbox_accuracy = bbox_correct / bbox_total if bbox_total > 0 else 0
    
    return {
        "class_accuracy": class_accuracy,
        "bbox_accuracy": bbox_accuracy,
        "class_correct": class_correct,
        "class_total": class_total,
        "bbox_correct": bbox_correct,
        "bbox_total": bbox_total,
        "detailed_results": results
    }


def _compare_detections(
    detections: np.ndarray,
    ref_detections: np.ndarray,
    target: Union[np.ndarray, torch.Tensor],
    conf_threshold: float,
    iou_threshold: float
) -> Dict[str, Any]:
    """
    Compare detections with reference detections and ground truth.
    
    Args:
        detections: Model detections [N, 6] (x1, y1, x2, y2, conf, class)
        ref_detections: Reference detections [M, 6]
        target: Ground truth [K, 5] (class, x1, y1, x2, y2)
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        
    Returns:
        Comparison results
    """
    # Convert target to numpy if needed
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Filter by confidence
    detections = detections[detections[:, 4] >= conf_threshold]
    ref_detections = ref_detections[ref_detections[:, 4] >= conf_threshold]
    
    # Initialize metrics
    class_correct = 0
    class_total = 0
    bbox_correct = 0
    bbox_total = 0
    
    # Compare with ground truth
    if len(target) > 0:
        # For each ground truth box
        for gt_box in target:
            gt_class = int(gt_box[0]) if len(gt_box) >= 5 else -1
            gt_bbox = gt_box[1:5] if len(gt_box) >= 5 else gt_box
            
            # Find best match in model detections
            best_iou = 0
            best_pred_class = -1
            
            for pred in detections:
                pred_bbox = pred[:4]
                pred_class = int(pred[5])
                
                # Calculate IoU
                iou = _calculate_iou(gt_bbox, pred_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred_class = pred_class
            
            # Find best match in reference detections
            best_ref_iou = 0
            best_ref_class = -1
            
            for ref_pred in ref_detections:
                ref_pred_bbox = ref_pred[:4]
                ref_pred_class = int(ref_pred[5])
                
                # Calculate IoU
                iou = _calculate_iou(gt_bbox, ref_pred_bbox)
                
                if iou > best_ref_iou:
                    best_ref_iou = iou
                    best_ref_class = ref_pred_class
            
            # Check if IoU is above threshold
            if best_iou >= iou_threshold:
                bbox_correct += 1
                
                # Check if class is correct
                if best_pred_class == gt_class:
                    class_correct += 1
                
                class_total += 1
            
            bbox_total += 1
    
    return {
        "class_correct": class_correct,
        "class_total": class_total,
        "bbox_correct": bbox_correct,
        "bbox_total": bbox_total
    }


def _compare_detections_no_gt(
    detections: np.ndarray,
    ref_detections: np.ndarray,
    conf_threshold: float,
    iou_threshold: float
) -> Dict[str, Any]:
    """
    Compare detections with reference detections without ground truth.
    
    Args:
        detections: Model detections [N, 6] (x1, y1, x2, y2, conf, class)
        ref_detections: Reference detections [M, 6]
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        
    Returns:
        Comparison results
    """
    # Filter by confidence
    detections = detections[detections[:, 4] >= conf_threshold]
    ref_detections = ref_detections[ref_detections[:, 4] >= conf_threshold]
    
    # Initialize metrics
    matched_detections = 0
    matched_classes = 0
    
    # Track matched reference detections
    ref_matched = np.zeros(len(ref_detections), dtype=bool)
    
    # For each model detection
    for pred in detections:
        pred_bbox = pred[:4]
        pred_class = int(pred[5])
        
        best_iou = 0
        best_idx = -1
        
        # Find best match in reference detections
        for i, ref_pred in enumerate(ref_detections):
            if ref_matched[i]:
                continue
                
            ref_pred_bbox = ref_pred[:4]
            
            # Calculate IoU
            iou = _calculate_iou(pred_bbox, ref_pred_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        
        # Check if IoU is above threshold
        if best_iou >= iou_threshold and best_idx >= 0:
            matched_detections += 1
            ref_matched[best_idx] = True
            
            # Check if class matches
            ref_class = int(ref_detections[best_idx][5])
            if pred_class == ref_class:
                matched_classes += 1
    
    # Calculate metrics
    detection_recall = matched_detections / len(ref_detections) if len(ref_detections) > 0 else 0
    class_accuracy = matched_classes / matched_detections if matched_detections > 0 else 0
    
    return {
        "matched_detections": matched_detections,
        "total_ref_detections": len(ref_detections),
        "detection_recall": detection_recall,
        "matched_classes": matched_classes,
        "class_accuracy": class_accuracy
    }


def _calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Get coordinates of intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou


def run_deployment_benchmark(
    model_path: str,
    output_dir: Optional[str] = None,
    backends: List[str] = ["pytorch", "onnx"],
    devices: List[str] = ["cpu", "cuda"],
    batch_sizes: List[int] = [1, 4, 8],
    num_runs: int = 100,
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    **kwargs
) -> Dict[str, Any]:
    """
    Run comprehensive deployment benchmark for a model.
    
    Args:
        model_path: Path to model file
        output_dir: Optional directory to save results
        backends: List of backends to benchmark
        devices: List of devices to benchmark
        batch_sizes: List of batch sizes to benchmark
        num_runs: Number of inference runs
        input_shape: Base input shape (batch dimension will be replaced)
        **kwargs: Additional arguments for specific backends
        
    Returns:
        Dictionary with benchmark results
    """
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results
    results = {
        "model_path": model_path,
        "performance": [],
        "memory": []
    }
    
    # Run benchmarks for each combination
    for backend in backends:
        for device in devices:
            # Skip invalid combinations
            if backend != "pytorch" and device == "cuda" and backend != "tensorrt":
                logger.info(f"Skipping {backend} on {device} (not supported)")
                continue
            
            # Benchmark memory usage
            try:
                memory_result = benchmark_memory_usage(
                    model_path=model_path,
                    backend=backend,
                    device=device,
                    input_shape=input_shape,
                    **kwargs
                )
                results["memory"].append(memory_result)
            except Exception as e:
                logger.error(f"Memory benchmark failed for {backend} on {device}: {e}")
            
            # Benchmark performance for each batch size
            for batch_size in batch_sizes:
                # Adjust input shape for batch size
                current_shape = (batch_size,) + input_shape[1:]
                
                try:
                    perf_result = benchmark_inference(
                        model_path=model_path,
                        input_shape=current_shape,
                        backend=backend,
                        device=device,
                        num_runs=num_runs,
                        **kwargs
                    )
                    results["performance"].append(perf_result)
                except Exception as e:
                    logger.error(f"Performance benchmark failed for {backend} on {device} with batch size {batch_size}: {e}")
    
    # Save results if output directory is provided
    if output_dir:
        # Save JSON results
        result_path = os.path.join(output_dir, "benchmark_results.json")
        with open(result_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            clean_results = _clean_for_json(results)
            json.dump(clean_results, f, indent=2)
        
        # Generate plots
        _generate_benchmark_plots(results, output_dir)
    
    return results


def compare_backends(
    model_paths: Dict[str, str],
    output_dir: Optional[str] = None,
    batch_size: int = 1,
    num_runs: int = 100,
    input_shape: Optional[Tuple[int, ...]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare performance of different backends for the same model.
    
    Args:
        model_paths: Dictionary mapping backend names to model paths
        output_dir: Optional directory to save results
        batch_size: Batch size for benchmark
        num_runs: Number of inference runs
        input_shape: Optional input shape (default: (batch_size, 3, 640, 640))
        **kwargs: Additional arguments for specific backends
        
    Returns:
        Dictionary with comparison results
    """
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set default input shape if not provided
    if input_shape is None:
        input_shape = (batch_size, 3, 640, 640)
    elif input_shape[0] != batch_size:
        input_shape = (batch_size,) + input_shape[1:]
    
    # Initialize results
    results = {
        "performance": [],
        "memory": []
    }
    
    # Run benchmarks for each backend
    for backend, model_path in model_paths.items():
        # Determine device based on backend
        device = "cpu"
        if backend == "pytorch" or backend == "tensorrt":
            device = "cuda"
        
        # Benchmark performance
        try:
            perf_result = benchmark_inference(
                model_path=model_path,
                input_shape=input_shape,
                backend=backend,
                device=device,
                num_runs=num_runs,
                **kwargs
            )
            results["performance"].append(perf_result)
        except Exception as e:
            logger.error(f"Performance benchmark failed for {backend}: {e}")
        
        # Benchmark memory usage
        try:
            memory_result = benchmark_memory_usage(
                model_path=model_path,
                backend=backend,
                device=device,
                input_shape=input_shape,
                **kwargs
            )
            results["memory"].append(memory_result)
        except Exception as e:
            logger.error(f"Memory benchmark failed for {backend}: {e}")
    
    # Save results if output directory is provided
    if output_dir:
        # Save JSON results
        result_path = os.path.join(output_dir, "backend_comparison.json")
        with open(result_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            clean_results = _clean_for_json(results)
            json.dump(clean_results, f, indent=2)
        
        # Generate comparison plots
        _generate_comparison_plots(results, output_dir)
    
    return results


def _clean_for_json(obj):
    """
    Convert numpy types to Python types for JSON serialization.
    
    Args:
        obj: Object to clean
        
    Returns:
        Cleaned object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    else:
        return obj


def _generate_benchmark_plots(results, output_dir):
    """
    Generate plots from benchmark results.
    
    Args:
        results: Benchmark results
        output_dir: Directory to save plots
    """
    # Create a DataFrame from performance results
    performance_data = pd.DataFrame(results["performance"])
    
    # Plot inference time by backend and batch size
    plt.figure(figsize=(10, 6))
    backends = performance_data["backend"].unique()
    devices = performance_data["device"].unique()
    batch_sizes = performance_data["batch_size"].unique()
    
    for backend in backends:
        for device in devices:
            data = performance_data[(performance_data["backend"] == backend) & 
                                    (performance_data["device"] == device)]
            if not data.empty:
                plt.plot(data["batch_size"], data["mean_time"] * 1000, 
                         marker='o', linestyle='-', 
                         label=f"{backend} ({device})")
    
    plt.xlabel("Batch Size")
    plt.ylabel("Inference Time (ms)")
    plt.title("Inference Time by Backend and Batch Size")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "inference_time.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot FPS by backend and batch size
    plt.figure(figsize=(10, 6))
    
    for backend in backends:
        for device in devices:
            data = performance_data[(performance_data["backend"] == backend) & 
                                    (performance_data["device"] == device)]
            if not data.empty:
                plt.plot(data["batch_size"], data["fps"], 
                         marker='o', linestyle='-', 
                         label=f"{backend} ({device})")
    
    plt.xlabel("Batch Size")
    plt.ylabel("Frames Per Second (FPS)")
    plt.title("Throughput by Backend and Batch Size")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "throughput.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot memory usage by backend
    if results["memory"]:
        memory_data = pd.DataFrame(results["memory"])
        
        plt.figure(figsize=(10, 6))
        backends_with_memory = memory_data["backend"].unique()
        
        model_sizes = []
        peak_memories = []
        labels = []
        
        for backend in backends_with_memory:
            for device in devices:
                data = memory_data[(memory_data["backend"] == backend) & 
                                   (memory_data["device"] == device)]
                if not data.empty:
                    model_size = data["model_size_mb"].values[0]
                    peak_memory = data["peak_memory_mb"].values[0]
                    
                    if model_size is not None:
                        model_sizes.append(model_size)
                        labels.append(f"{backend} ({device})")
                    
                    if peak_memory is not None:
                        peak_memories.append(peak_memory)
        
        # Create bar chart of model sizes
        if model_sizes:
            plt.figure(figsize=(10, 6))
            plt.bar(labels, model_sizes)
            plt.ylabel("Model Size (MB)")
            plt.title("Model Size by Backend")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "model_size.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create bar chart of peak memory usage
        if peak_memories and len(peak_memories) == len(labels):
            plt.figure(figsize=(10, 6))
            plt.bar(labels, peak_memories)
            plt.ylabel("Peak Memory (MB)")
            plt.title("Peak Memory Usage by Backend")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "peak_memory.png"), dpi=300, bbox_inches='tight')
            plt.close()


def _generate_comparison_plots(results, output_dir):
    """
    Generate comparison plots for different backends.
    
    Args:
        results: Comparison results
        output_dir: Directory to save plots
    """
    # Create DataFrames from results
    performance_data = pd.DataFrame(results["performance"])
    
    if not performance_data.empty:
        # Sort by inference time
        performance_data = performance_data.sort_values("mean_time")
        
        # Create bar chart of inference time
        plt.figure(figsize=(10, 6))
        plt.bar(performance_data["backend"], performance_data["mean_time"] * 1000)
        plt.ylabel("Inference Time (ms)")
        plt.title("Inference Time by Backend")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "backend_inference_time.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create bar chart of FPS
        plt.figure(figsize=(10, 6))
        plt.bar(performance_data["backend"], performance_data["fps"])
        plt.ylabel("Frames Per Second (FPS)")
        plt.title("Throughput by Backend")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "backend_throughput.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create bar chart of model size
    if "memory" in results and results["memory"]:
        memory_data = pd.DataFrame(results["memory"])
        
        if not memory_data.empty and "model_size_mb" in memory_data.columns:
            # Filter out None values
            memory_data = memory_data[memory_data["model_size_mb"].notna()]
            
            if not memory_data.empty:
                # Sort by model size
                memory_data = memory_data.sort_values("model_size_mb")
                
                plt.figure(figsize=(10, 6))
                plt.bar(memory_data["backend"], memory_data["model_size_mb"])
                plt.ylabel("Model Size (MB)")
                plt.title("Model Size by Backend")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "backend_model_size.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Create bar chart of peak memory usage if available
                if "peak_memory_mb" in memory_data.columns:
                    memory_data_with_peak = memory_data[memory_data["peak_memory_mb"].notna()]
                    
                    if not memory_data_with_peak.empty:
                        plt.figure(figsize=(10, 6))
                        plt.bar(memory_data_with_peak["backend"], memory_data_with_peak["peak_memory_mb"])
                        plt.ylabel("Peak Memory (MB)")
                        plt.title("Peak Memory Usage by Backend")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, "backend_peak_memory.png"), dpi=300, bbox_inches='tight')
                        plt.close()