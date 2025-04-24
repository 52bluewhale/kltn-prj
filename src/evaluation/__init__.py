"""
Evaluation utilities for YOLOv8 QAT models.

This module provides tools for evaluating and comparing regular and quantized models,
analyzing performance metrics, and visualizing results.
"""

from .metrics import (
    compute_map,
    compute_precision_recall,
    compute_confusion_matrix,
    compute_f1_score,
    calculate_accuracy,
    calculate_mean_average_precision
)

from .visualization import (
    plot_precision_recall_curve,
    plot_confusion_matrix,
    visualize_detections,
    visualize_activation_distributions,
    compare_detection_results,
    generate_evaluation_report
)

from .compare_models import (
    compare_fp32_int8_models,
    compare_model_outputs,
    compute_output_error,
    compare_layer_outputs,
    export_comparison_report
)

from .latency_testing import (
    measure_inference_time,
    benchmark_model,
    profile_model_layers,
    measure_throughput,
    export_benchmark_results
)

from .accuracy_drift import (
    track_accuracy_change,
    analyze_quantization_impact,
    identify_critical_layers,
    measure_drift_per_class,
    plot_accuracy_drift
)

from .memory_profiling import (
    measure_model_size,
    measure_memory_usage,
    profile_activation_memory,
    compare_memory_requirements,
    export_memory_profile
)

# Main API functions
def evaluate_model(model, dataloader, metrics=None):
    """
    Evaluate model performance on given dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation dataset
        metrics: List of metrics to compute (default: ['map', 'latency'])
        
    Returns:
        Dictionary with evaluation results
    """
    from .metrics import compute_evaluation_metrics
    return compute_evaluation_metrics(model, dataloader, metrics)

def compare_models(fp32_model, int8_model, dataloader, metrics=None):
    """
    Compare FP32 and INT8 models performance.
    
    Args:
        fp32_model: Floating point model
        int8_model: Quantized model
        dataloader: DataLoader with evaluation dataset
        metrics: List of metrics to compute
        
    Returns:
        Dictionary with comparison results
    """
    from .compare_models import compare_fp32_int8_models
    return compare_fp32_int8_models(fp32_model, int8_model, dataloader, metrics)

def generate_report(evaluation_results, output_path="./evaluation_report"):
    """
    Generate comprehensive evaluation report.
    
    Args:
        evaluation_results: Results from evaluate_model or compare_models
        output_path: Path to save report
        
    Returns:
        Path to generated report
    """
    from .visualization import generate_evaluation_report
    return generate_evaluation_report(evaluation_results, output_path)

def measure_performance(model, input_shape=(1, 3, 640, 640), num_runs=100, device="cuda"):
    """
    Measure model inference performance.
    
    Args:
        model: Model to benchmark
        input_shape: Shape of input tensor
        num_runs: Number of inference runs
        device: Device to run on
        
    Returns:
        Dictionary with performance metrics
    """
    from .latency_testing import benchmark_model
    return benchmark_model(model, input_shape, num_runs, device)