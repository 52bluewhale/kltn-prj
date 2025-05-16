#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration script for YOLOv8 quantization.

This script performs model calibration to determine appropriate quantization
parameters based on a representative dataset. It's a pre-step to quantization-aware
training that helps set the initial scaling factors and zero points.
"""

import os
import sys
import argparse
import yaml
import logging
import torch
import time
from pathlib import Path
from tqdm import tqdm

# Add project root to path to allow imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from src.models import (
    create_yolov8_model,
    prepare_model_for_qat
)
from src.data_utils import (
    create_dataloader,
    create_calibration_dataloader,
    get_dataset_from_yaml
)
from src.quantization import (
    load_quantization_config,
    calibrate_model,
    build_calibrator,
    create_qat_config_from_config_file
)
from src.evaluation import (
    evaluate_model,
    measure_quantization_error
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('calibrate')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Calibrate YOLOv8 model for quantization')
    
    # Dataset settings
    parser.add_argument('--data', type=str, default='dataset/vietnam-traffic-sign-detection/dataset.yaml',
                        help='path to dataset configuration yaml file')
    
    # Model settings
    parser.add_argument('--model', type=str, default='yolov8n', 
                        help='YOLOv8 model variant (yolov8n, yolov8s, yolov8m, etc)')
    parser.add_argument('--weights', type=str, required=True,
                        help='path to pretrained FP32 model weights')
    parser.add_argument('--img-size', type=int, default=640, 
                        help='input image size')
    
    # Calibration settings
    parser.add_argument('--calib-config', type=str, default='configs/quantization_config.yaml',
                        help='path to calibration/quantization configuration file')
    parser.add_argument('--method', type=str, default='histogram', 
                        choices=['minmax', 'histogram', 'percentile', 'entropy'],
                        help='calibration method')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='number of batches to use for calibration')
    parser.add_argument('--percentile', type=float, default=99.99,
                        help='percentile value for percentile calibration method')
    parser.add_argument('--batch-size', type=int, default=16, 
                        help='batch size for calibration')
    parser.add_argument('--workers', type=int, default=8, 
                        help='number of data loading workers')
    parser.add_argument('--device', type=str, default='', 
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='models/checkpoints/calibrated', 
                        help='path to save calibrated model')
    parser.add_argument('--export-config', action='store_true',
                        help='export calibration configuration')
    
    # Analysis settings
    parser.add_argument('--analyze', action='store_true',
                        help='analyze calibration results')
    parser.add_argument('--error-threshold', type=float, default=0.1,
                        help='error threshold for layer analysis')
    
    return parser.parse_args()


def load_fp32_model(args):
    """
    Load the pretrained FP32 model.
    
    Args:
        args: Command line arguments
    
    Returns:
        Pretrained FP32 model
    """
    # Get dataset information to determine number of classes
    dataset_info = get_dataset_from_yaml(args.data)
    num_classes = dataset_info.get('nc', 80)  # Default to 80 classes (COCO) if not specified
    
    logger.info(f"Loading pretrained FP32 model: {args.model} with {num_classes} classes")
    
    # Create model
    model = create_yolov8_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=False,
        pretrained_path=args.weights
    )
    
    logger.info(f"Model loaded successfully")
    return model, num_classes


def create_calibration_dataset(args):
    """
    Create dataset for calibration.
    
    Args:
        args: Command line arguments
    
    Returns:
        Calibration dataloader
    """
    logger.info(f"Creating calibration dataset from {args.data}")
    
    # Get dataset information
    dataset_info = get_dataset_from_yaml(args.data)
    
    # Check if validation set exists
    if 'val' not in dataset_info:
        raise ValueError("Validation set not found in dataset YAML")
    
    # Get validation image path
    val_img_path = dataset_info['val']
    
    # Create calibration dataloader
    calibration_loader = create_calibration_dataloader(
        img_path=val_img_path,
        batch_size=args.batch_size,
        img_size=args.img_size,
        workers=args.workers,
        cache=False
    )
    
    logger.info(f"Created calibration dataloader with {len(calibration_loader)} batches")
    return calibration_loader


def prepare_model_for_calibration(model, args):
    """
    Prepare model for calibration.
    
    Args:
        model: FP32 model
        args: Command line arguments
    
    Returns:
        Model prepared for calibration
    """
    logger.info(f"Preparing model for calibration")
    
    # Load quantization configuration
    try:
        quant_config = load_quantization_config(args.calib_config)
        logger.info(f"Loaded quantization configuration from {args.calib_config}")
    except Exception as e:
        logger.warning(f"Failed to load quantization configuration: {e}. Using default configuration.")
        quant_config = None
    
    # Prepare model for QAT (which includes preparation for calibration)
    prepared_model = prepare_model_for_qat(
        model=model,
        config_path=args.calib_config if quant_config is not None else None
    )
    
    logger.info(f"Model prepared for calibration")
    return prepared_model


def run_calibration(model, calibration_loader, args):
    """
    Run calibration on the model.
    
    Args:
        model: Model prepared for calibration
        calibration_loader: Calibration dataloader
        args: Command line arguments
    
    Returns:
        Calibrated model
    """
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running calibration using {args.method} method on {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Build calibrator with specific method
    calibrator_kwargs = {
        'device': device,
        'num_batches': min(args.num_batches, len(calibration_loader))
    }
    
    if args.method == 'percentile':
        calibrator_kwargs['percentile'] = args.percentile
    
    calibrator = build_calibrator(
        model=model,
        dataloader=calibration_loader,
        method=args.method,
        **calibrator_kwargs
    )
    
    # Start calibration
    start_time = time.time()
    calibrated_model = calibrator.calibrate(progress=True)
    calibration_time = time.time() - start_time
    
    logger.info(f"Calibration completed in {calibration_time:.2f} seconds")
    return calibrated_model


def analyze_calibration(model, calibrated_model, args):
    """
    Analyze calibration results.
    
    Args:
        model: Original FP32 model
        calibrated_model: Calibrated model
        args: Command line arguments
    
    Returns:
        Analysis results
    """
    logger.info(f"Analyzing calibration results")
    
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a random test input for analysis
    test_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    
    # Measure quantization error
    error_results = measure_quantization_error(
        fp32_model=model.to(device),
        int8_model=calibrated_model,
        test_input=test_input
    )
    
    # Find layers with high error
    high_error_layers = []
    for layer_name, error in error_results.items():
        if error['abs_error'] > args.error_threshold:
            high_error_layers.append((layer_name, error['abs_error']))
    
    # Sort by error
    high_error_layers.sort(key=lambda x: x[1], reverse=True)
    
    # Log results
    logger.info(f"Found {len(high_error_layers)} layers with error above threshold {args.error_threshold}")
    
    for layer_name, error in high_error_layers[:10]:  # Show top 10
        logger.info(f"Layer {layer_name}: Error = {error:.6f}")
    
    # Create analysis report
    analysis_report = {
        'high_error_layers': high_error_layers,
        'error_results': error_results,
        'error_threshold': args.error_threshold
    }
    
    return analysis_report


def save_calibrated_model(calibrated_model, analysis_report, args):
    """
    Save calibrated model and analysis report.
    
    Args:
        calibrated_model: Calibrated model
        analysis_report: Analysis report
        args: Command line arguments
    
    Returns:
        Path to saved model
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save calibrated model
    model_path = os.path.join(args.output_dir, f'yolov8_{args.model}_calibrated_{args.method}.pt')
    
    # Save model with metadata
    metadata = {
        'calibration_method': args.method,
        'img_size': args.img_size,
        'num_batches': args.num_batches,
        'batch_size': args.batch_size,
        'device': args.device,
        'percentile': args.percentile if args.method == 'percentile' else None,
        'calibration_config': args.calib_config,
        'original_weights': args.weights,
        'timestamp': time.strftime('%Y%m%d-%H%M%S')
    }
    
    # We need to save both the model state and the quantization parameters
    torch.save({
        'model_state_dict': calibrated_model.state_dict(),
        'metadata': metadata,
        'analysis': analysis_report if args.analyze else None
    }, model_path)
    
    logger.info(f"Calibrated model saved to {model_path}")
    
    # Export quantization configuration if requested
    if args.export_config:
        # Extract quantization parameters
        quant_params = {}
        for name, module in calibrated_model.named_modules():
            if hasattr(module, 'activation_post_process') and hasattr(module.activation_post_process, 'min_val'):
                observer = module.activation_post_process
                
                if hasattr(observer, 'min_val') and hasattr(observer, 'max_val'):
                    quant_params[name] = {
                        'min_val': observer.min_val.item() if hasattr(observer.min_val, 'item') else float(observer.min_val),
                        'max_val': observer.max_val.item() if hasattr(observer.max_val, 'item') else float(observer.max_val)
                    }
        
        # Create configuration file
        config_path = os.path.join(args.output_dir, f'calibration_config_{args.method}.yaml')
        
        # Load existing configuration if available
        if os.path.exists(args.calib_config):
            with open(args.calib_config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {'quantization': {}}
        
        # Add calibration parameters
        config['quantization']['calibration'] = {
            'method': args.method,
            'parameters': quant_params,
            'metadata': metadata
        }
        
        # Add high error layers if available
        if args.analyze and 'high_error_layers' in analysis_report:
            config['quantization']['critical_layers'] = [
                name for name, _ in analysis_report['high_error_layers']
            ]
        
        # Save configuration
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        logger.info(f"Calibration configuration saved to {config_path}")
    
    return model_path


def main():
    """Main calibration function"""
    # Parse arguments
    args = parse_args()
    
    # Load pretrained FP32 model
    model, num_classes = load_fp32_model(args)
    
    # Create calibration dataset
    calibration_loader = create_calibration_dataset(args)
    
    # Prepare model for calibration
    prepared_model = prepare_model_for_calibration(model, args)
    
    # Run calibration
    calibrated_model = run_calibration(prepared_model, calibration_loader, args)
    
    # Analyze calibration results if requested
    if args.analyze:
        analysis_report = analyze_calibration(model, calibrated_model, args)
    else:
        analysis_report = None
    
    # Save calibrated model
    model_path = save_calibrated_model(calibrated_model, analysis_report, args)
    
    logger.info(f"Calibration process completed successfully")
    
    return model_path


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error during calibration: {e}")
        sys.exit(1)