#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for YOLOv8 Quantization-Aware Training project.
This script provides a unified interface for training, quantization, evaluation, and export.
"""

import argparse
import os
import logging
import yaml
from pathlib import Path
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='YOLOv8 Quantization-Aware Training'
    )
    
    # Common arguments
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/vietnam-traffic-sign-detection/dataset.yaml',
                        help='Path to dataset YAML file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train base model parser
    train_fp32_parser = subparsers.add_parser('train_fp32', help='Train base FP32 model')
    train_fp32_parser.add_argument('--model', type=str, default='yolov8n.pt',
                              help='Base model to start from (yolov8n.pt, yolov8s.pt, etc.)')
    train_fp32_parser.add_argument('--epochs', type=int, default=100,
                              help='Number of training epochs')
    train_fp32_parser.add_argument('--batch-size', type=int, default=16,
                              help='Batch size')
    
    # Train QAT model parser
    train_qat_parser = subparsers.add_parser('train_qat', help='Train QAT model')
    train_qat_parser.add_argument('--model', type=str, required=True,
                             help='Path to base model checkpoint')
    train_qat_parser.add_argument('--qconfig', type=str, default='configs/qat_config.yaml',
                             help='QAT configuration file')
    train_qat_parser.add_argument('--epochs', type=int, default=10,
                             help='Number of QAT epochs')
    train_qat_parser.add_argument('--batch-size', type=int, default=16,
                             help='Batch size')
    
    # Evaluate model parser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    eval_parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    eval_parser.add_argument('--quantized', action='store_true',
                        help='Whether to evaluate as quantized model')
    
    # Export model parser
    export_parser = subparsers.add_parser('export', help='Export model')
    export_parser.add_argument('--model', type=str, required=True,
                         help='Path to model checkpoint')
    export_parser.add_argument('--format', type=str, default='onnx',
                         choices=['onnx', 'tflite'], help='Export format')
    export_parser.add_argument('--output', type=str, default='models/exported',
                         help='Output directory')
    
    # Analyze quantization parser
    analyze_parser = subparsers.add_parser('analyze', help='Analyze quantization effects')
    analyze_parser.add_argument('--model', type=str, required=True,
                           help='Path to QAT model checkpoint')
    analyze_parser.add_argument('--batch-size', type=int, default=1,
                           help='Batch size')
    analyze_parser.add_argument('--output', type=str, default='logs/quantization_analysis',
                           help='Output directory')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_fp32(args, config):
    """Train base FP32 model."""
    from scripts.train_fp32 import train_model
    
    logger.info(f"Training base FP32 model using {args.model}")
    
    # Ensure the checkpoints directory exists
    os.makedirs('models/checkpoints/fp32', exist_ok=True)
    
    # Train model
    trained_model = train_model(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        config=config
    )
    
    # Save model
    save_path = f"models/checkpoints/fp32/yolov8_{Path(args.model).stem}_finetuned.pt"
    torch.save(trained_model.state_dict(), save_path)
    
    logger.info(f"Model saved to {save_path}")
    return save_path

def train_qat(args, config):
    """Train QAT model."""
    from scripts.train_qat import train_qat_model
    
    logger.info(f"Training QAT model using {args.model}")
    
    # Load QAT-specific configuration
    qat_config = load_config(args.qconfig)
    
    # Ensure the checkpoints directory exists
    os.makedirs('models/checkpoints/qat', exist_ok=True)
    
    # Train QAT model
    trained_model = train_qat_model(
        model_path=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        config=qat_config
    )
    
    # Save model
    model_name = Path(args.model).stem
    save_path = f"models/checkpoints/qat/yolov8_{model_name}_qat.pt"
    torch.save(trained_model.state_dict(), save_path)
    
    logger.info(f"QAT model saved to {save_path}")
    return save_path

def evaluate(args, config):
    """Evaluate model."""
    from scripts.evaluate import evaluate_model
    
    logger.info(f"Evaluating model {args.model}")
    
    # Evaluate model
    metrics = evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        batch_size=args.batch_size,
        device=args.device,
        quantized=args.quantized,
        config=config
    )
    
    # Print metrics
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics

def export_model(args, config):
    """Export model to deployment format."""
    from scripts.export import export_model
    
    logger.info(f"Exporting model {args.model} to {args.format}")
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Export model
    export_path = export_model(
        model_path=args.model,
        output_dir=args.output,
        format=args.format,
        config=config
    )
    
    logger.info(f"Model exported to {export_path}")
    return export_path

def analyze_quantization(args, config):
    """Analyze quantization effects."""
    from scripts.analyze_quantization_error import analyze_model
    
    logger.info(f"Analyzing quantization effects for {args.model}")
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Analyze model
    results = analyze_model(
        model_path=args.model,
        data_yaml=args.data,
        batch_size=args.batch_size,
        output_dir=args.output,
        device=args.device,
        config=config
    )
    
    logger.info(f"Analysis results saved to {args.output}")
    return results

def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Execute command
    if args.command == 'train_fp32':
        train_fp32(args, config)
    elif args.command == 'train_qat':
        train_qat(args, config)
    elif args.command == 'evaluate':
        evaluate(args, config)
    elif args.command == 'export':
        export_model(args, config)
    elif args.command == 'analyze':
        analyze_quantization(args, config)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())