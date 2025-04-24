#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train YOLOv8 model with Quantization-Aware Training (QAT).
This script builds upon the base FP32 model to create a quantization-ready version.
"""

import os
import sys
import argparse
import yaml
import logging
import torch
from pathlib import Path
import time

# Add project root to path to allow imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from src.models import (
    create_yolov8_model, 
    prepare_model_for_qat,
    apply_qat_transforms
)
from src.data_utils import (
    create_qat_dataloader, 
    get_dataset_from_yaml
)
from src.training import (
    QATTrainer,
    build_loss_function,
    create_lr_scheduler
)
from src.evaluation import (
    evaluate_model,
    compare_fp32_int8_models
)
from src.quantization import (
    prepare_qat_model,
    create_qat_config_from_config_file,
    calibrate_model
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_qat')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train YOLOv8 model with Quantization-Aware Training')
    
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
    
    # QAT specific settings
    parser.add_argument('--qat-config', type=str, default='configs/quantization_config.yaml',
                        help='path to quantization configuration file')
    parser.add_argument('--start-epoch', type=int, default=10,
                        help='epoch to start applying quantization')
    parser.add_argument('--calibrate', action='store_true',
                        help='perform calibration before QAT')
    parser.add_argument('--calibration-batches', type=int, default=100,
                        help='number of batches to use for calibration')
    
    # Training settings
    parser.add_argument('--batch-size', type=int, default=16, 
                        help='total batch size for all GPUs')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='number of total epochs to run')
    parser.add_argument('--workers', type=int, default=8, 
                        help='number of data loading workers')
    parser.add_argument('--device', type=str, default='', 
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    # Optimizer settings
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning rate (lower than standard training)')
    parser.add_argument('--weight-decay', type=float, default=0.0005, 
                        help='weight decay')
    parser.add_argument('--optimizer', type=str, default='AdamW', 
                        help='optimizer type (SGD, Adam, AdamW)')
    
    # QAT distillation settings
    parser.add_argument('--distillation', action='store_true',
                        help='use knowledge distillation from FP32 model')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='temperature for knowledge distillation')
    
    # Learning rate scheduler settings
    parser.add_argument('--scheduler', type=str, default='qat', 
                        help='LR scheduler (qat, cosine, step, plateau)')
    
    # Saving settings
    parser.add_argument('--output-dir', type=str, default='models/checkpoints/qat', 
                        help='path to save models')
    parser.add_argument('--save-period', type=int, default=5, 
                        help='save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--export', action='store_true',
                        help='export final quantized model')
    parser.add_argument('--export-format', type=str, default='onnx',
                        help='export format (onnx, tensorrt, openvino)')
    
    # Logging settings
    parser.add_argument('--log-dir', type=str, default='logs/qat', 
                        help='path to save logs')
    
    return parser.parse_args()


def setup_training(args):
    """Setup training configuration based on arguments"""
    # Create configuration dictionary
    config = {
        'model': {
            'name': args.model,
            'img_size': args.img_size,
            'pretrained_weights': args.weights
        },
        'data': {
            'yaml_path': args.data,
            'img_size': args.img_size,
            'batch_size': args.batch_size,
            'workers': args.workers
        },
        'optimizer': {
            'type': args.optimizer.lower(),
            'lr': args.lr,
            'weight_decay': args.weight_decay
        },
        'qat': {
            'config_path': args.qat_config,
            'start_epoch': args.start_epoch,
            'calibrate': args.calibration_batches if args.calibrate else 0,
            'use_penalty': True,
            'penalty_factor': 0.01,
            'distillation': args.distillation,
            'temperature': args.temperature,
            'use_qat_scheduler': args.scheduler == 'qat',
            'monitor_error': True,
            'error_log_dir': os.path.join(args.log_dir, 'error_maps'),
            'save_error_maps': True,
            'progressive_stages': [
                {
                    'epoch': 0,  # Start immediately
                    'layers': ['model.0.conv'],  # First layer
                    'qconfig': 'first_layer'
                },
                {
                    'epoch': 5,  # After 5 epochs
                    'layers': ['model.\\d+\\.cv\\d+\\.conv'],  # Backbone layers
                    'qconfig': 'default'
                },
                {
                    'epoch': 15,  # After 15 epochs
                    'layers': ['model.\\d+\\.m\\.\\d+\\.cv\\d+\\.conv'],  # Feature layers
                    'qconfig': 'default'
                },
                {
                    'epoch': 25,  # After 25 epochs
                    'layers': ['model.\\d+\\.detect'],  # Detection head
                    'qconfig': 'sensitive'
                }
            ],
            'freeze_bn_epochs': 10  # Epochs after which to freeze BN
        },
        'scheduler': {
            'type': args.scheduler
        },
        'training': {
            'epochs': args.epochs,
            'device': args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
        },
        'output': {
            'dir': args.output_dir,
            'save_period': args.save_period,
            'export': args.export,
            'export_format': args.export_format
        },
        'logging': {
            'dir': args.log_dir
        }
    }
    
    # Customize scheduler configuration based on type
    if args.scheduler == 'qat':
        config['scheduler'].update({
            'warmup_epochs': 5,
            'initial_lr': args.lr / 10,
            'peak_lr': args.lr,
            'final_lr': args.lr / 100,
            'total_epochs': args.epochs
        })
    elif args.scheduler == 'cosine':
        config['scheduler'].update({
            'T_0': 10,
            'T_mult': 2,
            'eta_min': 1e-5
        })
    elif args.scheduler == 'step':
        config['scheduler'].update({
            'step_size': 20,
            'gamma': 0.1
        })
    elif args.scheduler == 'plateau':
        config['scheduler'].update({
            'patience': 5,
            'factor': 0.1
        })
    
    # Setup callbacks
    config['callbacks'] = [
        {
            'type': 'checkpoint',
            'filepath': os.path.join(args.output_dir, 'model_qat_{epoch:02d}_{val_loss:.4f}.pt'),
            'monitor': 'val_loss',
            'save_best_only': True
        },
        {
            'type': 'early_stopping',
            'monitor': 'val_loss',
            'patience': 20
        },
        {
            'type': 'tensorboard',
            'log_dir': os.path.join(args.log_dir, 'tensorboard')
        }
    ]
    
    return config


def create_dataloaders(config):
    """Create train and validation dataloaders optimized for QAT"""
    data_config = config['data']
    
    logger.info(f"Loading dataset from {data_config['yaml_path']}")
    dataset_info = get_dataset_from_yaml(data_config['yaml_path'])
    
    # Verify dataset has required splits
    required_keys = ['train', 'val']
    missing_keys = [key for key in required_keys if key not in dataset_info]
    if missing_keys:
        raise ValueError(f"Dataset YAML missing required keys: {missing_keys}")
    
    # Create train dataloader with QAT-specific augmentations
    train_loader, _ = create_qat_dataloader(
        dataset_yaml=data_config['yaml_path'],
        batch_size=data_config['batch_size'],
        img_size=data_config['img_size'],
        augment=True,
        shuffle=True,
        workers=data_config['workers'],
        mode='train',
        use_qat_transforms=True
    )
    
    # Create validation dataloader
    val_loader, _ = create_qat_dataloader(
        dataset_yaml=data_config['yaml_path'],
        batch_size=data_config['batch_size'],
        img_size=data_config['img_size'],
        augment=False,
        shuffle=False,
        workers=data_config['workers'],
        mode='val',
        use_qat_transforms=False
    )
    
    # Create calibration dataloader (smaller subset for calibration)
    if config['qat']['calibrate'] > 0:
        # Use a subset of the validation set for calibration
        calibration_loader, _ = create_qat_dataloader(
            dataset_yaml=data_config['yaml_path'],
            batch_size=data_config['batch_size'],
            img_size=data_config['img_size'],
            augment=False,
            shuffle=True,  # Shuffle to get diverse samples
            workers=data_config['workers'],
            mode='val',
            use_qat_transforms=False
        )
    else:
        calibration_loader = None
    
    logger.info(f"Created dataloaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    return train_loader, val_loader, calibration_loader, dataset_info


def load_and_prepare_model(config, num_classes):
    """Load FP32 model and prepare it for QAT"""
    model_config = config['model']
    qat_config = config['qat']
    device = torch.device(config['training']['device'])
    
    # Step 1: Load the pretrained FP32 model
    logger.info(f"Loading pretrained model from {model_config['pretrained_weights']}")
    model = create_yolov8_model(
        model_name=model_config['name'],
        num_classes=num_classes,
        pretrained=False,
        pretrained_path=model_config['pretrained_weights']
    )
    
    # Step 2: Load the QAT configuration
    logger.info(f"Loading QAT configuration from {qat_config['config_path']}")
    try:
        with open(qat_config['config_path'], 'r') as f:
            quant_config = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load QAT config: {e}. Using default configuration.")
        quant_config = {
            "quantization": {
                "default_qconfig": "default",
                "skip_layers": [
                    "model.\\d+\\.forward",
                    "model.\\d+\\.detect"
                ],
                "layer_configs": [
                    {
                        "pattern": "model.0.conv",
                        "qconfig": "first_layer"
                    },
                    {
                        "pattern": "model.\\d+\\.detect",
                        "qconfig": "sensitive"
                    }
                ]
            }
        }
    
    # Step 3: Prepare the model for QAT
    logger.info("Preparing model for Quantization-Aware Training...")
    
    # First, apply model transformations (fusions, etc.)
    model = apply_qat_transforms(model)
    
    # Then prepare for QAT with the configuration
    qat_model = prepare_model_for_qat(
        model=model,
        config_path=qat_config['config_path'] if os.path.exists(qat_config['config_path']) else None,
        skip_layers=quant_config.get("quantization", {}).get("skip_layers", [])
    )
    
    # Move model to device
    qat_model = qat_model.to(device)
    
    return qat_model, model  # Return both QAT model and original FP32 model


def calibrate_qat_model(model, calibration_loader, num_batches, device):
    """Calibrate the QAT model to set initial quantization parameters"""
    logger.info(f"Calibrating model with {num_batches} batches...")
    
    # Set model to evaluation mode for calibration
    model.eval()
    
    # Process calibration batches
    with torch.no_grad():
        for batch_idx, batch in enumerate(calibration_loader):
            if batch_idx >= num_batches:
                break
                
            # Extract images
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                images = batch[0]
            else:
                images = batch['image'] if 'image' in batch else batch['images']
                
            # Move to device
            images = images.to(device)
            
            # Forward pass to update observers
            _ = model(images)
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Calibrated with {batch_idx + 1}/{num_batches} batches")
    
    logger.info("Calibration complete")
    return model


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Setup training configuration
    config = setup_training(args)
    
    # Create output directories
    os.makedirs(config['output']['dir'], exist_ok=True)
    os.makedirs(config['logging']['dir'], exist_ok=True)
    os.makedirs(config['qat']['error_log_dir'], exist_ok=True)
    
    # Create dataloaders
    train_loader, val_loader, calibration_loader, dataset_info = create_dataloaders(config)
    
    # Determine number of classes from dataset
    num_classes = dataset_info.get('nc', 0)
    if num_classes <= 0:
        raise ValueError(f"Invalid number of classes: {num_classes}")
    
    logger.info(f"Training YOLOv8 model with QAT for {num_classes} classes")
    
    # Load and prepare model for QAT
    qat_model, fp32_model = load_and_prepare_model(config, num_classes)
    
    # Perform calibration if requested
    if config['qat']['calibrate'] > 0 and calibration_loader is not None:
        calibration_batches = min(config['qat']['calibrate'], len(calibration_loader))
        qat_model = calibrate_qat_model(
            qat_model, 
            calibration_loader, 
            calibration_batches, 
            config['training']['device']
        )
    
    # Create QAT loss function
    if config['qat']['distillation']:
        from src.training.loss import DistillationLoss
        criterion = DistillationLoss(
            teacher_model=fp32_model,
            temperature=config['qat']['temperature'],
            alpha=0.5  # Balance between distillation and task loss
        )
        logger.info("Using knowledge distillation loss")
    else:
        criterion = build_loss_function(
            loss_type='qat_penalty',
            config={'penalty_factor': config['qat']['penalty_factor']}
        )
        logger.info("Using QAT penalty loss")
    
    # Create optimizer
    optimizer_config = config['optimizer']
    if optimizer_config['type'] == 'sgd':
        optimizer = torch.optim.SGD(
            qat_model.parameters(),
            lr=optimizer_config['lr'],
            momentum=0.937,  # Default momentum
            weight_decay=optimizer_config['weight_decay']
        )
    elif optimizer_config['type'] == 'adam':
        optimizer = torch.optim.Adam(
            qat_model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
    elif optimizer_config['type'] == 'adamw':
        optimizer = torch.optim.AdamW(
            qat_model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")
    
    # Create learning rate scheduler
    scheduler = create_lr_scheduler(optimizer, config['scheduler']['type'], config['scheduler'])
    
    # Prepare QAT trainer configuration
    trainer_config = {
        'optimizer': optimizer,
        'criterion': criterion,
        'scheduler': scheduler,
        'callbacks': config['callbacks'],
        'device': config['training']['device'],
        'qat': config['qat']
    }
    
    # Create QAT trainer
    trainer = QATTrainer(qat_model, trainer_config)
    
    # Train model
    logger.info(f"Starting QAT training for {config['training']['epochs']} epochs")
    start_time = time.time()
    trainer.train(train_loader, val_loader, config['training']['epochs'])
    training_time = time.time() - start_time
    logger.info(f"QAT training completed in {training_time:.2f} seconds")
    
    # Save final QAT model
    final_qat_model_path = os.path.join(config['output']['dir'], 'yolov8_qat_final.pt')
    trainer.save_model(final_qat_model_path)
    logger.info(f"Final QAT model saved to {final_qat_model_path}")
    
    # Convert to fully quantized model
    logger.info("Converting QAT model to fully quantized model...")
    quantized_model = trainer.convert_model_to_quantized()
    
    # Save quantized model
    quantized_model_path = os.path.join(config['output']['dir'], 'yolov8_int8_quantized.pt')
    torch.save(quantized_model.state_dict(), quantized_model_path)
    logger.info(f"Quantized model saved to {quantized_model_path}")
    
    # Evaluate and compare models
    logger.info("Evaluating models...")
    
    # Move models to appropriate device
    device = torch.device(config['training']['device'])
    fp32_model = fp32_model.to(device)
    fp32_model.eval()
    quantized_model = quantized_model.to(device)
    quantized_model.eval()
    
    # Evaluate FP32 model
    fp32_metrics = evaluate_model(fp32_model, val_loader)
    
    # Evaluate quantized model
    int8_metrics = evaluate_model(quantized_model, val_loader)
    
    # Compare models
    comparison = compare_fp32_int8_models(
        fp32_model=fp32_model,
        int8_model=quantized_model,
        dataloader=val_loader,
        metrics=['accuracy', 'latency', 'output_error'],
        device=config['training']['device']
    )
    
    # Log evaluation results
    result_str = "Evaluation results:\n\n"
    result_str += "FP32 Model:\n"
    for k, v in fp32_metrics.items():
        result_str += f"{k}: {v}\n"
    
    result_str += "\nINT8 Model:\n"
    for k, v in int8_metrics.items():
        result_str += f"{k}: {v}\n"
    
    result_str += "\nComparison:\n"
    if 'accuracy_comparison' in comparison:
        acc_comp = comparison['accuracy_comparison']
        result_str += f"Accuracy change: {acc_comp.get('absolute_change', 0):.4f} ({acc_comp.get('relative_change', 0)*100:.2f}%)\n"
    
    if 'latency_comparison' in comparison:
        lat_comp = comparison['latency_comparison']
        result_str += f"Speedup: {lat_comp.get('speedup', 0):.2f}x\n"
        result_str += f"FP32 FPS: {lat_comp.get('fp32_fps', 0):.2f}\n"
        result_str += f"INT8 FPS: {lat_comp.get('int8_fps', 0):.2f}\n"
    
    logger.info(result_str)
    
    # Save evaluation results
    eval_path = os.path.join(config['logging']['dir'], 'qat_evaluation_results.txt')
    with open(eval_path, 'w') as f:
        f.write(result_str)
    
    # Export model if requested
    if config['output']['export']:
        export_format = config['output']['export_format'].lower()
        export_path = os.path.join(config['output']['dir'], f'yolov8_int8.{export_format}')
        
        logger.info(f"Exporting quantized model to {export_format} format...")
        
        try:
            from src.deployment.optimize import convert_to_target_format
            
            # Create export configuration
            export_config = {
                "model_input_shape": [1, 3, config['model']['img_size'], config['model']['img_size']],
                "opset_version": 13,
                "simplify": True,
                "dynamic": True
            }
            
            # Export model
            export_result = convert_to_target_format(
                model=quantized_model,
                output_path=export_path,
                target_format=export_format,
                config_path=None  # Use default export settings
            )
            
            if export_result:
                logger.info(f"Model exported successfully to {export_path}")
            else:
                logger.error(f"Failed to export model to {export_format} format")
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
    
    logger.info(f"QAT process completed. Final model saved to {quantized_model_path}")
    
    return quantized_model_path


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error during QAT training: {e}")
        sys.exit(1)