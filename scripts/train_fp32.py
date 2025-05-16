#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train YOLOv8 model in standard FP32 mode.
This script provides the first step in the QAT process by training a base model.
"""

import os
import sys
import argparse
import yaml
import logging
import torch
from pathlib import Path

# Add project root to path to allow imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Only go up one level, not two
sys.path.append(project_root)

from src.models import (
    create_yolov8_model, 
    prepare_model_for_qat
)
from src.data_utils import (
    create_dataloader, 
    get_dataset_from_yaml
)
from src.training import (
    Trainer,
    build_loss_function,
    create_lr_scheduler
)
from src.evaluation import evaluate_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_fp32')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train YOLOv8 model in FP32 mode')
    
    # Dataset settings
    parser.add_argument('--data', type=str, default='dataset/vietnam-traffic-sign-detection/dataset.yaml',
                        help='path to dataset configuration yaml file')
    
    # Model settings
    parser.add_argument('--model', type=str, default='yolov8n', 
                        help='YOLOv8 model variant (yolov8n, yolov8s, yolov8m, etc)')
    parser.add_argument('--weights', type=str, default=None,
                        help='initial weights path (default: None uses pretrained weights)')
    parser.add_argument('--img-size', type=int, default=640, 
                        help='input image size')
    
    # Training settings
    parser.add_argument('--batch-size', type=int, default=16, 
                        help='total batch size for all GPUs')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='number of total epochs to run')
    parser.add_argument('--workers', type=int, default=8, 
                        help='number of data loading workers')
    parser.add_argument('--device', type=str, default='', 
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    # Optimizer settings
    parser.add_argument('--lr', type=float, default=0.01, 
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, 
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.937, 
                        help='SGD momentum')
    parser.add_argument('--optimizer', type=str, default='SGD', 
                        help='optimizer type (SGD, Adam, AdamW)')
    
    # Loss function settings
    parser.add_argument('--loss', type=str, default='focal', 
                        help='loss function type (focal, cross_entropy)')
    
    # Learning rate scheduler settings
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        help='LR scheduler (cosine, step, plateau)')
    
    # Saving settings
    parser.add_argument('--output-dir', type=str, default='models/checkpoints/fp32', 
                        help='path to save models')
    parser.add_argument('--save-period', type=int, default=10, 
                        help='save checkpoint every x epochs (disabled if < 1)')
    
    # Logging settings
    parser.add_argument('--log-dir', type=str, default='logs/fp32', 
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
            'weight_decay': args.weight_decay,
            'momentum': args.momentum
        },
        'loss': {
            'type': args.loss
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
            'save_period': args.save_period
        },
        'logging': {
            'dir': args.log_dir
        }
    }
    
    # Customize scheduler configuration based on type
    if args.scheduler == 'cosine':
        config['scheduler'].update({
            'T_0': 10,
            'T_mult': 2,
            'eta_min': 1e-5
        })
    elif args.scheduler == 'step':
        config['scheduler'].update({
            'step_size': 30,
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
            'filepath': os.path.join(args.output_dir, 'model_{epoch:02d}_{val_loss:.4f}.pt'),
            'monitor': 'val_loss',
            'save_best_only': True
        },
        {
            'type': 'early_stopping',
            'monitor': 'val_loss',
            'patience': 15
        },
        {
            'type': 'tensorboard',
            'log_dir': os.path.join(args.log_dir, 'tensorboard')
        }
    ]
    
    return config


def create_dataloaders(config):
    """Create train and validation dataloaders"""
    data_config = config['data']
    
    logger.info(f"Loading dataset from {data_config['yaml_path']}")
    dataset_info = get_dataset_from_yaml(data_config['yaml_path'])

    # Add these debug lines
    import os
    abs_path = os.path.abspath(data_config['yaml_path'])
    logger.info(f"Absolute path: {abs_path}")
    logger.info(f"File exists: {os.path.exists(abs_path)}")
    
    # Try to load the YAML directly
    import yaml
    try:
        with open(abs_path, 'r') as f:
            yaml_content = yaml.safe_load(f)
            logger.info(f"YAML content: {yaml_content}")
    except Exception as e:
        logger.error(f"Failed to load YAML: {e}")
    
    # Verify dataset has required splits
    required_keys = ['train', 'val']
    missing_keys = [key for key in required_keys if key not in dataset_info]
    if missing_keys:
        raise ValueError(f"Dataset YAML missing required keys: {missing_keys}")
    
    # Create train dataloader
    train_loader, _ = create_dataloader(
        dataset_yaml=data_config['yaml_path'],
        batch_size=data_config['batch_size'],
        img_size=data_config['img_size'],
        augment=True,
        shuffle=True,
        workers=data_config['workers'],
        mode='train'
    )
    
    # Create validation dataloader
    val_loader, _ = create_dataloader(
        dataset_yaml=data_config['yaml_path'],
        batch_size=data_config['batch_size'],
        img_size=data_config['img_size'],
        augment=False,
        shuffle=False,
        workers=data_config['workers'],
        mode='val'
    )
    
    logger.info(f"Created dataloaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    return train_loader, val_loader, dataset_info


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Setup training configuration
    config = setup_training(args)
    
    # Create output directories
    os.makedirs(config['output']['dir'], exist_ok=True)
    os.makedirs(config['logging']['dir'], exist_ok=True)
    
    # Create dataloaders
    train_loader, val_loader, dataset_info = create_dataloaders(config)
    
    # Determine number of classes from dataset
    num_classes = dataset_info.get('nc', 0)
    if num_classes <= 0:
        raise ValueError(f"Invalid number of classes: {num_classes}")
    
    logger.info(f"Training YOLOv8 model for {num_classes} classes")
    
    # Create model
    model = create_yolov8_model(
        model_name=config['model']['name'],
        num_classes=num_classes,
        pretrained=True,
        pretrained_path=config['model']['pretrained_weights']
    )
    
    # Output model summary
    logger.info(f"Model created: {config['model']['name']} with {num_classes} classes")
    
    # Create optimizer
    optimizer_config = config['optimizer']
    if optimizer_config['type'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config['lr'],
            momentum=optimizer_config['momentum'],
            weight_decay=optimizer_config['weight_decay']
        )
    elif optimizer_config['type'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
    elif optimizer_config['type'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")
    
    # Create loss function
    criterion = build_loss_function(config['loss']['type'])
    
    # Create learning rate scheduler
    scheduler = create_lr_scheduler(optimizer, config['scheduler']['type'], config['scheduler'])
    
    # Create trainer
    trainer_config = {
        'optimizer': optimizer,
        'criterion': criterion,
        'scheduler': scheduler,
        'callbacks': config['callbacks'],
        'device': config['training']['device']
    }
    
    trainer = Trainer(model, trainer_config)
    
    # Train model
    logger.info(f"Starting training for {config['training']['epochs']} epochs")
    trainer.train(train_loader, val_loader, config['training']['epochs'])
    
    # Save final model
    final_model_path = os.path.join(config['output']['dir'], 'yolov8_fp32_final.pt')
    trainer.save_model(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = evaluate_model(model, val_loader)
    
    # Log evaluation results
    result_str = "Evaluation results:\n"
    for k, v in metrics.items():
        result_str += f"{k}: {v}\n"
    logger.info(result_str)
    
    # Save evaluation results
    eval_path = os.path.join(config['logging']['dir'], 'evaluation_results.txt')
    with open(eval_path, 'w') as f:
        f.write(result_str)
    
    # Save config
    config_path = os.path.join(config['logging']['dir'], 'training_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Training completed. Model saved to {final_model_path}")
    
    return final_model_path


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        sys.exit(1)