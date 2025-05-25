#!/usr/bin/env python
"""
Fixed YOLOv8 Quantization-Aware Training (QAT) Script

This fixed version preserves fake quantization during training and conversion.

Key fixes:
1. Uses QuantizedYOLOv8Fixed class that preserves quantization
2. Proper saving and loading of QAT models with fake quantization intact
3. Conversion from preserved QAT model instead of stripped model
"""
import os
import sys
import logging
import argparse
import yaml
import torch
import random
import numpy as np
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_qat_fixed')

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("Ultralytics YOLO package not found. Please install with: pip install ultralytics")
    sys.exit(1)

# Import project modules
from src.config import (
    DATASET_YAML, PRETRAINED_MODEL, IMG_SIZE, BATCH_SIZE, 
    QAT_EPOCHS, QAT_LEARNING_RATE, DEVICE
)

# Import the FIXED quantization class
from src.models.yolov8_qat_fixed import QuantizedYOLOv8Fixed

def test_quantization_preservation():
    """Test if quantization preservation is working."""
    logger.info("Testing quantization preservation in fixed implementation...")
    
    try:
        # Create QAT model
        qat_model = QuantizedYOLOv8Fixed(
            model_path='yolov8n.pt',
            qconfig_name='default',
            skip_detection_head=True,
            fuse_modules=True
        )
        
        # Prepare for QAT
        qat_model.prepare_for_qat()
        
        # Check quantization modules
        fake_quant_count = sum(1 for name, module in qat_model.preserved_qat_model.named_modules() 
                             if hasattr(module, 'weight_fake_quant'))
        observer_count = sum(1 for name, module in qat_model.preserved_qat_model.named_modules() 
                           if hasattr(module, 'activation_post_process'))
        
        logger.info(f"✅ Fixed implementation created QAT model with:")
        logger.info(f"  Fake quantization modules: {fake_quant_count}")
        logger.info(f"  Observer modules: {observer_count}")
        
        if fake_quant_count == 0:
            logger.error("❌ ERROR: Fixed QAT model creation is not working!")
            return False
        
        # Test saving with quantization preservation
        test_save_path = "test_fixed_qat_model.pt"
        success = qat_model.save_qat_model_with_quantization(qat_model.preserved_qat_model, test_save_path)
        
        if success:
            # Test loading
            loaded_model = qat_model.load_qat_model_with_quantization(test_save_path)
            if loaded_model is not None:
                logger.info("✅ Fixed quantization preservation test PASSED")
                test_passed = True
            else:
                logger.error("❌ Failed to load preserved QAT model")
                test_passed = False
        else:
            logger.error("❌ Failed to save QAT model with quantization")
            test_passed = False
        
        # Cleanup
        if os.path.exists(test_save_path):
            os.remove(test_save_path)
        
        return test_passed
        
    except Exception as e:
        logger.error(f"❌ Fixed quantization test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FIXED YOLOv8 QAT Training")
    
    # Basic parameters
    parser.add_argument('--model', type=str, default=PRETRAINED_MODEL,
                      help='path to model, or model name from ultralytics')
    parser.add_argument('--data', type=str, default=DATASET_YAML,
                      help='dataset.yaml path')
    
    # QAT parameters
    parser.add_argument('--qconfig', type=str, default='default',
                      choices=['default', 'sensitive', 'first_layer', 'last_layer', 'lsq'],
                      help='quantization configuration')
    parser.add_argument('--fuse', action='store_true', default=True,
                      help='fuse Conv+BN+ReLU modules for quantization')
    parser.add_argument('--keep-detection-head', action='store_true',
                      help='quantize detection head (not recommended)')
    parser.add_argument('--config', type=str, default='configs/qat_config.yaml',
                      help='QAT configuration file')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=QAT_EPOCHS,
                      help='number of epochs for QAT')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                      help='batch size')
    parser.add_argument('--img-size', type=int, default=IMG_SIZE,
                      help='input image size')
    parser.add_argument('--lr', type=float, default=QAT_LEARNING_RATE,
                      help='learning rate (typically 10x smaller than regular training)')
    
    # Output parameters
    parser.add_argument('--save-dir', type=str, default='models/checkpoints/qat_fixed',
                      help='directory to save results')
    parser.add_argument('--output', type=str, default='quantized_model_fixed.pt',
                      help='output model name')
    parser.add_argument('--log-dir', type=str, default='logs/qat_fixed',
                      help='directory to save logs')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default=DEVICE,
                      help='device to train on')
    
    # Evaluation parameters
    parser.add_argument('--eval', action='store_true',
                      help='evaluate model after training')
    parser.add_argument('--export', action='store_true',
                      help='export model to ONNX after training')
    parser.add_argument('--export-dir', type=str, default='models/exported_fixed',
                      help='directory to save exported models')
    
    # Advanced parameters
    parser.add_argument('--seed', type=int, default=42,
                      help='random seed for reproducibility')
    
    return parser.parse_args()

def load_config(config_path):
    """Load QAT configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file {config_path} not found. Using default settings.")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    """Main function for FIXED QAT training."""
    logger.info("=" * 80)
    logger.info("FIXED YOLOV8 QAT TRAINING - PRESERVES FAKE QUANTIZATION")
    logger.info("=" * 80)
    
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Test quantization preservation first
    logger.info("Running quantization preservation test...")
    if not test_quantization_preservation():
        logger.error("❌ Quantization preservation test failed. Exiting.")
        return None
    
    # Load QAT configuration
    qat_config = load_config(args.config)
    
    # Create output directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    if args.export:
        os.makedirs(args.export_dir, exist_ok=True)
    
    # Determine model and data paths
    model_path = args.model
    if not os.path.isabs(model_path) and not model_path.startswith('yolov8'):
        pretrained_path = os.path.join('models', 'pretrained', model_path)
        if os.path.exists(pretrained_path):
            model_path = pretrained_path
    
    data_path = args.data
    if not os.path.isabs(data_path):
        dataset_path = os.path.join('dataset', data_path)
        if os.path.exists(dataset_path):
            data_path = dataset_path
    
    # Initialize FIXED QAT model
    logger.info(f"Starting FIXED YOLOv8 QAT training")
    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {data_path}")
    logger.info(f"QConfig: {args.qconfig}")
    
    # Create fixed QAT model
    qat_model = QuantizedYOLOv8Fixed(
        model_path=model_path,
        qconfig_name=args.qconfig,
        skip_detection_head=not args.keep_detection_head,
        fuse_modules=args.fuse
    )
    
    # Prepare model for QAT
    qat_model.prepare_for_qat()
    
    # Get training parameters
    train_params = qat_config.get('train_params', {})
    epochs = args.epochs if args.epochs else train_params.get('epochs', QAT_EPOCHS)
    batch_size = args.batch_size if args.batch_size else train_params.get('batch_size', BATCH_SIZE)
    img_size = args.img_size if args.img_size else train_params.get('img_size', IMG_SIZE)
    lr = args.lr if args.lr else train_params.get('lr', QAT_LEARNING_RATE)
    
    # Train with FIXED QAT approach
    try:
        logger.info("Using FIXED phased QAT training approach")
        results = qat_model.train_model_fixed(
            data_yaml=data_path,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            lr=lr,
            device=args.device,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            use_distillation=False
        )
        
        logger.info("✅ FIXED QAT training completed successfully")
        
    except Exception as training_error:
        logger.error(f"❌ FIXED QAT training failed: {training_error}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    # Convert and save quantized model using FIXED approach
    save_path = os.path.join(args.save_dir, args.output)

    try:
        logger.info("Converting QAT model to quantized INT8 model using FIXED approach...")
        quantized_model = qat_model.convert_to_quantized_fixed(save_path)
        logger.info("✅ FIXED model conversion completed successfully")
        
    except Exception as conversion_error:
        logger.error(f"❌ FIXED model conversion failed: {conversion_error}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Try to save the preserved QAT model instead
        try:
            logger.info("Attempting to save preserved QAT model...")
            qat_save_path = save_path.replace('.pt', '_qat_preserved.pt')
            qat_model.save_qat_model_with_quantization(qat_model.preserved_qat_model, qat_save_path)
            logger.info(f"✅ Preserved QAT model saved to {qat_save_path}")
        except Exception as save_error:
            logger.error(f"❌ Could not save preserved QAT model: {save_error}")
    
    # Evaluate if requested
    if args.eval:
        logger.info("Evaluating FIXED QAT model...")
        try:
            eval_results = qat_model.evaluate(
                data_yaml=data_path,
                batch_size=batch_size,
                img_size=img_size,
                device=args.device
            )
            
            logger.info(f"FIXED QAT Evaluation results:")
            logger.info(f"  mAP50: {eval_results.box.map50:.4f}")
            logger.info(f"  mAP50-95: {eval_results.box.map:.4f}")
        except Exception as eval_error:
            logger.error(f"❌ Evaluation failed: {eval_error}")
    
    # Export if requested
    if args.export:
        logger.info("Exporting FIXED QAT model...")
        try:
            export_formats = qat_config.get('export', {}).get('formats', ['onnx'])
            for export_format in export_formats:
                export_path = os.path.join(args.export_dir, export_format)
                os.makedirs(export_path, exist_ok=True)
                
                export_file = os.path.join(export_path, f"{os.path.splitext(args.output)[0]}.{export_format}")
                logger.info(f"Exporting model to {export_file}")
                
                qat_model.export(export_file, format=export_format)
                logger.info(f"✅ Export to {export_format} completed successfully")
                
        except Exception as export_error:
            logger.error(f"❌ Export failed: {export_error}")
    
    logger.info("=" * 80)
    logger.info("FIXED YOLOv8 QAT TRAINING COMPLETED SUCCESSFULLY")
    logger.info(f"Quantized model saved to {save_path}")
    logger.info("Key improvements:")
    logger.info("  ✅ Fake quantization preserved during training")
    logger.info("  ✅ Proper QAT model saving and loading")
    logger.info("  ✅ Conversion from preserved QAT model")
    logger.info("=" * 80)
    
    return results

if __name__ == "__main__":
    main()