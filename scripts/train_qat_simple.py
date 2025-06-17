#!/usr/bin/env python
"""
Simplified QAT Training Script
Based on working train_qat_standard.py approach
"""
import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train_qat_simple')

# Import our simplified class
from src.models.yolov8_qat_simple import SimpleQuantizedYOLOv8

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple YOLOv8 QAT Training")
    
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                      help='Model path or name')
    parser.add_argument('--data', type=str, 
                      default='datasets/vietnam-traffic-sign-detection/dataset.yaml',
                      help='Dataset YAML path')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                      help='Image size')
    parser.add_argument('--lr', type=float, default=0.0005,
                      help='Learning rate')
    parser.add_argument('--device', type=str, default='0',
                      help='Device (cuda device or cpu)')
    parser.add_argument('--save-dir', type=str, 
                      default='models/checkpoints/qat_simple',
                      help='Save directory')
    parser.add_argument('--skip-detection-head', action='store_true', default=True,
                      help='Skip quantizing detection head')
    parser.add_argument('--convert-int8', action='store_true', default=True,
                      help='Convert to INT8 after training')
    parser.add_argument('--export-onnx', action='store_true', default=True,
                      help='Export to ONNX format')
    
    # Penalty loss arguments
    parser.add_argument('--penalty-loss', action='store_true', default=False,
                      help='Enable quantization penalty loss')
    parser.add_argument('--penalty-alpha', type=float, default=0.01,
                      help='Penalty loss weight')
    parser.add_argument('--penalty-warmup', type=int, default=5,
                      help='Penalty loss warmup epochs')

    # Advanced observer arguments
    parser.add_argument('--observer-type', type=str, default='minmax',
                      choices=['minmax', 'histogram', 'moving_average'],
                      help='Observer type for quantization')
    parser.add_argument('--per-channel-weights', action='store_true', default=True,
                      help='Use per-channel quantization for weights')
    parser.add_argument('--calibrate', action='store_true', default=False,
                      help='Run observer calibration before training')
    parser.add_argument('--calibration-batches', type=int, default=100,
                      help='Number of batches for calibration')

    # Analysis argument
    parser.add_argument('--analyze', action='store_true', default=False,
                      help='Run model analysis after training')

    return parser.parse_args()

def main():
    """Main function following the working approach."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    logger.info("🚀 Starting Simple YOLOv8 QAT Training")
    logger.info(f"📁 Model: {args.model}")
    logger.info(f"📊 Dataset: {args.data}")
    
    # Step 1: Initialize QAT model
    try:
        logger.info("📦 Initializing QAT model...")
        qat_model = SimpleQuantizedYOLOv8(
            model_path=args.model,
            skip_detection_head=args.skip_detection_head
        )
        
        # Step 2: Prepare for QAT
        try:
            logger.info("⚙️ Preparing model for QAT...")
            
            # Choose preparation method based on observer type
            if args.observer_type == 'minmax' and not args.per_channel_weights:
                # Use simple preparation for basic case
                qat_model.prepare_for_qat()
            else:
                # Use advanced preparation for other cases
                qat_model.prepare_for_qat_advanced(
                    observer_type=args.observer_type,
                    per_channel_weights=args.per_channel_weights
                )
            
            logger.info(f"✅ QAT prepared with {args.observer_type} observers")
            
            # Step 2.5: Calibration (if enabled)
            if args.calibrate:
                logger.info("🔧 Running observer calibration...")
                # Note: You would need to create a calibration dataloader here
                # For now, we'll skip this and add it as an exercise
                logger.info("⚠️ Calibration requested but not implemented in this example")
            
            # Step 2.6: Setup penalty loss (if enabled)
            if args.penalty_loss:
                logger.info("🎯 Setting up penalty loss...")
                qat_model.setup_penalty_loss(
                    alpha=args.penalty_alpha,
                    warmup_epochs=args.penalty_warmup
                )
                logger.info(f"✅ Penalty loss enabled (α={args.penalty_alpha})")
            
            logger.info("✅ QAT model initialization completed")
            
        except Exception as e:
            logger.error(f"❌ QAT model initialization failed: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return None

    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return None
    
    # Step 3: Train model
    try:
        logger.info("🏋️ Starting QAT training...")
        
        results = qat_model.train_model(
            data_yaml=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            lr=args.lr,
            device=args.device,
            save_dir=args.save_dir,
            log_dir=os.path.join(args.save_dir, 'logs')
        )
        
        logger.info("✅ QAT training completed")
        
    except Exception as e:
        logger.error(f"❌ QAT training failed: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return None
    
    # Step 4: Save QAT model
    qat_model_path = None
    try:
        qat_save_path = os.path.join(args.save_dir, "qat_model.pt")
        logger.info(f"💾 Saving QAT model to: {qat_save_path}")
        
        success = qat_model.save_qat_model(qat_save_path)
        if success:
            qat_model_path = qat_save_path
            logger.info("✅ QAT model saved successfully")
        else:
            logger.error("❌ Failed to save QAT model")
            
    except Exception as e:
        logger.error(f"❌ QAT model saving failed: {e}")
    
    # Step 5: Convert to INT8 (optional)
    int8_model_path = None
    if args.convert_int8:
        try:
            int8_save_path = os.path.join(args.save_dir, "int8_model.pt")
            logger.info(f"🔄 Converting to INT8...")
            
            quantized_model = qat_model.convert_to_quantized(int8_save_path)
            if quantized_model is not None:
                int8_model_path = int8_save_path
                logger.info("✅ INT8 conversion completed")
            else:
                logger.error("❌ INT8 conversion failed")
                
        except Exception as e:
            logger.error(f"❌ INT8 conversion failed: {e}")
    
    # Step 6: Export to ONNX (optional)
    onnx_model_path = None
    if args.export_onnx:
        try:
            onnx_save_path = os.path.join(args.save_dir, "qat_model.onnx")
            logger.info(f"📤 Exporting to ONNX...")
            
            onnx_model_path = qat_model.export_to_onnx(
                onnx_path=onnx_save_path,
                img_size=args.img_size
            )
            
            if onnx_model_path:
                logger.info("✅ ONNX export completed")
            else:
                logger.error("❌ ONNX export failed")
                
        except Exception as e:
            logger.error(f"❌ ONNX export failed: {e}")
    
    # Step 7: Evaluate model
    try:
        logger.info("📊 Evaluating model...")
        eval_results = qat_model.evaluate(
            data_yaml=args.data,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device
        )
        logger.info(f"📈 Evaluation - mAP50: {eval_results.box.map50:.4f}")
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
    
    # Step 8: Model Analysis
    if args.analyze:
        try:
            logger.info("📊 Running model analysis...")
            
            # Print comprehensive summary
            qat_model.print_model_summary()
            
            # Additional analysis
            analysis = qat_model.analyze_quantization_effects()
            
            logger.info("✅ Model analysis completed")
            
        except Exception as e:
            logger.error(f"❌ Model analysis failed: {e}")
    
    # Print final results
    print_results(qat_model_path, int8_model_path, onnx_model_path, args.save_dir, qat_model if args.analyze else None)
    
    return results

def print_results(qat_path, int8_path, onnx_path, save_dir, qat_model=None):
    """Print final results summary with optional model analysis."""
    print("\n" + "="*80)
    print("🎉 SIMPLE QAT TRAINING COMPLETED!")
    print("="*80)
    
    if qat_path and os.path.exists(qat_path):
        qat_size = os.path.getsize(qat_path) / (1024 * 1024)
        print(f"📁 QAT Model: {qat_path} ({qat_size:.2f} MB)")
    
    if int8_path and os.path.exists(int8_path):
        int8_size = os.path.getsize(int8_path) / (1024 * 1024)
        print(f"🚀 INT8 Model: {int8_path} ({int8_size:.2f} MB)")
    
    if onnx_path and os.path.exists(onnx_path):
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"📤 ONNX Model: {onnx_path} ({onnx_size:.2f} MB)")
    
    print(f"🏆 Training Directory: {save_dir}")
    
    # Add model summary if analysis was run
    if qat_model:
        try:
            qat_model.print_model_summary()
        except Exception as e:
            print(f"⚠️ Could not print model summary: {e}")
    
    print("="*80)

if __name__ == '__main__':
    main()