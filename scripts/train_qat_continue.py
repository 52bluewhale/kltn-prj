#!/usr/bin/env python
"""
Continue QAT from a saved model - handles both QAT and standard YOLO models
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
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('continue_qat')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Continue QAT from saved model")
    
    parser.add_argument('--qat-model', type=str, required=True,
                      help='Path to saved model (QAT or standard YOLO)')
    parser.add_argument('--data', type=str, 
                      default='datasets/vietnam-traffic-sign-detection/dataset.yaml',
                      help='Dataset YAML path')
    parser.add_argument('--output-dir', type=str, default='models/final',
                      help='Output directory for final models')
    parser.add_argument('--convert', action='store_true', default=True,
                      help='Convert to INT8 quantized model')
    parser.add_argument('--evaluate', action='store_true', default=True,
                      help='Evaluate the model')
    parser.add_argument('--export', action='store_true', default=True,
                      help='Export to ONNX')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device for evaluation')
    parser.add_argument('--force-reprepare', action='store_true',
                      help='Force re-preparation of QAT even if model seems to have quantization')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*80)
    logger.info("CONTINUING QAT FROM SAVED MODEL")
    logger.info("="*80)
    logger.info(f"Model: {args.qat_model}")
    logger.info(f"Output Dir: {args.output_dir}")
    
    try:
        # Import the QAT model class
        from src.models.yolov8_qat import QuantizedYOLOv8
        
        # Try to load as QAT model first
        logger.info("Attempting to load as QAT model...")
        qat_model = None
        
        try:
            qat_model = QuantizedYOLOv8.load_qat_model(args.qat_model, device=args.device)
            
            # Check if quantization is actually preserved
            if qat_model.verify_quantization_preserved() and not args.force_reprepare:
                logger.info("✅ QAT model loaded successfully with quantization preserved")
            else:
                logger.info("⚠️ QAT model loaded but quantization not preserved, will re-prepare...")
                qat_model = None  # Force re-preparation
                
        except Exception as qat_load_error:
            logger.info(f"Could not load as QAT model: {qat_load_error}")
            qat_model = None
        
        # If QAT loading failed, create new QAT model from standard YOLO model
        if qat_model is None:
            logger.info("Loading as standard YOLO model and re-preparing for QAT...")
            
            # Create new QAT model from the saved weights
            qat_model = QuantizedYOLOv8(
                model_path=args.qat_model,  # Use the trained model as base
                qconfig_name='default',
                skip_detection_head=True,
                fuse_modules=True
            )
            
            # Prepare for QAT
            logger.info("Re-preparing model for QAT...")
            qat_model.prepare_for_qat()
            
            # Verify quantization is now present
            if qat_model.verify_quantization_preserved():
                logger.info("✅ Model successfully re-prepared for QAT")
            else:
                logger.error("❌ Failed to re-prepare model for QAT")
                return False
        
        # Save the QAT model for future use
        qat_save_path = os.path.join(args.output_dir, "qat_model_preserved.pt")
        logger.info(f"Saving QAT model with quantization to {qat_save_path}")
        save_success = qat_model.save(qat_save_path, preserve_qat=True)
        if save_success:
            logger.info("✅ QAT model saved successfully")
        else:
            logger.warning("⚠️ QAT model save may have issues")
        
        # Convert to INT8 if requested
        if args.convert:
            logger.info("Converting QAT model to INT8...")
            int8_path = os.path.join(args.output_dir, "quantized_model.pt")
            
            try:
                quantized_model = qat_model.convert_to_quantized(int8_path)
                if quantized_model is not None:
                    logger.info(f"✅ INT8 model saved to {int8_path}")
                    
                    # Show compression stats
                    from src.quantization.utils import compare_model_sizes
                    size_info = compare_model_sizes(qat_model.original_model, quantized_model)
                    logger.info(f"📊 Compression Results:")
                    logger.info(f"  Original: {size_info['fp32_size_mb']:.2f} MB")
                    logger.info(f"  Quantized: {size_info['int8_size_mb']:.2f} MB")
                    logger.info(f"  Compression: {size_info['compression_ratio']:.2f}x")
                    logger.info(f"  Size reduction: {size_info['size_reduction_percent']:.1f}%")
                else:
                    logger.error("❌ INT8 conversion failed")
                    
            except Exception as conv_error:
                logger.error(f"❌ Conversion error: {conv_error}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Evaluate if requested
        if args.evaluate:
            logger.info("Evaluating model...")
            try:
                eval_results = qat_model.evaluate(
                    data_yaml=args.data,
                    device=args.device,
                    batch_size=16,
                    img_size=640
                )
                logger.info("✅ Evaluation completed:")
                logger.info(f"  📈 mAP@0.5: {eval_results.box.map50:.4f}")
                logger.info(f"  📈 mAP@0.5:0.95: {eval_results.box.map:.4f}")
                
            except Exception as eval_error:
                logger.error(f"❌ Evaluation error: {eval_error}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Export if requested
        if args.export:
            logger.info("Exporting to ONNX...")
            try:
                onnx_path = os.path.join(args.output_dir, "quantized_model.onnx")
                exported_path = qat_model.export(onnx_path, format='onnx', img_size=640)
                logger.info(f"✅ ONNX model exported to {exported_path}")
                
            except Exception as export_error:
                logger.error(f"❌ Export error: {export_error}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Summary
        logger.info("="*80)
        logger.info("📋 SUMMARY OF OUTPUTS:")
        logger.info("="*80)
        
        outputs = []
        if save_success:
            outputs.append(f"🔧 QAT Model: {qat_save_path}")
        if args.convert and os.path.exists(os.path.join(args.output_dir, "quantized_model.pt")):
            outputs.append(f"⚡ INT8 Model: {os.path.join(args.output_dir, 'quantized_model.pt')}")
        if args.export and os.path.exists(os.path.join(args.output_dir, "quantized_model.onnx")):
            outputs.append(f"🚀 ONNX Model: {os.path.join(args.output_dir, 'quantized_model.onnx')}")
        
        for output in outputs:
            logger.info(f"  {output}")
        
        logger.info("="*80)
        logger.info("✅ POST-TRAINING STEPS COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)