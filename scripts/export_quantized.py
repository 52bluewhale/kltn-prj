#!/usr/bin/env python
"""
QAT Model Conversion Script

This script converts your existing trained QAT model to properly saved formats
without needing to re-run the expensive training process.

Usage:
    python scripts/export_quantized.py --input models/checkpoints/qat/phase4_fine_tuning/weights/best.pt --output models/checkpoints/qat/quantized_model.pt
"""

import os
import sys
import logging
import argparse
import torch
import yaml
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('convert_qat')

# Import project modules
from src.models.yolov8_qat import QuantizedYOLOv8
from src.quantization.utils import get_model_size, compare_model_sizes, analyze_quantization_effects

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert QAT trained model")
    
    parser.add_argument('--input', type=str, 
                       default='models/checkpoints/qat/phase4_fine_tuning/weights/best.pt',
                       help='Path to trained QAT model weights')
    parser.add_argument('--output', type=str, 
                       default='models/checkpoints/qat/quantized_model.pt',
                       help='Output path for converted model')
    parser.add_argument('--qat-output', type=str,
                       default='models/checkpoints/qat/qat_model_fixed.pt',
                       help='Output path for QAT model with fixed save')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Original model path/name')
    parser.add_argument('--data', type=str, 
                       default='datasets/vietnam-traffic-sign-detection/dataset.yaml',
                       help='Dataset YAML path')
    parser.add_argument('--qconfig', type=str, default='default',
                       help='QConfig name used during training')
    
    # Conversion options
    parser.add_argument('--skip-detection-head', action='store_true', default=True,
                       help='Skip quantization of detection head (recommended)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for conversion')
    parser.add_argument('--export-onnx', action='store_true',
                       help='Export to ONNX after conversion')
    parser.add_argument('--export-dir', type=str, default='models/exported/onnx',
                       help='Directory for ONNX export')
    
    # Analysis options
    parser.add_argument('--analyze', action='store_true', default=True,
                       help='Analyze quantization effects')
    parser.add_argument('--eval', action='store_true',
                       help='Evaluate converted model')
    
    return parser.parse_args()

def load_trained_weights(weight_path, device='cpu'):
    """
    Load trained weights from the YOLOv8 checkpoint.
    
    Args:
        weight_path: Path to the trained weights (.pt file)
        device: Device to load on
        
    Returns:
        Model state dict or None if failed
    """
    logger.info(f"Loading trained weights from {weight_path}")
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(weight_path, map_location=device)
        
        # Extract model state dict
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                # Standard YOLOv8 checkpoint format
                model_state = checkpoint['model']
                if hasattr(model_state, 'state_dict'):
                    state_dict = model_state.state_dict()
                else:
                    state_dict = model_state
                    
                logger.info("✅ Loaded model state dict from YOLOv8 checkpoint")
                return state_dict
            else:
                logger.error("❌ No 'model' key found in checkpoint")
                return None
        else:
            # Direct state dict
            logger.info("✅ Loaded direct state dict")
            return checkpoint
            
    except Exception as e:
        logger.error(f"❌ Failed to load trained weights: {e}")
        return None

def create_qat_model_from_weights(model_path, weight_state_dict, qconfig_name='default', 
                                skip_detection_head=True, device='cpu'):
    """
    Create a QAT model and load the trained weights.
    
    Args:
        model_path: Original model path
        weight_state_dict: Trained weights state dict
        qconfig_name: QConfig name
        skip_detection_head: Whether to skip detection head quantization
        device: Device to use
        
    Returns:
        QuantizedYOLOv8 instance or None if failed
    """
    logger.info("Creating QAT model from trained weights...")
    
    try:
        # Create QAT model instance
        qat_model = QuantizedYOLOv8(
            model_path=model_path,
            qconfig_name=qconfig_name,
            skip_detection_head=skip_detection_head,
            fuse_modules=True
        )
        
        # Prepare for QAT (this creates the quantization structure)
        qat_model.prepare_for_qat()
        
        # Load the trained weights
        logger.info("Loading trained weights into QAT model...")
        
        # Handle potential key mismatches
        try:
            qat_model.model.model.load_state_dict(weight_state_dict, strict=True)
            logger.info("✅ Loaded weights with strict=True")
        except Exception as strict_error:
            logger.warning(f"Strict loading failed: {strict_error}")
            logger.info("Trying with strict=False...")
            
            missing_keys, unexpected_keys = qat_model.model.model.load_state_dict(
                weight_state_dict, strict=False
            )
            
            if missing_keys:
                logger.warning(f"Missing keys: {len(missing_keys)} keys")
                if len(missing_keys) <= 10:  # Show first few
                    for key in missing_keys[:10]:
                        logger.warning(f"  - {key}")
            
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 10:  # Show first few
                    for key in unexpected_keys[:10]:
                        logger.warning(f"  - {key}")
            
            logger.info("✅ Loaded weights with strict=False")
        
        # Move to device
        qat_model.model.model.to(device)
        
        # Verify quantization structure
        fake_quant_count = sum(1 for n, m in qat_model.model.model.named_modules() 
                             if 'FakeQuantize' in type(m).__name__)
        
        qconfig_count = sum(1 for n, m in qat_model.model.model.named_modules() 
                          if hasattr(m, 'qconfig') and m.qconfig is not None)
        
        logger.info(f"QAT model created successfully:")
        logger.info(f"  - FakeQuantize modules: {fake_quant_count}")
        logger.info(f"  - Modules with qconfig: {qconfig_count}")
        
        if fake_quant_count > 0:
            logger.info("✅ QAT structure verified")
            return qat_model
        else:
            logger.error("❌ No quantization structure found")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to create QAT model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def save_qat_model(qat_model, output_path):
    """
    Save QAT model using the fixed save method.
    
    Args:
        qat_model: QuantizedYOLOv8 instance
        output_path: Path to save QAT model
        
    Returns:
        Success status
    """
    logger.info(f"Saving QAT model to {output_path}")
    
    try:
        # Create directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Count FakeQuantize modules
        fake_quant_count = sum(1 for n, m in qat_model.model.model.named_modules() 
                             if 'FakeQuantize' in type(m).__name__)
        
        # Save using fixed method (state dict only)
        save_dict = {
            'state_dict': qat_model.model.model.state_dict(),
            'qat_info': {
                'qconfig_name': qat_model.qconfig_name,
                'skip_detection_head': qat_model.skip_detection_head,
                'fuse_modules': qat_model.fuse_modules,
                'is_prepared': qat_model.is_prepared,
                'pytorch_version': torch.__version__,
                'model_path': qat_model.model_path,
                'num_classes': 58,
                'img_size': 640,
                'dataset': 'vietnamese_traffic_signs'
            },
            'fake_quant_count': fake_quant_count,
            'quantization_preserved': True,
            'save_method': 'state_dict_only',
            'conversion_ready': True
        }
        
        # Save
        torch.save(save_dict, output_path)
        
        # Verify
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"✅ QAT model saved successfully: {file_size:.2f} MB")
            logger.info(f"✅ Preserved {fake_quant_count} FakeQuantize modules")
            return True
        else:
            logger.error("❌ Save file not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to save QAT model: {e}")
        return False

def convert_to_quantized(qat_model, output_path):
    """
    Convert QAT model to quantized INT8 model.
    
    Args:
        qat_model: QuantizedYOLOv8 instance
        output_path: Path to save quantized model
        
    Returns:
        Quantized model or None if failed
    """
    logger.info(f"Converting QAT model to quantized INT8...")
    
    try:
        # Set model to eval mode for conversion
        qat_model.model.model.eval()
        
        # Convert using PyTorch's convert function
        quantized_model = torch.quantization.convert(qat_model.model.model, inplace=False)
        
        # Calculate model sizes
        original_size = get_model_size(qat_model.original_model)
        quantized_size = get_model_size(quantized_model)
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
        size_reduction = (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
        
        logger.info(f"Model size comparison:")
        logger.info(f"  Original FP32 model: {original_size:.2f} MB")
        logger.info(f"  Quantized INT8 model: {quantized_size:.2f} MB")
        logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
        logger.info(f"  Size reduction: {size_reduction:.2f}%")
        
        # Save quantized model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        save_dict = {
            'model': quantized_model,
            'model_info': {
                'architecture': 'yolov8n',
                'num_classes': 58,
                'quantized': True,
                'format': 'int8',
                'compression_ratio': compression_ratio,
                'size_reduction_percent': size_reduction
            },
            'save_method': 'quantized_converted',
            'deployment_ready': True
        }
        
        torch.save(save_dict, output_path)
        
        logger.info(f"✅ Quantized model saved to {output_path}")
        return quantized_model
        
    except Exception as e:
        logger.error(f"❌ Quantization conversion failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def export_to_onnx(model, export_path, img_size=640):
    """
    Export model to ONNX format.
    
    Args:
        model: Model to export (can be YOLO object or torch model)
        export_path: Path to save ONNX model
        img_size: Input image size
        
    Returns:
        Success status
    """
    logger.info(f"Exporting model to ONNX: {export_path}")
    
    try:
        # Create directory
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        # Handle different model types
        if hasattr(model, 'export'):
            # YOLO object with export method
            exported_path = model.export(
                format='onnx',
                imgsz=img_size,
                simplify=True,
                opset=12
            )
            logger.info(f"✅ ONNX export successful: {exported_path}")
            return True
        else:
            # Direct PyTorch model export
            model.eval()
            dummy_input = torch.randn(1, 3, img_size, img_size)
            
            torch.onnx.export(
                model,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
            
            logger.info(f"✅ ONNX export successful: {export_path}")
            return True
            
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")
        return False

def evaluate_model(model, data_yaml, device='cpu'):
    """
    Evaluate model performance.
    
    Args:
        model: Model to evaluate
        data_yaml: Dataset YAML path
        device: Device for evaluation
        
    Returns:
        Evaluation results or None if failed
    """
    logger.info("Evaluating model performance...")
    
    try:
        if hasattr(model, 'val'):
            # YOLO object with val method
            results = model.val(
                data=data_yaml,
                device=device,
                verbose=True
            )
            
            # Extract metrics
            if hasattr(results, 'box'):
                map50 = results.box.map50
                map50_95 = results.box.map
                
                logger.info(f"Evaluation results:")
                logger.info(f"  mAP50: {map50:.4f}")
                logger.info(f"  mAP50-95: {map50_95:.4f}")
                
                return results
            else:
                logger.info("✅ Evaluation completed (detailed metrics not available)")
                return results
        else:
            logger.warning("⚠️ Model doesn't have evaluation method")
            return None
            
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return None

def main():
    """Main conversion function."""
    args = parse_args()
    
    logger.info("="*80)
    logger.info("QAT MODEL CONVERSION SCRIPT")
    logger.info("="*80)
    logger.info(f"Input weights: {args.input}")
    logger.info(f"QAT output: {args.qat_output}")
    logger.info(f"Quantized output: {args.output}")
    logger.info(f"Device: {args.device}")
    
    # Step 1: Load trained weights
    logger.info("\n" + "="*50)
    logger.info("STEP 1: Loading trained weights")
    logger.info("="*50)
    
    if not os.path.exists(args.input):
        logger.error(f"❌ Input weights file not found: {args.input}")
        return
    
    weight_state_dict = load_trained_weights(args.input, args.device)
    if weight_state_dict is None:
        logger.error("❌ Failed to load trained weights")
        return
    
    # Step 2: Create QAT model from weights
    logger.info("\n" + "="*50)
    logger.info("STEP 2: Creating QAT model from weights")
    logger.info("="*50)
    
    qat_model = create_qat_model_from_weights(
        model_path=args.model,
        weight_state_dict=weight_state_dict,
        qconfig_name=args.qconfig,
        skip_detection_head=args.skip_detection_head,
        device=args.device
    )
    
    if qat_model is None:
        logger.error("❌ Failed to create QAT model")
        return
    
    # Step 3: Save QAT model with fixed method
    logger.info("\n" + "="*50)
    logger.info("STEP 3: Saving QAT model with fixed method")
    logger.info("="*50)
    
    qat_save_success = save_qat_model(qat_model, args.qat_output)
    if not qat_save_success:
        logger.error("❌ Failed to save QAT model")
        return
    
    # Step 4: Convert to quantized INT8 model
    logger.info("\n" + "="*50)
    logger.info("STEP 4: Converting to quantized INT8 model")
    logger.info("="*50)
    
    quantized_model = convert_to_quantized(qat_model, args.output)
    if quantized_model is None:
        logger.error("❌ Failed to convert to quantized model")
        return
    
    # Step 5: Analysis (if requested)
    if args.analyze:
        logger.info("\n" + "="*50)
        logger.info("STEP 5: Analyzing quantization effects")
        logger.info("="*50)
        
        analysis_results = analyze_quantization_effects(quantized_model)
        logger.info(f"Quantization analysis:")
        logger.info(f"  Total modules: {analysis_results['total_modules']}")
        logger.info(f"  Quantized modules: {analysis_results['quantized_modules']}")
        logger.info(f"  Quantization ratio: {analysis_results['quantized_ratio']:.2f}")
        
        # Save analysis
        analysis_path = os.path.join(os.path.dirname(args.output), "conversion_analysis.yaml")
        with open(analysis_path, 'w') as f:
            yaml.dump(analysis_results, f)
        logger.info(f"Analysis saved to {analysis_path}")
    
    # Step 6: Export to ONNX (if requested)
    if args.export_onnx:
        logger.info("\n" + "="*50)
        logger.info("STEP 6: Exporting to ONNX")
        logger.info("="*50)
        
        onnx_path = os.path.join(args.export_dir, "quantized_model.onnx")
        export_success = export_to_onnx(qat_model.model, onnx_path)
        
        if export_success:
            logger.info(f"✅ ONNX export successful: {onnx_path}")
        else:
            logger.error("❌ ONNX export failed")
    
    # Step 7: Evaluation (if requested)
    if args.eval:
        logger.info("\n" + "="*50)
        logger.info("STEP 7: Evaluating converted model")
        logger.info("="*50)
        
        eval_results = evaluate_model(qat_model.model, args.data, args.device)
        if eval_results:
            logger.info("✅ Evaluation completed")
        else:
            logger.warning("⚠️ Evaluation not available")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("CONVERSION COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"✅ QAT model saved: {args.qat_output}")
    logger.info(f"✅ Quantized model saved: {args.output}")
    if args.export_onnx:
        logger.info(f"✅ ONNX model saved: {os.path.join(args.export_dir, 'quantized_model.onnx')}")
    logger.info("="*80)

if __name__ == "__main__":
    main()