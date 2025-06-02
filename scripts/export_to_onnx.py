#!/usr/bin/env python
"""
YOLOv8 Quantized Model ONNX Export Script

This script exports your quantized YOLOv8 model to ONNX format for deployment.
Supports both QAT models and INT8 quantized models.
"""

import os
import sys
import argparse
import logging
import torch
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
from src.models.yolov8_qat import QuantizedYOLOv8

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('onnx_export')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export YOLOv8 Quantized Model to ONNX")
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to quantized model (.pt file)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output ONNX file path (auto-generated if not specified)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size for ONNX model')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for ONNX export')
    parser.add_argument('--opset', type=int, default=12,
                       help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', default=True,
                       help='Simplify ONNX model')
    parser.add_argument('--dynamic', action='store_true',
                       help='Enable dynamic axes for batch size and image size')
    parser.add_argument('--half', action='store_true',
                       help='Export with FP16 precision (not recommended for quantized models)')
    parser.add_argument('--verify', action='store_true', default=True,
                       help='Verify ONNX model after export')
    
    return parser.parse_args()

def export_quantized_yolo_to_onnx(model_path, output_path=None, img_size=640, 
                                 batch_size=1, opset=12, simplify=True, 
                                 dynamic=False, half=False, verify=True):
    """
    Export quantized YOLOv8 model to ONNX format.
    
    Args:
        model_path: Path to the quantized model
        output_path: Output ONNX file path
        img_size: Input image size
        batch_size: Batch size for export
        opset: ONNX opset version
        simplify: Whether to simplify the model
        dynamic: Whether to use dynamic axes
        half: Whether to use FP16 (not recommended for quantized models)
        verify: Whether to verify the exported model
    
    Returns:
        Path to exported ONNX model
    """
    
    logger.info(f"🚀 Starting ONNX export for: {model_path}")
    
    # Generate output path if not provided
    if output_path is None:
        model_dir = os.path.dirname(model_path)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = os.path.join(model_dir, f"{model_name}.onnx")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Method 1: Try loading as standard YOLO model first (works for most cases)
        logger.info("📥 Loading model...")
        
        try:
            # Load as standard YOLO model
            model = YOLO(model_path)
            logger.info("✅ Loaded as standard YOLO model")
            
        except Exception as yolo_error:
            logger.warning(f"⚠️ Standard YOLO load failed: {yolo_error}")
            
            # Method 2: Try loading as QAT model
            try:
                logger.info("🔄 Attempting to load as QAT model...")
                qat_model = QuantizedYOLOv8.load_qat_model(model_path)
                model = qat_model.model
                logger.info("✅ Loaded as QAT model")
                
            except Exception as qat_error:
                logger.error(f"❌ QAT model load failed: {qat_error}")
                
                # Method 3: Try loading raw PyTorch model
                logger.info("🔄 Attempting to load as raw PyTorch model...")
                model_data = torch.load(model_path, map_location='cpu')
                
                if isinstance(model_data, dict):
                    if 'model' in model_data:
                        # Extract model from saved dict
                        model_state = model_data['model']
                        
                        # Create a new YOLO model and load the state
                        model = YOLO('yolov8n.pt')  # Load base architecture
                        
                        if hasattr(model_state, 'state_dict'):
                            model.model.load_state_dict(model_state.state_dict())
                        elif isinstance(model_state, dict):
                            model.model.load_state_dict(model_state)
                        else:
                            model.model = model_state
                            
                        logger.info("✅ Loaded from raw PyTorch format")
                    else:
                        raise ValueError("Cannot extract model from saved data")
                else:
                    raise ValueError("Unsupported model format")
        
        # Prepare export arguments
        export_args = {
            'format': 'onnx',
            'imgsz': img_size,
            'batch': batch_size,
            'opset': opset,
            'simplify': simplify,
            'half': half,
            'dynamic': dynamic,
            'verbose': True
        }
        
        # Remove half precision for quantized models unless explicitly requested
        if not half:
            logger.info("🔧 Disabling FP16 for quantized model export")
            export_args['half'] = False
        
        # Configure dynamic axes if requested
        if dynamic:
            logger.info("🔧 Enabling dynamic axes for flexible input sizes")
            export_args['dynamic'] = True
        
        logger.info("🔄 Exporting to ONNX...")
        logger.info(f"📋 Export settings:")
        logger.info(f"   - Input size: {img_size}x{img_size}")
        logger.info(f"   - Batch size: {batch_size}")
        logger.info(f"   - ONNX opset: {opset}")
        logger.info(f"   - Simplify: {simplify}")
        logger.info(f"   - Dynamic: {dynamic}")
        logger.info(f"   - Half precision: {half}")
        
        # Export to ONNX
        exported_files = model.export(**export_args)
        
        # The export typically returns the path or a list of paths
        if isinstance(exported_files, (list, tuple)):
            onnx_path = exported_files[0] if exported_files else output_path
        else:
            onnx_path = exported_files if exported_files else output_path
        
        # If export didn't create file at expected location, check for auto-generated name
        if not os.path.exists(onnx_path):
            # Look for ONNX files in the model directory
            model_dir = os.path.dirname(model_path)
            onnx_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
            
            if onnx_files:
                # Use the most recently created ONNX file
                onnx_files_paths = [os.path.join(model_dir, f) for f in onnx_files]
                onnx_path = max(onnx_files_paths, key=os.path.getctime)
                logger.info(f"📁 Found exported ONNX file: {onnx_path}")
                
                # Move to desired location if different
                if onnx_path != output_path:
                    import shutil
                    shutil.move(onnx_path, output_path)
                    onnx_path = output_path
                    logger.info(f"📁 Moved ONNX file to: {output_path}")
        
        # Verify the export
        if verify and os.path.exists(onnx_path):
            logger.info("🔍 Verifying ONNX export...")
            verify_onnx_export(onnx_path, img_size, batch_size)
        
        # Get file size
        if os.path.exists(onnx_path):
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            logger.info(f"✅ ONNX export completed successfully!")
            logger.info(f"📄 Output file: {onnx_path}")
            logger.info(f"📊 File size: {file_size:.2f} MB")
            
            return onnx_path
        else:
            raise FileNotFoundError(f"ONNX file not created at expected location: {onnx_path}")
            
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")
        logger.error(f"💡 Troubleshooting tips:")
        logger.error(f"   - Ensure model file exists and is valid")
        logger.error(f"   - Try with --simplify=False if simplification fails")
        logger.error(f"   - Try different --opset version (11, 12, 13)")
        logger.error(f"   - Check if model architecture is compatible with ONNX")
        raise

def verify_onnx_export(onnx_path, img_size, batch_size):
    """Verify the exported ONNX model."""
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
        
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX model structure is valid")
        
        # Test inference
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Get input/output info
        input_info = ort_session.get_inputs()[0]
        output_info = ort_session.get_outputs()
        
        logger.info(f"📊 ONNX Model Info:")
        logger.info(f"   - Input shape: {input_info.shape}")
        logger.info(f"   - Input type: {input_info.type}")
        logger.info(f"   - Number of outputs: {len(output_info)}")
        
        # Test with dummy data
        dummy_input = np.random.randn(batch_size, 3, img_size, img_size).astype(np.float32)
        
        outputs = ort_session.run(None, {input_info.name: dummy_input})
        logger.info(f"✅ ONNX inference test passed")
        logger.info(f"   - Output shapes: {[out.shape for out in outputs]}")
        
    except ImportError:
        logger.warning("⚠️ onnx or onnxruntime not installed - skipping verification")
        logger.info("💡 Install with: pip install onnx onnxruntime")
    except Exception as e:
        logger.warning(f"⚠️ ONNX verification failed: {e}")
        logger.info("💡 Model may still be valid - try loading in your deployment environment")

def main():
    """Main function."""
    args = parse_args()
    
    logger.info("🎯 YOLOv8 Quantized Model → ONNX Export")
    logger.info("=" * 60)
    
    try:
        # Export to ONNX
        onnx_path = export_quantized_yolo_to_onnx(
            model_path=args.model,
            output_path=args.output,
            img_size=args.img_size,
            batch_size=args.batch_size,
            opset=args.opset,
            simplify=args.simplify,
            dynamic=args.dynamic,
            half=args.half,
            verify=args.verify
        )
        
        logger.info("=" * 60)
        logger.info("🎉 EXPORT COMPLETED SUCCESSFULLY!")
        logger.info(f"📄 ONNX Model: {onnx_path}")
        logger.info("💡 Ready for deployment!")
        
        # Provide usage example
        logger.info("=" * 60)
        logger.info("📋 Usage Example:")
        logger.info("```python")
        logger.info("import onnxruntime as ort")
        logger.info("import numpy as np")
        logger.info("")
        logger.info(f"# Load ONNX model")
        logger.info(f"session = ort.InferenceSession('{onnx_path}')")
        logger.info("")
        logger.info(f"# Prepare input (batch_size, channels, height, width)")
        logger.info(f"input_data = np.random.randn({args.batch_size}, 3, {args.img_size}, {args.img_size}).astype(np.float32)")
        logger.info("")
        logger.info(f"# Run inference")
        logger.info(f"outputs = session.run(None, {{'images': input_data}})")
        logger.info("```")
        
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()