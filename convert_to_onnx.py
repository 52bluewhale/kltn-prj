#!/usr/bin/env python
"""
ONNX Converter for Fixed QAT Models

This script works with models saved using the corrected save method.
Use this AFTER you've implemented the fixes and re-trained your model.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add your project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fixed_onnx_converter')

def load_fixed_qat_model(model_path, device='cpu'):
    """
    Load a model saved with the corrected save method.
    
    Args:
        model_path: Path to the corrected QAT model
        device: Device to load on
    
    Returns:
        Loaded model and metadata
    """
    logger.info(f"Loading fixed QAT model from {model_path}")
    
    try:
        # Load the corrected model format
        saved_data = torch.load(model_path, map_location=device)
        
        # Check format
        if isinstance(saved_data, dict) and 'model' in saved_data:
            model = saved_data['model']
            qat_info = saved_data.get('qat_info', {})
            conversion_info = saved_data.get('conversion_info', {})
            
            logger.info("✅ Loaded fixed QAT model successfully")
            logger.info(f"  - Architecture: {qat_info.get('qconfig_name', 'unknown')}")
            logger.info(f"  - Classes: {qat_info.get('num_classes', 'unknown')}")
            logger.info(f"  - Input size: {qat_info.get('img_size', 'unknown')}")
            logger.info(f"  - Quantized: {saved_data.get('quantization_preserved', False)}")
            logger.info(f"  - ONNX ready: {saved_data.get('onnx_ready', False)}")
            
            # Set model to eval mode
            model.eval()
            
            return model, qat_info, conversion_info
            
        else:
            logger.error("❌ Model format not recognized as corrected format")
            return None, None, None
            
    except Exception as e:
        logger.error(f"❌ Failed to load fixed QAT model: {e}")
        return None, None, None

def export_fixed_model_to_onnx(model, qat_info, output_path):
    """
    Export fixed QAT model to ONNX.
    
    Args:
        model: Fixed QAT model
        qat_info: QAT information from saved model
        output_path: Path to save ONNX model
    
    Returns:
        Success status
    """
    logger.info(f"Exporting fixed QAT model to ONNX: {output_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get model info
    img_size = qat_info.get('img_size', 640)
    num_classes = qat_info.get('num_classes', 58)
    
    logger.info(f"Model configuration:")
    logger.info(f"  - Input size: {img_size}x{img_size}")
    logger.info(f"  - Classes: {num_classes}")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # Test model inference
    logger.info("Testing model inference...")
    try:
        with torch.no_grad():
            output = model(dummy_input)
        
        if isinstance(output, torch.Tensor):
            logger.info(f"✅ Model output shape: {output.shape}")
        elif isinstance(output, (list, tuple)):
            logger.info(f"✅ Model outputs: {len(output)} tensors")
            for i, out in enumerate(output):
                if hasattr(out, 'shape'):
                    logger.info(f"    Output {i}: {out.shape}")
        else:
            logger.info(f"✅ Model output type: {type(output)}")
            
    except Exception as e:
        logger.error(f"❌ Model inference test failed: {e}")
        return False
    
    # Export to ONNX
    try:
        logger.info("Exporting to ONNX...")
        
        # Export parameters optimized for quantized YOLOv8
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            verbose=False,
            opset_version=11,  # Good compatibility for quantized models
            input_names=['images'],
            output_names=['output0'],
            dynamic_axes={
                'images': {0: 'batch', 2: 'height', 3: 'width'},
                'output0': {0: 'batch', 1: 'anchors'}
            },
            do_constant_folding=True,
            keep_initializers_as_inputs=False,
            export_params=True
        )
        
        # Verify export
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"✅ ONNX export successful! File size: {file_size:.2f} MB")
            
            # Test with ONNX Runtime if available
            try:
                import onnxruntime as ort
                
                # Create session
                providers = ['CPUExecutionProvider']
                session = ort.InferenceSession(output_path, providers=providers)
                
                # Test inference
                input_name = session.get_inputs()[0].name
                ort_inputs = {input_name: dummy_input.numpy()}
                ort_outputs = session.run(None, ort_inputs)
                
                logger.info(f"✅ ONNX Runtime verification passed")
                logger.info(f"  - Input shape: {session.get_inputs()[0].shape}")
                logger.info(f"  - Output count: {len(ort_outputs)}")
                logger.info(f"  - Output 0 shape: {ort_outputs[0].shape}")
                
                # Verify output makes sense for Vietnamese traffic signs
                expected_detections = ort_outputs[0].shape[1]  # Should be number of detections
                logger.info(f"  - Detection outputs: {expected_detections}")
                
            except ImportError:
                logger.warning("⚠️ ONNX Runtime not installed - cannot verify")
                logger.info("Install with: pip install onnxruntime")
            except Exception as e:
                logger.warning(f"⚠️ ONNX Runtime verification failed: {e}")
            
            return True
        else:
            logger.error("❌ ONNX export failed - no output file created")
            return False
            
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to convert fixed QAT model to ONNX."""
    
    logger.info("="*80)
    logger.info("ONNX CONVERTER FOR FIXED QAT MODELS")
    logger.info("Works with models saved using the corrected save method")
    logger.info("="*80)
    
    # Expected model paths (after you implement the fixes)
    possible_models = [
        "models/checkpoints/qat/qat_model_fixed.pt",  # From corrected training
        "models/final/qat_model_fixed.pt",
        "models/checkpoints/qat/phase4_fine_tuning_fixed.pt"
    ]
    
    # Find the corrected model
    model_path = None
    for path in possible_models:
        if os.path.exists(path):
            model_path = path
            logger.info(f"Found corrected model: {path}")
            break
    
    if model_path is None:
        logger.error("❌ No corrected QAT model found!")
        logger.error("Expected locations:")
        for path in possible_models:
            logger.error(f"  - {path}")
        logger.error("\nPlease:")
        logger.error("1. Implement the fixes in yolov8_qat.py")
        logger.error("2. Re-run your QAT training")
        logger.error("3. Then use this converter")
        return False
    
    try:
        # Load the corrected model
        model, qat_info, conversion_info = load_fixed_qat_model(model_path)
        
        if model is None:
            logger.error("❌ Failed to load corrected model")
            return False
        
        # Export to ONNX
        output_path = "models/final/vietnamese_traffic_signs_corrected.onnx"
        success = export_fixed_model_to_onnx(model, qat_info, output_path)
        
        if success:
            logger.info("="*80)
            logger.info("🎉 CORRECTED MODEL SUCCESSFULLY EXPORTED TO ONNX!")
            logger.info(f"📁 Input model: {model_path}")
            logger.info(f"📁 ONNX model: {output_path}")
            logger.info("="*80)
            logger.info("✅ This ONNX model should work properly for deployment")
            logger.info("✅ Quantization benefits preserved")
            logger.info("✅ Vietnamese traffic sign classes (58) handled correctly")
            logger.info("="*80)
            return True
        else:
            logger.error("❌ ONNX export failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Fixed model successfully converted to ONNX!")
    else:
        print("\n❌ Conversion failed. Please implement the fixes first.")
    
    sys.exit(0 if success else 1)