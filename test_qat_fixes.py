#!/usr/bin/env python
"""
Test script to verify QAT fixes are working
"""
import sys
import os
import torch
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def test_basic_quantization():
    """Test basic quantization functionality."""
    print("\n" + "="*50)
    print("TESTING BASIC QUANTIZATION")
    print("="*50)
    
    try:
        # Test basic quantization
        model = torch.nn.Conv2d(3, 16, 3)
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        prepared = torch.quantization.prepare_qat(model)
        
        print("✓ Basic quantization test PASSED")
        return True
    except Exception as e:
        print(f"❌ Basic quantization test FAILED: {e}")
        return False

def test_model_loading():
    """Test YOLOv8 model loading."""
    print("\n" + "="*50)
    print("TESTING YOLOV8 MODEL LOADING")
    print("="*50)
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✓ YOLOv8 model loading test PASSED")
        return True
    except Exception as e:
        print(f"❌ YOLOv8 model loading test FAILED: {e}")
        return False

def test_qat_model_creation():
    """Test QAT model creation."""
    print("\n" + "="*50)
    print("TESTING QAT MODEL CREATION")
    print("="*50)
    
    try:
        from src.models.yolov8_qat import QuantizedYOLOv8
        
        qat_model = QuantizedYOLOv8(
            model_path="yolov8n.pt",
            qconfig_name='default',
            skip_detection_head=True,
            fuse_modules=True
        )
        
        print("✓ QAT model creation test PASSED")
        return True
    except Exception as e:
        print(f"❌ QAT model creation test FAILED: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_qat_preparation():
    """Test QAT model preparation."""
    print("\n" + "="*50)
    print("TESTING QAT MODEL PREPARATION")
    print("="*50)
    
    try:
        from src.models.yolov8_qat import QuantizedYOLOv8
        
        qat_model = QuantizedYOLOv8(
            model_path="yolov8n.pt",
            qconfig_name='default',
            skip_detection_head=True,
            fuse_modules=True
        )
        
        # Test preparation
        qat_model.prepare_for_qat()
        
        print("✓ QAT model preparation test PASSED")
        return True
    except Exception as e:
        print(f"❌ QAT model preparation test FAILED: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_model_saving():
    """Test model saving without pickling issues."""
    print("\n" + "="*50)
    print("TESTING MODEL SAVING")
    print("="*50)
    
    try:
        from src.models.yolov8_qat import QuantizedYOLOv8
        import tempfile
        
        qat_model = QuantizedYOLOv8(
            model_path="yolov8n.pt",
            qconfig_name='default',
            skip_detection_head=True,
            fuse_modules=True
        )
        
        # Prepare model
        qat_model.prepare_for_qat()
        
        # Test saving
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            temp_path = tmp.name
        
        # This should not cause a pickling error anymore
        qat_model._save_model_with_quantization_info(qat_model.model.model, temp_path)
        
        # Clean up
        try:
            os.unlink(temp_path)
        except:
            pass
        
        print("✓ Model saving test PASSED")
        return True
    except Exception as e:
        print(f"❌ Model saving test FAILED: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Run all tests."""
    print("RUNNING QAT FIXES VERIFICATION TESTS")
    print("="*60)
    
    tests = [
        test_basic_quantization,
        test_model_loading,
        test_qat_model_creation,
        test_qat_preparation,
        test_model_saving
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your QAT fixes are working correctly.")
        print("You can now run the full QAT training script.")
        return True
    else:
        print(f"💥 {total - passed} tests failed. Please fix the issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)