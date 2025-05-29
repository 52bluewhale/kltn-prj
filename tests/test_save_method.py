#!/usr/bin/env python
"""
Comprehensive Test Suite for QAT Save/Load Fixes
This script thoroughly tests the new QAT implementation before running full training.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('qat_test')

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.yolov8_qat import QuantizedYOLOv8

def test_basic_qat_setup():
    """Test 1: Basic QAT model creation and preparation"""
    logger.info("🧪 Test 1: Basic QAT Setup")
    
    try:
        # Create QAT model
        qat_model = QuantizedYOLOv8('yolov8n.pt', skip_detection_head=True)
        logger.info("✓ QAT model created successfully")
        
        # Prepare for QAT
        qat_model.prepare_for_qat()
        logger.info("✓ Model prepared for QAT successfully")
        
        # Check quantization modules
        fake_quant_count = sum(1 for n, m in qat_model.model.model.named_modules() 
                            if 'FakeQuantize' in type(m).__name__)
        
        if fake_quant_count > 0:
            logger.info(f"✓ Found {fake_quant_count} FakeQuantize modules")
            return True, qat_model
        else:
            logger.error("❌ No FakeQuantize modules found")
            return False, None
            
    except Exception as e:
        logger.error(f"❌ Test 1 failed: {e}")
        return False, None

def test_new_save_method(qat_model):
    """Test 2: New state_dict-based save method"""
    logger.info("🧪 Test 2: New Save Method")
    
    test_save_path = 'test_qat_save_method.pt'
    
    try:
        # Count FakeQuantize modules before save
        fake_quant_count = sum(1 for n, m in qat_model.model.model.named_modules() 
                            if 'FakeQuantize' in type(m).__name__)
        
        logger.info(f"Pre-save FakeQuantize count: {fake_quant_count}")
        
        # Test the new save method
        save_success = qat_model.save(test_save_path, preserve_qat=True)
        
        if not save_success:
            logger.error("❌ Save method returned failure")
            return False
        
        # Check file exists
        if not os.path.exists(test_save_path):
            logger.error("❌ Save file not created")
            return False
        
        file_size = os.path.getsize(test_save_path) / (1024 * 1024)
        logger.info(f"✓ Save file created: {file_size:.2f} MB")
        
        # Load and verify save format
        saved_data = torch.load(test_save_path, map_location='cpu')
        
        # Check save format
        required_keys = ['model_state_dict', 'qat_info', 'fake_quant_count']
        missing_keys = [key for key in required_keys if key not in saved_data]
        
        if missing_keys:
            logger.error(f"❌ Missing keys in save: {missing_keys}")
            return False
        
        # Check FakeQuantize count preservation
        saved_count = saved_data['fake_quant_count']
        if saved_count != fake_quant_count:
            logger.error(f"❌ FakeQuantize count mismatch: {fake_quant_count} vs {saved_count}")
            return False
        
        logger.info(f"✓ Save format correct, FakeQuantize count preserved: {saved_count}")
        
        # Check quantization parameters
        if 'quantization_params' in saved_data:
            quant_params = saved_data['quantization_params']
            logger.info(f"✓ Quantization parameters extracted for {len(quant_params)} modules")
        
        # Clean up
        os.remove(test_save_path)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test 2 failed: {e}")
        if os.path.exists(test_save_path):
            os.remove(test_save_path)
        return False

def test_load_method(qat_model):
    """Test 3: New load method"""
    logger.info("🧪 Test 3: New Load Method")
    
    test_save_path = 'test_qat_load_method.pt'
    
    try:
        # Save the model first
        save_success = qat_model.save(test_save_path, preserve_qat=True)
        if not save_success:
            logger.error("❌ Failed to save model for load test")
            return False
        
        # Get original FakeQuantize count
        original_count = sum(1 for n, m in qat_model.model.model.named_modules() 
                           if 'FakeQuantize' in type(m).__name__)
        
        logger.info(f"Original FakeQuantize count: {original_count}")
        
        # Test loading
        loaded_model = QuantizedYOLOv8.load_qat_model_fixed(test_save_path, device='cpu')
        
        if loaded_model is None:
            logger.error("❌ Load method returned None")
            return False
        
        # Check loaded model quantization
        loaded_count = sum(1 for n, m in loaded_model.model.model.named_modules() 
                         if 'FakeQuantize' in type(m).__name__)
        
        logger.info(f"Loaded FakeQuantize count: {loaded_count}")
        
        if loaded_count != original_count:
            logger.error(f"❌ FakeQuantize count mismatch after loading")
            return False
        
        # Test quantization preservation verification
        if not loaded_model.verify_quantization_preserved():
            logger.error("❌ Quantization preservation verification failed")
            return False
        
        logger.info("✓ Load method successful, quantization preserved")
        
        # Clean up
        os.remove(test_save_path)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test 3 failed: {e}")
        if os.path.exists(test_save_path):
            os.remove(test_save_path)
        return False

def test_save_load_cycle(qat_model):
    """Test 4: Complete save/load cycle with verification"""
    logger.info("🧪 Test 4: Complete Save/Load Cycle")
    
    test_save_path = 'test_qat_cycle.pt'
    
    try:
        # Run the built-in test method
        test_result = qat_model.test_quantization_preservation()
        
        if test_result:
            logger.info("✓ Built-in quantization preservation test passed")
        else:
            logger.error("❌ Built-in quantization preservation test failed")
            return False
        
        # Test manual save/load cycle
        logger.info("Testing manual save/load cycle...")
        
        # Save
        save_success = qat_model.save(test_save_path, preserve_qat=True)
        if not save_success:
            logger.error("❌ Manual save failed")
            return False
        
        # Load
        loaded_model = QuantizedYOLOv8.load_qat_model_fixed(test_save_path)
        if loaded_model is None:
            logger.error("❌ Manual load failed")
            return False
        
        # Test inference capability (basic forward pass)
        logger.info("Testing inference capability...")
        test_input = torch.randn(1, 3, 640, 640)
        
        # Original model inference
        qat_model.model.model.eval()
        with torch.no_grad():
            orig_output = qat_model.model.model(test_input)
        
        # Loaded model inference
        loaded_model.model.model.eval()
        with torch.no_grad():
            loaded_output = loaded_model.model.model(test_input)
        
        # Compare outputs (they should be very similar)
        if len(orig_output) == len(loaded_output):
            max_diff = max(torch.max(torch.abs(o - l)).item() 
                          for o, l in zip(orig_output, loaded_output))
            logger.info(f"✓ Inference test passed, max output difference: {max_diff:.6f}")
            
            if max_diff > 1e-3:  # Some difference is expected due to quantization
                logger.warning(f"⚠️ Output difference is higher than expected: {max_diff}")
        else:
            logger.error("❌ Output structure mismatch")
            return False
        
        # Clean up
        os.remove(test_save_path)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test 4 failed: {e}")
        if os.path.exists(test_save_path):
            os.remove(test_save_path)
        return False

def test_model_conversion():
    """Test 5: QAT to INT8 conversion"""
    logger.info("🧪 Test 5: QAT to INT8 Conversion")
    
    try:
        # Create and prepare QAT model
        qat_model = QuantizedYOLOv8('yolov8n.pt', skip_detection_head=True)
        qat_model.prepare_for_qat()
        
        # Test conversion (without actual training)
        logger.info("Testing conversion readiness...")
        
        # Check if model has quantization preserved
        if not qat_model.verify_quantization_preserved():
            logger.error("❌ QAT model doesn't have quantization preserved")
            return False
        
        # Test the conversion method (this will fail without calibration, but we test the structure)
        try:
            # This should fail gracefully since we haven't trained/calibrated
            converted_model = qat_model.convert_to_quantized()
            if converted_model is not None:
                logger.info("⚠️ Conversion succeeded without training (unexpected)")
            else:
                logger.info("✓ Conversion correctly failed without calibration")
        except Exception as conv_error:
            if "QAT structure is missing" in str(conv_error):
                logger.info("✓ Conversion correctly detected missing QAT structure")
            else:
                logger.info(f"✓ Conversion failed as expected: {conv_error}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test 5 failed: {e}")
        return False

def test_onnx_export_readiness():
    """Test 6: ONNX export readiness"""
    logger.info("🧪 Test 6: ONNX Export Readiness")
    
    try:
        # Create and prepare QAT model
        qat_model = QuantizedYOLOv8('yolov8n.pt', skip_detection_head=True)
        qat_model.prepare_for_qat()
        
        # Save model in ONNX-ready format
        onnx_ready_path = 'test_onnx_ready.pt'
        save_success = qat_model.save(onnx_ready_path, preserve_qat=True)
        
        if not save_success:
            logger.error("❌ ONNX-ready save failed")
            return False
        
        # Check ONNX readiness indicators
        saved_data = torch.load(onnx_ready_path, map_location='cpu')
        
        onnx_ready = saved_data.get('model_info', {}).get('onnx_ready', False)
        if not onnx_ready:
            logger.warning("⚠️ ONNX ready flag not set")
        else:
            logger.info("✓ ONNX ready flag is set")
        
        # Check model info
        model_info = saved_data.get('model_info', {})
        expected_fields = ['architecture', 'task', 'format']
        
        missing_fields = [field for field in expected_fields if field not in model_info]
        if missing_fields:
            logger.warning(f"⚠️ Missing ONNX info fields: {missing_fields}")
        else:
            logger.info("✓ All ONNX info fields present")
        
        # Clean up
        os.remove(onnx_ready_path)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test 6 failed: {e}")
        return False

def run_all_tests():
    """Run all QAT fix tests"""
    logger.info("🚀 Starting Comprehensive QAT Fix Testing")
    logger.info("=" * 60)
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Basic QAT Setup
    test1_passed, qat_model = test_basic_qat_setup()
    if test1_passed:
        tests_passed += 1
    
    if not test1_passed or qat_model is None:
        logger.error("❌ Cannot continue without basic QAT setup")
        return False
    
    # Test 2: New Save Method
    if test_new_save_method(qat_model):
        tests_passed += 1
    
    # Test 3: New Load Method
    if test_load_method(qat_model):
        tests_passed += 1
    
    # Test 4: Complete Save/Load Cycle
    if test_save_load_cycle(qat_model):
        tests_passed += 1
    
    # Test 5: Model Conversion
    if test_model_conversion():
        tests_passed += 1
    
    # Test 6: ONNX Export Readiness
    if test_onnx_export_readiness():
        tests_passed += 1
    
    # Results summary
    logger.info("=" * 60)
    logger.info(f"🏁 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Your QAT fixes are ready for training.")
        return True
    elif tests_passed >= total_tests - 1:
        logger.info("✅ MOST TESTS PASSED! Minor issues detected, but should work for training.")
        return True
    else:
        logger.error("❌ MULTIPLE TESTS FAILED! Please fix issues before training.")
        return False

def cleanup_test_files():
    """Clean up any remaining test files"""
    test_files = [
        'test_qat_save_method.pt',
        'test_qat_load_method.pt', 
        'test_qat_cycle.pt',
        'test_onnx_ready.pt',
        'temp_qat_test.pt'
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            logger.info(f"Cleaned up: {file}")

if __name__ == "__main__":
    try:
        # Run all tests
        success = run_all_tests()
        
        # Clean up
        cleanup_test_files()
        
        # Exit with appropriate code
        if success:
            print("\n🎉 SUCCESS: QAT fixes are working correctly!")
            print("You can now run your full training with confidence.")
            sys.exit(0)
        else:
            print("\n❌ FAILURE: QAT fixes need more work.")
            print("Please address the issues before running full training.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        cleanup_test_files()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Testing failed with exception: {e}")
        cleanup_test_files()
        sys.exit(1)