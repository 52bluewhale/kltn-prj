#!/usr/bin/env python
"""
Quick test script to verify QAT save/load functionality
"""
import sys
import os
import logging
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_qat_save_load():
    """Test QAT save/load functionality"""
    print("="*80)
    print("TESTING QAT SAVE/LOAD FUNCTIONALITY")
    print("="*80)
    
    try:
        # Import the QAT model
        from src.models.yolov8_qat import QuantizedYOLOv8
        
        print("Step 1: Creating QAT model...")
        qat_model = QuantizedYOLOv8('yolov8n.pt', skip_detection_head=True)
        
        print("Step 2: Preparing model for QAT...")
        qat_model.prepare_for_qat()
        
        print("Step 3: Testing quantization preservation...")
        test_result = qat_model.test_quantization_preservation()
        
        if test_result:
            print("✅ QAT save/load test PASSED")
            return True
        else:
            print("❌ QAT save/load test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ QAT save/load test exception: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_basic_verification():
    """Test the basic model verification"""
    print("\n" + "="*80)
    print("TESTING BASIC MODEL VERIFICATION")
    print("="*80)
    
    try:
        from src.models.yolov8_qat import QuantizedYOLOv8
        
        # Create and prepare model
        qat_model = QuantizedYOLOv8('yolov8n.pt', skip_detection_head=True)
        qat_model.prepare_for_qat()
        
        # Test current quantization status
        result = qat_model.verify_quantization_preserved()
        
        if result:
            print("✅ Basic verification test PASSED")
            return True
        else:
            print("❌ Basic verification test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Basic verification test exception: {e}")
        return False

if __name__ == "__main__":
    print("Running QAT functionality tests...\n")
    
    # Test 1: Basic verification
    test1_result = test_basic_verification()
    
    # Test 2: Save/load cycle
    test2_result = test_qat_save_load()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Basic verification: {'PASSED' if test1_result else 'FAILED'}")
    print(f"Save/load cycle: {'PASSED' if test2_result else 'FAILED'}")
    
    if test1_result and test2_result:
        print("✅ ALL TESTS PASSED - QAT functionality is working correctly")
        exit(0)
    else:
        print("❌ SOME TESTS FAILED - Check the logs above for details")
        exit(1)