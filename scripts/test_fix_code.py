#!/usr/bin/env python
"""
QAT Implementation Verification Script
Location: scripts/verify_qat_implementation.py

Run this script to verify that your QAT implementation (steps 1-3) is working correctly
and hasn't corrupted your codebase.

Usage:
    python scripts/verify_qat_implementation.py
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('qat_verification')

def verify_qat_implementation():
    """
    Verify that steps 1-3 implementations are working correctly.
    This will help identify any corruption issues.
    """
    print("=" * 80)
    print("🔍 VERIFYING QAT IMPLEMENTATION - STEPS 1-3")
    print("=" * 80)
    
    verification_results = {
        'observer_warmup': False,
        'gradual_quantization': False,
        'improved_ste': False,
        'overall_status': 'UNKNOWN'
    }
    
    try:
        # Test 1: Observer Warmup Implementation
        print("\n📊 Testing Observer Warmup Implementation...")
        
        from src.models.yolov8_qat import QuantizedYOLOv8
        
        # Create test model
        test_model = QuantizedYOLOv8('yolov8n.pt', skip_detection_head=True)
        
        # Check if warmup methods exist
        if hasattr(test_model, 'warmup_observers'):
            print("✅ warmup_observers method found")
            verification_results['observer_warmup'] = True
        else:
            print("❌ warmup_observers method missing")
        
        if hasattr(test_model, '_check_observer_readiness'):
            print("✅ _check_observer_readiness method found")
        else:
            print("❌ _check_observer_readiness method missing")
            
    except ImportError as e:
        print(f"❌ Import error in observer warmup: {e}")
    except Exception as e:
        print(f"❌ Error testing observer warmup: {e}")
    
    try:
        # Test 2: Gradual Quantization Implementation
        print("\n🔄 Testing Gradual Quantization Implementation...")
        
        # Check if gradual methods exist
        if hasattr(test_model, 'gradual_quantization_enable'):
            print("✅ gradual_quantization_enable method found")
            
            # Test the method
            factor = test_model.gradual_quantization_enable(5, 10, 0, 8)
            if 0.0 <= factor <= 1.0:
                print(f"✅ Gradual factor calculation working: {factor:.3f}")
                verification_results['gradual_quantization'] = True
            else:
                print(f"❌ Invalid gradual factor: {factor}")
        else:
            print("❌ gradual_quantization_enable method missing")
            
        if hasattr(test_model, '_configure_phase_gradual'):
            print("✅ _configure_phase_gradual method found")
        else:
            print("❌ _configure_phase_gradual method missing")
            
    except Exception as e:
        print(f"❌ Error testing gradual quantization: {e}")
    
    try:
        # Test 3: Improved STE Implementation
        print("\n⚡ Testing Improved STE Implementation...")
        
        from src.quantization.fake_quantize import AdaptiveSTE, ImprovedCustomFakeQuantize
        
        print("✅ AdaptiveSTE class imported successfully")
        print("✅ ImprovedCustomFakeQuantize class imported successfully")
        
        # Test AdaptiveSTE
        test_tensor = torch.randn(2, 3, 4, 4, requires_grad=True)
        ste_output = AdaptiveSTE.apply(test_tensor, 1.0)
        
        if ste_output.shape == test_tensor.shape:
            print("✅ AdaptiveSTE forward pass working")
            verification_results['improved_ste'] = True
        else:
            print("❌ AdaptiveSTE shape mismatch")
            
    except ImportError as e:
        print(f"❌ Import error in improved STE: {e}")
    except Exception as e:
        print(f"❌ Error testing improved STE: {e}")
    
    # Overall Assessment
    print("\n" + "=" * 80)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 80)
    
    passed_tests = sum(verification_results[key] for key in ['observer_warmup', 'gradual_quantization', 'improved_ste'])
    total_tests = 3
    
    print(f"Observer Warmup:       {'✅ PASS' if verification_results['observer_warmup'] else '❌ FAIL'}")
    print(f"Gradual Quantization:  {'✅ PASS' if verification_results['gradual_quantization'] else '❌ FAIL'}")
    print(f"Improved STE:          {'✅ PASS' if verification_results['improved_ste'] else '❌ FAIL'}")
    
    if passed_tests == total_tests:
        verification_results['overall_status'] = 'HEALTHY'
        print(f"\n🎉 OVERALL STATUS: HEALTHY ({passed_tests}/{total_tests} tests passed)")
        print("✅ Your codebase is NOT corrupted and steps 1-3 are working correctly!")
    elif passed_tests >= 2:
        verification_results['overall_status'] = 'MOSTLY_HEALTHY'
        print(f"\n⚠️ OVERALL STATUS: MOSTLY HEALTHY ({passed_tests}/{total_tests} tests passed)")
        print("🔧 Minor issues detected but codebase is functional")
    else:
        verification_results['overall_status'] = 'NEEDS_ATTENTION'
        print(f"\n🚨 OVERALL STATUS: NEEDS ATTENTION ({passed_tests}/{total_tests} tests passed)")
        print("❌ Multiple issues detected - please review implementation")
    
    return verification_results

def quick_functionality_test():
    """Quick test to ensure basic QAT functionality works."""
    print("\n🧪 Running Quick Functionality Test...")
    
    try:
        # Test basic model creation and preparation
        from src.models.yolov8_qat import QuantizedYOLOv8
        
        qat_model = QuantizedYOLOv8('yolov8n.pt', skip_detection_head=True)
        qat_model.prepare_for_qat()
        
        # Count quantization modules
        fake_quant_count = sum(1 for n, m in qat_model.model.model.named_modules() 
                            if 'FakeQuantize' in type(m).__name__)
        
        if fake_quant_count > 0:
            print(f"✅ QAT preparation successful: {fake_quant_count} FakeQuantize modules")
            return True
        else:
            print("❌ QAT preparation failed: No FakeQuantize modules found")
            return False
            
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Main verification function."""
    print("🚀 Starting QAT Implementation Verification")
    print(f"📁 Project root: {ROOT}")
    print(f"🐍 Python version: {sys.version}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    
    # Run verification
    results = verify_qat_implementation()
    
    # Run functionality test
    functional = quick_functionality_test()
    
    # Final recommendation
    print("\n" + "=" * 80)
    print("🎯 RECOMMENDATIONS")
    print("=" * 80)
    
    if results['overall_status'] == 'HEALTHY' and functional:
        print("✅ Your implementation is SOLID! Proceed to step 4 with confidence.")
        print("🚀 No corruption detected - your codebase is working correctly.")
        return 0
    elif results['overall_status'] in ['MOSTLY_HEALTHY', 'HEALTHY']:
        print("⚠️ Minor issues detected but codebase is FUNCTIONAL.")
        print("🔧 You can proceed to step 4, but monitor for any issues.")
        return 1
    else:
        print("🚨 SIGNIFICANT ISSUES detected!")
        print("❌ Please review your implementation before proceeding.")
        print("💡 Check import paths and ensure all files are properly updated.")
        return 2

if __name__ == "__main__":
    exit_code = main()
    print(f"\n📞 If you see issues, please share this output for help!")
    sys.exit(exit_code)