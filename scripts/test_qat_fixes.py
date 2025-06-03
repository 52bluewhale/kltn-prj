#!/usr/bin/env python
"""
QAT Fixes Verification Script

Run this script to verify that the QAT fixes are working correctly.
This will test the quantization preservation throughout the training process.
"""

import os
import sys
import logging
import torch
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_qat')

def test_quantizer_state_manager():
    """Test the fixed quantizer state manager."""
    logger.info("🧪 Testing Quantizer State Manager...")
    
    try:
        # Create a simple model for testing
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, 3)
                self.conv2 = torch.nn.Conv2d(16, 32, 3)
                self.relu = torch.nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                return x
        
        # Prepare model for QAT
        model = TestModel()
        model.train()
        
        # Apply qconfig
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare for QAT
        prepared_model = torch.quantization.prepare_qat(model, inplace=False)
        
        # Count initial FakeQuantize modules
        initial_count = sum(1 for n, m in prepared_model.named_modules() 
                           if 'FakeQuantize' in type(m).__name__)
        logger.info(f"✅ Initial FakeQuantize modules: {initial_count}")
        
        if initial_count == 0:
            logger.error("❌ No FakeQuantize modules found after prepare_qat!")
            return False
        
        # Test quantizer state manager
        from src.quantization.quantizer_state_manager import QuantizerStateManager
        manager = QuantizerStateManager(prepared_model)
        
        # Test phase transitions
        phases = [
            ("phase1_weight_only", True, False),
            ("phase2_activations", True, True),
            ("phase3_full_quant", True, True),
            ("phase4_fine_tuning", True, True)
        ]
        
        for phase_name, weights_enabled, activations_enabled in phases:
            logger.info(f"Testing {phase_name}...")
            
            success = manager.set_phase_state(
                phase_name=phase_name,
                weights_enabled=weights_enabled,
                activations_enabled=activations_enabled
            )
            
            if not success:
                logger.error(f"❌ Phase {phase_name} failed!")
                return False
            
            # Verify FakeQuantize modules still exist
            current_count = manager.count_fake_quantize_modules()
            if current_count != initial_count:
                logger.error(f"❌ FakeQuantize count changed: {initial_count} → {current_count}")
                return False
        
        logger.info("✅ Quantizer State Manager test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Quantizer State Manager test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_qat_model_creation():
    """Test QAT model creation and preparation."""
    logger.info("🧪 Testing QAT Model Creation...")
    
    try:
        from src.models.yolov8_qat import QuantizedYOLOv8
        
        # Create QAT model
        qat_model = QuantizedYOLOv8(
            model_path='yolov8n.pt',
            qconfig_name='default',
            skip_detection_head=True,
            fuse_modules=True
        )
        
        # Prepare for QAT
        prepared_model = qat_model.prepare_for_qat()
        
        if prepared_model is None:
            logger.error("❌ QAT preparation returned None!")
            return False
        
        # Check quantization structure
        fake_quant_count = qat_model._count_fake_quantize_modules()
        logger.info(f"✅ QAT model has {fake_quant_count} FakeQuantize modules")
        
        if fake_quant_count == 0:
            logger.error("❌ No FakeQuantize modules in QAT model!")
            return False
        
        # Test phase configuration
        if hasattr(qat_model, 'quantizer_manager'):
            success = qat_model._configure_quantizers_dynamically("phase1_weight_only")
            if not success:
                logger.error("❌ Phase configuration failed!")
                return False
            
            # Verify quantization still exists
            final_count = qat_model._count_fake_quantize_modules()
            if final_count == 0:
                logger.error("❌ Quantization lost after phase configuration!")
                return False
            
            logger.info(f"✅ After phase config: {final_count} FakeQuantize modules")
        
        logger.info("✅ QAT Model Creation test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ QAT Model Creation test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_model_saving():
    """Test QAT model saving with quantization preservation."""
    logger.info("🧪 Testing QAT Model Saving...")
    
    try:
        from src.models.yolov8_qat import QuantizedYOLOv8
        
        # Create and prepare QAT model
        qat_model = QuantizedYOLOv8(
            model_path='yolov8n.pt',
            qconfig_name='default',
            skip_detection_head=True
        )
        qat_model.prepare_for_qat()
        
        # Check quantization before save
        fake_quant_count_before = qat_model._count_fake_quantize_modules()
        logger.info(f"Before save: {fake_quant_count_before} FakeQuantize modules")
        
        if fake_quant_count_before == 0:
            logger.error("❌ No quantization to save!")
            return False
        
        # Save model
        test_save_path = "test_qat_model.pt"
        success = qat_model.save(test_save_path, preserve_qat=True)
        
        if not success:
            logger.error("❌ Model save failed!")
            return False
        
        # Verify saved file
        if not os.path.exists(test_save_path):
            logger.error("❌ Saved file does not exist!")
            return False
        
        # Load and verify saved model
        saved_data = torch.load(test_save_path, map_location='cpu')
        
        if not isinstance(saved_data, dict):
            logger.error("❌ Saved data is not a dictionary!")
            return False
        
        saved_fake_quant_count = saved_data.get('fake_quant_count', 0)
        logger.info(f"✅ Saved model has {saved_fake_quant_count} FakeQuantize modules")
        
        if saved_fake_quant_count != fake_quant_count_before:
            logger.error(f"❌ FakeQuantize count mismatch: {fake_quant_count_before} vs {saved_fake_quant_count}")
            return False
        
        # Cleanup
        os.remove(test_save_path)
        
        logger.info("✅ QAT Model Saving test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ QAT Model Saving test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_penalty_loss_integration():
    """Test penalty loss integration."""
    logger.info("🧪 Testing Penalty Loss Integration...")
    
    try:
        from src.models.yolov8_qat import QuantizedYOLOv8
        
        # Create QAT model
        qat_model = QuantizedYOLOv8(
            model_path='yolov8n.pt',
            skip_detection_head=True
        )
        qat_model.prepare_for_qat()
        
        # Setup penalty loss
        qat_model.setup_penalty_loss_integration(alpha=0.01, warmup_epochs=5)
        
        # Verify penalty setup
        penalty_working = qat_model.verify_penalty_setup()
        
        if penalty_working:
            logger.info("✅ Penalty Loss Integration test PASSED")
            return True
        else:
            logger.error("❌ Penalty loss verification failed!")
            return False
        
    except Exception as e:
        logger.error(f"❌ Penalty Loss Integration test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_all_tests():
    """Run all QAT fix verification tests."""
    print("="*80)
    print("🧪 QAT FIXES VERIFICATION TESTS")
    print("="*80)
    
    tests = [
        ("Quantizer State Manager", test_quantizer_state_manager),
        ("QAT Model Creation", test_qat_model_creation),
        ("QAT Model Saving", test_model_saving),
        ("Penalty Loss Integration", test_penalty_loss_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "="*80)
    print(f"📊 TEST RESULTS: {passed}/{total} PASSED")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! QAT fixes are working correctly.")
        print("✅ You can now run QAT training with confidence.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("⚠️ QAT training may not work correctly until issues are resolved.")
    
    print("="*80)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n🚀 Ready to run QAT training:")
        print("python scripts/train_qat.py --model yolov8n.pt --data datasets/vietnam-traffic-sign-detection/dataset.yaml --epochs 5")
    else:
        print("\n🔧 Please fix the failing tests before running QAT training.")
    
    sys.exit(0 if success else 1)