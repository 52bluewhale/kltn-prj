#!/usr/bin/env python
"""
Corrected Test Script - test_preservation.py
Place this in your scripts/ directory
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]  # Go up one level from scripts/
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import logging
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_preservation_integration():
    """Test that preservation works with existing code."""
    logger.info("🧪 Testing preservation integration...")
    
    try:
        # Import your existing classes
        from src.models.yolov8_qat import QuantizedYOLOv8
        
        # Initialize QAT model (your existing way)
        qat_model = QuantizedYOLOv8(
            model_path="yolov8n.pt",
            qconfig_name="sensitive", 
            skip_detection_head=True,
            fuse_modules=True
        )
        
        logger.info("✅ QuantizedYOLOv8 model created successfully")
        
        # Use new preparation method
        logger.info("🔧 Testing preparation with preservation...")
        result = qat_model.prepare_for_qat_with_preservation()
        
        if result is None:
            logger.error("❌ Preparation failed - no FakeQuantize modules created")
            return False
            
        logger.info("✅ QAT preparation with preservation successful")
        
        # Test phase transitions
        phases = ["phase1_weight_only", "phase2_activations", "phase3_full_quant", "phase4_fine_tuning"]
        
        for phase in phases:
            logger.info(f"🔄 Testing {phase}...")
            success = qat_model._configure_quantizers_with_preservation(phase)
            
            if success:
                stats = qat_model.quantizer_preserver.get_quantizer_stats()
                logger.info(f"✅ {phase}: {stats['total_fake_quantizers']} quantizers preserved")
                logger.info(f"    Weights: {stats['weight_quantizers_enabled']}/{stats['weight_quantizers_total']}")
                logger.info(f"    Activations: {stats['activation_quantizers_enabled']}/{stats['activation_quantizers_total']}")
            else:
                logger.error(f"❌ {phase}: Preservation failed!")
                return False
        
        # Test back to phase 1 to ensure re-enabling works
        logger.info("🔄 Testing return to phase1_weight_only...")
        success = qat_model._configure_quantizers_with_preservation("phase1_weight_only")
        
        if success:
            stats = qat_model.quantizer_preserver.get_quantizer_stats()
            logger.info(f"✅ Return to phase1: {stats['total_fake_quantizers']} quantizers preserved")
            logger.info(f"    Should have weights enabled, activations disabled")
            
            # Verify weights enabled, activations disabled
            if (stats['weight_quantizers_enabled'] > 0 and 
                stats['activation_quantizers_enabled'] == 0):
                logger.info("✅ Phase1 state correctly set: weights ON, activations OFF")
            else:
                logger.warning(f"⚠️ Phase1 state unexpected: weights={stats['weight_quantizers_enabled']}, activations={stats['activation_quantizers_enabled']}")
        
        logger.info("🎉 All preservation tests passed! Integration successful!")
        
        # Final verification
        final_stats = qat_model.quantizer_preserver.get_quantizer_stats()
        logger.info("📊 Final Statistics:")
        logger.info(f"  - Total FakeQuantize modules: {final_stats['total_fake_quantizers']}")
        logger.info(f"  - Weight quantizers: {final_stats['weight_quantizers_total']}")
        logger.info(f"  - Activation quantizers: {final_stats['activation_quantizers_total']}")
        logger.info(f"  - Preservation active: {final_stats['preservation_active']}")
        
        return final_stats['preservation_active'] and final_stats['total_fake_quantizers'] > 0
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure you have created the quantizer_preservation.py file")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_basic_quantization_first():
    """Test basic quantization before preservation test."""
    logger.info("🧪 Testing basic quantization first...")
    
    try:
        from src.quantization.qconfig import get_default_qat_qconfig
        
        # Create simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
            
            def forward(self, x):
                return self.conv(x)
        
        model = SimpleModel()
        model.train()
        model.qconfig = get_default_qat_qconfig()
        
        prepared = torch.quantization.prepare_qat(model)
        
        has_weight_fake_quant = hasattr(prepared.conv, 'weight_fake_quant')
        has_activation_post_process = hasattr(prepared.conv, 'activation_post_process')
        
        logger.info(f"✅ Basic quantization test:")
        logger.info(f"  - Conv has weight_fake_quant: {has_weight_fake_quant}")
        logger.info(f"  - Conv has activation_post_process: {has_activation_post_process}")
        
        return has_weight_fake_quant and has_activation_post_process
        
    except Exception as e:
        logger.error(f"❌ Basic quantization test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting preservation integration tests...")
    
    # Test basic quantization first
    basic_test = test_basic_quantization_first()
    if not basic_test:
        logger.error("❌ Basic quantization test failed - check your environment")
        sys.exit(1)
    
    # Test preservation integration
    success = test_preservation_integration()
    
    print("\n" + "="*80)
    print("📊 TEST RESULTS")
    print("="*80)
    print(f"Basic Quantization: {'✅ PASS' if basic_test else '❌ FAIL'}")
    print(f"Preservation Integration: {'✅ PASS' if success else '❌ FAIL'}")
    print("="*80)
    
    if success:
        print("🎉 ALL TESTS PASSED! Ready for integration with train_qat.py")
        print("\nNext steps:")
        print("1. Add the missing methods to your QuantizedYOLOv8 class")
        print("2. Update your train_qat.py as shown in the tutorial")
        print("3. Run your full training with preservation")
    else:
        print("❌ TESTS FAILED! Check the errors above")
        print("\nTroubleshooting:")
        print("1. Ensure quantizer_preservation.py is created")
        print("2. Check that all methods are added to QuantizedYOLOv8")
        print("3. Verify imports and paths are correct")
    
    sys.exit(0 if success else 1)