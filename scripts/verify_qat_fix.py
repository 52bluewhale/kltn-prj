#!/usr/bin/env python
"""
QAT Fix Verification Script

This script verifies that the QAT fix is working correctly by:
1. Testing quantization preservation
2. Running a mini QAT training session
3. Verifying conversion works properly
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
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the fixed implementation
from src.models.yolov8_qat_fixed import QuantizedYOLOv8Fixed

def verify_quantization_preservation():
    """Verify that quantization is preserved during save/load."""
    logger.info("=" * 60)
    logger.info("TESTING QUANTIZATION PRESERVATION")
    logger.info("=" * 60)
    
    try:
        # Create QAT model
        logger.info("1. Creating QAT model...")
        qat_model = QuantizedYOLOv8Fixed(
            model_path='yolov8n.pt',
            qconfig_name='default',
            skip_detection_head=True,
            fuse_modules=True
        )
        
        # Prepare for QAT
        logger.info("2. Preparing model for QAT...")
        qat_model.prepare_for_qat()
        
        # Check initial quantization modules
        initial_fake_quant = sum(1 for name, module in qat_model.preserved_qat_model.named_modules() 
                               if hasattr(module, 'weight_fake_quant'))
        initial_observers = sum(1 for name, module in qat_model.preserved_qat_model.named_modules() 
                              if hasattr(module, 'activation_post_process'))
        
        logger.info(f"   Initial fake quantization modules: {initial_fake_quant}")
        logger.info(f"   Initial observer modules: {initial_observers}")
        
        if initial_fake_quant == 0:
            logger.error("❌ FAILED: No fake quantization modules created!")
            return False
        
        # Test saving with quantization preservation
        logger.info("3. Testing quantization-preserving save...")
        test_save_path = "test_qat_preservation.pt"  # Simple filename, no directory
        success = qat_model.save_qat_model_with_quantization(qat_model.preserved_qat_model, test_save_path)
        
        if not success:
            logger.error("❌ FAILED: Could not save with quantization preservation!")
            return False
        
        # Test loading with quantization preservation
        logger.info("4. Testing quantization-preserving load...")
        loaded_model = qat_model.load_qat_model_with_quantization(test_save_path)
        
        if loaded_model is None:
            logger.error("❌ FAILED: Could not load with quantization preservation!")
            return False
        
        # Verify quantization modules are preserved
        loaded_fake_quant = sum(1 for name, module in loaded_model.named_modules() 
                              if hasattr(module, 'weight_fake_quant'))
        loaded_observers = sum(1 for name, module in loaded_model.named_modules() 
                             if hasattr(module, 'activation_post_process'))
        
        logger.info(f"   Loaded fake quantization modules: {loaded_fake_quant}")
        logger.info(f"   Loaded observer modules: {loaded_observers}")
        
        # Check if quantization is preserved
        if loaded_fake_quant == initial_fake_quant and loaded_observers == initial_observers:
            logger.info("✅ SUCCESS: Quantization preservation test PASSED!")
            success = True
        else:
            logger.error("❌ FAILED: Quantization modules not preserved correctly!")
            success = False
        
        # Cleanup
        if os.path.exists(test_save_path):
            os.remove(test_save_path)
        
        return success
        
    except Exception as e:
        logger.error(f"❌ FAILED: Exception during quantization preservation test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def verify_phase_configuration():
    """Verify that phase-based quantization configuration works."""
    logger.info("=" * 60)
    logger.info("TESTING PHASE CONFIGURATION")
    logger.info("=" * 60)
    
    try:
        # Create QAT model
        logger.info("1. Creating and preparing QAT model...")
        qat_model = QuantizedYOLOv8Fixed(
            model_path='yolov8n.pt',
            qconfig_name='default',
            skip_detection_head=True,
            fuse_modules=True
        )
        qat_model.prepare_for_qat()
        
        # Test different phase configurations
        phases_to_test = [
            ("weight_only", True, False),
            ("activation_phase", True, True),
            ("full_quantization", True, True),
            ("fine_tuning", True, True)
        ]
        
        for phase_name, expected_weights, expected_activations in phases_to_test:
            logger.info(f"2. Testing {phase_name} configuration...")
            
            # Configure phase
            qat_model._configure_phase(phase_name)
            
            # Count active quantizers
            active_weights = 0
            active_activations = 0
            
            for name, module in qat_model.model.model.named_modules():
                if hasattr(module, 'weight_fake_quant') and not isinstance(module.weight_fake_quant, torch.nn.Identity):
                    active_weights += 1
                if hasattr(module, 'activation_post_process') and not isinstance(module.activation_post_process, torch.nn.Identity):
                    active_activations += 1
            
            logger.info(f"   Active weight quantizers: {active_weights}")
            logger.info(f"   Active activation quantizers: {active_activations}")
            
            # Verify configuration
            weights_correct = (active_weights > 0) == expected_weights
            activations_correct = (active_activations > 0) == expected_activations
            
            if weights_correct and activations_correct:
                logger.info(f"   ✅ {phase_name} configuration CORRECT")
            else:
                logger.error(f"   ❌ {phase_name} configuration INCORRECT")
                return False
        
        logger.info("✅ SUCCESS: Phase configuration test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ FAILED: Exception during phase configuration test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def verify_conversion_capability():
    """Verify that conversion from QAT to quantized model works."""
    logger.info("=" * 60)
    logger.info("TESTING CONVERSION CAPABILITY")
    logger.info("=" * 60)
    
    try:
        # Create QAT model
        logger.info("1. Creating and preparing QAT model...")
        qat_model = QuantizedYOLOv8Fixed(
            model_path='yolov8n.pt',
            qconfig_name='default',
            skip_detection_head=True,
            fuse_modules=True
        )
        qat_model.prepare_for_qat()
        
        # Verify we have fake quantization modules
        fake_quant_count = sum(1 for name, module in qat_model.preserved_qat_model.named_modules() 
                             if hasattr(module, 'weight_fake_quant'))
        
        logger.info(f"   QAT model has {fake_quant_count} fake quantization modules")
        
        if fake_quant_count == 0:
            logger.error("❌ FAILED: No fake quantization modules to convert!")
            return False
        
        # Test conversion
        logger.info("2. Testing conversion to quantized model...")
        test_save_path = "test_quantized_conversion.pt"  # Simple filename, no directory
        
        try:
            quantized_model = qat_model.convert_to_quantized_fixed(test_save_path)
            
            if quantized_model is not None:
                logger.info("   ✅ Conversion completed successfully")
                
                # Check if the quantized model has the expected structure
                quantized_modules = sum(1 for name, module in quantized_model.named_modules() 
                                      if 'quantized' in module.__class__.__name__.lower())
                
                logger.info(f"   Quantized model has {quantized_modules} quantized modules")
                
                # Cleanup
                if os.path.exists(test_save_path):
                    os.remove(test_save_path)
                
                logger.info("✅ SUCCESS: Conversion capability test PASSED!")
                return True
            else:
                logger.error("❌ FAILED: Conversion returned None!")
                return False
                
        except Exception as conv_error:
            logger.error(f"❌ FAILED: Conversion failed with error: {conv_error}")
            return False
        
    except Exception as e:
        logger.error(f"❌ FAILED: Exception during conversion test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_mini_training_test():
    """Run a mini training test to verify the full pipeline."""
    logger.info("=" * 60)
    logger.info("TESTING MINI TRAINING PIPELINE")
    logger.info("=" * 60)
    
    try:
        # This is a conceptual test - we can't run full training without data
        # But we can test the setup and configuration
        
        logger.info("1. Setting up mini training test...")
        qat_model = QuantizedYOLOv8Fixed(
            model_path='yolov8n.pt',
            qconfig_name='default',
            skip_detection_head=True,
            fuse_modules=True
        )
        qat_model.prepare_for_qat()
        
        # Test saving QAT model during "training"
        logger.info("2. Testing QAT model saving during training simulation...")
        test_save_dir = "test_mini_training"
        os.makedirs(test_save_dir, exist_ok=True)
        
        # Simulate saving after each phase
        phases = ["phase1_weight_only", "phase2_activations", "phase3_full_quant", "phase4_fine_tuning"]
        
        for phase in phases:
            qat_model._configure_phase(phase.replace("phase1_", "").replace("phase2_", "").replace("phase3_", "").replace("phase4_", ""))
            
            phase_save_path = os.path.join(test_save_dir, f"{phase}_qat_model.pt")
            success = qat_model.save_qat_model_with_quantization(qat_model.preserved_qat_model, phase_save_path)
            
            if success:
                logger.info(f"   ✅ {phase} QAT model saved successfully")
                
                # Test loading
                loaded = qat_model.load_qat_model_with_quantization(phase_save_path)
                if loaded is not None:
                    logger.info(f"   ✅ {phase} QAT model loaded successfully")
                else:
                    logger.error(f"   ❌ {phase} QAT model failed to load")
                    return False
            else:
                logger.error(f"   ❌ {phase} QAT model failed to save")
                return False
        
        # Test conversion from the final phase
        logger.info("3. Testing conversion from final phase...")
        final_qat_path = os.path.join(test_save_dir, "phase4_fine_tuning_qat_model.pt")
        
        if os.path.exists(final_qat_path):
            # Load the final QAT model
            final_qat_model = qat_model.load_qat_model_with_quantization(final_qat_path)
            
            if final_qat_model is not None:
                qat_model.preserved_qat_model = final_qat_model
                
                # Test conversion
                try:
                    test_quantized_path = "test_final_quantized.pt"  # Simple filename, no directory
                    quantized_model = qat_model.convert_to_quantized_fixed(test_quantized_path)
                    
                    if quantized_model is not None:
                        logger.info("   ✅ Final conversion successful")
                        
                        # Cleanup
                        if os.path.exists(test_quantized_path):
                            os.remove(test_quantized_path)
                    else:
                        logger.error("   ❌ Final conversion failed")
                        return False
                        
                except Exception as conv_error:
                    logger.error(f"   ❌ Final conversion failed: {conv_error}")
                    return False
            else:
                logger.error("   ❌ Failed to load final QAT model")
                return False
        
        # Cleanup
        import shutil
        if os.path.exists(test_save_dir):
            shutil.rmtree(test_save_dir)
        
        logger.info("✅ SUCCESS: Mini training pipeline test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ FAILED: Exception during mini training test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all verification tests."""
    logger.info("🔧 QAT FIX VERIFICATION SUITE")
    logger.info("=" * 80)
    
    tests = [
        ("Quantization Preservation", verify_quantization_preservation),
        ("Phase Configuration", verify_phase_configuration), 
        ("Conversion Capability", verify_conversion_capability),
        ("Mini Training Pipeline", run_mini_training_test)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        
        try:
            if test_func():
                passed_tests += 1
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} test FAILED with exception: {e}")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("🏁 VERIFICATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! The QAT fix is working correctly.")
        logger.info("\nYou can now run the fixed training script:")
        logger.info("python scripts/train_qat_fixed.py --model yolov8n.pt --data your_dataset.yaml --epochs 5")
        return True
    else:
        logger.error(f"❌ {total_tests - passed_tests} tests failed. The fix needs more work.")
        return False

if __name__ == "__main__":
    main()