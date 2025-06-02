#!/usr/bin/env python
"""
Test script to verify all phase transitions work correctly.
Run this BEFORE full training to ensure the fix works.
"""
import sys
import os
import logging
import torch
from pathlib import Path

# FIXED: Add project root to path correctly
ROOT = Path(__file__).resolve().parents[1]  # Go up one level from scripts/ to project root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

print(f"🔍 Project root: {ROOT}")
print(f"🔍 Python path includes: {[p for p in sys.path if 'kltn-prj' in p]}")

def test_all_phase_transitions():
    """Test ALL phase transitions comprehensively."""
    
    print("🧪 COMPREHENSIVE Phase Transition Test...")
    print("="*60)
    
    try:
        # Import the fixed QAT model
        from src.models.yolov8_qat import QuantizedYOLOv8
        print("✅ Successfully imported QuantizedYOLOv8")
        
        # Create a test model
        print("📦 Creating test QAT model...")
        qat_model = QuantizedYOLOv8('yolov8n.pt', skip_detection_head=True)
        print("✅ Model created successfully")
        
        # Prepare for QAT
        print("⚙️ Preparing model for QAT...")
        qat_model.prepare_for_qat()
        print("✅ Model prepared for QAT")
        
        # Test sequence: Phase 1 → 2 → 3 → 4 → 1 (full cycle)
        test_sequence = [
            ("phase1_weight_only", "Phase 1 (Weight Only)", True, False),
            ("phase2_activations", "Phase 2 (Add Activations)", True, True),
            ("phase3_full_quant", "Phase 3 (Full Quantization)", True, True),
            ("phase4_fine_tuning", "Phase 4 (Fine-tuning)", True, True),
            ("phase1_weight_only", "Phase 1 Again (Cycle Test)", True, False),
            ("phase2_activations", "Phase 2 Again (Cycle Test)", True, True),
        ]
        
        all_passed = True
        results = []
        
        print("\n🔄 Testing Full Phase Transition Sequence...")
        print("-" * 60)
        
        for i, (phase_name, description, expected_weights, expected_activations) in enumerate(test_sequence, 1):
            print(f"\n📍 Step {i}: {description}")
            print(f"   Expected: Weights={expected_weights}, Activations={expected_activations}")
            
            # Apply phase transition
            success = qat_model._configure_quantizers_dynamically(phase_name)
            
            if not success:
                print(f"❌ {description} - Configuration FAILED")
                all_passed = False
                results.append((description, False, "Config Failed", 0, 0))
                continue
            
            # Get current state
            if hasattr(qat_model, 'quantizer_manager'):
                state = qat_model.quantizer_manager.get_current_state()
                
                actual_weights_enabled = state['weight_active'] > 0
                actual_activations_enabled = state['activation_active'] > 0
                
                # Check if results match expectations
                weights_ok = actual_weights_enabled == expected_weights
                activations_ok = actual_activations_enabled == expected_activations
                
                if weights_ok and activations_ok:
                    print(f"✅ {description} - SUCCESS")
                    print(f"   Actual: Weight={state['weight_active']}/{state['total_weight']}, "
                          f"Activation={state['activation_active']}/{state['total_activation']}")
                    results.append((description, True, "Success", state['weight_active'], state['activation_active']))
                else:
                    print(f"❌ {description} - State MISMATCH")
                    print(f"   Expected: Weights={expected_weights}, Activations={expected_activations}")
                    print(f"   Actual: Weights={actual_weights_enabled}, Activations={actual_activations_enabled}")
                    print(f"   Counts: Weight={state['weight_active']}/{state['total_weight']}, "
                          f"Activation={state['activation_active']}/{state['total_activation']}")
                    all_passed = False
                    results.append((description, False, "State Mismatch", state['weight_active'], state['activation_active']))
            else:
                print(f"❌ {description} - QuantizerStateManager missing")
                all_passed = False
                results.append((description, False, "Manager Missing", 0, 0))
        
        # Print comprehensive results
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE TEST RESULTS")
        print("="*60)
        
        for i, (description, passed, status, weights, activations) in enumerate(results, 1):
            status_icon = "✅" if passed else "❌"
            print(f"{status_icon} Step {i}: {description}")
            print(f"   Status: {status}")
            if weights > 0 or activations > 0:
                print(f"   Quantizers: Weight={weights}, Activation={activations}")
        
        print("\n" + "="*60)
        if all_passed:
            print("🎉 ALL PHASE TRANSITIONS WORKING PERFECTLY!")
            print("✅ Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 1 cycle PASSED")
            print("✅ Your phased QAT training should work without quantizer loss!")
        else:
            print("❌ SOME PHASE TRANSITIONS FAILED!")
            print("⚠️  Check the failed steps above for specific issues.")
        
        print("="*60)
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases that might cause issues."""
    
    print("\n🔬 Testing Edge Cases...")
    print("-" * 40)
    
    try:
        from src.models.yolov8_qat import QuantizedYOLOv8
        
        qat_model = QuantizedYOLOv8('yolov8n.pt', skip_detection_head=True)
        qat_model.prepare_for_qat()
        
        # Test 1: Rapid phase switching
        print("🔄 Test 1: Rapid Phase Switching...")
        rapid_phases = ["phase1_weight_only", "phase2_activations", "phase1_weight_only", "phase2_activations"]
        
        for phase in rapid_phases:
            success = qat_model._configure_quantizers_dynamically(phase)
            if not success:
                print(f"❌ Rapid switching failed at {phase}")
                return False
        
        print("✅ Rapid phase switching works")
        
        # Test 2: Emergency restoration - NOW WITH TORCH IMPORT!
        print("🚨 Test 2: Emergency Restoration...")
        if hasattr(qat_model, 'quantizer_manager'):
            # Force some quantizers to Identity (simulate loss)
            original_count = 0
            for name, module in qat_model.model.model.named_modules():
                if hasattr(module, 'weight_fake_quant') and original_count < 5:
                    module.weight_fake_quant = torch.nn.Identity()  # ← NOW TORCH IS IMPORTED!
                    original_count += 1
            
            print(f"   Simulated loss of {original_count} quantizers")
            
            # Try emergency restoration
            restored = qat_model.quantizer_manager.emergency_restore_all_quantizers()
            if restored:
                print("✅ Emergency restoration works")
            else:
                print("❌ Emergency restoration failed")
                return False
        
        # Test 3: State consistency check
        print("🔍 Test 3: State Consistency Check...")
        qat_model._configure_quantizers_dynamically("phase3_full_quant")
        
        if hasattr(qat_model, 'quantizer_manager'):
            state1 = qat_model.quantizer_manager.get_current_state()
            state2 = qat_model.quantizer_manager.get_current_state()
            
            if state1 == state2:
                print("✅ State consistency maintained")
            else:
                print("❌ State consistency broken")
                return False
        
        print("✅ All edge cases passed")
        return True
        
    except Exception as e:
        print(f"❌ Edge case testing failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 COMPREHENSIVE QUANTIZER FIX TESTING")
    print("="*60)
    
    # Test all phase transitions
    main_test_passed = test_all_phase_transitions()
    
    # Test edge cases
    edge_test_passed = test_edge_cases()
    
    overall_success = main_test_passed and edge_test_passed
    
    print("\n" + "="*60)
    print("🏁 FINAL TEST SUMMARY")
    print("="*60)
    print(f"📊 Main Phase Transitions: {'✅ PASSED' if main_test_passed else '❌ FAILED'}")
    print(f"🔬 Edge Cases: {'✅ PASSED' if edge_test_passed else '❌ FAILED'}")
    print(f"🎯 Overall Result: {'✅ ALL GOOD' if overall_success else '❌ NEEDS WORK'}")
    
    if overall_success:
        print("\n🚀 READY FOR TRAINING!")
        print("Run your phased QAT training with confidence:")
        print("python scripts/train_qat.py --model yolov8n.pt --data datasets/vietnam-traffic-sign-detection/dataset.yaml --epochs 5 --batch-size 8 --lr 0.0005 --qconfig default --phased-training --quant-penalty --fuse --save-dir models/checkpoints/qat/COMPREHENSIVE_TEST --device cpu")
    else:
        print("\n⚠️  ISSUES FOUND - Review failed tests above")
    
    print("="*60)
    
    exit(0 if overall_success else 1)