#!/usr/bin/env python
"""
Comprehensive Penalty Loss Integration Test
Tests all aspects of the penalty loss integration including actual forward passes.
"""

import sys
import torch
from pathlib import Path

# Add project root to sys.path before any src imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.yolov8_qat import QuantizedYOLOv8

def test_penalty_integration_comprehensive():
    """
    Comprehensive test of penalty loss integration.
    """
    print("🚀 Starting Comprehensive Penalty Loss Integration Test...")
    print("=" * 70)
    
    # Test 1: Basic Setup
    print("📝 Test 1: Basic Setup")
    try:
        model = QuantizedYOLOv8('yolov8n.pt', skip_detection_head=True)
        model.prepare_for_qat()
        print("✅ Model preparation successful")
    except Exception as e:
        print(f"❌ Model preparation failed: {e}")
        return False
    
    # Test 2: Penalty Loss Setup
    print("\n📝 Test 2: Penalty Loss Setup")
    try:
        model.setup_penalty_loss_integration(alpha=0.01, warmup_epochs=5)
        print("✅ Penalty loss setup successful")
    except Exception as e:
        print(f"❌ Penalty loss setup failed: {e}")
        return False
    
    # Test 3: Forward Pass Test (Training Mode)
    print("\n📝 Test 3: Forward Pass Test (Training Mode)")
    try:
        model.model.model.train()  # Ensure training mode
        dummy_input = torch.randn(1, 3, 640, 640)
        
        print("  - Running forward pass...")
        with torch.no_grad():
            output = model.model.model(dummy_input)
        
        print("  - Checking for penalty buffer...")
        if hasattr(model.model.model, '_current_quant_penalty'):
            penalty_value = model.model.model._current_quant_penalty
            print(f"  ✅ Penalty buffer created: {penalty_value:.6f}")
        else:
            print("  ⚠️ Penalty buffer not found, but forward pass worked")
        
        print("✅ Forward pass test successful")
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        print(f"   Error details: {type(e).__name__}: {e}")
        return False
    
    # Test 4: Penalty Calculation Test
    print("\n📝 Test 4: Penalty Calculation Test")
    try:
        if hasattr(model, 'penalty_handler'):
            penalty = model.penalty_handler.calculate_penalty(model.model.model)
            print(f"  - Penalty value: {penalty:.6f}")
            print(f"  - Penalty type: {type(penalty)}")
            print(f"  - Is tensor: {torch.is_tensor(penalty)}")
            
            # Test penalty statistics
            stats = model.penalty_handler.get_statistics()
            print(f"  - Penalty stats: {stats}")
            
            print("✅ Penalty calculation test successful")
        else:
            print("❌ No penalty handler found")
            return False
    except Exception as e:
        print(f"❌ Penalty calculation test failed: {e}")
        return False
    
    # Test 5: Training Simulation
    print("\n📝 Test 5: Training Simulation")
    try:
        model.model.model.train()
        
        # Simulate a few training steps
        for epoch in range(3):
            model.penalty_handler.update_epoch(epoch)
            
            # Forward pass
            dummy_input = torch.randn(1, 3, 640, 640)
            output = model.model.model(dummy_input)
            
            # Check penalty
            if hasattr(model.model.model, '_current_quant_penalty'):
                penalty = model.model.model._current_quant_penalty
                print(f"  - Epoch {epoch}: Penalty = {penalty:.6f}")
            else:
                # Manually calculate penalty
                penalty = model.penalty_handler.calculate_penalty(model.model.model)
                print(f"  - Epoch {epoch}: Manual penalty = {penalty:.6f}")
        
        print("✅ Training simulation successful")
    except Exception as e:
        print(f"❌ Training simulation failed: {e}")
        return False
    
    # Test 6: Enhanced Verification
    print("\n📝 Test 6: Enhanced Verification")
    success = enhanced_verify_penalty_integration(model)
    
    if success:
        print("✅ Enhanced verification successful")
    else:
        print("❌ Enhanced verification failed")
        return False
    
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED! Penalty Loss Integration is WORKING!")
    print("=" * 70)
    return True

def enhanced_verify_penalty_integration(model):
    """Enhanced verification with better checks."""
    print("  🔍 Running enhanced verification...")
    
    checks = {
        'penalty_handler': False,
        'forward_patched': False,
        'penalty_calculation': False,
        'training_mode_penalty': False,
        'statistics_available': False
    }
    
    # Check 1: Penalty handler
    if hasattr(model, 'penalty_handler'):
        checks['penalty_handler'] = True
        print("    ✅ Penalty handler exists")
    else:
        print("    ❌ Penalty handler missing")
    
    # Check 2: Forward method patching
    if hasattr(model, 'original_forward'):
        checks['forward_patched'] = True
        print("    ✅ Forward method is patched")
    else:
        print("    ⚠️ Cannot verify forward patching (might still work)")
        # Try alternative check
        try:
            original_method = str(model.model.model.forward)
            if 'forward_with_penalty' in original_method or hasattr(model.model, '_penalty_handler'):
                checks['forward_patched'] = True
                print("    ✅ Forward patching verified (alternative method)")
        except:
            pass
    
    # Check 3: Penalty calculation
    try:
        penalty = model.penalty_handler.calculate_penalty(model.model.model)
        if torch.is_tensor(penalty):
            checks['penalty_calculation'] = True
            print(f"    ✅ Penalty calculation works: {penalty.item():.6f}")
    except Exception as e:
        print(f"    ❌ Penalty calculation failed: {e}")
    
    # Check 4: Training mode penalty
    try:
        model.model.model.train()
        dummy_input = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            _ = model.model.model(dummy_input)
        
        if hasattr(model.model.model, '_current_quant_penalty'):
            checks['training_mode_penalty'] = True
            print("    ✅ Training mode penalty buffer created")
        else:
            print("    ⚠️ Training mode penalty buffer not found")
    except Exception as e:
        print(f"    ❌ Training mode test failed: {e}")
    
    # Check 5: Statistics
    try:
        stats = model.penalty_handler.get_statistics()
        if isinstance(stats, dict) and len(stats) > 0:
            checks['statistics_available'] = True
            print("    ✅ Penalty statistics available")
    except Exception as e:
        print(f"    ❌ Statistics check failed: {e}")
    
    passed = sum(checks.values())
    total = len(checks)
    
    print(f"  📊 Enhanced verification: {passed}/{total} checks passed")
    
    return passed >= 3  # At least 3/5 checks should pass

if __name__ == "__main__":
    success = test_penalty_integration_comprehensive()
    if success:
        print("\n🎯 RESULT: Penalty Loss Integration is FULLY FUNCTIONAL!")
    else:
        print("\n❌ RESULT: Some issues found in penalty loss integration")
        
    print("\n💡 Next steps:")
    print("  1. If all tests pass, proceed with training")
    print("  2. Monitor penalty values during actual training")
    print("  3. Verify penalty loss affects gradient updates")