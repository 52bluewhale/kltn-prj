#!/usr/bin/env python3
"""
Phase 1 Runner: QAT Manual Training Loop Test
==============================================

Simple execution script for Phase 1 testing.
Run this to validate that manual training loops preserve QAT quantization.

Usage:
    python scripts/manual_qat_training/run_phase1.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import our Phase 1 implementation
from scripts.manual_qat_training.phase1_basic_qat_test import phase1_main, generate_phase1_report
from scripts.manual_qat_training.utils.qat_validation import QATValidator

def main():
    """Execute Phase 1 with comprehensive reporting."""
    
    print("🚀 MANUAL QAT TRAINING - PHASE 1 EXECUTION")
    print("=" * 60)
    print("Objective: Prove manual training preserves QAT quantization")
    print("Expected: All 90 FakeQuantize modules preserved throughout training")
    print("=" * 60)
    
    # Validate environment
    try:
        import torch
        import ultralytics
        print(f"✅ Environment check:")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   Ultralytics: {ultralytics.__version__}")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
    except ImportError as e:
        print(f"❌ Environment check failed: {e}")
        return False
    
    # Execute Phase 1
    try:
        print(f"\n🎯 Starting Phase 1 execution...")
        success = phase1_main()
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 PHASE 1 EXECUTION: COMPLETE SUCCESS!")
            print("=" * 60)
            print("✅ Key Achievements:")
            print("   1. Manual training loop implemented successfully")
            print("   2. All 90 FakeQuantize modules preserved throughout training")
            print("   3. QAT model saved and reloaded successfully")
            print("   4. Approach validated for proceeding to Phase 2")
            
            print(f"\n🎯 Next Steps:")
            print("   1. Proceed to Phase 2: YOLOv8 Loss Integration")
            print("   2. Replace simple MSE loss with real YOLOv8 detection loss")
            print("   3. Validate training effectiveness with real loss functions")
            
            print(f"\n📁 Output Files:")
            print("   - QAT Model: F:/kltn-prj/models/checkpoints/phase1_test/qat_manual_trained.pt")
            print("   - Contains: Model with preserved quantization structure")
            
            return True
        else:
            print("\n" + "=" * 60)
            print("❌ PHASE 1 EXECUTION: FAILED")
            print("=" * 60)
            print("Issues identified:")
            print("   - Manual training loop did not preserve QAT quantization")
            print("   - Requires debugging before proceeding to Phase 2")
            
            print(f"\n🔧 Debugging Steps:")
            print("   1. Check QAT model preparation")
            print("   2. Validate FakeQuantize module creation")
            print("   3. Inspect training step operations")
            
            return False
            
    except Exception as e:
        print(f"\n❌ Phase 1 execution failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_diagnostic_only():
    """Run just the diagnostic part without full training."""
    print("🔍 RUNNING DIAGNOSTIC MODE ONLY")
    print("=" * 40)
    
    # This would be useful for quick testing
    # Implementation would go here
    pass

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 1: Manual QAT Training Test')
    parser.add_argument('--diagnostic-only', action='store_true', 
                       help='Run diagnostic checks only')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs (default: 2)')
    parser.add_argument('--batches', type=int, default=3,
                       help='Batches per epoch (default: 3)')
    
    args = parser.parse_args()
    
    if args.diagnostic_only:
        run_diagnostic_only()
    else:
        # For this phase, we'll use the hardcoded values in phase1_main()
        # In later phases, we can use args.epochs and args.batches
        success = main()
        sys.exit(0 if success else 1)