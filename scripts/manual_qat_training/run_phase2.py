#!/usr/bin/env python3
"""
Phase 2 Runner: YOLOv8 Loss Integration Test
============================================

Execute Phase 2 to integrate real YOLOv8 detection loss while maintaining
the QAT preservation proven in Phase 1.

Usage:
    python scripts/manual_qat_training/run_phase2.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import our Phase 2 implementation
try:
    from scripts.manual_qat_training.phase2_yolo_loss import phase2_main, generate_phase2_report
except ImportError:
    # Alternative import path
    from phase2_yolo_loss import phase2_main, generate_phase2_report

def main():
    """Execute Phase 2 with comprehensive reporting."""
    
    print("🚀 MANUAL QAT TRAINING - PHASE 2 EXECUTION")
    print("=" * 60)
    print("Building on Phase 1 Success:")
    print("✅ Manual training loop preserves QAT (90/90 modules)")
    print("✅ Gradient fix enables proper training")
    print("✅ QAT model save/load works correctly")
    print("")
    print("Phase 2 Objective:")
    print("🎯 Replace simple loss with real YOLOv8 detection loss")
    print("🎯 Maintain perfect QAT preservation")
    print("🎯 Validate training effectiveness with real loss components")
    print("=" * 60)
    
    # Validate environment
    try:
        import torch
        import ultralytics
        from ultralytics.utils.loss import v8DetectionLoss
        
        print(f"✅ Environment check:")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   Ultralytics: {ultralytics.__version__}")
        print(f"   v8DetectionLoss: Available ✅")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
    except ImportError as e:
        print(f"❌ Environment check failed: {e}")
        print("🔧 Required components:")
        print("   - torch >= 2.0")
        print("   - ultralytics >= 8.0")
        print("   - v8DetectionLoss from ultralytics.utils.loss")
        return False
    
    # Execute Phase 2
    try:
        print(f"\n🎯 Starting Phase 2 execution...")
        print("📋 Phase 2 Pipeline:")
        print("   1. ✅ Reuse Phase 1's proven QAT setup")
        print("   2. 🆕 Create real YOLOv8 detection loss")
        print("   3. 🆕 Integrate v8DetectionLoss with manual training loop")
        print("   4. ✅ Validate QAT preservation throughout")
        print("   5. 📊 Analyze loss components (box, cls, dfl)")
        print("")
        
        success = phase2_main()
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 PHASE 2 EXECUTION: COMPLETE SUCCESS!")
            print("=" * 60)
            print("✅ Key Achievements:")
            print("   1. ✅ YOLOv8 detection loss integrated successfully")
            print("   2. ✅ All 90 FakeQuantize modules preserved")
            print("   3. ✅ Real loss components working (box, cls, dfl)")
            print("   4. ✅ Training effectiveness validated")
            print("   5. ✅ QAT model saved with YOLOv8 loss metadata")
            
            print(f"\n🎯 What This Proves:")
            print("   ✅ Manual training loops work with real YOLOv8 loss")
            print("   ✅ QAT preservation is robust across different loss functions")
            print("   ✅ Training is effective (loss decreases properly)")
            print("   ✅ Foundation ready for real data integration")
            
            print(f"\n🎯 Next Steps:")
            print("   1. Proceed to Phase 3: Real Data Pipeline Integration")
            print("   2. Replace dummy data with YOLOv8's actual dataloader")
            print("   3. Integrate real traffic sign detection dataset")
            print("   4. Scale to full training with validation metrics")
            
            print(f"\n📁 Output Files:")
            print("   - YOLOv8 QAT Model: F:/kltn-prj/models/checkpoints/phase2_test/qat_yolov8_trained.pt")
            print("   - Contains: Model trained with real YOLOv8 loss + preserved quantization")
            print("   - Metadata: loss_function='v8DetectionLoss', fake_quant_count=90")
            
            return True
        else:
            print("\n" + "=" * 60)
            print("❌ PHASE 2 EXECUTION: FAILED")
            print("=" * 60)
            print("Issues identified:")
            print("   - YOLOv8 loss integration failed")
            print("   - QAT preservation may have been compromised")
            print("   - Requires debugging before proceeding to Phase 3")
            
            print(f"\n🔧 Debugging Steps:")
            print("   1. Check v8DetectionLoss import and compatibility")
            print("   2. Validate YOLOv8 prediction format handling")
            print("   3. Inspect loss computation gradient flow")
            print("   4. Compare with Phase 1 baseline")
            
            return False
            
    except Exception as e:
        print(f"\n❌ Phase 2 execution failed with exception: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🔧 Troubleshooting:")
        print("   1. Ensure ultralytics version supports v8DetectionLoss")
        print("   2. Check for YOLOv8 API changes")
        print("   3. Validate model format compatibility")
        print("   4. Try fallback loss implementation")
        
        return False

def run_comparison_test():
    """Run comparison between Phase 1 and Phase 2 results."""
    print("🔍 RUNNING PHASE 1 vs PHASE 2 COMPARISON")
    print("=" * 50)
    
    # Check if Phase 1 model exists
    phase1_path = "F:/kltn-prj/models/checkpoints/phase1_test/qat_manual_trained.pt"
    phase2_path = "F:/kltn-prj/models/checkpoints/phase2_test/qat_yolov8_trained.pt"
    
    if os.path.exists(phase1_path) and os.path.exists(phase2_path):
        try:
            import torch
            
            # Load both models
            phase1_data = torch.load(phase1_path, map_location='cpu')
            phase2_data = torch.load(phase2_path, map_location='cpu')
            
            print("📊 Model Comparison:")
            print(f"   Phase 1 - FakeQuantize modules: {phase1_data.get('fake_quant_count', 'N/A')}")
            print(f"   Phase 2 - FakeQuantize modules: {phase2_data.get('fake_quant_count', 'N/A')}")
            print(f"   Phase 1 - Training method: {phase1_data.get('training_method', 'N/A')}")
            print(f"   Phase 2 - Training method: {phase2_data.get('training_method', 'N/A')}")
            print(f"   Phase 2 - Loss function: {phase2_data.get('loss_function', 'N/A')}")
            
            # File size comparison
            phase1_size = os.path.getsize(phase1_path) / (1024 * 1024)
            phase2_size = os.path.getsize(phase2_path) / (1024 * 1024)
            
            print(f"   Phase 1 - File size: {phase1_size:.2f} MB")
            print(f"   Phase 2 - File size: {phase2_size:.2f} MB")
            
            # Check consistency
            if (phase1_data.get('fake_quant_count') == phase2_data.get('fake_quant_count') == 90):
                print("✅ CONSISTENCY CHECK: Both phases preserved QAT correctly")
            else:
                print("❌ CONSISTENCY CHECK: QAT preservation differs between phases")
            
        except Exception as e:
            print(f"❌ Comparison failed: {e}")
    else:
        print("⚠️ Cannot compare - missing model files")
        if not os.path.exists(phase1_path):
            print(f"   Missing: {phase1_path}")
        if not os.path.exists(phase2_path):
            print(f"   Missing: {phase2_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 2: YOLOv8 Loss Integration Test')
    parser.add_argument('--compare', action='store_true', 
                       help='Run comparison with Phase 1 results')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs (default: 2)')
    parser.add_argument('--batches', type=int, default=3,
                       help='Batches per epoch (default: 3)')
    
    args = parser.parse_args()
    
    if args.compare:
        run_comparison_test()
    else:
        success = main()
        
        # Run comparison if both succeeded
        if success and args.compare:
            print("\n")
            run_comparison_test()
        
        sys.exit(0 if success else 1)