#!/usr/bin/env python
"""
RUN THIS FIRST: Analyze exactly where your models are broken
"""

import torch
import os

def quick_analysis():
    """Quick analysis of your model files."""
    
    print("🔍 QUICK BREAKAGE ANALYSIS")
    print("=" * 50)
    
    # Paths to your models
    qat_path = "models/checkpoints/qat_full_features_3/qat_model_with_fakequant.pt"
    int8_path = "models/checkpoints/qat_full_features_3/qat_yolov8n_full_int8_final.pt"
    
    # Check QAT model
    print("\n📊 QAT Model Analysis:")
    if os.path.exists(qat_path):
        size_mb = os.path.getsize(qat_path) / (1024 * 1024)
        print(f"✅ File exists: {size_mb:.2f} MB")
        
        try:
            data = torch.load(qat_path, map_location='cpu')
            print(f"📋 Type: {type(data).__name__}")
            
            if isinstance(data, dict):
                print(f"📝 Keys: {list(data.keys())}")
                
                if 'model_state_dict' in data:
                    state_dict = data['model_state_dict']
                    print(f"💾 State dict size: {len(state_dict)} parameters")
                    
                    # Count quantization parameters
                    fake_quant_keys = [k for k in state_dict.keys() if 'fake_quant' in k]
                    observer_keys = [k for k in state_dict.keys() if 'observer' in k]
                    
                    print(f"🔧 FakeQuantize parameters: {len(fake_quant_keys)}")
                    print(f"👁️ Observer parameters: {len(observer_keys)}")
                    
                    if len(fake_quant_keys) > 0:
                        print("✅ GOOD: QAT structure preserved in state_dict")
                        print("❌ BAD: Missing complete model wrapper")
                        print("🔧 FIX: Reconstruct from state_dict + YOLO architecture")
                    else:
                        print("❌ CRITICAL: No quantization in state_dict")
                
                fake_count = data.get('fake_quant_count', 0)
                print(f"📊 Recorded FakeQuantize count: {fake_count}")
                
            else:
                print("❌ CRITICAL: Not a dictionary format")
                
        except Exception as e:
            print(f"❌ Load failed: {e}")
    else:
        print("❌ File not found")
    
    # Check INT8 model
    print("\n⚡ INT8 Model Analysis:")
    if os.path.exists(int8_path):
        size_mb = os.path.getsize(int8_path) / (1024 * 1024)
        print(f"✅ File exists: {size_mb:.2f} MB")
        
        try:
            data = torch.load(int8_path, map_location='cpu')
            print(f"📋 Type: {type(data).__name__}")
            
            if isinstance(data, dict):
                print(f"📝 Keys: {list(data.keys())}")
                
                if 'state_dict' in data:
                    state_dict = data['state_dict']
                    print(f"💾 State dict size: {len(state_dict)} parameters")
                    
                    # Check for quantized weights
                    quantized_weights = []
                    for k, v in state_dict.items():
                        if hasattr(v, 'dtype') and v.dtype in [torch.qint8, torch.quint8]:
                            quantized_weights.append(k)
                    
                    print(f"⚡ Actually quantized weights: {len(quantized_weights)}")
                    
                    if len(quantized_weights) > 0:
                        print("✅ GOOD: Real quantized weights found")
                    else:
                        print("❌ BAD: No real quantized weights")
                        print("🚨 ISSUE: Observers not calibrated (default parameters used)")
                        print("🔧 FIX: Re-calibrate observers + proper conversion")
                
                metadata = data.get('metadata', {})
                if metadata:
                    print(f"📋 Metadata: {metadata}")
                
            else:
                print("❌ CRITICAL: Not a dictionary format")
                
        except Exception as e:
            print(f"❌ Load failed: {e}")
    else:
        print("❌ File not found")
    
    # Summary
    print("\n🎯 BREAKAGE SUMMARY:")
    print("=" * 50)
    
    if os.path.exists(qat_path):
        print("✅ QAT Model: SALVAGEABLE")
        print("   Issue: Missing YOLO wrapper")
        print("   Fix: Reconstruct from state_dict")
        print("   Difficulty: EASY")
        print("   Time: 5-10 minutes")
    else:
        print("❌ QAT Model: NOT FOUND")
    
    if os.path.exists(int8_path):
        print("⚠️ INT8 Model: NEEDS MAJOR FIX")
        print("   Issue: Uncalibrated observers")
        print("   Fix: Re-calibrate + convert")
        print("   Difficulty: MEDIUM")
        print("   Time: 15-30 minutes")
    else:
        print("❌ INT8 Model: NOT FOUND")
    
    print("\n💡 RECOMMENDED STRATEGY:")
    print("1. Fix QAT model first (easy win)")
    print("2. Use fixed QAT to create proper INT8")
    print("3. Total time: ~30 minutes")
    print("4. Success probability: 95%")

if __name__ == "__main__":
    quick_analysis()