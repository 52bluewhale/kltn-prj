#!/usr/bin/env python
"""
Quick test script to verify the save fix works
Run this after updating your methods
"""

import sys
import os
sys.path.append('.')

from src.models.yolov8_qat import QuantizedYOLOv8

def test_save_fix():
    print("🧪 Testing QAT Save Fix...")
    
    # Create a small test model
    qat_model = QuantizedYOLOv8('yolov8n.pt', skip_detection_head=True)
    
    # Prepare for QAT with preservation
    print("⚙️ Preparing model with preservation...")
    qat_model.prepare_for_qat_with_preservation()
    
    # Check if quantizers exist
    if hasattr(qat_model, 'quantizer_preserver'):
        stats = qat_model.quantizer_preserver.get_quantizer_stats()
        print(f"📊 Model prepared with {stats['total_fake_quantizers']} FakeQuantize modules")
        
        if stats['total_fake_quantizers'] == 0:
            print("❌ No quantizers found - preparation failed")
            return False
    else:
        print("❌ Quantizer preserver not found")
        return False
    
    # Test save
    save_path = "models/test_save_fix.pt"
    print(f"💾 Testing save to: {save_path}")
    
    success = qat_model.save(save_path, preserve_qat=True)
    
    if success:
        print("✅ Save successful - now testing load...")
        
        # Test that the saved file contains quantizers
        import torch
        try:
            saved_data = torch.load(save_path, map_location='cpu')
            if 'fake_quant_count' in saved_data:
                count = saved_data['fake_quant_count']
                print(f"✅ Saved file contains {count} FakeQuantize modules")
                
                if count > 0:
                    print("🎉 SAVE FIX TEST PASSED!")
                    return True
                else:
                    print("❌ Save fix test failed - no quantizers in saved file")
                    return False
            else:
                print("❌ Save fix test failed - wrong save format")
                return False
                
        except Exception as e:
            print(f"❌ Error loading saved file: {e}")
            return False
    else:
        print("❌ Save failed")
        return False

if __name__ == "__main__":
    success = test_save_fix()
    if success:
        print("\n🎯 Your save fix is working correctly!")
        print("🚀 You can now run full training with confidence")
    else:
        print("\n💥 Save fix needs more work")
        print("🔧 Check the error messages above")