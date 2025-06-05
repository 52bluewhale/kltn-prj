#!/usr/bin/env python
"""Test the fixes"""

import torch
import os
from src.models.yolov8_qat import QuantizedYOLOv8

def test_fixes():
    print("🧪 Testing QAT fixes...")
    
    # Test 1: Create QAT model
    qat_model = QuantizedYOLOv8('yolov8n.pt', skip_detection_head=True)
    qat_model.prepare_for_qat_with_preservation()
    
    # Test 2: Check observer design
    if hasattr(qat_model, 'quantizer_preserver'):
        stats = qat_model.quantizer_preserver.get_quantizer_stats()
        print(f"✅ Fixed design active: {stats.get('observers_always_active', False)}")
    
    # Test 3: Test save/load
    test_path = 'test_qat_model.pt'
    save_success = qat_model.save(test_path, preserve_qat=True)
    print(f"✅ Save test: {'PASSED' if save_success else 'FAILED'}")
    
    if save_success:
        loaded_model = QuantizedYOLOv8.load_qat_model(test_path)
        print(f"✅ Load test: {'PASSED' if loaded_model else 'FAILED'}")
        
        if loaded_model:
            ready = loaded_model.verify_qat_model_ready()
            print(f"✅ Ready test: {'PASSED' if ready else 'FAILED'}")
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("🎉 Fix testing complete!")

if __name__ == "__main__":
    test_fixes()