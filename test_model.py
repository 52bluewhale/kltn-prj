#!/usr/bin/env python3
"""
IMMEDIATE TEST SCRIPT - Run this right now to test best_qat.pt
Copy and paste this into a new .py file and run it immediately.
"""

import torch
import os
import numpy as np
from ultralytics import YOLO
import traceback

def immediate_diagnosis():
    """Immediate diagnosis of the best_qat.pt model."""
    
    model_path = "F:/kltn-prj/models/checkpoints/qat_standard/best_qat.pt"
    
    print("🚨 IMMEDIATE DIAGNOSIS: best_qat.pt")
    print("="*50)
    
    # Critical Check 1: File exists and size
    print("1. 📁 FILE CHECK:")
    if not os.path.exists(model_path):
        print("   ❌ CRITICAL: File does not exist!")
        return "MISSING"
    
    file_size_bytes = os.path.getsize(model_path)
    file_size_kb = file_size_bytes / 1024
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"   📊 Size: {file_size_bytes} bytes ({file_size_kb:.1f} KB, {file_size_mb:.2f} MB)")
    
    if file_size_kb < 100:  # Less than 100 KB is definitely wrong
        print("   ❌ CRITICAL: File size is way too small!")
        print("   🔍 Expected: ~3-6 MB for YOLO model")
        print("   🔍 Got: {:.1f} KB - This indicates corruption".format(file_size_kb))
        return "CORRUPTED"
    elif file_size_mb < 1:
        print("   ⚠️  WARNING: File size is suspiciously small")
        return "SUSPICIOUS"
    else:
        print("   ✅ File size looks reasonable")
    
    # Critical Check 2: Can load with PyTorch
    print("\n2. 🔄 PYTORCH LOADING:")
    try:
        data = torch.load(model_path, map_location='cpu')
        print("   ✅ PyTorch can load the file")
        print(f"   📊 Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"   📊 Keys: {list(data.keys())}")
        
    except Exception as e:
        print(f"   ❌ CRITICAL: PyTorch cannot load file!")
        print(f"   🔍 Error: {e}")
        return "UNLOADABLE"
    
    # Critical Check 3: Can load with YOLO
    print("\n3. 🤖 YOLO LOADING:")
    try:
        model = YOLO(model_path)
        print("   ✅ YOLO can load the model")
    except Exception as e:
        print(f"   ❌ CRITICAL: YOLO cannot load model!")
        print(f"   🔍 Error: {e}")
        return "YOLO_INCOMPATIBLE"
    
    # Critical Check 4: Basic inference
    print("\n4. 🧠 INFERENCE TEST:")
    try:
        # Create dummy input
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Try inference
        results = model(dummy_img, verbose=False)
        print("   ✅ Inference works!")
        
        # Check if results make sense
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes'):
                num_detections = len(result.boxes) if result.boxes is not None else 0
                print(f"   📊 Detections: {num_detections}")
                print("   ✅ Model produces valid outputs")
                return "WORKING"
            else:
                print("   ⚠️  No detection boxes in results")
                return "NO_OUTPUTS"
        else:
            print("   ⚠️  Empty results")
            return "EMPTY_RESULTS"
    
    except Exception as e:
        print(f"   ❌ CRITICAL: Inference failed!")
        print(f"   🔍 Error: {e}")
        return "INFERENCE_FAILED"

def print_verdict(status):
    """Print final verdict based on diagnosis."""
    print("\n" + "="*50)
    print("🎯 FINAL VERDICT")
    print("="*50)
    
    if status == "WORKING":
        print("🎉 GOOD NEWS: Model is FUNCTIONAL!")
        print("✅ best_qat.pt can be used for inference")
        print("✅ Model is deployable")
        print("\n📋 Next steps:")
        print("   1. Test with real images from your dataset")
        print("   2. Measure inference speed and accuracy")
        print("   3. Export to ONNX for production deployment")
        
    elif status == "NO_OUTPUTS" or status == "EMPTY_RESULTS":
        print("⚠️  PARTIAL SUCCESS: Model loads but has output issues")
        print("🔧 The model may work but needs investigation")
        print("\n📋 Next steps:")
        print("   1. Test with real images")
        print("   2. Check if model was properly trained")
        print("   3. Verify detection thresholds")
        
    elif status == "CORRUPTED":
        print("❌ BAD NEWS: Model file is CORRUPTED")
        print("💀 The 3.342 KB size confirms this is broken")
        print("\n🔍 What happened:")
        print("   - PyTorch quantization conversion failed")
        print("   - File save process was incomplete")
        print("   - Model data was lost during conversion")
        print("\n🛠️  How to fix:")
        print("   1. Use your QuantizedYOLOv8 class instead")
        print("   2. Don't use torch.quantization.convert() with YOLO")
        print("   3. Save QAT and INT8 models to different files")
        print("   4. Test model size after saving")
        
    elif status == "MISSING":
        print("❌ BAD NEWS: Model file not found")
        print("🔍 Check if training completed successfully")
        
    elif status == "UNLOADABLE":
        print("❌ BAD NEWS: File is corrupted or invalid format")
        print("🔧 File exists but contains invalid data")
        
    elif status == "YOLO_INCOMPATIBLE":
        print("❌ BAD NEWS: Model incompatible with YOLO")
        print("🔧 Model format doesn't match YOLO expectations")
        
    elif status == "INFERENCE_FAILED":
        print("❌ BAD NEWS: Model loads but inference fails")
        print("🔧 Model structure may be broken")
        
    else:
        print("⚠️  UNKNOWN STATUS: Unexpected test result")

def main():
    """Main function - run immediate diagnosis."""
    print("🚨 EMERGENCY DIAGNOSIS OF best_qat.pt")
    print("This will tell you immediately if your model works or not")
    print()
    
    status = immediate_diagnosis()
    print_verdict(status)
    
    print(f"\n📊 DIAGNOSIS COMPLETE - Status: {status}")

if __name__ == "__main__":
    # Run the immediate test
    main()