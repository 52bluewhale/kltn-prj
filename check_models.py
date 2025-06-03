#!/usr/bin/env python
"""
Check if your saved models contain FakeQuantize modules
Run this to verify what's in your saved model files
"""

import torch
import os
from pathlib import Path

def check_model_for_quantizers(model_path):
    """Check if a saved model contains FakeQuantize modules."""
    print(f"\n🔍 Checking: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return False
    
    try:
        # Load the model
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ File loaded successfully")
        
        # Check file size
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"📦 File size: {file_size:.2f} MB")
        
        # Check what's in the checkpoint
        if isinstance(checkpoint, dict):
            print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
            
            # Look for model in different locations
            model = None
            if 'model' in checkpoint:
                model = checkpoint['model']
                print("✅ Found 'model' key in checkpoint")
            elif 'state_dict' in checkpoint:
                print("✅ Found 'state_dict' key in checkpoint")
                # This is a state dict, we need to reconstruct the model
                return check_state_dict_for_quantizers(checkpoint['state_dict'])
            else:
                print("⚠️ No standard model key found, trying to load as full model")
                model = checkpoint
        else:
            print("✅ Checkpoint is a direct model object")
            model = checkpoint
        
        if model is None:
            print("❌ Could not extract model from checkpoint")
            return False
        
        # Check if it's a YOLO model with .model attribute
        if hasattr(model, 'model'):
            print("✅ Found YOLO model with .model attribute")
            actual_model = model.model
        else:
            print("✅ Using model directly")
            actual_model = model
        
        # Count FakeQuantize modules
        fake_quant_count = 0
        weight_fake_quant_count = 0
        activation_post_process_count = 0
        
        for name, module in actual_model.named_modules():
            # Count FakeQuantize modules
            if 'FakeQuantize' in type(module).__name__:
                fake_quant_count += 1
            
            # Count modules with quantization attributes
            if hasattr(module, 'weight_fake_quant'):
                weight_fake_quant_count += 1
            
            if hasattr(module, 'activation_post_process'):
                activation_post_process_count += 1
        
        # Report results
        print(f"📊 Quantization Analysis:")
        print(f"  - FakeQuantize modules: {fake_quant_count}")
        print(f"  - Modules with weight_fake_quant: {weight_fake_quant_count}")
        print(f"  - Modules with activation_post_process: {activation_post_process_count}")
        
        # Check a few specific modules
        print(f"📋 Sample module inspection:")
        count = 0
        for name, module in actual_model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and count < 3:
                has_weight_fq = hasattr(module, 'weight_fake_quant')
                has_act_pp = hasattr(module, 'activation_post_process')
                print(f"  - {name}: weight_fake_quant={has_weight_fq}, activation_post_process={has_act_pp}")
                count += 1
        
        # Determine if this is a QAT model
        is_qat_model = fake_quant_count > 0 or (weight_fake_quant_count > 0 and activation_post_process_count > 0)
        
        if is_qat_model:
            print(f"🎉 SUCCESS: This IS a QAT model with quantization!")
            print(f"   Ready for: torch.quantization.convert() to INT8")
        else:
            print(f"❌ RESULT: This is NOT a QAT model (no quantization found)")
            print(f"   This is a regular FP32 model")
        
        return is_qat_model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def check_state_dict_for_quantizers(state_dict):
    """Check state dict for quantization-related parameters."""
    print(f"🔍 Checking state_dict for quantization parameters...")
    
    fake_quant_params = 0
    observer_params = 0
    scale_params = 0
    zero_point_params = 0
    
    for key in state_dict.keys():
        if 'fake_quant' in key:
            fake_quant_params += 1
        if 'observer' in key:
            observer_params += 1
        if '.scale' in key:
            scale_params += 1
        if 'zero_point' in key:
            zero_point_params += 1
    
    print(f"📊 State Dict Quantization Parameters:")
    print(f"  - fake_quant parameters: {fake_quant_params}")
    print(f"  - observer parameters: {observer_params}")
    print(f"  - scale parameters: {scale_params}")
    print(f"  - zero_point parameters: {zero_point_params}")
    
    # Show some example keys
    quantization_keys = [k for k in state_dict.keys() if any(q in k for q in ['fake_quant', 'observer', 'scale', 'zero_point'])]
    if quantization_keys:
        print(f"📋 Example quantization keys:")
        for key in quantization_keys[:5]:  # Show first 5
            print(f"  - {key}")
    
    has_quantization = fake_quant_params > 0 or observer_params > 0 or scale_params > 0
    
    if has_quantization:
        print(f"🎉 SUCCESS: State dict contains quantization parameters!")
    else:
        print(f"❌ RESULT: State dict does not contain quantization parameters")
    
    return has_quantization

def main():
    """Check all possible model locations."""
    print("🔍 CHECKING YOUR SAVED MODELS FOR FAKEQUANTIZE MODULES")
    print("=" * 80)
    
    # Define possible model locations
    base_dir = "models/checkpoints/qat_full_features"
    
    model_paths = [
        f"{base_dir}/weights/best.pt",
        f"{base_dir}/weights/last.pt", 
        f"{base_dir}/qat_model_with_fakequant.pt",  # This likely doesn't exist
    ]
    
    results = {}
    
    for model_path in model_paths:
        try:
            is_qat = check_model_for_quantizers(model_path)
            results[model_path] = is_qat
        except Exception as e:
            print(f"❌ Failed to check {model_path}: {e}")
            results[model_path] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY OF YOUR SAVED MODELS")
    print("=" * 80)
    
    qat_models = []
    regular_models = []
    missing_models = []
    
    for path, is_qat in results.items():
        if not os.path.exists(path):
            missing_models.append(path)
        elif is_qat:
            qat_models.append(path)
        else:
            regular_models.append(path)
    
    if qat_models:
        print("🎉 QAT MODELS FOUND (with FakeQuantize modules):")
        for model in qat_models:
            print(f"  ✅ {model}")
        print(f"\n💡 USE THESE FOR INT8 CONVERSION!")
    else:
        print("❌ NO QAT MODELS FOUND")
    
    if regular_models:
        print(f"\n📦 REGULAR FP32 MODELS FOUND:")
        for model in regular_models:
            print(f"  📁 {model}")
    
    if missing_models:
        print(f"\n❌ MISSING FILES:")
        for model in missing_models:
            print(f"  🚫 {model}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    if qat_models:
        print(f"1. ✅ Use {qat_models[0]} for INT8 conversion")
        print(f"2. ✅ This model contains your preserved quantizers")
        print(f"3. ✅ You can proceed with torch.quantization.convert()")
    else:
        print(f"1. ❌ Your quantizers were lost during YOLOv8's automatic save")
        print(f"2. 🔧 You need to use the fixed save method I provided")
        print(f"3. 🔄 Re-run training with the preservation fix")

if __name__ == "__main__":
    main()