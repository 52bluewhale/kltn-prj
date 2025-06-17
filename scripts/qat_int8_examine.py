#!/usr/bin/env python
"""
INT8 Model Content Examiner
Comprehensive inspection of the quantized YOLOv8 model
"""
import torch
import os
from pathlib import Path
import json
from collections import OrderedDict

def examine_int8_model(model_path):
    """
    Comprehensive examination of the INT8 model file.
    
    Args:
        model_path: Path to the INT8 model file
    """
    print("="*80)
    print("🔍 COMPREHENSIVE INT8 MODEL EXAMINATION")
    print("="*80)
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found at {model_path}")
        return None
    
    # Basic file info
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"📁 File Info:")
    print(f"   Path: {model_path}")
    print(f"   Size: {file_size:.2f} MB")
    print()
    
    try:
        # Load the model
        print("📦 Loading model...")
        saved_data = torch.load(model_path, map_location='cpu')
        print(f"✅ Model loaded successfully")
        print(f"📊 Top-level type: {type(saved_data)}")
        print()
        
        # Examine top-level structure
        print("🔍 TOP-LEVEL STRUCTURE:")
        if isinstance(saved_data, dict):
            print(f"   Dictionary with {len(saved_data)} keys:")
            for key in saved_data.keys():
                value = saved_data[key]
                print(f"   - '{key}': {type(value)} {get_shape_info(value)}")
        else:
            print(f"   Direct object: {type(saved_data)}")
            if hasattr(saved_data, 'state_dict'):
                print("   Has state_dict method")
            if hasattr(saved_data, '__dict__'):
                attrs = list(saved_data.__dict__.keys())[:10]  # First 10 attributes
                print(f"   Attributes: {attrs}...")
        print()
        
        # Look for the actual model
        model_object = None
        state_dict = None
        metadata = None
        
        if isinstance(saved_data, dict):
            # Check for different possible keys
            possible_model_keys = ['model', 'model_state_dict', 'state_dict', 'quantized_model']
            possible_metadata_keys = ['metadata', 'qat_info', 'info', 'config']
            
            for key in possible_model_keys:
                if key in saved_data:
                    model_object = saved_data[key]
                    print(f"🎯 Found model at key: '{key}'")
                    print(f"   Type: {type(model_object)}")
                    break
            
            for key in possible_metadata_keys:
                if key in saved_data:
                    metadata = saved_data[key]
                    print(f"📋 Found metadata at key: '{key}'")
                    break
        else:
            # Direct model object
            model_object = saved_data
        
        # Examine the model object
        if model_object is not None:
            print("\n🔍 MODEL OBJECT ANALYSIS:")
            print(f"   Type: {type(model_object)}")
            
            # Try to get state_dict
            if hasattr(model_object, 'state_dict'):
                print("   ✅ Has state_dict() method")
                state_dict = model_object.state_dict()
            elif isinstance(model_object, dict):
                print("   📊 Model object is already a state_dict")
                state_dict = model_object
            else:
                print("   ❓ Unclear how to get state_dict")
            
            # Check for quantization-specific attributes
            if hasattr(model_object, 'named_modules'):
                print("   ✅ Has named_modules() method")
                quantized_modules = []
                for name, module in model_object.named_modules():
                    if 'quantized' in type(module).__name__.lower():
                        quantized_modules.append((name, type(module).__name__))
                
                if quantized_modules:
                    print(f"   🎯 Found {len(quantized_modules)} quantized modules:")
                    for name, module_type in quantized_modules[:5]:  # Show first 5
                        print(f"      - {name}: {module_type}")
                    if len(quantized_modules) > 5:
                        print(f"      ... and {len(quantized_modules) - 5} more")
                else:
                    print("   ❓ No obviously quantized modules found")
        
        # Examine state_dict if we have it
        if state_dict is not None:
            print(f"\n🔍 STATE_DICT ANALYSIS:")
            print(f"   Total parameters: {len(state_dict)}")
            
            # Categorize parameters
            weight_params = []
            bias_params = []
            bn_params = []
            quantization_params = []
            other_params = []
            
            for name, param in state_dict.items():
                if 'weight' in name:
                    weight_params.append(name)
                elif 'bias' in name:
                    bias_params.append(name)
                elif any(bn_key in name for bn_key in ['running_mean', 'running_var', 'num_batches_tracked']):
                    bn_params.append(name)
                elif any(quant_key in name for quant_key in ['scale', 'zero_point', '_packed_params']):
                    quantization_params.append(name)
                else:
                    other_params.append(name)
            
            print(f"   📊 Parameter categories:")
            print(f"      - Weight parameters: {len(weight_params)}")
            print(f"      - Bias parameters: {len(bias_params)}")
            print(f"      - BatchNorm parameters: {len(bn_params)}")
            print(f"      - Quantization parameters: {len(quantization_params)}")
            print(f"      - Other parameters: {len(other_params)}")
            
            # Show examples of each category
            if quantization_params:
                print(f"\n   🎯 QUANTIZATION PARAMETERS (first 10):")
                for i, name in enumerate(quantization_params[:10]):
                    param = state_dict[name]
                    print(f"      {i+1}. {name}: {param.shape if hasattr(param, 'shape') else type(param)}")
            
            if weight_params:
                print(f"\n   ⚖️ WEIGHT PARAMETERS (first 10):")
                for i, name in enumerate(weight_params[:10]):
                    param = state_dict[name]
                    print(f"      {i+1}. {name}: {param.shape if hasattr(param, 'shape') else type(param)}")
        
        # Examine metadata
        if metadata is not None:
            print(f"\n🔍 METADATA ANALYSIS:")
            print(f"   Type: {type(metadata)}")
            if isinstance(metadata, dict):
                print(f"   Keys: {list(metadata.keys())}")
                for key, value in metadata.items():
                    if isinstance(value, (dict, list)):
                        print(f"   - {key}: {type(value)} with {len(value)} items")
                    else:
                        print(f"   - {key}: {value}")
        
        # Test basic inference capability
        print(f"\n🧪 BASIC INFERENCE TEST:")
        try:
            if model_object is not None and hasattr(model_object, '__call__'):
                # Try to create a dummy input
                dummy_input = torch.randn(1, 3, 640, 640)
                print(f"   📥 Created dummy input: {dummy_input.shape}")
                
                # Set to eval mode
                if hasattr(model_object, 'eval'):
                    model_object.eval()
                    print("   ✅ Set model to eval mode")
                
                # Try inference
                with torch.no_grad():
                    try:
                        output = model_object(dummy_input)
                        print(f"   ✅ Inference successful!")
                        if isinstance(output, (list, tuple)):
                            print(f"   📤 Output: {len(output)} tensors")
                            for i, out in enumerate(output):
                                if hasattr(out, 'shape'):
                                    print(f"      - Output {i}: {out.shape}")
                        else:
                            if hasattr(output, 'shape'):
                                print(f"   📤 Output shape: {output.shape}")
                            else:
                                print(f"   📤 Output type: {type(output)}")
                    except Exception as e:
                        print(f"   ❌ Inference failed: {e}")
            else:
                print("   ❓ Cannot test inference - no callable model found")
        except Exception as e:
            print(f"   ❌ Inference test error: {e}")
        
        print("\n" + "="*80)
        print("🎯 SUMMARY & RECOMMENDATIONS:")
        print("="*80)
        
        # Generate recommendations based on findings
        recommendations = []
        
        if state_dict is not None:
            recommendations.append("✅ State dict accessible - can rebuild model architecture")
        
        if len(quantization_params) > 0:
            recommendations.append(f"✅ Found {len(quantization_params)} quantization parameters - model is quantized")
        
        if model_object is not None and hasattr(model_object, '__call__'):
            recommendations.append("✅ Model is callable - can run inference directly")
        else:
            recommendations.append("⚠️ Need to reconstruct model architecture for inference")
        
        if metadata is not None:
            recommendations.append("✅ Metadata available - contains training configuration")
        
        for rec in recommendations:
            print(f"   {rec}")
        
        print(f"\n📋 NEXT STEPS:")
        if hasattr(model_object, '__call__') and model_object is not None:
            print("   1. ✅ Model can be used directly for inference")
            print("   2. 🔧 Build preprocessing pipeline")
            print("   3. 🔧 Build postprocessing pipeline") 
            print("   4. 📊 Implement evaluation metrics")
        else:
            print("   1. 🔧 Reconstruct YOLOv8 architecture")
            print("   2. 🔧 Load state_dict into architecture")
            print("   3. 🔧 Build preprocessing pipeline")
            print("   4. 🔧 Build postprocessing pipeline")
            print("   5. 📊 Implement evaluation metrics")
        
        return {
            'saved_data': saved_data,
            'model_object': model_object,
            'state_dict': state_dict,
            'metadata': metadata,
            'file_info': {
                'path': model_path,
                'size_mb': file_size
            },
            'analysis': {
                'has_model': model_object is not None,
                'has_state_dict': state_dict is not None,
                'has_metadata': metadata is not None,
                'quantization_params_count': len(quantization_params) if 'quantization_params' in locals() else 0,
                'is_callable': hasattr(model_object, '__call__') if model_object else False
            }
        }
        
    except Exception as e:
        print(f"❌ ERROR during examination: {e}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return None

def get_shape_info(obj):
    """Helper function to get shape/size information about objects."""
    if hasattr(obj, 'shape'):
        return f"shape={obj.shape}"
    elif hasattr(obj, '__len__'):
        try:
            return f"len={len(obj)}"
        except:
            return ""
    else:
        return ""

# Example usage
if __name__ == "__main__":
    # Update this path to match your model location
    model_path = "models/checkpoints/qat_full_features_6/qat_yolov8n_full_int8_final.pt"
    
    result = examine_int8_model(model_path)
    
    if result:
        print(f"\n🎉 Examination completed successfully!")
        print(f"🔧 Ready to proceed with custom evaluation pipeline")
    else:
        print(f"\n💥 Examination failed - check the error messages above")