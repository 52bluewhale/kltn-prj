#!/usr/bin/env python
"""
Simple script to check if a model is quantized by analyzing the saved tensors directly
Usage: python simple_quantization_checker.py model.pt
"""
import sys
import torch
import os

def check_quantization(model_path):
    """Simple check for quantized tensors in a model file."""
    
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return False
    
    try:
        # Load the model file
        print(f"🔍 Loading: {model_path}")
        try:
            saved_data = torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            saved_data = torch.load(model_path, map_location='cpu')
        
        # Get file size
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"📏 File size: {file_size_mb:.2f} MB")
        
        # Extract state_dict
        state_dict = None
        if isinstance(saved_data, dict):
            if 'state_dict' in saved_data:
                state_dict = saved_data['state_dict']
            elif 'model' in saved_data:
                # Try to get state_dict from model
                model = saved_data['model']
                if hasattr(model, 'state_dict'):
                    state_dict = model.state_dict()
        else:
            # Direct model object
            if hasattr(saved_data, 'state_dict'):
                state_dict = saved_data.state_dict()
        
        if state_dict is None:
            print("❌ Could not extract state_dict")
            return False
        
        # Analyze tensors
        total_tensors = 0
        quantized_tensors = 0
        quantized_params = 0
        total_params = 0
        
        print(f"\n🔬 ANALYZING TENSORS...")
        print("-" * 60)
        
        for name, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                total_tensors += 1
                total_params += tensor.numel()
                
                if tensor.is_quantized:
                    quantized_tensors += 1
                    quantized_params += tensor.numel()
                    
                    # Show details for first few quantized tensors
                    if quantized_tensors <= 5:
                        print(f"✅ {name[:45]:45} | {str(tensor.dtype):15} | {tensor.shape}")
                        
                        # Safely get quantization parameters
                        try:
                            scale = tensor.q_scale()
                            zero_point = tensor.q_zero_point()
                            qscheme = tensor.qscheme()
                            print(f"   Scale: {scale:.6f}, Zero Point: {zero_point}, Scheme: {qscheme}")
                        except Exception as e:
                            # Handle per-channel quantization or other schemes
                            try:
                                scales = tensor.q_per_channel_scales()
                                zero_points = tensor.q_per_channel_zero_points()
                                qscheme = tensor.qscheme()
                                print(f"   Per-channel: {len(scales)} scales, Scheme: {qscheme}")
                            except:
                                print(f"   Quantization details: {tensor.dtype} (scheme details unavailable)")
        
        if quantized_tensors > 5:
            print(f"... and {quantized_tensors - 5} more quantized tensors")
        
        # Results
        print(f"\n📊 SUMMARY")
        print("-" * 60)
        print(f"Total tensors: {total_tensors}")
        print(f"Quantized tensors: {quantized_tensors}")
        print(f"Total parameters: {total_params:,}")
        print(f"Quantized parameters: {quantized_params:,}")
        
        if quantized_tensors > 0:
            quant_ratio = quantized_params / total_params * 100
            print(f"Quantization ratio: {quant_ratio:.1f}%")
        
        # Final verdict
        print(f"\n🎯 VERDICT")
        print("=" * 60)
        if quantized_tensors > 0:
            print("✅ THIS MODEL IS QUANTIZED (INT8)")
            print(f"   Found {quantized_tensors} quantized tensors")
            print(f"   {quantized_params:,} parameters are quantized")
            
            # Size analysis
            if file_size_mb < 8:  # YOLOv8n FP32 is ~12MB
                print(f"   Small file size ({file_size_mb:.2f} MB) confirms compression")
            
            return True
        else:
            print("❌ THIS MODEL IS NOT QUANTIZED")
            print("   No quantized tensors found")
            print("   This appears to be a regular FP32 model")
            return False
            
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python simple_quantization_checker.py model.pt")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    print("🔍 SIMPLE QUANTIZATION CHECKER")
    print("=" * 60)
    
    is_quantized = check_quantization(model_path)
    
    # Exit code: 0 if quantized, 1 if not
    sys.exit(0 if is_quantized else 1)

if __name__ == "__main__":
    main()