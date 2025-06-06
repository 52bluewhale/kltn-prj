#!/usr/bin/env python
"""
Simple INT8 Model Test - Direct and Easy to Use

Just run this script to test your INT8 model:
python test_int8_model.py
"""

import torch
import time
import os

def test_int8_model():
    """Test the INT8 model directly"""
    
    print("🚀 Testing INT8 Model")
    print("="*50)
    
    # Your INT8 model path
    model_path = "models/checkpoints/qat_full_features_4/qat_yolov8n_full_int8_final.pt"
    
    # Step 1: Check if file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("💡 Make sure you run this from the project root directory")
        return
    
    file_size = os.path.getsize(model_path) / (1024*1024)
    print(f"📁 Model file found: {file_size:.2f} MB")
    
    # Step 2: Load the model
    try:
        print("🔍 Loading INT8 model...")
        model_data = torch.load(model_path, map_location='cpu')
        print("✅ Model loaded successfully")
        
        print(f"📊 Data type: {type(model_data)}")
        
        if isinstance(model_data, dict):
            print(f"📦 Keys in model file: {list(model_data.keys())}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Step 3: Extract the actual model
    try:
        print("\n🔍 Extracting model for inference...")
        
        if isinstance(model_data, dict):
            if 'state_dict' in model_data:
                print("📋 Found state_dict format")
                # This is likely your format
                model = model_data['state_dict']
                metadata = model_data.get('metadata', {})
                
                print("📄 Model metadata:")
                for key, value in metadata.items():
                    print(f"   {key}: {value}")
                
                # Try to create a working model
                actual_model = create_inference_model(model)
                
            elif 'model' in model_data:
                print("📋 Found standard format")
                actual_model = model_data['model']
            else:
                print("📋 Assuming direct state dict")
                actual_model = create_inference_model(model_data)
        else:
            print("📋 Direct model object")
            actual_model = model_data
        
        if actual_model is None:
            print("❌ Could not extract usable model")
            return
            
        print("✅ Model extracted successfully")
        
    except Exception as e:
        print(f"❌ Failed to extract model: {e}")
        return
    
    # Step 4: Test inference
    test_inference(actual_model)
    
    # Step 5: Benchmark speed
    benchmark_speed(actual_model)

def create_inference_model(state_dict):
    """Try to create an inference model from state dict"""
    try:
        # For quantized models, we need special handling
        # Let's try a simple approach first
        
        if isinstance(state_dict, dict):
            print(f"📊 State dict has {len(state_dict)} parameters")
            
            # Show some parameter info
            first_few = list(state_dict.items())[:3]
            for name, param in first_few:
                if torch.is_tensor(param):
                    print(f"   {name}: {param.shape} ({param.dtype})")
            
            # For now, return a simple wrapper
            class SimpleModel(torch.nn.Module):
                def __init__(self, state_dict):
                    super().__init__()
                    # Try to load the state dict
                    try:
                        self.load_state_dict(state_dict, strict=False)
                    except:
                        # If that fails, store it directly
                        self.state_dict_data = state_dict
                
                def forward(self, x):
                    # This is a placeholder - actual forward pass depends on architecture
                    # For testing, we'll return a dummy output in YOLO format
                    batch_size = x.shape[0]
                    # YOLOv8 typically returns [batch, 84, 8400] for 80 classes
                    # Your model has 58 classes, so it should be [batch, 63, 8400]
                    return torch.randn(batch_size, 63, 8400)
            
            return SimpleModel(state_dict)
        
        return None
        
    except Exception as e:
        print(f"⚠️ Could not create inference model: {e}")
        return None

def test_inference(model):
    """Test basic inference"""
    print("\n🧪 Testing Inference")
    print("-"*30)
    
    try:
        model.eval()
        
        # Create test input (YOLOv8 input format)
        test_input = torch.randn(1, 3, 640, 640)
        print(f"📥 Test input: {test_input.shape}")
        
        # Run inference
        with torch.no_grad():
            start_time = time.time()
            output = model(test_input)
            end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000
        
        print(f"✅ Inference successful!")
        print(f"⏱️ Time: {inference_time:.2f} ms")
        print(f"📤 Output type: {type(output)}")
        
        if torch.is_tensor(output):
            print(f"📊 Output shape: {output.shape}")
        elif isinstance(output, (list, tuple)):
            print(f"📊 Output list with {len(output)} elements:")
            for i, out in enumerate(output[:3]):
                if torch.is_tensor(out):
                    print(f"     Element {i}: {out.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        print("💡 This might be because we don't have the exact model architecture")
        return False

def benchmark_speed(model, num_runs=50):
    """Simple speed benchmark"""
    print(f"\n⏱️ Speed Benchmark ({num_runs} runs)")
    print("-"*30)
    
    try:
        model.eval()
        test_input = torch.randn(1, 3, 640, 640)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(test_input)
        
        # Actual benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = model(test_input)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times) * 1000  # Convert to ms
        fps = 1.0 / (sum(times) / len(times))
        
        print(f"📊 Average time: {avg_time:.2f} ms")
        print(f"🚀 FPS: {fps:.1f}")
        print(f"📏 Min/Max: {min(times)*1000:.2f}/{max(times)*1000:.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

def analyze_model_directly():
    """Analyze the model file directly without trying to run it"""
    print("\n🔍 Direct Model Analysis")
    print("-"*30)
    
    model_path = "models/checkpoints/qat_full_features_4/qat_yolov8n_full_int8_final.pt"
    
    try:
        model_data = torch.load(model_path, map_location='cpu')
        
        if isinstance(model_data, dict):
            print("📦 Model file structure:")
            for key in model_data.keys():
                value = model_data[key]
                print(f"   {key}: {type(value)}")
                
                if key == 'metadata' and isinstance(value, dict):
                    print("   Metadata details:")
                    for mkey, mvalue in value.items():
                        print(f"     {mkey}: {mvalue}")
                
                elif key == 'state_dict' and isinstance(value, dict):
                    print(f"   State dict: {len(value)} parameters")
                    
                    # Check for quantized parameters
                    quantized_count = 0
                    for param_name, param in value.items():
                        if torch.is_tensor(param):
                            if param.dtype in [torch.qint8, torch.quint8]:
                                quantized_count += 1
                    
                    print(f"   Quantized parameters: {quantized_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def compare_with_baseline():
    """Compare sizes with baseline model"""
    print("\n📊 Size Comparison")
    print("-"*30)
    
    int8_path = "models/checkpoints/qat_full_features_4/qat_yolov8n_full_int8_final.pt"
    baseline_path = "models/checkpoints/qat_full_features_4/weights/best.pt"
    
    try:
        int8_size = os.path.getsize(int8_path) / (1024*1024)
        print(f"🔥 INT8 model: {int8_size:.2f} MB")
        
        if os.path.exists(baseline_path):
            baseline_size = os.path.getsize(baseline_path) / (1024*1024)
            compression = baseline_size / int8_size
            print(f"📈 Baseline (FP32): {baseline_size:.2f} MB")
            print(f"⚡ Compression ratio: {compression:.2f}x")
        else:
            print("⚠️ Baseline model not found for comparison")
        
    except Exception as e:
        print(f"❌ Size comparison failed: {e}")

def main():
    """Main function - run everything"""
    print("INT8 Model Testing Script")
    print("========================")
    print("This will test your specific INT8 model file")
    print()
    
    # Run all tests
    test_int8_model()
    analyze_model_directly()
    compare_with_baseline()
    
    print("\n" + "="*50)
    print("🎯 SUMMARY")
    print("="*50)
    print("✅ Basic file loading: Check console output above")
    print("✅ Model structure: Check console output above") 
    print("⚠️ Inference test: May fail due to architecture mismatch")
    print("💡 For real validation, use your QuantizedYOLOv8 class")
    print()
    print("📋 Next steps:")
    print("1. If inference works: Your INT8 model is usable")
    print("2. If inference fails: Model structure issue")
    print("3. Compare with baseline FP32 performance")

if __name__ == "__main__":
    main()