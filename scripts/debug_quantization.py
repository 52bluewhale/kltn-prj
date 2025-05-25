#!/usr/bin/env python
"""
Debug Quantization Analysis

This script helps debug why quantized modules aren't being detected properly
and provides a comprehensive analysis of the quantization state.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from collections import defaultdict

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: Ultralytics YOLO package not found. Please install with: pip install ultralytics")
    sys.exit(1)

def analyze_model_quantization(model_path, verbose=True):
    """
    Comprehensive analysis of model quantization state.
    
    Args:
        model_path: Path to model file
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with analysis results
    """
    print(f"Analyzing quantization in: {model_path}")
    print("=" * 80)
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    analysis = {
        'file_size_mb': os.path.getsize(model_path) / (1024 * 1024),
        'checkpoint_type': str(type(checkpoint)),
        'quantized_layers': 0,
        'total_layers': 0,
        'quantized_parameters': 0,
        'total_parameters': 0,
        'quantized_layer_types': defaultdict(int),
        'all_layer_types': defaultdict(int),
        'quantization_detected': False,
        'detection_methods': []
    }
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        print("Found state_dict format checkpoint")
        state_dict = checkpoint['state_dict']
        metadata = checkpoint.get('metadata', {})
        
        print(f"Metadata: {metadata}")
        print(f"State dict keys: {len(state_dict)}")
        
        # Analyze state dict for quantization indicators
        quantized_keys = []
        quantized_types = set()
        
        for key, tensor in state_dict.items():
            analysis['total_parameters'] += tensor.numel()
            
            # Check for quantization indicators in key names
            if any(q_indicator in key.lower() for q_indicator in [
                'quantized', 'qweight', 'qbias', 'scale', 'zero_point', 
                '_packed_params', 'weight_ih_l0_packed', 'weight_hh_l0_packed'
            ]):
                quantized_keys.append(key)
                analysis['quantized_parameters'] += tensor.numel()
                analysis['quantization_detected'] = True
                analysis['detection_methods'].append('state_dict_key_names')
            
            # Check tensor dtype for quantization
            if tensor.dtype in [torch.qint8, torch.quint8, torch.qint32]:
                quantized_keys.append(key)
                quantized_types.add(str(tensor.dtype))
                analysis['quantized_parameters'] += tensor.numel()
                analysis['quantization_detected'] = True
                analysis['detection_methods'].append('tensor_dtype')
                
            # Count layer types based on key patterns
            if '.weight' in key or '.bias' in key:
                layer_name = key.split('.weight')[0].split('.bias')[0]
                layer_parts = layer_name.split('.')
                if len(layer_parts) > 1:
                    layer_type = 'unknown'
                    # Try to infer layer type from key pattern
                    if 'conv' in key.lower():
                        layer_type = 'Conv'
                    elif 'linear' in key.lower() or 'fc' in key.lower():
                        layer_type = 'Linear'
                    elif 'bn' in key.lower() or 'norm' in key.lower():
                        layer_type = 'BatchNorm'
                    
                    analysis['all_layer_types'][layer_type] += 1
                    analysis['total_layers'] += 1
                    
                    if key in quantized_keys:
                        analysis['quantized_layer_types'][layer_type] += 1
                        analysis['quantized_layers'] += 1
        
        if verbose:
            print(f"\nQuantization Detection Results:")
            print(f"Quantization detected: {analysis['quantization_detected']}")
            print(f"Detection methods: {analysis['detection_methods']}")
            print(f"Quantized keys found: {len(quantized_keys)}")
            if quantized_keys:
                print("Sample quantized keys:")
                for key in quantized_keys[:10]:
                    tensor = state_dict[key]
                    print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
            
            if quantized_types:
                print(f"Quantized tensor types found: {quantized_types}")
        
        # Try to load with a base model to check for quantized modules
        try:
            print(f"\nTrying to load with base model...")
            
            # Find a base model
            possible_base_models = [
                'yolov8n.pt',
                'models/pretrained/yolov8n.pt',
                'models/checkpoints/qat/phase4_fine_tuning/weights/best.pt'
            ]
            
            base_model = None
            for model_path_candidate in possible_base_models:
                if os.path.exists(model_path_candidate):
                    try:
                        base_model = YOLO(model_path_candidate)
                        print(f"Using base model: {model_path_candidate}")
                        break
                    except:
                        continue
            
            if base_model is None:
                print("Creating default YOLOv8n model")
                base_model = YOLO('yolov8n.pt')
            
            # Load the state dict
            missing_keys, unexpected_keys = base_model.model.load_state_dict(state_dict, strict=False)
            
            print("✓ Successfully loaded state dict into base model")
            
            # Analyze the loaded model for quantized modules
            quantized_modules = analyze_model_modules(base_model.model)
            analysis.update(quantized_modules)
            
        except Exception as e:
            print(f"Failed to load with base model: {e}")
    
    else:
        print("Unknown checkpoint format")
    
    return analysis

def analyze_model_modules(model):
    """
    Analyze PyTorch model modules for quantization.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with module analysis
    """
    analysis = {
        'torch_quantized_modules': 0,
        'fake_quantize_modules': 0,
        'quantized_module_types': defaultdict(int),
        'total_modules': 0,
        'module_analysis': []
    }
    
    print(f"\nAnalyzing model modules...")
    
    quantized_module_indicators = [
        'Quantized', 'quantized', 'QuantizedConv', 'QuantizedLinear',
        'qconv', 'qlinear', 'FakeQuantize', 'fake_quant'
    ]
    
    for name, module in model.named_modules():
        analysis['total_modules'] += 1
        module_type = module.__class__.__name__
        
        # Check for quantized modules
        is_quantized = False
        quantization_type = None
        
        # Method 1: Check class name
        for indicator in quantized_module_indicators:
            if indicator in module_type:
                is_quantized = True
                quantization_type = 'class_name'
                break
        
        # Method 2: Check for quantization attributes
        if hasattr(module, 'weight_fake_quant') or hasattr(module, 'activation_post_process'):
            is_quantized = True
            quantization_type = 'fake_quantize'
            analysis['fake_quantize_modules'] += 1
        
        # Method 3: Check for PyTorch quantized operations
        if any(q_op in str(type(module)) for q_op in ['quantized', 'Quantized']):
            is_quantized = True
            quantization_type = 'torch_quantized'
            analysis['torch_quantized_modules'] += 1
        
        # Method 4: Check tensor dtypes in module
        if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
            if module.weight.dtype in [torch.qint8, torch.quint8, torch.qint32]:
                is_quantized = True
                quantization_type = 'tensor_dtype'
        
        if is_quantized:
            analysis['quantized_module_types'][module_type] += 1
            analysis['module_analysis'].append({
                'name': name,
                'type': module_type,
                'quantization_type': quantization_type
            })
    
    print(f"Total modules: {analysis['total_modules']}")
    print(f"Fake quantize modules: {analysis['fake_quantize_modules']}")
    print(f"PyTorch quantized modules: {analysis['torch_quantized_modules']}")
    print(f"Quantized module types: {dict(analysis['quantized_module_types'])}")
    
    if analysis['module_analysis']:
        print(f"\nFirst 10 quantized modules:")
        for i, module_info in enumerate(analysis['module_analysis'][:10]):
            print(f"  {i+1}. {module_info['name']} ({module_info['type']}) - {module_info['quantization_type']}")
    
    return analysis

def improved_quantization_analysis(model):
    """
    Improved quantization analysis function to replace the one in utils.py
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with quantization analysis
    """
    analysis = {
        'quantized_modules': 0,
        'total_modules': 0,
        'quantized_ratio': 0.0,
        'quantization_types': defaultdict(int),
        'quantized_layer_details': []
    }
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only count leaf modules
            analysis['total_modules'] += 1
            
            is_quantized = False
            quantization_type = 'none'
            
            # Check for different types of quantization
            module_class = module.__class__.__name__
            
            # 1. PyTorch native quantized modules
            if any(indicator in module_class.lower() for indicator in [
                'quantized', 'quantizedconv', 'quantizedlinear', 'quantizedrelu'
            ]):
                is_quantized = True
                quantization_type = 'pytorch_quantized'
            
            # 2. Fake quantization modules (QAT)
            elif hasattr(module, 'weight_fake_quant') or hasattr(module, 'activation_post_process'):
                is_quantized = True
                quantization_type = 'fake_quantize'
            
            # 3. Check for quantized tensors
            elif hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
                if module.weight.dtype in [torch.qint8, torch.quint8, torch.qint32]:
                    is_quantized = True
                    quantization_type = 'tensor_dtype'
            
            # 4. Check module string representation
            elif 'quantized' in str(type(module)).lower():
                is_quantized = True
                quantization_type = 'string_match'
            
            if is_quantized:
                analysis['quantized_modules'] += 1
                analysis['quantization_types'][quantization_type] += 1
                analysis['quantized_layer_details'].append({
                    'name': name,
                    'type': module_class,
                    'quantization_type': quantization_type
                })
    
    if analysis['total_modules'] > 0:
        analysis['quantized_ratio'] = analysis['quantized_modules'] / analysis['total_modules']
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description="Debug quantization analysis")
    parser.add_argument('model', type=str, help='Path to model file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    try:
        # Comprehensive analysis
        analysis = analyze_model_quantization(args.model, verbose=args.verbose)
        
        print(f"\n" + "="*80)
        print("QUANTIZATION ANALYSIS SUMMARY")
        print("="*80)
        print(f"File size: {analysis['file_size_mb']:.2f} MB")
        print(f"Quantization detected: {analysis['quantization_detected']}")
        print(f"Detection methods: {analysis['detection_methods']}")
        print(f"Total parameters: {analysis['total_parameters']:,}")
        print(f"Quantized parameters: {analysis['quantized_parameters']:,}")
        
        if analysis['total_parameters'] > 0:
            param_ratio = analysis['quantized_parameters'] / analysis['total_parameters']
            print(f"Parameter quantization ratio: {param_ratio:.4f}")
        
        if 'torch_quantized_modules' in analysis:
            print(f"PyTorch quantized modules: {analysis['torch_quantized_modules']}")
            print(f"Fake quantize modules: {analysis['fake_quantize_modules']}")
            print(f"Total modules analyzed: {analysis['total_modules']}")
        
        # Recommendations
        print(f"\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        if not analysis['quantization_detected']:
            print("❌ No quantization detected!")
            print("Possible issues:")
            print("1. Quantization conversion failed during training")
            print("2. Model was saved before quantization")
            print("3. Quantization analysis function needs updating")
            print("\nNext steps:")
            print("1. Check your QAT training logs for conversion errors")
            print("2. Try exporting from the QAT model instead:")
            print("   python scripts/export.py --model models/checkpoints/qat/phase4_fine_tuning/weights/best.pt --model-type qat")
        else:
            print("✅ Quantization detected!")
            print("The model appears to be properly quantized.")
            print("If export fails, the issue might be with model loading, not quantization.")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()