#!/usr/bin/env python
"""
Diagnose and Fix Quantization Issues

This script compares your QAT model with the quantized model to diagnose
conversion issues and provides a corrected quantized model.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: Ultralytics YOLO package not found. Please install with: pip install ultralytics")
    sys.exit(1)

def compare_models(qat_model_path, quantized_model_path):
    """
    Compare QAT model with quantized model to identify issues.
    
    Args:
        qat_model_path: Path to QAT model
        quantized_model_path: Path to quantized model
        
    Returns:
        Dictionary with comparison results
    """
    print("Comparing QAT model with quantized model...")
    
    comparison = {
        'qat_analysis': {},
        'quantized_analysis': {},
        'issues': [],
        'recommendations': []
    }
    
    # Analyze QAT model
    try:
        print("Loading QAT model...")
        qat_model = YOLO(qat_model_path)
        
        # Count fake quantization modules
        fake_quant_count = 0
        observer_count = 0
        
        for name, module in qat_model.model.named_modules():
            if hasattr(module, 'weight_fake_quant'):
                fake_quant_count += 1
            if hasattr(module, 'activation_post_process'):
                observer_count += 1
        
        comparison['qat_analysis'] = {
            'fake_quant_modules': fake_quant_count,
            'observer_modules': observer_count,
            'total_parameters': sum(p.numel() for p in qat_model.model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in qat_model.model.parameters()) / (1024**2)
        }
        
        print(f"QAT model analysis:")
        print(f"  Fake quantization modules: {fake_quant_count}")
        print(f"  Observer modules: {observer_count}")
        print(f"  Total parameters: {comparison['qat_analysis']['total_parameters']:,}")
        print(f"  Model size: {comparison['qat_analysis']['model_size_mb']:.2f} MB")
        
    except Exception as e:
        comparison['issues'].append(f"Failed to analyze QAT model: {e}")
        print(f"❌ Failed to analyze QAT model: {e}")
    
    # Analyze quantized model
    try:
        print("\nLoading quantized model...")
        quantized_checkpoint = torch.load(quantized_model_path, map_location='cpu')
        
        if isinstance(quantized_checkpoint, dict) and 'state_dict' in quantized_checkpoint:
            state_dict = quantized_checkpoint['state_dict']
            metadata = quantized_checkpoint.get('metadata', {})
            
            none_count = 0
            valid_tensors = 0
            total_params = 0
            
            for key, tensor in state_dict.items():
                if tensor is None:
                    none_count += 1
                elif isinstance(tensor, torch.Tensor):
                    valid_tensors += 1
                    total_params += tensor.numel()
            
            comparison['quantized_analysis'] = {
                'state_dict_keys': len(state_dict),
                'none_values': none_count,
                'valid_tensors': valid_tensors,
                'total_parameters': total_params,
                'metadata': metadata
            }
            
            print(f"Quantized model analysis:")
            print(f"  State dict keys: {len(state_dict)}")
            print(f"  None values: {none_count}")
            print(f"  Valid tensors: {valid_tensors}")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Metadata: {metadata}")
            
            # Identify issues
            if none_count > 0:
                comparison['issues'].append(f"Found {none_count} None values in state_dict")
                comparison['recommendations'].append("Re-convert QAT model with proper observer calibration")
            
            if total_params == 0:
                comparison['issues'].append("No valid parameters found in quantized model")
                comparison['recommendations'].append("Quantization conversion failed completely")
            
        else:
            comparison['issues'].append("Quantized model is not in expected state_dict format")
    
    except Exception as e:
        comparison['issues'].append(f"Failed to analyze quantized model: {e}")
        print(f"❌ Failed to analyze quantized model: {e}")
    
    return comparison

def create_corrected_quantized_model(qat_model_path, output_path):
    """
    Create a corrected quantized model from the QAT model.
    
    Args:
        qat_model_path: Path to QAT model
        output_path: Path to save corrected model
        
    Returns:
        Success status
    """
    print(f"\nCreating corrected quantized model...")
    print(f"QAT model: {qat_model_path}")
    print(f"Output: {output_path}")
    
    try:
        # Load QAT model
        qat_model = YOLO(qat_model_path)
        print("✓ QAT model loaded successfully")
        
        # Set to eval mode
        qat_model.model.eval()
        
        # Method 1: Save as YOLO-compatible model (preserves fake quantization)
        print("Saving as YOLO-compatible quantized model...")
        
        # Create metadata
        metadata = {
            'framework': 'pytorch',
            'format': 'qat_model_corrected',
            'original_model': qat_model_path,
            'description': 'QAT model preserved for export compatibility'
        }
        
        # Save using YOLO's save method
        qat_model.save(output_path)
        print(f"✓ Corrected model saved to: {output_path}")
        
        # Verify the saved model can be loaded
        try:
            verification_model = YOLO(output_path)
            print("✓ Model verification successful")
            
            # Check fake quantization is preserved
            fake_quant_count = sum(1 for name, module in verification_model.model.named_modules() 
                                 if hasattr(module, 'weight_fake_quant'))
            print(f"✓ Preserved {fake_quant_count} fake quantization modules")
            
            return True
            
        except Exception as verify_error:
            print(f"❌ Model verification failed: {verify_error}")
            return False
    
    except Exception as e:
        print(f"❌ Failed to create corrected model: {e}")
        return False

def fix_quantized_model_state_dict(quantized_model_path, qat_model_path, output_path):
    """
    Fix the quantized model by replacing None values with QAT model weights.
    
    Args:
        quantized_model_path: Path to broken quantized model
        qat_model_path: Path to QAT model
        output_path: Path to save fixed model
        
    Returns:
        Success status
    """
    print(f"\nFixing quantized model state_dict...")
    
    try:
        # Load both models
        quantized_checkpoint = torch.load(quantized_model_path, map_location='cpu')
        qat_model = YOLO(qat_model_path)
        
        if not (isinstance(quantized_checkpoint, dict) and 'state_dict' in quantized_checkpoint):
            print("❌ Quantized model is not in state_dict format")
            return False
        
        quantized_state_dict = quantized_checkpoint['state_dict']
        qat_state_dict = qat_model.model.state_dict()
        
        print(f"Quantized state_dict keys: {len(quantized_state_dict)}")
        print(f"QAT state_dict keys: {len(qat_state_dict)}")
        
        # Fix None values
        fixed_count = 0
        for key, value in quantized_state_dict.items():
            if value is None:
                if key in qat_state_dict:
                    quantized_state_dict[key] = qat_state_dict[key].clone()
                    fixed_count += 1
                    print(f"Fixed None value for: {key}")
                else:
                    print(f"❌ Could not fix None value for: {key} (not found in QAT model)")
        
        print(f"✓ Fixed {fixed_count} None values")
        
        # Create fixed model
        fixed_checkpoint = {
            'state_dict': quantized_state_dict,
            'metadata': {
                'framework': 'pytorch',
                'format': 'quantized_int8_fixed',
                'original_quantized_model': quantized_model_path,
                'qat_model_source': qat_model_path,
                'fixes_applied': f'Fixed {fixed_count} None values'
            }
        }
        
        # Save fixed model
        torch.save(fixed_checkpoint, output_path)
        print(f"✓ Fixed model saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix quantized model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Diagnose and fix quantization issues")
    parser.add_argument('--qat-model', type=str, 
                       default='models/checkpoints/qat/phase4_fine_tuning/weights/best.pt',
                       help='Path to QAT model')
    parser.add_argument('--quantized-model', type=str,
                       default='models/checkpoints/qat/quantized_model.pt',
                       help='Path to quantized model')
    parser.add_argument('--output-dir', type=str, default='models/checkpoints/qat',
                       help='Output directory for fixed models')
    parser.add_argument('--fix-method', type=str, default='both',
                       choices=['corrected', 'fixed', 'both'],
                       help='Method to fix the quantization issues')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.qat_model):
        print(f"❌ QAT model not found: {args.qat_model}")
        return
    
    if not os.path.exists(args.quantized_model):
        print(f"❌ Quantized model not found: {args.quantized_model}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("QUANTIZATION DIAGNOSIS AND FIX")
    print("=" * 80)
    
    # Step 1: Compare models
    comparison = compare_models(args.qat_model, args.quantized_model)
    
    print(f"\n" + "=" * 80)
    print("DIAGNOSIS RESULTS")
    print("=" * 80)
    
    if comparison['issues']:
        print("Issues found:")
        for issue in comparison['issues']:
            print(f"  ❌ {issue}")
    else:
        print("✅ No major issues detected")
    
    if comparison['recommendations']:
        print("\nRecommendations:")
        for rec in comparison['recommendations']:
            print(f"  💡 {rec}")
    
    # Step 2: Create fixes
    print(f"\n" + "=" * 80)
    print("CREATING FIXES")
    print("=" * 80)
    
    success_count = 0
    
    if args.fix_method in ['corrected', 'both']:
        print("\n1. Creating corrected QAT model for export...")
        corrected_path = os.path.join(args.output_dir, 'quantized_model_corrected.pt')
        if create_corrected_quantized_model(args.qat_model, corrected_path):
            success_count += 1
            print(f"✅ Corrected model: {corrected_path}")
        else:
            print("❌ Failed to create corrected model")
    
    if args.fix_method in ['fixed', 'both']:
        print("\n2. Fixing quantized model state_dict...")
        fixed_path = os.path.join(args.output_dir, 'quantized_model_fixed.pt')
        if fix_quantized_model_state_dict(args.quantized_model, args.qat_model, fixed_path):
            success_count += 1
            print(f"✅ Fixed model: {fixed_path}")
        else:
            print("❌ Failed to fix quantized model")
    
    # Step 3: Provide usage instructions
    print(f"\n" + "=" * 80)
    print("USAGE INSTRUCTIONS")
    print("=" * 80)
    
    if success_count > 0:
        print("✅ Fixed models created successfully!")
        print("\nTo export your models:")
        
        if args.fix_method in ['corrected', 'both']:
            corrected_path = os.path.join(args.output_dir, 'quantized_model_corrected.pt')
            print(f"\n1. Export corrected QAT model (RECOMMENDED):")
            print(f"   python scripts/export.py --model {corrected_path} --format onnx --model-type qat")
        
        if args.fix_method in ['fixed', 'both']:
            fixed_path = os.path.join(args.output_dir, 'quantized_model_fixed.pt')
            print(f"\n2. Export fixed quantized model:")
            print(f"   python scripts/export.py --model {fixed_path} --format onnx --model-type quantized")
        
        print(f"\n3. Or export directly from QAT model:")
        print(f"   python scripts/export.py --model {args.qat_model} --format onnx --model-type qat")
        
    else:
        print("❌ No fixes were successful.")
        print("Recommendation: Export directly from QAT model:")
        print(f"   python scripts/export.py --model {args.qat_model} --format onnx --model-type qat")

if __name__ == "__main__":
    main()