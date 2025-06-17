#!/usr/bin/env python
"""
Convert existing QAT checkpoint to Ultralytics-compatible format
"""
import torch
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def convert_qat_checkpoint(input_path, output_path):
    """
    Convert your current QAT checkpoint to Ultralytics-compatible format.
    
    Args:
        input_path: Path to your current QAT model
        output_path: Path for the converted model
    """
    print(f"Converting {input_path} to Ultralytics format...")
    
    try:
        # Load your current checkpoint
        checkpoint = torch.load(input_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            print("✅ Found state_dict format checkpoint")
            
            # You'll need to reconstruct the model architecture
            # This is a simplified approach - you may need to adjust based on your exact model
            
            # Option 1: If you have access to your QuantizedYOLOv8 instance
            # Recreate the model with the same configuration
            from src.models.yolov8_qat import QuantizedYOLOv8
            
            # Extract QAT info
            qat_info = checkpoint.get('qat_info', {})
            qconfig_name = qat_info.get('qconfig_name', 'default')
            skip_detection_head = qat_info.get('skip_detection_head', True)
            fuse_modules = qat_info.get('fuse_modules', True)
            
            print(f"QAT Config: {qconfig_name}, Skip head: {skip_detection_head}")
            
            # Create new QAT model (you'll need to adjust the model_path)
            qat_model = QuantizedYOLOv8(
                model_path='yolov8n.pt',  # Adjust this
                qconfig_name=qconfig_name,
                skip_detection_head=skip_detection_head,
                fuse_modules=fuse_modules
            )
            
            # Prepare for QAT to recreate the structure
            qat_model.prepare_for_qat_with_preservation()
            
            # Load the state dict
            qat_model.model.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Create Ultralytics-compatible checkpoint
            converted_checkpoint = {
                'model': qat_model.model.model,  # ✅ Correct key for Ultralytics
                'epoch': 0,
                'best_fitness': 0.0,
                'date': str(datetime.now()),
                'qat_info': qat_info,
                'fake_quant_count': checkpoint.get('fake_quant_count', 0),
                'conversion_info': {
                    'original_format': 'state_dict_with_preservation',
                    'converted_format': 'ultralytics_compatible',
                    'conversion_time': str(datetime.now())
                }
            }
            
            # Save converted checkpoint
            torch.save(converted_checkpoint, output_path)
            
            # Verify conversion
            verification = torch.load(output_path, map_location='cpu')
            has_model = 'model' in verification
            fake_quant_count = sum(1 for n, m in verification['model'].named_modules() 
                                 if 'FakeQuantize' in type(m).__name__)
            
            print(f"✅ Conversion successful!")
            print(f"   - Has 'model' key: {has_model}")
            print(f"   - FakeQuantize modules: {fake_quant_count}")
            print(f"   - Output: {output_path}")
            
            del verification
            return True
            
        else:
            print("❌ Unexpected checkpoint format")
            return False
            
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_fix_existing_checkpoint(input_path, output_path, model_reference):
    """
    Quick fix if you have access to the original model.
    
    Args:
        input_path: Your current QAT checkpoint
        output_path: Fixed checkpoint path
        model_reference: Reference to your QAT model instance
    """
    print(f"Quick-fixing {input_path}...")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(input_path, map_location='cpu')
        
        # Create new format
        fixed_checkpoint = {
            'model': model_reference.model.model,  # Actual model object
            'epoch': 0,
            'best_fitness': 0.0,
            'date': str(datetime.now()),
            'qat_info': checkpoint.get('qat_info', {}),
            'fake_quant_count': checkpoint.get('fake_quant_count', 0),
        }
        
        # Save fixed checkpoint
        torch.save(fixed_checkpoint, output_path)
        print(f"✅ Quick fix complete: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Quick fix failed: {e}")
        return False

if __name__ == "__main__":
    # Convert your existing checkpoint
    input_file = "models/checkpoints/qat_full_features_6/qat_model_with_fakequant.pt"
    output_file = "models/checkpoints/qat_full_features_6/qat_model_ultralytics_compatible.pt"
    
    success = convert_qat_checkpoint(input_file, output_file)
    
    if success:
        print("\n🎉 Conversion successful!")
        print(f"Now you can use: yolo val model={output_file} ...")
    else:
        print("\n💥 Conversion failed. Check the error messages above.")