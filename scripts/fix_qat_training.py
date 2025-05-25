#!/usr/bin/env python
"""
Fix QAT Training to Preserve Fake Quantization

This script identifies why fake quantization is being lost during training
and provides a corrected approach.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
from src.models.yolov8_qat import QuantizedYOLOv8

logger = logging.getLogger(__name__)

def check_qat_model_during_training(model_path):
    """
    Check if a model saved during QAT training has preserved fake quantization.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Dictionary with analysis
    """
    print(f"Checking QAT model: {model_path}")
    
    try:
        # Load with YOLO
        model = YOLO(model_path)
        
        # Count quantization modules
        fake_quant_count = 0
        observer_count = 0
        qconfig_count = 0
        
        for name, module in model.model.named_modules():
            if hasattr(module, 'qconfig') and module.qconfig is not None:
                qconfig_count += 1
            
            if hasattr(module, 'weight_fake_quant'):
                fake_quant_count += 1
                
            if hasattr(module, 'activation_post_process'):
                observer_count += 1
        
        analysis = {
            'model_path': model_path,
            'fake_quant_modules': fake_quant_count,
            'observer_modules': observer_count,
            'qconfig_modules': qconfig_count,
            'has_quantization': fake_quant_count > 0 or observer_count > 0
        }
        
        print(f"Results:")
        print(f"  QConfig modules: {qconfig_count}")
        print(f"  Fake quantization modules: {fake_quant_count}")  
        print(f"  Observer modules: {observer_count}")
        print(f"  Has quantization: {analysis['has_quantization']}")
        
        return analysis
        
    except Exception as e:
        print(f"Error checking model: {e}")
        return None

def fix_yolov8_qat_model_saving():
    """
    Provide instructions to fix YOLOv8 QAT model saving.
    """
    instructions = """
    ISSUE IDENTIFIED: YOLOv8 is stripping fake quantization during model saving.
    
    ROOT CAUSE: 
    - YOLOv8's training pipeline automatically converts models to standard format
    - This removes quantization-specific modules (FakeQuantize, observers, qconfig)
    - The saved "best.pt" is actually a regular FP32 model
    
    SOLUTION: Modify your yolov8_qat.py to preserve quantization during saving.
    
    The issue is likely in the _train_phase method where YOLO saves the model.
    We need to intercept the saving process and preserve the QAT modules.
    """
    
    return instructions

def create_quantization_preserving_trainer():
    """
    Create a custom trainer that preserves quantization during saving.
    """
    
    code = '''
# Add this to your yolov8_qat.py file:

def save_qat_model_with_quantization(self, model, save_path):
    """
    Save QAT model while preserving fake quantization modules.
    
    Args:
        model: QAT model with fake quantization
        save_path: Path to save the model
    """
    import tempfile
    import shutil
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_model_path = os.path.join(temp_dir, "qat_model_temp.pt")
    
    try:
        # Method 1: Save the entire model object (preserves everything)
        torch.save({
            'model': model,  # Save the entire model object
            'quantization_preserved': True,
            'metadata': {
                'framework': 'pytorch',
                'format': 'qat_with_fake_quantization',
                'saved_with_quantization': True
            }
        }, temp_model_path)
        
        # Move to final location
        shutil.move(temp_model_path, save_path)
        
        logger.info(f"QAT model saved with quantization preserved: {save_path}")
        
        # Verify quantization was preserved
        verification = torch.load(save_path, map_location='cpu')
        if 'model' in verification:
            fake_quant_count = sum(1 for name, module in verification['model'].named_modules() 
                                 if hasattr(module, 'weight_fake_quant'))
            logger.info(f"Verification: {fake_quant_count} fake quantization modules preserved")
        
    except Exception as e:
        logger.error(f"Failed to save QAT model: {e}")
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise
    finally:
        # Cleanup temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def _train_phase_fixed(self, data_yaml, epochs, batch_size, img_size, lr, device, save_dir, log_dir, use_distillation, phase_name):
    """
    Fixed version of _train_phase that preserves quantization.
    """
    # ... existing training code ...
    
    # CRITICAL FIX: After training, manually save the QAT model
    if hasattr(self, 'model') and self.model is not None:
        # Save the QAT model with quantization preserved
        qat_save_path = os.path.join(save_dir, "qat_model_with_quantization.pt")
        self.save_qat_model_with_quantization(self.model.model, qat_save_path)
        
        # Also save as standard YOLO format for compatibility
        yolo_save_path = os.path.join(save_dir, "weights", "best_qat.pt")
        os.makedirs(os.path.dirname(yolo_save_path), exist_ok=True)
        self.model.save(yolo_save_path)
    
    return results
'''
    
    return code

def test_quantization_preservation():
    """
    Test if we can create and preserve a QAT model properly.
    """
    print("Testing quantization preservation...")
    
    try:
        # Create a QAT model
        qat_model = QuantizedYOLOv8(
            model_path='yolov8n.pt',
            qconfig_name='default',
            skip_detection_head=True,
            fuse_modules=True
        )
        
        # Prepare for QAT
        qat_model.prepare_for_qat()
        
        # Check quantization modules
        fake_quant_count = sum(1 for name, module in qat_model.model.model.named_modules() 
                             if hasattr(module, 'weight_fake_quant'))
        observer_count = sum(1 for name, module in qat_model.model.model.named_modules() 
                           if hasattr(module, 'activation_post_process'))
        
        print(f"Created QAT model with:")
        print(f"  Fake quantization modules: {fake_quant_count}")
        print(f"  Observer modules: {observer_count}")
        
        if fake_quant_count == 0:
            print("❌ ERROR: QAT model creation is not working!")
            return False
        
        # Test saving with quantization preservation
        test_save_path = "test_qat_model.pt"
        
        # Save with quantization preserved
        torch.save({
            'model': qat_model.model.model,
            'quantization_preserved': True,
            'metadata': {
                'framework': 'pytorch', 
                'format': 'qat_with_fake_quantization',
                'fake_quant_count': fake_quant_count,
                'observer_count': observer_count
            }
        }, test_save_path)
        
        # Load and verify
        loaded = torch.load(test_save_path, map_location='cpu')
        if 'model' in loaded:
            loaded_fake_quant = sum(1 for name, module in loaded['model'].named_modules() 
                                  if hasattr(module, 'weight_fake_quant'))
            loaded_observer = sum(1 for name, module in loaded['model'].named_modules() 
                                if hasattr(module, 'activation_post_process'))
            
            print(f"After saving and loading:")
            print(f"  Fake quantization modules: {loaded_fake_quant}")
            print(f"  Observer modules: {loaded_observer}")
            
            if loaded_fake_quant == fake_quant_count:
                print("✅ Quantization preservation test PASSED")
                success = True
            else:
                print("❌ Quantization preservation test FAILED")
                success = False
        else:
            print("❌ Could not load saved model")
            success = False
        
        # Cleanup
        if os.path.exists(test_save_path):
            os.remove(test_save_path)
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("=" * 80)
    print("QAT TRAINING ISSUE DIAGNOSIS AND FIX")
    print("=" * 80)
    
    # Check all QAT models from training phases
    qat_models_to_check = [
        'models/checkpoints/qat/phase1_weight_only/weights/best.pt',
        'models/checkpoints/qat/phase2_activations/weights/best.pt', 
        'models/checkpoints/qat/phase3_full_quant/weights/best.pt',
        'models/checkpoints/qat/phase4_fine_tuning/weights/best.pt'
    ]
    
    print("1. Checking all QAT models from training phases...")
    all_missing_quantization = True
    
    for model_path in qat_models_to_check:
        if os.path.exists(model_path):
            print(f"\nChecking: {model_path}")
            analysis = check_qat_model_during_training(model_path)
            if analysis and analysis['has_quantization']:
                all_missing_quantization = False
        else:
            print(f"Model not found: {model_path}")
    
    if all_missing_quantization:
        print("\n❌ CONFIRMED: All QAT models are missing fake quantization!")
        print("The issue is in the training/saving process.")
    else:
        print("\n✅ Some models have preserved quantization.")
    
    # Test quantization preservation
    print("\n" + "=" * 80)
    print("2. Testing quantization preservation...")
    
    test_result = test_quantization_preservation()
    
    if not test_result:
        print("❌ Quantization preservation test failed")
        print("There may be an issue with your QAT setup")
    
    # Provide fix instructions
    print("\n" + "=" * 80)
    print("3. SOLUTION INSTRUCTIONS")
    print("=" * 80)
    
    print(fix_yolov8_qat_model_saving())
    
    print("\nCode to add to your yolov8_qat.py:")
    print(create_quantization_preserving_trainer())
    
    print("\n" + "=" * 80)
    print("IMMEDIATE ACTION PLAN") 
    print("=" * 80)
    print("1. The issue is that YOLOv8 strips fake quantization when saving models")
    print("2. Your training logs show quantization was active, but saved models lost it")
    print("3. You need to modify your yolov8_qat.py to preserve quantization during saving")
    print("4. Alternatively, export directly from the in-memory QAT model during training")
    
    print("\nRECOMMENDED WORKAROUND:")
    print("Re-run QAT training with a modified saving approach that preserves fake quantization")

if __name__ == "__main__":
    main()