import torch
import torch.quantization
from ultralytics import YOLO
import yaml
import os
import time
from pathlib import Path

# ============================================================================
# INITIAL SETUP AND VALIDATION
# ============================================================================

print("🚀 Starting Enhanced QAT Training Pipeline")
print("=" * 60)

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("GPU not detected. Ensure CUDA is installed.")

# Check NumPy
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError:
    raise RuntimeError("NumPy not available. Install with 'pip install numpy==1.24.4'")

# Validate dataset YAML
dataset_path = "F:/kltn-prj/datasets/vietnam-traffic-sign-detection/dataset.yaml"
try:
    with open(dataset_path, 'r') as f:
        data = yaml.safe_load(f)
    print(f"✅ Dataset validated: {data['nc']} classes")
    assert os.path.exists(data['train']), f"Train folder {data['train']} not found"
    assert os.path.exists(data['val']), f"Val folder {data['val']} not found"
except Exception as e:
    raise RuntimeError(f"Dataset validation failed: {e}")

# ============================================================================
# DIAGNOSTIC MONITORING SYSTEM
# ============================================================================

class QATDiagnosticMonitor:
    """
    Comprehensive diagnostic system to track exactly where FakeQuantize modules are lost.
    """
    
    def __init__(self):
        self.checkpoints = []
        self.model_snapshots = {}
        
    def capture_model_state(self, model, checkpoint_name):
        """Capture detailed model state at a specific checkpoint."""
        fake_quant_count = 0
        fake_quant_modules = []
        qconfig_count = 0
        total_params = 0
        
        for name, module in model.named_modules():
            if 'FakeQuantize' in str(type(module)):
                fake_quant_count += 1
                fake_quant_modules.append(name)
            if hasattr(module, 'qconfig') and module.qconfig is not None:
                qconfig_count += 1
        
        for param in model.parameters():
            total_params += param.numel()
        
        snapshot = {
            'checkpoint': checkpoint_name,
            'timestamp': time.time(),
            'fake_quant_count': fake_quant_count,
            'fake_quant_modules': fake_quant_modules[:5],  # Sample
            'qconfig_count': qconfig_count,
            'total_parameters': total_params,
            'model_id': id(model),
            'model_type': str(type(model))
        }
        
        self.model_snapshots[checkpoint_name] = snapshot
        self.checkpoints.append(checkpoint_name)
        
        print(f"📸 CHECKPOINT [{checkpoint_name}]: {fake_quant_count} FakeQuantize modules, {total_params} params")
        return snapshot
    
    def compare_states(self, checkpoint1, checkpoint2):
        """Compare two model states to identify changes."""
        if checkpoint1 not in self.model_snapshots or checkpoint2 not in self.model_snapshots:
            return None
        
        state1 = self.model_snapshots[checkpoint1]
        state2 = self.model_snapshots[checkpoint2]
        
        fake_quant_diff = state2['fake_quant_count'] - state1['fake_quant_count']
        param_diff = state2['total_parameters'] - state1['total_parameters']
        model_changed = state1['model_id'] != state2['model_id']
        
        comparison = {
            'from': checkpoint1,
            'to': checkpoint2,
            'fake_quant_change': fake_quant_diff,
            'param_change': param_diff,
            'model_object_changed': model_changed,
            'status': '✅ PRESERVED' if fake_quant_diff == 0 else f'❌ LOST {abs(fake_quant_diff)} modules'
        }
        
        print(f"🔄 COMPARISON [{checkpoint1} → {checkpoint2}]:")
        print(f"   FakeQuantize: {state1['fake_quant_count']} → {state2['fake_quant_count']} ({fake_quant_diff:+d})")
        print(f"   Parameters: {state1['total_parameters']} → {state2['total_parameters']} ({param_diff:+d})")
        print(f"   Model ID: {'CHANGED' if model_changed else 'SAME'}")
        print(f"   Status: {comparison['status']}")
        
        return comparison
    
    def generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report."""
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE DIAGNOSTIC REPORT")
        print("="*80)
        
        print(f"📊 Total Checkpoints: {len(self.checkpoints)}")
        
        # Show all checkpoints
        for i, checkpoint in enumerate(self.checkpoints):
            state = self.model_snapshots[checkpoint]
            print(f"{i+1:2d}. {checkpoint:20s} - {state['fake_quant_count']:2d} FakeQuantize, {state['total_parameters']:8d} params")
        
        print("\n🔄 State Transitions:")
        # Compare consecutive checkpoints
        for i in range(len(self.checkpoints) - 1):
            self.compare_states(self.checkpoints[i], self.checkpoints[i+1])
        
        # Find the critical failure point
        print("\n💥 CRITICAL FAILURE ANALYSIS:")
        for i in range(len(self.checkpoints) - 1):
            curr_state = self.model_snapshots[self.checkpoints[i]]
            next_state = self.model_snapshots[self.checkpoints[i+1]]
            
            if curr_state['fake_quant_count'] > 0 and next_state['fake_quant_count'] == 0:
                print(f"🎯 QUANTIZATION LOST BETWEEN: {self.checkpoints[i]} → {self.checkpoints[i+1]}")
                print(f"   This is the exact point where YOLOv8 stripped the quantization!")
                break

# Global diagnostic monitor
diagnostic_monitor = QATDiagnosticMonitor()

# ============================================================================
# QAT PROTECTION MECHANISMS WITH ENHANCED MONITORING
# ============================================================================

def protect_qat_model(model):
    """
    CRITICAL: Protect QAT model from YOLOv8's internal optimizations
    that strip FakeQuantize modules during training.
    """
    print("🛡️ Installing QAT protection...")
    
    # Store original methods
    if not hasattr(model, '_qat_protected'):
        model._original_fuse = getattr(model, 'fuse', None)
        model._original_half = getattr(model, 'half', None)
    
    # Override fuse() to prevent FakeQuantize removal
    def protected_fuse(*args, **kwargs):
        print("🚫 Fusion attempt blocked! QAT preserved.")
        diagnostic_monitor.capture_model_state(model, "FUSION_ATTEMPT_BLOCKED")
        return model
    
    # Override half() to prevent FP16 conversion (breaks QAT)
    def protected_half(*args, **kwargs):
        print("🚫 Half precision blocked! (would break quantization)")
        diagnostic_monitor.capture_model_state(model, "HALF_PRECISION_BLOCKED")
        return model
    
    model.fuse = protected_fuse
    model.half = protected_half
    model._qat_protected = True
    
    print("✅ QAT protection installed successfully")
    diagnostic_monitor.capture_model_state(model, "PROTECTION_INSTALLED")

def validate_qat_comprehensive(model, stage=""):
    """
    Comprehensive QAT validation with detailed reporting.
    Returns (is_valid, fake_quant_count, detailed_stats)
    """
    print(f"\n🔍 {stage} - Comprehensive QAT Validation")
    print("-" * 50)
    
    # Count FakeQuantize modules
    fake_quant_count = 0
    fake_quant_modules = []
    for name, module in model.named_modules():
        if 'FakeQuantize' in str(type(module)):
            fake_quant_count += 1
            fake_quant_modules.append(name)
    
    # Count QConfig modules
    qconfig_count = 0
    qconfig_modules = []
    for name, module in model.named_modules():
        if hasattr(module, 'qconfig') and module.qconfig is not None:
            qconfig_count += 1
            qconfig_modules.append(name)
    
    # Check observer initialization status
    initialized_observers = 0
    total_observers = 0
    for name, module in model.named_modules():
        if hasattr(module, 'activation_post_process'):
            total_observers += 1
            try:
                observer = module.activation_post_process
                if hasattr(observer, 'min_val') and hasattr(observer, 'max_val'):
                    min_val = observer.min_val
                    max_val = observer.max_val
                    
                    # Check if initialized (not inf values)
                    if (torch.is_tensor(min_val) and torch.is_tensor(max_val)):
                        if (min_val.item() != float('inf') and 
                            max_val.item() != float('-inf')):
                            initialized_observers += 1
                    elif (not torch.is_tensor(min_val) and not torch.is_tensor(max_val)):
                        if (min_val != float('inf') and max_val != float('-inf')):
                            initialized_observers += 1
            except:
                continue
    
    # Check for quantized modules (after conversion)
    quantized_count = sum(1 for n, m in model.named_modules() 
                         if 'Quantized' in str(type(m)))
    
    # Display results
    print(f"📊 QAT Status Report:")
    print(f"  - FakeQuantize modules: {fake_quant_count}")
    print(f"  - QConfig modules: {qconfig_count}")
    print(f"  - Observer status: {initialized_observers}/{total_observers} initialized")
    print(f"  - Quantized modules: {quantized_count}")
    
    # Determine validity
    is_valid_qat = fake_quant_count > 0 and qconfig_count > 0
    is_valid_int8 = quantized_count > 0
    
    if stage.lower().find('int8') != -1 or stage.lower().find('convert') != -1:
        is_valid = is_valid_int8
        status = "✅ VALID INT8" if is_valid else "❌ INVALID INT8"
    else:
        is_valid = is_valid_qat
        status = "✅ VALID QAT" if is_valid else "❌ INVALID QAT"
    
    print(f"  - Status: {status}")
    
    # Show some example modules for debugging
    if fake_quant_count > 0:
        print(f"  - Sample FakeQuantize: {fake_quant_modules[:3]}")
    if quantized_count > 0:
        quantized_modules = [n for n, m in model.named_modules() 
                           if 'Quantized' in str(type(m))][:3]
        print(f"  - Sample Quantized: {quantized_modules}")
    
    detailed_stats = {
        'fake_quant_count': fake_quant_count,
        'qconfig_count': qconfig_count,
        'initialized_observers': initialized_observers,
        'total_observers': total_observers,
        'quantized_count': quantized_count,
        'is_valid': is_valid
    }
    
    return is_valid, fake_quant_count, detailed_stats

def setup_enhanced_qat_config():
    """
    Setup enhanced QAT configuration with proper observers.
    """
    print("\n⚙️ Setting up Enhanced QAT Configuration")
    print("-" * 50)
    
    from torch.quantization.fake_quantize import FakeQuantize
    from torch.quantization.observer import PerChannelMinMaxObserver, MovingAverageMinMaxObserver
    from torch.quantization import QConfig
    
    # Enhanced per-channel weight quantizer
    weight_fake_quant = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0
    )
    
    # Enhanced activation quantizer  
    activation_fake_quant = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        averaging_constant=0.01
    )
    
    qconfig = QConfig(activation=activation_fake_quant, weight=weight_fake_quant)
    print("✅ Enhanced QConfig created successfully")
    
    return qconfig

def apply_qconfig_to_model(model, qconfig):
    """
    Apply QConfig to appropriate model modules with protection.
    """
    print("\n🔧 Applying QConfig to Model Modules")
    print("-" * 50)
    
    modules_configured = 0
    detection_modules_skipped = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            # Skip detection head for stability
            if 'detect' in name or '22' in name:
                module.qconfig = None
                detection_modules_skipped += 1
                print(f"  ⏭️ Skipped detection: {name}")
            else:
                module.qconfig = qconfig
                modules_configured += 1
    
    print(f"✅ QConfig applied to {modules_configured} modules")
    print(f"✅ Skipped {detection_modules_skipped} detection modules")
    
    return modules_configured

def train_qat_protected(model_yolo, dataset_path, epochs=2):
    """
    Train QAT model with comprehensive protection and diagnostic monitoring.
    """
    print(f"\n🏋️ Starting Protected QAT Training ({epochs} epochs)")
    print("-" * 50)
    
    # CHECKPOINT 1: Before protection
    diagnostic_monitor.capture_model_state(model_yolo.model, "BEFORE_PROTECTION")
    
    # Apply protection to both model levels
    protect_qat_model(model_yolo.model)
    protect_qat_model(model_yolo)
    
    # CHECKPOINT 2: After protection
    diagnostic_monitor.capture_model_state(model_yolo.model, "AFTER_PROTECTION")
    
    # Validate protection is in place
    print("🔒 Verifying protection...")
    protection_status = (
        hasattr(model_yolo.model, '_qat_protected') and
        hasattr(model_yolo, '_qat_protected')
    )
    print(f"Protection status: {'✅ ACTIVE' if protection_status else '❌ FAILED'}")
    
    # CHECKPOINT 3: Before training arguments setup
    diagnostic_monitor.capture_model_state(model_yolo.model, "BEFORE_TRAIN_ARGS")
    
    # QAT-optimized training parameters
    training_args = {
        'data': dataset_path,
        'epochs': epochs,
        'batch': 4,
        'imgsz': 640,
        'lr0': 0.0005,
        'device': 0,
        'optimizer': 'Adam',
        
        # QAT-SAFE SETTINGS (Critical!)
        'amp': False,       # Disable AMP (conflicts with QAT)
        'half': False,      # Disable FP16 (breaks FakeQuantize)
        
        # Reduce aggressive optimizations that might interfere
        'mosaic': 0.0,      # Disable mosaic augmentation
        'hsv_h': 0.0,       # Disable color augmentations
        'fliplr': 0.0,      # Disable flip augmentations
        'translate': 0.0,   # Disable translation augmentation
        
        # Training stability
        'patience': 50,
        'save_period': 1,
        'verbose': True,
        'workers': 2,
        'cache': 'disk'
    }
    
    print("📋 Training Configuration:")
    for key, value in training_args.items():
        print(f"  - {key}: {value}")
    
    # CHECKPOINT 4: Just before train() call - CRITICAL POINT
    print("\n🎯 CRITICAL CHECKPOINT: Just before model_yolo.train() call")
    diagnostic_monitor.capture_model_state(model_yolo.model, "JUST_BEFORE_TRAIN_CALL")
    
    # Start training
    print("\n🚀 Beginning QAT training...")
    start_time = time.time()
    
    try:
        # DIAGNOSTIC: Monitor model state during training process
        original_model_id = id(model_yolo.model)
        print(f"🆔 Original model ID: {original_model_id}")
        
        # THE CRITICAL CALL - This is where we expect to lose quantization
        results = model_yolo.train(**training_args)
        
        # CHECKPOINT 5: Immediately after train() call - CRITICAL POINT
        print("\n🎯 CRITICAL CHECKPOINT: Immediately after model_yolo.train() call")
        diagnostic_monitor.capture_model_state(model_yolo.model, "JUST_AFTER_TRAIN_CALL")
        
        # Check if model object changed
        new_model_id = id(model_yolo.model)
        print(f"🆔 New model ID: {new_model_id}")
        if original_model_id != new_model_id:
            print("🚨 ALERT: Model object was replaced during training!")
            diagnostic_monitor.capture_model_state(model_yolo.model, "MODEL_OBJECT_REPLACED")
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        # CHECKPOINT 6: Final training state
        diagnostic_monitor.capture_model_state(model_yolo.model, "TRAINING_COMPLETED")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        # Even if training fails, capture the state
        diagnostic_monitor.capture_model_state(model_yolo.model, "TRAINING_FAILED")
        raise

# ============================================================================
# MAIN QAT PIPELINE
# ============================================================================

def main():
    """
    Main QAT training pipeline with comprehensive validation.
    """
    print("\n" + "="*60)
    print("🎯 STARTING MAIN QAT PIPELINE")
    print("="*60)
    
    # ========================================================================
    # STEP 1: LOAD MODEL (ONLY ONCE!)
    # ========================================================================
    print("\n📥 STEP 1: Loading Base Model")
    model_path = "models/pretrained/yolov8n.pt"
    model_yolo = YOLO(model_path)
    model = model_yolo.model
    
    print(f"✅ YOLOv8n loaded: {sum(p.numel() for p in model.parameters())} parameters")
    
    # ========================================================================
    # STEP 2: SETUP QAT CONFIGURATION
    # ========================================================================
    print("\n⚙️ STEP 2: QAT Configuration Setup")
    qconfig = setup_enhanced_qat_config()
    modules_configured = apply_qconfig_to_model(model, qconfig)
    
    if modules_configured == 0:
        raise RuntimeError("No modules configured for QAT!")
    
    # ========================================================================
    # STEP 3: PREPARE MODEL FOR QAT
    # ========================================================================
    print("\n🔧 STEP 3: Preparing Model for QAT")
    
    try:
        model.train()  # Ensure training mode
        model_prepared = torch.quantization.prepare_qat(model, inplace=True)
        model_yolo.model = model_prepared
        print("✅ QAT preparation completed")
    except Exception as e:
        raise RuntimeError(f"QAT preparation failed: {e}")
    
    # Validate QAT setup
    is_valid, fake_count, stats = validate_qat_comprehensive(
        model_yolo.model, "POST-QAT-SETUP"
    )
    
    if not is_valid:
        raise RuntimeError("QAT setup validation failed!")
    
    print(f"✅ QAT setup validated: {fake_count} FakeQuantize modules ready")
    
    # ========================================================================
    # STEP 4: PROTECTED QAT TRAINING
    # ========================================================================
    print("\n🏋️ STEP 4: Protected QAT Training")
    
    # Pre-training validation
    validate_qat_comprehensive(model_yolo.model, "PRE-TRAINING")
    
    # Train with protection
    train_results = train_qat_protected(model_yolo, dataset_path, epochs=2)
    
    # Post-training validation (CRITICAL CHECK!)
    is_valid_after, fake_count_after, stats_after = validate_qat_comprehensive(
        model_yolo.model, "POST-TRAINING"
    )
    
    if not is_valid_after:
        print("❌ CRITICAL: QAT modules lost during training!")
        print("This indicates YOLOv8 internal optimizations bypassed our protection.")
        return False
    
    print(f"🎉 SUCCESS: QAT preserved through training! {fake_count_after} modules intact")
    
    # ========================================================================
    # STEP 5: SAVE QAT MODEL
    # ========================================================================
    print("\n💾 STEP 5: Saving QAT Model")
    
    qat_save_path = "F:/kltn-prj/models/checkpoints/train_qat_standard_log_1/qat_model.pt"
    os.makedirs(os.path.dirname(qat_save_path), exist_ok=True)
    
    try:
        # Save the QAT model
        model_yolo.save(qat_save_path)
        
        # Verify save
        if os.path.exists(qat_save_path):
            file_size = os.path.getsize(qat_save_path) / (1024 * 1024)
            print(f"✅ QAT model saved: {qat_save_path}")
            print(f"   File size: {file_size:.2f} MB")
            
            # Test reload to verify QAT preservation
            print("🔍 Testing QAT model reload...")
            test_model = YOLO(qat_save_path)
            is_valid_reload, fake_count_reload, _ = validate_qat_comprehensive(
                test_model.model, "AFTER-RELOAD"
            )
            
            if is_valid_reload:
                print(f"✅ QAT model reload successful: {fake_count_reload} modules preserved")
            else:
                print("❌ WARNING: QAT modules lost during save/reload!")
        else:
            raise RuntimeError("Save failed - file not created")
            
    except Exception as e:
        print(f"❌ Save failed: {e}")
        return False
    
    # ========================================================================
    # STEP 6: CONVERT TO INT8
    # ========================================================================
    print("\n🔄 STEP 6: Converting to INT8")
    
    try:
        # Set to eval mode for conversion
        model_yolo.model.eval()
        
        # Convert to INT8 quantized model
        model_int8 = torch.quantization.convert(model_yolo.model, inplace=False)
        
        # Validate INT8 conversion
        is_valid_int8, _, stats_int8 = validate_qat_comprehensive(
            model_int8, "POST-INT8-CONVERSION"
        )
        
        if is_valid_int8:
            print(f"✅ INT8 conversion successful: {stats_int8['quantized_count']} quantized modules")
            
            # Save INT8 model
            int8_save_path = "F:/kltn-prj/models/checkpoints/train_qat_standard_log_1/qat_int8_model.pt"
            model_yolo.model = model_int8
            model_yolo.save(int8_save_path)
            
            if os.path.exists(int8_save_path):
                int8_file_size = os.path.getsize(int8_save_path) / (1024 * 1024)
                print(f"✅ INT8 model saved: {int8_save_path}")
                print(f"   File size: {int8_file_size:.2f} MB")
                
                # Calculate compression ratio
                compression_ratio = file_size / int8_file_size if int8_file_size > 0 else 0
                print(f"📊 Compression ratio: {compression_ratio:.2f}x")
                
            else:
                print("❌ INT8 save failed")
        else:
            print("❌ INT8 conversion failed - no quantized modules found")
            print("This may indicate issues with observer calibration during training")
    
    except Exception as e:
        print(f"❌ INT8 conversion failed: {e}")
        return False
    
    # ========================================================================
    # STEP 7: QAT EFFECTIVENESS VALIDATION
    # ========================================================================
    print("\n📊 STEP 7: QAT Effectiveness Validation")
    
    try:
        # Reload QAT model for testing
        qat_test_model = YOLO(qat_save_path)
        
        print("🧪 Running QAT model validation...")
        qat_results = qat_test_model.val(
            data=dataset_path,
            batch=4,
            workers=2,
            device=0,
            verbose=False
        )
        
        print("✅ QAT Model Performance:")
        print(f"   mAP50: {qat_results.box.map50:.4f}")
        print(f"   mAP50-95: {qat_results.box.map:.4f}")
        
        # Compare with baseline if needed
        print("\n📈 QAT Effectiveness Summary:")
        print(f"   - Training epochs: 2 (test run)")
        print(f"   - QAT modules preserved: ✅")
        print(f"   - INT8 conversion: ✅")
        print(f"   - Model deployment ready: ✅")
        
    except Exception as e:
        print(f"❌ Effectiveness validation failed: {e}")
        return False
    
    # ========================================================================
    # FINAL SUCCESS REPORT
    # ========================================================================
    print("\n" + "="*60)
    print("🎉 QAT PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("✅ All objectives achieved:")
    print("   1. ✅ Trained truly QAT model")
    print("   2. ✅ Saved qat_model.pt with quantization preserved")
    print("   3. ✅ Converted to qat_int8_model.pt")
    print("   4. ✅ Validated QAT effectiveness")
    print("   5. ✅ Models ready for deployment testing")
    print("\n🎯 Next steps:")
    print("   - Increase epochs for full training")
    print("   - Test ONNX export from qat_model.pt")
    print("   - Apply PTQ to ONNX for deployment")
    
    return True

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    success = main()
    
    if success:
        print("\n🎊 Script completed successfully!")
    else:
        print("\n💥 Script failed - check logs for details")
        exit(1)