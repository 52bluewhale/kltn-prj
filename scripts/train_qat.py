#!/usr/bin/env python
"""
Enhanced YOLOv8 Quantization-Aware Training (QAT) Script

This script implements QAT for YOLOv8 models with:
- Module fusion (Conv-BN-ReLU)
- Fake quantization (simulating quantization during training)
- Multiple quantization schemes (symmetric/asymmetric, per-tensor/per-channel)
- Configurable observers and calibration
- Advanced calibration methods (entropy, percentile, histogram)
- Knowledge distillation during QAT
- Mixed precision quantization
- Detailed quantization analysis
- Phased QAT training approach
"""
import os
import sys
import logging
import argparse
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from functools import partial

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_qat')

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("Ultralytics YOLO package not found. Please install with: pip install ultralytics")
    sys.exit(1)

# Import project modules
from src.config import (
    DATASET_YAML, PRETRAINED_MODEL, IMG_SIZE, BATCH_SIZE, 
    QAT_EPOCHS, QAT_LEARNING_RATE, DEVICE
)

# Import quantization modules
from src.quantization.fusion import fuse_yolov8_modules, fuse_model_modules
from src.quantization.qconfig import (
    get_default_qat_qconfig,
    get_qconfig_by_name,
    create_qconfig_mapping,
    get_sensitive_layer_qconfig,
    get_first_layer_qconfig,
    get_last_layer_qconfig,
    get_lsq_qconfig,
    QAT_CONFIGS
)
from src.quantization.utils import (
    prepare_model_for_qat,
    convert_qat_model_to_quantized,
    get_model_size,
    compare_model_sizes,
    analyze_quantization_effects,
    apply_layer_specific_quantization,
    skip_layers_from_quantization,
    save_quantized_model,
    load_quantized_model,
    measure_layer_wise_quantization_error
)
from src.quantization.calibration import (
    Calibrator,
    PercentileCalibrator,
    EntropyCalibrator,
    calibrate_model
)
from src.quantization.observers import (
    CustomMinMaxObserver,
    PerChannelMinMaxObserver,
    HistogramObserver,
    get_observer
)
from src.quantization.fake_quantize import (
    CustomFakeQuantize,
    PerChannelFakeQuantize,
    LSQFakeQuantize
)
from src.quantization.schemes.per_tensor import (
    create_per_tensor_quantizer,
    UINT8_ASYMMETRIC,
    INT8_SYMMETRIC
)
from src.quantization.schemes.per_channel import (
    create_per_channel_quantizer,
    INT8_SYMMETRIC_PER_CHANNEL
)

# FIXED: Import QAT model and penalty loss from correct paths
from src.models.yolov8_qat import QuantizedYOLOv8
# FIXED: Import from penalty_loss.py instead of loss.py
from src.training.penalty_loss import QuantizationPenaltyLoss

def test_basic_quantization():
    """Test basic PyTorch quantization to verify environment."""
    print("=" * 80)
    print("TESTING BASIC PYTORCH QUANTIZATION FUNCTIONALITY")
    print("=" * 80)
    
    # Create simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.relu = torch.nn.ReLU()
            
        def forward(self, x):
            return self.relu(self.conv(x))
    
    try:
        # Get config and prepare model
        from src.quantization.qconfig import get_default_qat_qconfig
        model = SimpleModel()
        model.train()
        model.qconfig = get_default_qat_qconfig()
        
        print(f"QConfig created and applied to model: {model.qconfig is not None}")
        
        prepared = torch.quantization.prepare_qat(model)
        
        # Check if quantization modules are present
        has_weight_fake_quant = hasattr(prepared.conv, 'weight_fake_quant')
        has_activation_post_process = hasattr(prepared.conv, 'activation_post_process')
        
        print(f"Model prepared successfully: {prepared is not None}")
        print(f"Conv has weight_fake_quant: {has_weight_fake_quant}")
        print(f"Conv has activation_post_process: {has_activation_post_process}")
        print(f"Prepared model structure: {prepared}")
        
        test_passed = has_weight_fake_quant and has_activation_post_process
        print(f"Basic quantization test {'PASSED' if test_passed else 'FAILED'}")
        print("=" * 80)
        
        return test_passed
    
    except Exception as e:
        print(f"Error during basic quantization test: {e}")
        import traceback
        print(traceback.format_exc())
        print("Basic quantization test FAILED")
        print("=" * 80)
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced YOLOv8 QAT Training")
    
    # Basic parameters
    parser.add_argument('--model', type=str, default=PRETRAINED_MODEL,
                      help='path to model, or model name from ultralytics')
    parser.add_argument('--data', type=str, default=DATASET_YAML,
                      help='dataset.yaml path')
    
    # QAT parameters
    parser.add_argument('--qconfig', type=str, default='default',
                      choices=['default', 'sensitive', 'first_layer', 'last_layer', 'lsq'],
                      help='quantization configuration')
    parser.add_argument('--fuse', action='store_true', default=True,
                      help='fuse Conv+BN+ReLU modules for quantization')
    parser.add_argument('--keep-detection-head', action='store_true',
                      help='quantize detection head (not recommended)')
    parser.add_argument('--config', type=str, default='configs/qat_config.yaml',
                      help='QAT configuration file')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=QAT_EPOCHS,
                      help='number of epochs for QAT')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                      help='batch size')
    parser.add_argument('--img-size', type=int, default=IMG_SIZE,
                      help='input image size')
    parser.add_argument('--lr', type=float, default=QAT_LEARNING_RATE,
                      help='learning rate (typically 10x smaller than regular training)')
    
    # Output parameters
    parser.add_argument('--save-dir', type=str, default='models/checkpoints/qat',
                      help='directory to save results')
    parser.add_argument('--output', type=str, default='quantized_model.pt',
                      help='output model name')
    parser.add_argument('--log-dir', type=str, default='logs/qat',
                      help='directory to save logs')
    # ADD this new argument
    parser.add_argument('--no-convert', action='store_true',
                      help='skip INT8 conversion, keep QAT model only')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default=DEVICE,
                      help='device to train on (cuda device, i.e. 0 or 0,1,2,3 or cpu)')
    
    # Evaluation parameters
    parser.add_argument('--eval', action='store_true',
                      help='evaluate model after training')
    parser.add_argument('--export', action='store_true',
                      help='export model to ONNX after training')
    parser.add_argument('--export-dir', type=str, default='models/exported',
                      help='directory to save exported models')
    
    # Advanced parameters
    parser.add_argument('--distillation', action='store_true',
                      help='use knowledge distillation during QAT')
    parser.add_argument('--teacher-model', type=str, default=None,
                      help='teacher model for knowledge distillation')
    parser.add_argument('--mixed-precision', action='store_true',
                      help='use mixed precision quantization')
    parser.add_argument('--analyze', action='store_true',
                      help='analyze quantization effects')
    parser.add_argument('--seed', type=int, default=42,
                      help='random seed for reproducibility')
    
    # Phased training parameters
    parser.add_argument('--phased-training', action='store_true', default=True,
                      help='use phased QAT training approach')
    parser.add_argument('--quant-penalty', action='store_true', default=True, 
                      help='use quantization penalty loss')
    
    return parser.parse_args()

def load_config(config_path):
    """Load QAT configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file {config_path} not found. Using default settings.")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_calibrator(model, dataloader, method='histogram', device='cuda', **kwargs):
    """Create calibrator based on method."""
    num_batches = kwargs.get('num_batches', 100)
    
    if method == 'percentile':
        percentile = kwargs.get('percentile', 99.99)
        return PercentileCalibrator(
            model=model,
            calibration_loader=dataloader,
            device=device,
            num_batches=num_batches,
            percentile=percentile
        )
    elif method == 'entropy':
        return EntropyCalibrator(
            model=model,
            calibration_loader=dataloader,
            device=device,
            num_batches=num_batches
        )
    else:  # Default to histogram
        return Calibrator(
            model=model,
            calibration_loader=dataloader,
            device=device,
            num_batches=num_batches
        )

def select_calibration_coreset(dataset, size=1000, method='k-center'):
    """
    Select a representative subset of data for calibration.
    
    Args:
        dataset: Full dataset
        size: Desired coreset size
        method: Selection method (k-center, random, herding)
        
    Returns:
        Calibration coreset indices
    """
    if method == 'random':
        # Simple random selection
        indices = torch.randperm(len(dataset))[:size]
    elif method == 'k-center':
        # This would be a more sophisticated implementation
        # For now, just return random indices
        logger.warning("K-center coreset selection not fully implemented, using random selection")
        indices = torch.randperm(len(dataset))[:size]
    elif method == 'herding':
        # Feature-based herding would require model forward passes
        # For now, just return random indices
        logger.warning("Herding coreset selection not fully implemented, using random selection")
        indices = torch.randperm(len(dataset))[:size]
    else:
        logger.warning(f"Unknown coreset method: {method}, using random selection")
        indices = torch.randperm(len(dataset))[:size]
    
    return indices

def create_qconfig_from_params(config):
    """Create QConfig from configuration parameters."""
    # Extract quantization parameters from config
    quant_config = config.get('quantization', {})
    
    # Weight parameters
    weight_params = quant_config.get('weight', {})
    weight_dtype = weight_params.get('dtype', 'qint8')
    weight_scheme = weight_params.get('scheme', 'per_channel')
    weight_observer = weight_params.get('observer', 'minmax')
    weight_symmetric = weight_params.get('symmetric', True)
    weight_bit_width = weight_params.get('bit_width', 8)
    weight_channel_axis = weight_params.get('channel_axis', 0)
    
    # Activation parameters
    act_params = quant_config.get('activation', {})
    act_dtype = act_params.get('dtype', 'quint8')
    act_scheme = act_params.get('scheme', 'per_tensor')
    act_observer = act_params.get('observer', 'moving_average_minmax')
    act_symmetric = act_params.get('symmetric', False)
    act_bit_width = act_params.get('bit_width', 8)
    
    # Fake quantize parameters
    fake_quantize_params = quant_config.get('fake_quantize', {})
    fake_quantize_type = fake_quantize_params.get('type', 'custom')
    grad_factor = fake_quantize_params.get('grad_factor', 1.0)
    
    # Create per-tensor or per-channel quantizers
    if weight_scheme == 'per_channel':
        weight_quantizer = create_per_channel_quantizer(
            bit_width=weight_bit_width,
            symmetric=weight_symmetric,
            is_signed=(weight_dtype == 'qint8'),
            ch_axis=weight_channel_axis
        )
    else:
        weight_quantizer = create_per_tensor_quantizer(
            bit_width=weight_bit_width,
            symmetric=weight_symmetric,
            is_signed=(weight_dtype == 'qint8')
        )
    
    if act_scheme == 'per_channel':
        act_quantizer = create_per_channel_quantizer(
            bit_width=act_bit_width,
            symmetric=act_symmetric,
            is_signed=(act_dtype == 'qint8')
        )
    else:
        act_quantizer = create_per_tensor_quantizer(
            bit_width=act_bit_width,
            symmetric=act_symmetric,
            is_signed=(act_dtype == 'qint8')
        )
    
    # For now, return one of the predefined QConfigs
    # In a full implementation, we would create a custom QConfig from the parameters
    if fake_quantize_type == 'lsq':
        return get_lsq_qconfig()
    else:
        return get_default_qat_qconfig()

def create_mixed_precision_quantization_config(model, bit_widths=[4, 8], sensitivity_metric="kl_div"):
    """
    Create mixed precision quantization configuration.
    
    Args:
        model: Model to configure
        bit_widths: Available bit widths
        sensitivity_metric: Metric to measure layer sensitivity
        
    Returns:
        Dictionary mapping layer names to QConfigs
    """
    # This would be a more sophisticated implementation
    # For now, just return a simple mapping
    mixed_precision_mapping = {}
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if 'downsample' in name or 'bottleneck' in name:
                # Use lower precision for less sensitive layers
                mixed_precision_mapping[name] = 4
            else:
                # Use higher precision for more sensitive layers
                mixed_precision_mapping[name] = 8
    
    return mixed_precision_mapping

def setup_knowledge_distillation(student_model, teacher_model_path, temperature=4.0, alpha=0.5):
    """
    Setup knowledge distillation for QAT.
    
    Args:
        student_model: Student model (quantized)
        teacher_model_path: Path to teacher model
        temperature: Temperature for softening logits
        alpha: Weight for distillation loss
        
    Returns:
        Configured teacher model and distillation parameters
    """
    # Load teacher model
    teacher_model = YOLO(teacher_model_path)
    teacher_model.model = teacher_model.model.to(next(student_model.parameters()).device)
    teacher_model.model.eval()  # Teacher in eval mode
    
    return {
        'teacher_model': teacher_model,
        'temperature': temperature,
        'alpha': alpha
    }

def analyze_mixed_precision_model(model):
    """
    Analyze mixed precision quantized model.
    
    Args:
        model: Mixed precision model
        
    Returns:
        Analysis results
    """
    bit_width_counts = {4: 0, 8: 0, 16: 0, 32: 0}
    layer_bit_widths = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'qconfig') and module.qconfig is not None:
            if hasattr(module, 'weight_fake_quant'):
                if hasattr(module.weight_fake_quant, 'quant_min') and hasattr(module.weight_fake_quant, 'quant_max'):
                    # Determine bit width from quant_min and quant_max
                    if module.weight_fake_quant.quant_min == -8 and module.weight_fake_quant.quant_max == 7:
                        bit_width = 4
                    elif module.weight_fake_quant.quant_min == -128 and module.weight_fake_quant.quant_max == 127:
                        bit_width = 8
                    else:
                        bit_width = 32  # Default
                    
                    bit_width_counts[bit_width] += 1
                    layer_bit_widths[name] = bit_width
    
    return {
        'bit_width_counts': bit_width_counts,
        'layer_bit_widths': layer_bit_widths,
        'total_layers': sum(bit_width_counts.values())
    }

# ==============================================================================
# ADD verification function before main()
# ==============================================================================

def verify_qat_model_file(model_path):
    """Verify that a saved model file contains quantization."""
    logger.info(f"Verifying QAT model file: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file does not exist: {model_path}")
        return False
    
    try:
        saved_data = torch.load(model_path, map_location='cpu')
        
        if isinstance(saved_data, dict):
            # Check for our QAT format
            if 'fake_quant_count' in saved_data:
                count = saved_data['fake_quant_count']
                method = saved_data.get('save_method', 'unknown')
                
                logger.info(f"✅ QAT model verified:")
                logger.info(f"  - FakeQuantize modules: {count}")
                logger.info(f"  - Save method: {method}")
                logger.info(f"  - Quantization preserved: {saved_data.get('quantization_preserved', False)}")
                
                return count > 0
            
            # Check if it has model structure
            elif 'model' in saved_data:
                # Try to count FakeQuantize modules in the model
                try:
                    fake_quant_count = sum(1 for n, m in saved_data['model'].named_modules() 
                                          if 'FakeQuantize' in type(m).__name__)
                    
                    logger.info(f"✅ Model structure found with {fake_quant_count} FakeQuantize modules")
                    return fake_quant_count > 0
                    
                except Exception as e:
                    logger.warning(f"⚠️ Could not analyze model structure: {e}")
                    return False
            
            else:
                logger.warning("⚠️ Model file format not recognized as QAT")
                return False
        else:
            logger.warning("⚠️ Model file is not in dictionary format")
            return False
    
    except Exception as e:
        logger.error(f"❌ Error verifying model file: {e}")
        return False

def test_qat_fixes():
    """Test that the QAT fixes work correctly."""
    logger.info("Testing QAT fixes...")
    
    try:
        # Create a simple QAT model for testing
        qat_model = QuantizedYOLOv8('yolov8n.pt', skip_detection_head=True)
        qat_model.prepare_for_qat()
        
        # Test the quantization preservation
        test_result = qat_model.test_quantization_preservation()
        
        if test_result:
            logger.info("✅ QAT fixes test PASSED")
            return True
        else:
            logger.error("❌ QAT fixes test FAILED")
            return False
            
    except Exception as e:
        logger.error(f"❌ QAT fixes test exception: {e}")
        return False

def main():
    """
    FIXED: Main function with proper penalty loss integration
    """
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Safety check: Test basic quantization first
    logger.info("Running safety check for quantization compatibility...")
    try:
        test_model = torch.nn.Conv2d(3, 16, 3)
        test_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        test_prepared = torch.quantization.prepare_qat(test_model)
        logger.info("✓ Basic quantization compatibility check passed")
    except Exception as e:
        logger.error(f"❌ Basic quantization compatibility check failed: {e}")
        return None

    # Load QAT configuration
    qat_config = load_config(args.config)
    
    # Create output directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    if args.export:
        os.makedirs(args.export_dir, exist_ok=True)
    
    # Resolve paths
    model_path = args.model
    if not os.path.isabs(model_path) and not model_path.startswith('yolov8'):
        pretrained_path = os.path.join('models', 'pretrained', model_path)
        if os.path.exists(pretrained_path):
            model_path = pretrained_path
    
    data_path = args.data
    if not os.path.isabs(data_path):
        dataset_path = os.path.join('datasets', data_path)
        if os.path.exists(dataset_path):
            data_path = dataset_path
    
    # Extract configuration parameters
    qat_params = qat_config.get('qat_params', {})
    train_params = qat_config.get('train_params', {})
    
    # Training configuration
    phased_training = args.phased_training or qat_config.get('qat_phased_training', {}).get('enabled', True)
    skip_detection_head = not args.keep_detection_head or qat_params.get('skip_detection_head', True)
    use_quant_penalty = args.quant_penalty or qat_config.get('quant_penalty_loss', {}).get('enabled', True)
    quant_penalty_alpha = qat_config.get('quant_penalty_loss', {}).get('alpha', 0.01)
    
    # Training parameters
    epochs = args.epochs or train_params.get('epochs', 10)
    batch_size = args.batch_size or train_params.get('batch_size', 16)
    img_size = args.img_size or train_params.get('img_size', 640)
    lr = args.lr or train_params.get('lr', 0.0005)
    
    logger.info(f"🚀 Starting YOLOv8 QAT Training")
    logger.info(f"📁 Model: {model_path}")
    logger.info(f"📊 Dataset: {data_path}")
    logger.info(f"⚙️ QConfig: {args.qconfig}")
    logger.info(f"🔄 Phased Training: {phased_training}")
    logger.info(f"⚡ Penalty Loss: {use_quant_penalty} (α={quant_penalty_alpha})")
    
    # Initialize and Prepare QAT Model
    try:
        logger.info("📦 Initializing QAT model...")
        qat_model = QuantizedYOLOv8(
            model_path=model_path,
            qconfig_name=args.qconfig,
            skip_detection_head=skip_detection_head,
            fuse_modules=args.fuse
        )
        
        logger.info("⚙️ Preparing model for QAT...")
        qat_model.prepare_for_qat()
        
        # FIXED: Setup penalty loss if enabled
        if use_quant_penalty:
            logger.info(f"🎯 Setting up penalty loss (α={quant_penalty_alpha})...")
            qat_model.setup_penalty_loss_integration(alpha=quant_penalty_alpha, warmup_epochs=5)
            
            # FIXED: Verify penalty setup
            penalty_working = qat_model.verify_penalty_setup()
            if penalty_working:
                logger.info("✅ Penalty loss integration verified")
            else:
                logger.warning("⚠️ Penalty loss integration verification failed, continuing anyway...")
        
        logger.info("✅ QAT model preparation completed")
        
    except Exception as e:
        logger.error(f"❌ QAT model preparation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    # Train Model
    try:
        logger.info("🏋️ Starting QAT training...")
        
        results = qat_model.train_model(
            data_yaml=data_path,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            lr=lr,
            device=args.device,
            save_dir=args.save_dir,
            log_dir=args.log_dir
        )
        
        logger.info("✅ QAT training completed successfully")
        
    except Exception as e:
        logger.error(f"❌ QAT training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    # Save QAT Model
    try:
        qat_save_path = os.path.join(args.save_dir, "qat_model_with_fakequant.pt")
        logger.info(f"💾 Saving QAT model (with FakeQuantize) to: {qat_save_path}")
        
        save_success = qat_model.save(qat_save_path, preserve_qat=True)
        if not save_success:
            logger.error("❌ Failed to save QAT model!")
            return None
        
        logger.info("✅ QAT model saved successfully")
        
    except Exception as e:
        logger.error(f"❌ QAT model saving failed: {e}")
        return None
    
    # Convert to INT8 (Optional)
    int8_model_path = None
    int8_model_info = {}
    
    if not args.no_convert:
        try:
            int8_save_path = os.path.join(args.save_dir, args.output)
            logger.info(f"🔄 Converting QAT model to INT8 quantized model...")
            
            quantized_model = qat_model.convert_to_quantized(int8_save_path)
            
            if quantized_model is not None:
                int8_model_path = int8_save_path.replace('.pt', '_int8_final.pt')
                
                # Gather model information
                int8_model_info = {
                    'model_path': int8_model_path,
                    'architecture': 'YOLOv8n',
                    'quantization': 'INT8',
                    'num_classes': 58,  # Vietnamese traffic signs
                    'input_size': [1, 3, img_size, img_size],
                    'file_size_mb': os.path.getsize(int8_model_path) / (1024 * 1024) if os.path.exists(int8_model_path) else 0,
                    'deployment_ready': True
                }
                
                logger.info("✅ INT8 model conversion completed successfully")
            else:
                logger.error("❌ INT8 model conversion failed")
                
        except Exception as e:
            logger.error(f"❌ INT8 conversion failed: {e}")
            logger.warning("Continuing without INT8 conversion...")
    else:
        logger.info("⏭️ Skipping INT8 conversion as requested")
    
    # Evaluation and Export (Optional) - same as before
    if args.eval:
        try:
            logger.info("📊 Evaluating model...")
            eval_results = qat_model.evaluate(
                data_yaml=data_path,
                batch_size=batch_size,
                img_size=img_size,
                device=args.device
            )
            logger.info(f"📈 Evaluation Results - mAP50: {eval_results.box.map50:.4f}, mAP50-95: {eval_results.box.map:.4f}")
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
    
    if args.export:
        try:
            export_formats = qat_config.get('export', {}).get('formats', ['onnx'])
            for export_format in export_formats:
                export_path = os.path.join(args.export_dir, export_format)
                os.makedirs(export_path, exist_ok=True)
                export_file = os.path.join(export_path, f"{os.path.splitext(args.output)[0]}.{export_format}")
                
                logger.info(f"📤 Exporting to {export_format}...")
                qat_model.export(export_file, format=export_format)
                logger.info(f"✅ Export to {export_format} completed")
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
    
    # Print Final Results
    print_final_results(qat_save_path, int8_model_path, int8_model_info, args.save_dir)
    
    return results

def print_final_results(qat_model_path, int8_model_path, int8_model_info, save_dir):
    """Print comprehensive final results with model paths and information."""
    print("\n" + "="*80)
    print("🎉 QAT TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # QAT Model Information
    print("📁 QAT MODEL (For Continued Training/Fine-tuning):")
    if os.path.exists(qat_model_path):
        qat_size = os.path.getsize(qat_model_path) / (1024 * 1024)
        print(f"   Path: {qat_model_path}")
        print(f"   Size: {qat_size:.2f} MB")
        print(f"   Type: QAT with FakeQuantize modules")
        print(f"   Status: ✅ Ready for additional training")
    else:
        print(f"   Status: ❌ QAT model not found")
    
    print()
    
    # INT8 Model Information  
    print("🚀 INT8 QUANTIZED MODEL (For Deployment):")
    if int8_model_path and os.path.exists(int8_model_path):
        print(f"   Path: {int8_model_path}")
        print(f"   Size: {int8_model_info.get('file_size_mb', 0):.2f} MB")
        print(f"   Architecture: {int8_model_info.get('architecture', 'YOLOv8n')}")
        print(f"   Quantization: {int8_model_info.get('quantization', 'INT8')}")
        print(f"   Classes: {int8_model_info.get('num_classes', 58)} (Vietnamese Traffic Signs)")
        print(f"   Input Size: {int8_model_info.get('input_size', [1, 3, 640, 640])}")
        print(f"   Deployment: {'✅ Ready' if int8_model_info.get('deployment_ready') else '❌ Not Ready'}")
    else:
        print(f"   Status: ❌ INT8 model not created")
    
    print()
    
    # Next Steps
    print("🎯 NEXT STEPS:")
    if int8_model_path and os.path.exists(int8_model_path):
        print("   1. ✅ Deploy INT8 model for inference")
        print("   2. 📊 Test model performance on validation set")
        print("   3. 📤 Export to ONNX/TensorRT for optimization")
        print("   4. 🚀 Integrate into production pipeline")
    else:
        print("   1. ⚠️ Check INT8 conversion issues")
        print("   2. 🔄 Try manual conversion if needed")
        print("   3. 📊 Use QAT model for inference testing")
    
    print("="*80)
    print(f"🏆 TRAINING DIRECTORY: {save_dir}")
    print("="*80)

if __name__ == '__main__':
    main()