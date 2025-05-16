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

# Import QAT model and loss
from src.models.yolov8_qat import QuantizedYOLOv8
from src.training.loss import QATPenaltyLoss

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

def main():
    """Main function for QAT training with phased approach."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load QAT configuration
    qat_config = load_config(args.config)
    
    # Create output directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    if args.export:
        os.makedirs(args.export_dir, exist_ok=True)
    
    # Determine model path (relative to project root if not absolute)
    model_path = args.model
    if not os.path.isabs(model_path) and not model_path.startswith('yolov8'):
        # Check if it's a path relative to models/pretrained
        pretrained_path = os.path.join('models', 'pretrained', model_path)
        if os.path.exists(pretrained_path):
            model_path = pretrained_path
    
    # Determine data path (relative to project root if not absolute)
    data_path = args.data
    if not os.path.isabs(data_path):
        # Check if it's a path relative to dataset
        dataset_path = os.path.join('dataset', data_path)
        if os.path.exists(dataset_path):
            data_path = dataset_path
    
    # Initialize QAT model
    logger.info(f"Starting YOLOv8 QAT training")
    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {data_path}")
    logger.info(f"QConfig: {args.qconfig}")
    
    # Get additional QAT parameters from config
    qat_params = qat_config.get('qat_params', {})
    phased_training = args.phased_training
    if not phased_training and 'qat_phased_training' in qat_config:
        phased_training = qat_config.get('qat_phased_training', {}).get('enabled', True)
        
    skip_detection_head = not args.keep_detection_head
    if not skip_detection_head and 'skip_detection_head' in qat_params:
        skip_detection_head = qat_params['skip_detection_head']
    
    # Check for mixed precision quantization
    use_mixed_precision = args.mixed_precision
    if not use_mixed_precision and 'use_mixed_precision' in qat_params:
        use_mixed_precision = qat_params['use_mixed_precision']
    
    # Create QConfig from configuration if available
    custom_qconfig = None
    if qat_config and 'quantization' in qat_config:
        custom_qconfig = create_qconfig_from_params(qat_config)
    
    # Initialize QAT model from module
    qat_model = QuantizedYOLOv8(
        model_path=model_path,
        qconfig_name=args.qconfig,
        skip_detection_head=skip_detection_head,
        fuse_modules=args.fuse,
        custom_qconfig=custom_qconfig
    )
    
    # # Setup for advanced calibration if specified
    # calibration_config = qat_config.get('calibration', {})
    # if calibration_config:
    #     calibration_method = calibration_config.get('method', 'histogram')
    #     num_batches = calibration_config.get('num_batches', 100)
    #     percentile = calibration_config.get('percentile', 99.99)
    #     use_coreset = calibration_config.get('use_coreset', False)
        
    #     if use_coreset:
    #         coreset_method = calibration_config.get('coreset_method', 'random')
    #         coreset_size = calibration_config.get('coreset_size', 1000)
    #         qat_model.setup_calibration(
    #             method=calibration_method,
    #             num_batches=num_batches,
    #             percentile=percentile,
    #             use_coreset=True,
    #             coreset_method=coreset_method,
    #             coreset_size=coreset_size
    #         )
    #     else:
    #         qat_model.setup_calibration(
    #             method=calibration_method,
    #             num_batches=num_batches,
    #             percentile=percentile
    #         )
    
    # Setup for mixed precision quantization if specified
    if use_mixed_precision:
        layer_bit_widths = create_mixed_precision_quantization_config(
            qat_model.model,
            bit_widths=[4, 8],
            sensitivity_metric="kl_div"
        )
        qat_model.setup_mixed_precision(layer_bit_widths)
    
    # Setup for knowledge distillation if specified
    use_distillation = args.distillation
    temp = 4.0
    alpha = 0.5
    
    if not use_distillation and 'use_distillation' in qat_params:
        use_distillation = qat_params['use_distillation']
    
    if use_distillation:
        teacher_model_path = args.teacher_model
        if not teacher_model_path and 'distillation' in qat_config:
            distill_config = qat_config['distillation']
            if distill_config.get('enabled', False):
                teacher_model_path = distill_config.get('teacher_model', model_path)
                temp = distill_config.get('temperature', 4.0)
                alpha = distill_config.get('alpha', 0.5)
        
        if teacher_model_path:
            qat_model.setup_distillation(
                teacher_model_path=teacher_model_path,
                temperature=temp,
                alpha=alpha
            )
    
    # Setup for quantization penalty loss if specified
    use_quant_penalty = args.quant_penalty
    quant_penalty_alpha = 0.01
    
    if 'quant_penalty_loss' in qat_config:
        penalty_config = qat_config['quant_penalty_loss']
        use_quant_penalty = penalty_config.get('enabled', True)
        quant_penalty_alpha = penalty_config.get('alpha', 0.01)
    
    if use_quant_penalty:
        qat_model.setup_quant_penalty_loss(alpha=quant_penalty_alpha)
    
    # Prepare model for QAT
    qat_model.prepare_for_qat()
    
    # Get training parameters from config
    train_params = qat_config.get('train_params', {})
    epochs = args.epochs if args.epochs else train_params.get('epochs', QAT_EPOCHS)
    batch_size = args.batch_size if args.batch_size else train_params.get('batch_size', BATCH_SIZE)
    img_size = args.img_size if args.img_size else train_params.get('img_size', IMG_SIZE)
    lr = args.lr if args.lr else train_params.get('lr', QAT_LEARNING_RATE)
    
    # Train with QAT - either phased or standard approach
    if phased_training:
        logger.info("Using phased QAT training approach")
        results = qat_model.train_model(
            data_yaml=data_path,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            lr=lr,
            device=args.device,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            use_distillation=use_distillation
        )
    else:
        logger.info("Using standard QAT training approach (non-phased)")
        # Original non-phased training method would be called here
        results = qat_model.train_standard(
            data_yaml=data_path,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            lr=lr,
            device=args.device,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            use_distillation=use_distillation
        )
    
    # Convert and save quantized model
    save_path = os.path.join(args.save_dir, args.output)
    quantized_model = qat_model.convert_to_quantized(save_path)
    
    # Analyze quantization effects if requested
    if args.analyze or qat_params.get('analyze_quantization_effects', False):
        logger.info("Analyzing quantization effects...")
        
        # Basic analysis
        analysis_results = analyze_quantization_effects(quantized_model)
        logger.info(f"Quantization ratio: {analysis_results['quantized_ratio']:.2f}")
        logger.info(f"Quantized modules: {analysis_results['quantized_modules']} / {analysis_results['total_modules']}")
        
        # Size comparison
        size_results = compare_model_sizes(qat_model.original_model, quantized_model)
        logger.info(f"FP32 model size: {size_results['fp32_size_mb']:.2f} MB")
        logger.info(f"INT8 model size: {size_results['int8_size_mb']:.2f} MB")
        logger.info(f"Compression ratio: {size_results['compression_ratio']:.2f}x")
        logger.info(f"Size reduction: {size_results['size_reduction_percent']:.2f}%")
        
        # Mixed precision analysis if applicable
        if use_mixed_precision:
            mp_analysis = analyze_mixed_precision_model(quantized_model)
            logger.info("Mixed precision bit-width distribution:")
            for bit_width, count in mp_analysis['bit_width_counts'].items():
                percentage = (count / mp_analysis['total_layers']) * 100 if mp_analysis['total_layers'] > 0 else 0
                logger.info(f"  {bit_width}-bit: {count} layers ({percentage:.2f}%)")
        
        # Save analysis results
        analysis_path = os.path.join(args.save_dir, "quantization_analysis.yaml")
        with open(analysis_path, 'w') as f:
            yaml.dump({
                'quantization_analysis': analysis_results,
                'size_comparison': size_results,
                'mixed_precision_analysis': mp_analysis if use_mixed_precision else None
            }, f)
        
        logger.info(f"Analysis results saved to {analysis_path}")
    
    # Evaluate if requested
    if args.eval:
        eval_results = qat_model.evaluate(
            data_yaml=data_path,
            batch_size=batch_size,
            img_size=img_size,
            device=args.device
        )
        
        logger.info(f"Evaluation results:")
        logger.info(f"  mAP50: {eval_results.box.map50:.4f}")
        logger.info(f"  mAP50-95: {eval_results.box.map:.4f}")
    
    # Export if requested
    if args.export:
        export_formats = qat_config.get('export', {}).get('formats', ['onnx'])
        for export_format in export_formats:
            export_path = os.path.join(args.export_dir, export_format)
            os.makedirs(export_path, exist_ok=True)
            
            export_file = os.path.join(export_path, f"{os.path.splitext(args.output)[0]}.{export_format}")
            logger.info(f"Exporting model to {export_file}")
            
            try:
                qat_model.export(export_file, format=export_format)
                logger.info(f"Export to {export_format} completed successfully")
            except Exception as e:
                logger.error(f"Export to {export_format} failed: {e}")
    
    logger.info(f"YOLOv8 QAT training completed")
    logger.info(f"Quantized model saved to {save_path}")
    
    return results

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     # Test basic quantization support
#     quantization_working = test_basic_quantization()
#     if not quantization_working:
#         print("WARNING: Basic quantization test failed - environment issues may prevent QAT")
#         print("You can continue but QAT may not work correctly")
#         response = input("Continue despite quantization issues? (y/n): ")
#         if response.lower() != 'y':
#             print("Exiting due to quantization issues")
#             import sys
#             sys.exit(1)
    
#     # Call the main function
#     main()