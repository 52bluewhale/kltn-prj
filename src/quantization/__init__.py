from .observers import (
    CustomMinMaxObserver,
    PerChannelMinMaxObserver,
    HistogramObserver,
    get_observer
)

from .fake_quantize import (
    CustomFakeQuantize,
    PerChannelFakeQuantize,
    LSQFakeQuantize,
    create_fake_quantizer,
    get_fake_quantize_from_config
)

from .qconfig import (
    create_qconfig,
    get_default_qat_qconfig,
    get_sensitive_layer_qconfig,
    get_first_layer_qconfig,
    get_last_layer_qconfig,
    get_lsq_qconfig,
    QAT_CONFIGS,
    get_qconfig_by_name,
    create_qconfig_mapping,
    prepare_qat_config_from_yaml
)

from .qat_modules import (
    QATConv2d,
    QATBatchNorm2d,
    QATLinear,
    QATReLU
)

from .fusion import (
    fuse_conv_bn,
    fuse_conv_bn_relu,
    fuse_conv_bn_silu,
    fuse_yolov8_modules,
    find_modules_to_fuse,
    fuse_model_modules
)

from .utils import (
    load_quantization_config,
    prepare_model_for_qat,
    convert_qat_model_to_quantized,
    apply_layer_specific_quantization,
    skip_layers_from_quantization,
    get_model_size,
    compare_model_sizes,
    save_quantized_model,
    load_quantized_model,
    analyze_quantization_effects,
    get_quantization_parameters,
    measure_layer_wise_quantization_error
)

from .calibration import (
    Calibrator,
    calibrate_model,
    PercentileCalibrator,
    EntropyCalibrator,
    build_calibrator
)

from .schemes import (
    get_weight_quantizer,
    get_activation_quantizer,
    INT8_SYMMETRIC,
    INT8_SYMMETRIC_PER_CHANNEL,
    UINT8_ASYMMETRIC,
)

# Main API functions for easy access
def prepare_qat_model(model, config=None, config_path=None, skip_layers=None):
    """
    Prepare model for quantization-aware training.
    
    Args:
        model: Model to prepare
        config: Quantization configuration dictionary
        config_path: Path to configuration file
        skip_layers: List of layer regex patterns to skip
        
    Returns:
        Prepared model
    """
    if config_path is not None:
        config = load_quantization_config(config_path)
    
    # Prepare model for QAT
    model = prepare_model_for_qat(model, config, inplace=True)
    
    # Skip specified layers from quantization
    if skip_layers is not None:
        model = skip_layers_from_quantization(model, skip_layers)
    
    return model

def quantize_model(qat_model):
    """
    Convert QAT model to quantized model.
    
    Args:
        qat_model: QAT model to convert
        
    Returns:
        Quantized model
    """
    return convert_qat_model_to_quantized(qat_model, inplace=False)

def calibrate_and_quantize(model, dataloader, method='histogram', num_batches=100, device='cuda'):
    """
    Calibrate and quantize model in one step.
    
    Args:
        model: Model to calibrate and quantize
        dataloader: DataLoader for calibration
        method: Calibration method
        num_batches: Number of batches for calibration
        device: Device to use
        
    Returns:
        Calibrated and quantized model
    """
    # Calibrate model
    calibrated_model = calibrate_model(model, dataloader, method, num_batches, device)
    
    # Convert to quantized model
    quantized_model = convert_qat_model_to_quantized(calibrated_model, inplace=False)
    
    return quantized_model

def create_qat_config_from_config_file(config_path):
    """
    Create QAT configuration from config file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        QAT configuration
    """
    config = load_quantization_config(config_path)
    return prepare_qat_config_from_yaml(config)