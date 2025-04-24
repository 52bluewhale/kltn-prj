"""
KLTN Project: YOLOv8 Quantization-Aware Training
"""

__version__ = '0.1.0'

# Import main modules for easy access
from .config import load_config, merge_configs, get_default_config

# Import main components from submodules
from .models import (
    prepare_model_for_qat,
    convert_qat_model_to_quantized
)

from .quantization import (
    get_weight_quantizer,
    get_activation_quantizer,
    create_qconfig,
    prepare_qat_config_from_yaml,
    INT8_SYMMETRIC,
    INT8_SYMMETRIC_PER_CHANNEL,
    UINT8_ASYMMETRIC,
    load_quantization_config
)

from .training import (
    create_trainer,
    QATTrainer,
    QATPenaltyLoss,
    build_loss_function,
    create_lr_scheduler,
    ModelCheckpoint
)

from .data_utils import (
    create_dataloader,
    create_qat_dataloader,
    get_dataset_from_yaml,
    get_qat_specific_transforms
)

from .evaluation import (
    evaluate_model,
    compare_models,
    measure_performance,
    generate_report
)

from .deployment import (
    prepare_model_for_deployment,
    deploy_model,
    create_deployer
)

# Ensure submodules are importable
__all__ = [
    # Config
    'load_config', 'merge_configs', 'get_default_config',
    
    # Models
    'prepare_model_for_qat', 'convert_qat_model_to_quantized',
    
    # Quantization
    'get_weight_quantizer', 'get_activation_quantizer', 'create_qconfig',
    'prepare_qat_config_from_yaml', 'INT8_SYMMETRIC', 'INT8_SYMMETRIC_PER_CHANNEL',
    'UINT8_ASYMMETRIC', 'load_quantization_config',
    
    # Training
    'create_trainer', 'QATTrainer', 'QATPenaltyLoss', 'build_loss_function',
    'create_lr_scheduler', 'ModelCheckpoint',
    
    # Data
    'create_dataloader', 'create_qat_dataloader', 'get_dataset_from_yaml',
    'get_qat_specific_transforms',
    
    # Evaluation
    'evaluate_model', 'compare_models', 'measure_performance', 'generate_report',
    
    # Deployment
    'prepare_model_for_deployment', 'deploy_model', 'create_deployer'
]