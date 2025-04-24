from .trainer import Trainer, QATTrainer
from .loss import QATPenaltyLoss, FocalLoss, DistillationLoss
from .callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ProgressiveQuantization,
    TensorBoardLogger,
    QuantizationErrorMonitor,
    LearningRateScheduler
)
from .lr_scheduler import (
    CosineAnnealingWarmRestarts,
    StepLR,
    ReduceLROnPlateau,
    get_qat_scheduler
)

# Main API
def create_trainer(model, config, qat_mode=False):
    """
    Create appropriate trainer based on mode.
    
    Args:
        model: Model to train
        config: Training configuration
        qat_mode: Whether to use QAT trainer
        
    Returns:
        Trainer instance
    """
    if qat_mode:
        return QATTrainer(model, config)
    else:
        return Trainer(model, config)

def build_loss_function(loss_type, config=None):
    """
    Build loss function based on type.
    
    Args:
        loss_type: Type of loss function
        config: Loss configuration
        
    Returns:
        Loss function
    """
    if loss_type == "qat_penalty":
        from .loss import FocalLoss  # Import a base criterion to use
        penalty_factor = config.get("penalty_factor", 0.01) if config else 0.01
        base_criterion = config.get("base_criterion", FocalLoss()) if config else FocalLoss()
        return QATPenaltyLoss(base_criterion, penalty_factor)
    elif loss_type == "focal":
        gamma = config.get("gamma", 2.0) if config else 2.0
        alpha = config.get("alpha", 0.25) if config else 0.25
        return FocalLoss(gamma, alpha)
    elif loss_type == "distillation":
        teacher_model = config.get("teacher_model", None)
        temp = config.get("temperature", 1.0) if config else 1.0
        alpha = config.get("alpha", 0.5) if config else 0.5
        return DistillationLoss(teacher_model, temp, alpha)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def create_lr_scheduler(optimizer, scheduler_type, config=None):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        config: Scheduler configuration
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        T_0 = config.get("T_0", 10) if config else 10
        T_mult = config.get("T_mult", 2) if config else 2
        eta_min = config.get("eta_min", 1e-6) if config else 1e-6
        return CosineAnnealingWarmRestarts(optimizer, T_0, T_mult, eta_min)
    elif scheduler_type == "step":
        step_size = config.get("step_size", 30) if config else 30
        gamma = config.get("gamma", 0.1) if config else 0.1
        return StepLR(optimizer, step_size, gamma)
    elif scheduler_type == "plateau":
        patience = config.get("patience", 10) if config else 10
        factor = config.get("factor", 0.1) if config else 0.1
        return ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
    elif scheduler_type == "qat":
        return get_qat_scheduler(optimizer, config)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")