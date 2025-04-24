import torch
import math
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Cosine annealing scheduler with warm restarts.
    """
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        """
        Initialize cosine annealing scheduler.
        
        Args:
            optimizer: Optimizer
            T_0: Initial restart period
            T_mult: Restart period multiplier
            eta_min: Minimum learning rate
            last_epoch: Last epoch
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        Calculate learning rate.
        
        Returns:
            List of learning rates
        """
        if self.T_cur == -1:
            return self.base_lrs
        
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.T_cur / self.T_0)) / 2
                for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        """
        Update learning rate.
        
        Args:
            epoch: Current epoch (unused)
        """
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_0:
                self.T_cur = self.T_cur - self.T_0
                self.T_0 = self.T_0 * self.T_mult
        else:
            if epoch >= self.T_0:
                if self.T_mult > 1:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_0 = self.T_0 * self.T_mult ** n
                else:
                    self.T_cur = epoch % self.T_0
            else:
                self.T_cur = epoch
                
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class StepLR(_LRScheduler):
    """
    Step learning rate scheduler.
    """
    
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        """
        Initialize step scheduler.
        
        Args:
            optimizer: Optimizer
            step_size: Epoch interval for reduction
            gamma: Learning rate reduction factor
            last_epoch: Last epoch
        """
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        Calculate learning rate.
        
        Returns:
            List of learning rates
        """
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return self.base_lrs
        
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]


class ReduceLROnPlateau:
    """
    Reduce learning rate when metric stops improving.
    """
    
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, 
                 threshold=1e-4, min_lr=0, verbose=False):
        """
        Initialize reduce on plateau scheduler.
        
        Args:
            optimizer: Optimizer
            mode: 'min' or 'max'
            factor: Learning rate reduction factor
            patience: Epochs to wait before reducing
            threshold: Threshold for measuring improvement
            min_lr: Minimum learning rate
            verbose: Whether to print messages
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.verbose = verbose
        
        # State
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.cooldown_counter = 0
        self.reduced = False

        # For compatibility with PyTorch LR schedulers
        self.last_epoch = -1
    
    def step(self, metric):
        """
        Update learning rate based on metric.
        
        Args:
            metric: Measured value
        """
        self.last_epoch += 1

        if metric is None:
            logger.warning("ReduceLROnPlateau requires a metric value, but none was provided")
            return

        if self.mode == 'min':
            is_better = metric < self.best - self.threshold
        else:
            is_better = metric > self.best + self.threshold
        
        # Update best metric
        if is_better:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1
            
            # Check if patience is exhausted
            if self.wait >= self.patience:
                self._reduce_lr()
                self.wait = 0
    
    def _reduce_lr(self):
        """
        Reduce learning rate.
        """
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            
            if self.verbose:
                print(f"Reducing learning rate from {old_lr} to {new_lr}")

    # For compatibility with PyTorch LR schedulers
    def get_last_lr(self):
        """Return last computed learning rate by current scheduler."""
        return [group['lr'] for group in self.optimizer.param_groups]



class QATLearningRateScheduler(_LRScheduler):
    """
    Learning rate scheduler specifically for QAT.
    Manages different phases of QAT training.
    """
    
    def __init__(self, optimizer, warmup_epochs=5, initial_lr=1e-4, 
                 peak_lr=1e-3, final_lr=1e-5, quant_start_epoch=10, 
                 total_epochs=100, last_epoch=-1):
        """
        Initialize QAT scheduler.
        
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            initial_lr: Initial learning rate
            peak_lr: Peak learning rate after warmup
            final_lr: Final learning rate
            quant_start_epoch: Epoch to start quantization
            total_epochs: Total number of epochs
            last_epoch: Last epoch
        """
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.peak_lr = peak_lr
        self.final_lr = final_lr
        self.quant_start_epoch = quant_start_epoch
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        Calculate learning rate based on training phase.
        
        Returns:
            List of learning rates
        """
        epoch = self.last_epoch
        
        if epoch < self.warmup_epochs:
            # Warmup phase: linearly increase LR
            factor = epoch / self.warmup_epochs
            lr = self.initial_lr + (self.peak_lr - self.initial_lr) * factor
        elif epoch < self.quant_start_epoch:
            # Pre-quantization phase: maintain peak LR
            lr = self.peak_lr
        else:
            # Quantization phase: cosine decay to final LR
            progress = (epoch - self.quant_start_epoch) / (self.total_epochs - self.quant_start_epoch)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.final_lr + (self.peak_lr - self.final_lr) * factor
        
        return [lr for _ in self.base_lrs]


def get_qat_scheduler(optimizer, config):
    """
    Create QAT-specific learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        config: QAT configuration
        
    Returns:
        Learning rate scheduler
    """
    scheduler_config = config.get('scheduler', {})
    
    warmup_epochs = scheduler_config.get('warmup_epochs', 5)
    initial_lr = scheduler_config.get('initial_lr', 1e-4)
    peak_lr = scheduler_config.get('peak_lr', 1e-3)
    final_lr = scheduler_config.get('final_lr', 1e-5)
    quant_start_epoch = config.get('start_epoch', 10)
    total_epochs = scheduler_config.get('total_epochs', 100)
    
    return QATLearningRateScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        initial_lr=initial_lr,
        peak_lr=peak_lr,
        final_lr=final_lr,
        quant_start_epoch=quant_start_epoch,
        total_epochs=total_epochs
    )