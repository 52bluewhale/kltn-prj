import os
import re
import logging
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Setup logging
logger = logging.getLogger(__name__)

class Callback:
    """
    Base callback class for model training hooks.
    """
    
    def on_train_begin(self, trainer):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer):
        """Called at the beginning of an epoch."""
        pass
    
    def on_epoch_end(self, trainer, logs=None):
        """Called at the end of an epoch."""
        pass
    
    def on_batch_begin(self, trainer, batch_idx):
        """Called at the beginning of a batch."""
        pass
    
    def on_batch_end(self, trainer, batch_idx, logs=None):
        """Called at the end of a batch."""
        pass


class ModelCheckpoint(Callback):
    """
    Callback to save model checkpoints during training.
    """
    
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=True):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Whether to save only the best model
            verbose: Whether to display messages
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        # Initialize best metric
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
    
    def on_epoch_end(self, trainer, logs=None):
        """
        Save model checkpoint at the end of an epoch.
        
        Args:
            trainer: Trainer instance
            logs: Dictionary of logs
        """
        if logs is None or self.monitor not in logs:
            return
        
        current_metric = logs[self.monitor]
        
        # Check if model improved
        if (self.mode == 'min' and current_metric < self.best_metric) or \
           (self.mode == 'max' and current_metric > self.best_metric):
            # Update best metric
            self.best_metric = current_metric
            
            # Format filepath
            formatted_filepath = self.filepath.format(
                epoch=trainer.current_epoch + 1,
                **{k: v for k, v in logs.items() if isinstance(v, (int, float))}
            )
            
            # Save model
            if self.verbose:
                logger.info(f"Saving model to {formatted_filepath}")
            
            trainer.save_model(formatted_filepath)
        elif not self.save_best_only:
            # Save model even if not improved
            formatted_filepath = self.filepath.format(
                epoch=trainer.current_epoch + 1,
                **{k: v for k, v in logs.items() if isinstance(v, (int, float))}
            )
            
            # Save model
            if self.verbose:
                logger.info(f"Saving model to {formatted_filepath}")
            
            trainer.save_model(formatted_filepath)


class EarlyStopping(Callback):
    """
    Callback to stop training early if monitored metric stops improving.
    """
    
    def __init__(self, monitor='val_loss', patience=10, mode='min', min_delta=0, verbose=True):
        """
        Initialize early stopping callback.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement before stopping
            mode: 'min' or 'max'
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to display messages
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        
        # Initialize state
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
    
    def on_epoch_end(self, trainer, logs=None):
        """
        Check if training should be stopped.
        
        Args:
            trainer: Trainer instance
            logs: Dictionary of logs
            
        Returns:
            Whether to continue training
        """
        if logs is None or self.monitor not in logs:
            return True
        
        current_metric = logs[self.monitor]
        
        # Check if model improved
        if (self.mode == 'min' and current_metric < self.best_metric - self.min_delta) or \
           (self.mode == 'max' and current_metric > self.best_metric + self.min_delta):
            # Update best metric
            self.best_metric = current_metric
            self.wait = 0
        else:
            # Increment wait counter
            self.wait += 1
            
            # Check if patience is exhausted
            if self.wait >= self.patience:
                if self.verbose:
                    logger.info(f"Early stopping triggered after {trainer.current_epoch + 1} epochs")
                return False
        
        return True


class TensorBoardLogger(Callback):
    """
    Callback to log metrics to TensorBoard.
    """
    
    def __init__(self, log_dir='logs/tensorboard', histogram_freq=0):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save logs
            histogram_freq: Frequency of weight histogram logging
        """
        super().__init__()
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        
        if SummaryWriter is None:
            logger.warning("TensorBoard not available. Install tensorflow or tensorboard.")
            self.writer = None
        else:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
    
    def on_epoch_end(self, trainer, logs=None):
        """
        Log metrics to TensorBoard.
        
        Args:
            trainer: Trainer instance
            logs: Dictionary of logs
        """
        if self.writer is None or logs is None:
            return
        
        # Log scalars
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, trainer.current_epoch)
        
        # Log histograms
        if self.histogram_freq > 0 and trainer.current_epoch % self.histogram_freq == 0:
            for name, param in trainer.model.named_parameters():
                self.writer.add_histogram(name, param.data.cpu().numpy(), trainer.current_epoch)
    
    def on_train_end(self, trainer):
        """
        Close TensorBoard writer.
        
        Args:
            trainer: Trainer instance
        """
        if self.writer is not None:
            self.writer.close()


class LearningRateScheduler(Callback):
    """
    Callback to adjust learning rate during training.
    """
    
    def __init__(self, scheduler, monitor='val_loss', verbose=True):
        """
        Initialize learning rate scheduler.
        
        Args:
            scheduler: Learning rate scheduler
            monitor: Metric to monitor for ReduceLROnPlateau
            verbose: Whether to display messages
        """
        super().__init__()
        self.scheduler = scheduler
        self.monitor = monitor
        self.verbose = verbose
    
    def on_epoch_end(self, trainer, logs=None):
        """
        Adjust learning rate at the end of an epoch.
        
        Args:
            trainer: Trainer instance
            logs: Dictionary of logs
        """
        if self.scheduler is None:
            return
        
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if logs is not None and self.monitor in logs:
                self.scheduler.step(logs[self.monitor])
        else:
            self.scheduler.step()
        
        if self.verbose:
            lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else trainer.optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {lr:.6f}")


class ProgressiveQuantization(Callback):
    """
    Callback to progressively apply quantization to model layers.
    """
    
    def __init__(self, progressive_stages, start_epoch=0, verbose=True):
        """
        Initialize progressive quantization callback.
        
        Args:
            progressive_stages: List of dictionaries with stage configuration
            start_epoch: Epoch to start quantization
            verbose: Whether to display messages
        """
        super().__init__()
        self.progressive_stages = progressive_stages
        self.start_epoch = start_epoch
        self.verbose = verbose
        self.current_stage = -1
    
    def on_epoch_begin(self, trainer):
        """
        Apply quantization settings for the current epoch.
        
        Args:
            trainer: Trainer instance
        """
        # Check if quantization should start
        if trainer.current_epoch < self.start_epoch:
            return
        
        # Find current stage
        stage_idx = -1
        for i, stage in enumerate(self.progressive_stages):
            if trainer.current_epoch >= self.start_epoch + stage.get('epoch', 0):
                stage_idx = i
        
        # Skip if stage hasn't changed
        if stage_idx == self.current_stage:
            return
        
        # Update current stage
        self.current_stage = stage_idx
        
        if stage_idx >= 0:
            # Apply quantization for current stage
            stage = self.progressive_stages[stage_idx]
            layer_patterns = stage.get('layers', [])
            qconfig_name = stage.get('qconfig', 'default')
            
            if self.verbose:
                logger.info(f"Applying {qconfig_name} quantization to {len(layer_patterns)} layer patterns in stage {stage_idx + 1}")
            
            # Apply quantization to matching layers
            self._apply_quantization_to_layers(trainer.model, layer_patterns, qconfig_name)
    
    def _apply_quantization_to_layers(self, model, layer_patterns, qconfig_name):
        """
        Apply quantization to layers matching patterns.
        
        Args:
            model: Model to apply quantization to
            layer_patterns: List of regex patterns for layer names
            qconfig_name: Name of QConfig to apply
        """
        try:
            # More robust import approach
            try:
                from ..quantization.qconfig import get_qconfig_by_name
            except ImportError:
                import sys
                if 'quantization.qconfig' in sys.modules:
                    get_qconfig_by_name = sys.modules['quantization.qconfig'].get_qconfig_by_name
                else:
                    raise ImportError("Could not import quantization utilities")
            
            # Get QConfig
            qconfig = get_qconfig_by_name(qconfig_name)
            
            # Apply to matching layers
            for name, module in model.named_modules():
                if any(re.match(pattern, name) for pattern in layer_patterns):
                    if hasattr(module, 'qconfig'):
                        module.qconfig = qconfig
                        if self.verbose:
                            logger.info(f"Applied {qconfig_name} to {name}")
        except ImportError:
            logger.error("Could not import quantization utilities")


class QuantizationErrorMonitor(Callback):
    """
    Callback to monitor quantization error during training.
    """
    
    def __init__(self, log_dir='logs/qat_error', save_maps=False, freq=5, verbose=True):
        """
        Initialize quantization error monitor.
        
        Args:
            log_dir: Directory to save logs
            save_maps: Whether to save error maps as images
            freq: Frequency of error monitoring in epochs
            verbose: Whether to display messages
        """
        super().__init__()
        self.log_dir = log_dir
        self.save_maps = save_maps
        self.freq = freq
        self.verbose = verbose
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        if SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
    
    def on_epoch_end(self, trainer, logs=None):
        """
        Monitor quantization error at the end of an epoch.
        
        Args:
            trainer: Trainer instance
            logs: Dictionary of logs
        """
        # Skip if not at monitoring frequency
        if trainer.current_epoch % self.freq != 0:
            return
        
        # Calculate quantization error for each layer
        layer_errors = self._calculate_quantization_error(trainer.model)
        
        # Log errors
        if self.writer is not None:
            for layer_name, error_stats in layer_errors.items():
                for stat_name, stat_value in error_stats.items():
                    self.writer.add_scalar(f"quant_error/{layer_name}/{stat_name}", 
                                           stat_value, trainer.current_epoch)
        
        # Save error maps
        if self.save_maps:
            self._save_error_maps(trainer.model, trainer.current_epoch)
        
        # Print summary
        if self.verbose:
            avg_error = np.mean([stats['mean'] for stats in layer_errors.values()])
            max_error = np.max([stats['max'] for stats in layer_errors.values()])
            logger.info(f"Quantization error - Avg: {avg_error:.6f}, Max: {max_error:.6f}")
    
    def _calculate_quantization_error(self, model):
        """
        Calculate quantization error for each layer with fake quantization.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary of layer errors
        """
        errors = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight_fake_quant') and hasattr(module, 'weight'):
                # Calculate error for weights
                orig_weight = module.weight
                quant_weight = module.weight_fake_quant(orig_weight)
                
                # Use CPU tensors to avoid memory issues
                error = (orig_weight - quant_weight).abs().detach().cpu()
                
                # Calculate statistics using .item() to avoid tensor references
                errors[name] = {
                    'mean': float(error.mean().item()),
                    'max': float(error.max().item()),
                    'std': float(error.std().item()),
                    'norm': float(torch.norm(error).item())
                }
                
                # Explicitly delete tensors to free memory
                del error
                
        return errors
    
    def _save_error_maps(self, model, epoch):
        """
        Save quantization error maps as images.
        
        Args:
            model: Model to analyze
            epoch: Current epoch
        """
        for name, module in model.named_modules():
            if hasattr(module, 'weight_fake_quant') and hasattr(module, 'weight'):
                # Calculate error for weights
                orig_weight = module.weight
                quant_weight = module.weight_fake_quant(orig_weight)
                error = (orig_weight - quant_weight).abs().detach().cpu().numpy()
                
                # Create error map
                if len(error.shape) == 4:  # Conv weights
                    # Average over input channels and kernel dimensions
                    error_map = np.mean(error, axis=(1, 2, 3))
                elif len(error.shape) == 2:  # Linear weights
                    # Average over input dimensions
                    error_map = np.mean(error, axis=1)
                else:
                    continue
                
                # Create figure
                plt.figure(figsize=(10, 5))
                plt.bar(range(len(error_map)), error_map)
                plt.title(f"Quantization Error - {name}")
                plt.xlabel("Output Channel")
                plt.ylabel("Average Error")
                
                # Get the figure before closing it
                fig = plt.gcf()
                
                # Save figure
                save_path = os.path.join(self.log_dir, f"error_map_{name.replace('.', '_')}_{epoch}.png")
                plt.savefig(save_path)
                
                # Log to TensorBoard
                if self.writer is not None:
                    self.writer.add_figure(f"error_map/{name}", fig, epoch)
                    
                # Now close the figure
                plt.close()
    
    def on_train_end(self, trainer):
        """
        Clean up at the end of training.
        
        Args:
            trainer: Trainer instance
        """
        if self.writer is not None:
            self.writer.close()