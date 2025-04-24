# Main training loop with QAT support:
#     - Progressive quantization
#     - Specialized learning rate scheduling for QAT

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import logging
import time
from tqdm import tqdm
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

class Trainer:
    """
    Base trainer class for model training.
    """
    
    def __init__(self, model, config):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.device = self._get_device()
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup loss function
        self.criterion = self._create_criterion()
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup callbacks
        self.callbacks = self._setup_callbacks()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': {}
        }
    
    def _get_device(self):
        """
        Get appropriate device for training.
        
        Returns:
            torch.device
        """
        device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _create_optimizer(self):
        """
        Create optimizer from configuration.
        
        Returns:
            Optimizer
        """
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        lr = optimizer_config.get('lr', 1e-3)
        weight_decay = optimizer_config.get('weight_decay', 0)
        
        if optimizer_type.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def _create_criterion(self):
        """
        Create loss function from configuration.
        
        Returns:
            Loss function
        """
        from .loss import FocalLoss
        
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'focal')
        
        if loss_type == 'focal':
            gamma = loss_config.get('gamma', 2.0)
            alpha = loss_config.get('alpha', 0.25)
            return FocalLoss(gamma=gamma, alpha=alpha)
        elif hasattr(nn, loss_type):
            return getattr(nn, loss_type)()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def _create_scheduler(self):
        """
        Create learning rate scheduler from configuration.
        
        Returns:
            Learning rate scheduler
        """
        scheduler_config = self.config.get('scheduler', {})
        if not scheduler_config:
            return None
        
        scheduler_type = scheduler_config.get('type', None)
        if not scheduler_type:
            return None
        
        from .lr_scheduler import (
            CosineAnnealingWarmRestarts,
            StepLR,
            ReduceLROnPlateau
        )
        
        if scheduler_type == 'cosine':
            T_0 = scheduler_config.get('T_0', 10)
            T_mult = scheduler_config.get('T_mult', 2)
            eta_min = scheduler_config.get('eta_min', 1e-6)
            return CosineAnnealingWarmRestarts(self.optimizer, T_0, T_mult, eta_min)
        elif scheduler_type == 'step':
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            return StepLR(self.optimizer, step_size, gamma)
        elif scheduler_type == 'plateau':
            patience = scheduler_config.get('patience', 10)
            factor = scheduler_config.get('factor', 0.1)
            return ReduceLROnPlateau(self.optimizer, patience=patience, factor=factor)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def _setup_callbacks(self):
        """
        Setup callbacks from configuration.
        
        Returns:
            List of callbacks
        """
        from .callbacks import (
            ModelCheckpoint,
            EarlyStopping,
            TensorBoardLogger
        )
        
        callbacks = []
        callback_configs = self.config.get('callbacks', [])
        
        for callback_config in callback_configs:
            callback_type = callback_config.get('type', '')
            
            if callback_type == 'checkpoint':
                filepath = callback_config.get('filepath', 'checkpoints/model_{epoch:02d}_{val_loss:.4f}.pt')
                monitor = callback_config.get('monitor', 'val_loss')
                mode = callback_config.get('mode', 'min')
                save_best_only = callback_config.get('save_best_only', True)
                callbacks.append(ModelCheckpoint(filepath, monitor, mode, save_best_only))
            
            elif callback_type == 'early_stopping':
                monitor = callback_config.get('monitor', 'val_loss')
                patience = callback_config.get('patience', 10)
                mode = callback_config.get('mode', 'min')
                callbacks.append(EarlyStopping(monitor, patience, mode))
            
            elif callback_type == 'tensorboard':
                log_dir = callback_config.get('log_dir', 'logs/tensorboard')
                callbacks.append(TensorBoardLogger(log_dir))
        
        return callbacks
    
    def train(self, train_loader, val_loader=None, epochs=10):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        # Reset training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': {}
        }
        
        # Call on_train_begin for callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(self)
        
        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Call on_epoch_begin for callbacks
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_begin'):
                    callback.on_epoch_begin(self)
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                self.training_history['val_loss'].append(val_loss)
            
            # Call on_epoch_end for callbacks
            stop_training = False
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    result = callback.on_epoch_end(self, logs={
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    })
                    if result is False:
                        stop_training = True
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print epoch summary
            log_message = f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}"
            if val_loss is not None:
                log_message += f" - val_loss: {val_loss:.4f}"
            logger.info(log_message)
            
            # Check if training should be stopped
            if stop_training:
                logger.info("Early stopping triggered")
                break
        
        # Call on_train_end for callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(self)
        
        return self.training_history
    
    def _train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Call on_batch_begin for callbacks
            for callback in self.callbacks:
                if hasattr(callback, 'on_batch_begin'):
                    callback.on_batch_begin(self, batch_idx)
            
            # Extract data and labels
            data, targets = self._process_batch(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            
            # Calculate loss
            loss = self._compute_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Call on_batch_end for callbacks
            for callback in self.callbacks:
                if hasattr(callback, 'on_batch_end'):
                    callback.on_batch_end(self, batch_idx, logs={'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader):
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        # Progress bar
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Extract data and labels
                data, targets = self._process_batch(batch)
                
                # Forward pass
                outputs = self.model(data)
                
                # Calculate loss
                loss = self._compute_loss(outputs, targets)
                
                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(val_loader)
    
    def _process_batch(self, batch):
        """
        Process batch data.
        
        Args:
            batch: Batch data
            
        Returns:
            data, targets
        """
        # Handle different data formats
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            data, targets = batch[0], batch[1]
        elif isinstance(batch, dict) and 'input' in batch and 'target' in batch:
            data, targets = batch['input'], batch['target']
        else:
            raise ValueError(f"Unsupported batch format: {type(batch)}")
        
        # Move to device
        data = data.to(self.device)
        if isinstance(targets, torch.Tensor):
            targets = targets.to(self.device)
        elif isinstance(targets, (tuple, list)):
            targets = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in targets]
        
        return data, targets
    
    def _compute_loss(self, outputs, targets):
        """
        Compute loss with quantization penalty if needed.
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            Loss value
        """
        # Regular loss computation 
        loss = super()._compute_loss(outputs, targets)
        
        # Check if additional QAT-specific penalty should be applied
        if self.qat_config.get('use_penalty', False) and not isinstance(self.criterion, QATPenaltyLoss):
            # Calculate quantization penalty
            penalty = 0.0
            penalty_factor = self.qat_config.get('penalty_factor', 0.01)
            
            # Find modules with fake quantization
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight') and hasattr(module, 'weight_fake_quant'):
                    # Compute error between original and quantized weights
                    orig_weight = module.weight
                    quant_weight = module.weight_fake_quant(orig_weight)
                    error = torch.mean((orig_weight - quant_weight) ** 2)
                    penalty = penalty + error
            
            # Add penalty to loss
            loss = loss + penalty_factor * penalty
        
        return loss
    
    def save_model(self, filepath):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
            
        Returns:
            Success flag
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.current_epoch,
                'best_metric': self.best_metric,
                'training_history': self.training_history
            }, filepath)
            
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Success flag
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_metric = checkpoint.get('best_metric', float('inf'))
            self.training_history = checkpoint.get('training_history', {
                'train_loss': [],
                'val_loss': [],
                'metrics': {}
            })
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


class QATTrainer(Trainer):
    """
    Trainer for Quantization-Aware Training.
    Extends base Trainer with QAT-specific functionality.
    """
    
    def __init__(self, model, config):
        """
        Initialize QAT trainer.
        
        Args:
            model: Model to train (should be prepared for QAT)
            config: Training configuration
        """
        super().__init__(model, config)
        
        # QAT-specific configuration
        self.qat_config = config.get('qat', {})
        self.quantization_start_epoch = self.qat_config.get('start_epoch', 0)
        self.freeze_bn_epochs = self.qat_config.get('freeze_bn_epochs', 3)
        self.progressive_stages = self.qat_config.get('progressive_stages', [])
        
        # QAT-specific callbacks
        self._setup_qat_callbacks()
    
    def _create_criterion(self):
        """
        Create loss function with QAT penalty if needed.
        
        Returns:
            Loss function
        """
        # Get base criterion
        base_criterion = super()._create_criterion()
        
        # Check if QAT penalty should be added
        use_qat_penalty = self.qat_config.get('use_penalty', False)
        if use_qat_penalty:
            from .loss import QATPenaltyLoss
            penalty_factor = self.qat_config.get('penalty_factor', 0.01)
            return QATPenaltyLoss(base_criterion, penalty_factor)
        
        return base_criterion
    
    def _create_scheduler(self):
        """
        Create QAT-specific learning rate scheduler.
        
        Returns:
            Learning rate scheduler
        """
        # Check if QAT scheduler should be used
        use_qat_scheduler = self.qat_config.get('use_qat_scheduler', False)
        if use_qat_scheduler:
            from .lr_scheduler import get_qat_scheduler
            return get_qat_scheduler(self.optimizer, self.qat_config)
        
        return super()._create_scheduler()
    
    def _setup_qat_callbacks(self):
        """
        Setup QAT-specific callbacks.
        """
        from .callbacks import (
            ProgressiveQuantization,
            QuantizationErrorMonitor
        )
        
        # Add progressive quantization callback if specified
        if self.progressive_stages:
            self.callbacks.append(ProgressiveQuantization(
                self.progressive_stages,
                start_epoch=self.quantization_start_epoch
            ))
        
        # Add quantization error monitor if specified
        monitor_error = self.qat_config.get('monitor_error', False)
        if monitor_error:
            log_dir = self.qat_config.get('error_log_dir', 'logs/qat_error')
            save_maps = self.qat_config.get('save_error_maps', False)
            self.callbacks.append(QuantizationErrorMonitor(
                log_dir, save_maps=save_maps
            ))
    
    def _train_epoch(self, train_loader):
        """
        Train for one epoch with QAT specifics.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        # Check if BN layers should be frozen
        if self.current_epoch >= self.quantization_start_epoch + self.freeze_bn_epochs:
            # Freeze batch normalization layers to preserve statistics
            self._freeze_bn_layers()
        
        return super()._train_epoch(train_loader)
    
    def _freeze_bn_layers(self):
        """
        Freeze batch normalization layers.
        """
        def _freeze_bn_stats(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        
        self.model.apply(_freeze_bn_stats)
    
    def _compute_loss(self, outputs, targets):
        """
        Compute loss with quantization penalty if needed.
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            Loss value
        """
        # # Regular loss computation for QATPenaltyLoss
        # return super()._compute_loss(outputs, targets)

        # Regular loss computation 
        loss = super()._compute_loss(outputs, targets)
    
        # Check if additional QAT-specific penalty should be applied
        if self.qat_config.get('use_penalty', False) and not isinstance(self.criterion, QATPenaltyLoss):
            # Calculate quantization penalty
            penalty = 0.0
            penalty_factor = self.qat_config.get('penalty_factor', 0.01)
            
            # Find modules with fake quantization
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight') and hasattr(module, 'weight_fake_quant'):
                    # Compute error between original and quantized weights
                    orig_weight = module.weight
                    quant_weight = module.weight_fake_quant(orig_weight)
                    error = torch.mean((orig_weight - quant_weight) ** 2)
                    penalty = penalty + error
            
            # Add penalty to loss
            loss = loss + penalty_factor * penalty
    
    return loss
    
    def convert_model_to_quantized(self):
        """
        Convert QAT model to fully quantized model.
        
        Returns:
            Quantized model
        """
        try:
            from torch.quantization import convert
            
            # Try multiple import paths to be more robust
            try:
                from ..quantization.utils import convert_qat_model_to_quantized
            except ImportError:
                try:
                    from src.quantization.utils import convert_qat_model_to_quantized
                except ImportError:
                    import sys
                    if 'quantization.utils' in sys.modules:
                        convert_qat_model_to_quantized = sys.modules['quantization.utils'].convert_qat_model_to_quantized
                    else:
                        # Fallback to PyTorch's convert
                        logger.warning("Could not import custom conversion function, using PyTorch's convert")
                        convert_qat_model_to_quantized = convert
            
            # Make sure model is in eval mode
            self.model.eval()
            
            # Convert model to quantized version
            quantized_model = convert_qat_model_to_quantized(self.model, inplace=False)
            
            return quantized_model
        except Exception as e:
            logger.error(f"Error converting model to quantized: {e}")
            logger.error("Falling back to PyTorch's convert function")
            
            # Fallback to PyTorch's convert
            from torch.quantization import convert
            self.model.eval()
            return convert(self.model)
    
    def evaluate_quantized_model(self, test_loader):
        """
        Evaluate the quantized version of the model.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        # Convert model to quantized version
        quantized_model = self.convert_model_to_quantized()
        quantized_model = quantized_model.to(self.device)
        
        # Evaluate
        quantized_model.eval()
        total_loss = 0
        metrics = {}
        
        with torch.no_grad():
            for batch in test_loader:
                # Process batch
                data, targets = self._process_batch(batch)
                
                # Forward pass
                outputs = quantized_model(data)
                
                # Calculate loss
                loss = self._compute_loss(outputs, targets)
                total_loss += loss.item()
                
                # Calculate additional metrics if needed
                # ...
        
        # Compute average loss
        avg_loss = total_loss / len(test_loader)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def save_model(self, filepath):
        """
        Save QAT model with additional information.
        
        Args:
            filepath: Path to save model
            
        Returns:
            Success flag
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model with QAT info
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.current_epoch,
                'best_metric': self.best_metric,
                'training_history': self.training_history,
                'qat_config': self.qat_config
            }, filepath)
            
            logger.info(f"QAT model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving QAT model: {e}")
            return False