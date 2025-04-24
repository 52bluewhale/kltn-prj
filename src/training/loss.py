import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QATPenaltyLoss(nn.Module):
    """
    Loss function with additional penalty term for quantization error.
    Helps to guide the model towards better quantization-friendly weights.
    """
    
    def __init__(self, base_criterion, penalty_factor=0.01):
        """
        Initialize QAT penalty loss.
        
        Args:
            base_criterion: Base loss function
            penalty_factor: Weight for quantization penalty term
        """
        super().__init__()
        self.base_criterion = base_criterion
        self.penalty_factor = penalty_factor
        self.model = None
        
        # Try to get model from criterion
        if hasattr(base_criterion, 'model'):
            self.model = base_criterion.model
    
    def forward(self, outputs, targets):
        """
        Forward pass to compute loss with quantization penalty.
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            Loss value
        """
        # Compute base loss
        base_loss = self.base_criterion(outputs, targets)
        
        # Compute quantization penalty
        quant_penalty = self._compute_quantization_penalty()
        
        # Combine losses
        total_loss = base_loss + self.penalty_factor * quant_penalty
        
        return total_loss
    
    def _compute_quantization_penalty(self):
        """
        Compute penalty term based on quantization error.
        
        Returns:
            Penalty term
        """
        penalty = torch.tensor(0.0, device=next(self.base_criterion.parameters()).device
                              if hasattr(self.base_criterion, 'parameters') else 'cpu')
        
        # Collect all fake quantized parameters and their original values
        for name, module in self._find_quantized_modules():
            if hasattr(module, 'weight') and hasattr(module, 'weight_fake_quant'):
                # Compute error between original and quantized weights
                orig_weight = module.weight
                quant_weight = module.weight_fake_quant(orig_weight)
                error = torch.mean((orig_weight - quant_weight) ** 2)
                penalty = penalty + error
        
        return penalty
    
    def _find_quantized_modules(self):
        """
        Find modules with fake quantization.
        
        Returns:
            List of (name, module) pairs with fake quantization
        """
        if self.model is not None:
            return [(name, module) for name, module in self.model.named_modules()
                   if hasattr(module, 'weight_fake_quant')]
        elif hasattr(self.base_criterion, 'model'):
            # If base_criterion has access to model
            model = self.base_criterion.model
            return [(name, module) for name, module in model.named_modules()
                   if hasattr(module, 'weight_fake_quant')]
        else:
            # If no model is directly accessible, penalty will be zero
            return []


class FocalLoss(nn.Module):
    """
    Focal Loss for object detection.
    Reduces loss contribution from easy examples and focuses on hard ones.
    """
    
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        """
        Initialize focal loss.
        
        Args:
            gamma: Focusing parameter for hard examples
            alpha: Weighting factor for positive examples
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Loss value
        """
        # Convert inputs to probabilities with sigmoid
        p = torch.sigmoid(inputs)
        
        # Prepare targets
        targets = targets.float()
        
        # Calculate binary cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Apply focal weighting
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        # Combine factors
        focal_loss = alpha_factor * modulating_factor * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for model compression.
    Combines regular loss with KL divergence from teacher model.
    """
    
    def __init__(self, teacher_model, temperature=1.0, alpha=0.5):
        """
        Initialize distillation loss.
        
        Args:
            teacher_model: Teacher model for knowledge distillation
            temperature: Temperature for softening probability distributions
            alpha: Weight for distillation loss (1-alpha for regular loss)
        """
        super().__init__()
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Ensure teacher model is in eval mode
        self.teacher_model.eval()
        
        # Regular loss function for task-specific loss
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, student_outputs, targets):
        """
        Compute distillation loss.
        
        Args:
            student_outputs: Student model predictions
            targets: Ground truth labels
            
        Returns:
            Combined loss value
        """
        # Get device
        device = student_outputs.device
        
        # Convert targets if necessary
        if len(targets.shape) == 1:
            # Classification targets
            hard_targets = targets
        else:
            # One-hot encoded targets
            hard_targets = targets.argmax(dim=1)
        
        # Compute regular task loss
        task_loss = self.criterion(student_outputs, hard_targets).mean()
        
        # Compute teacher outputs (no gradient needed)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(student_outputs.detach())
        
        # Compute distillation loss (KL divergence)
        soft_targets = F.softmax(teacher_outputs / self.temperature, dim=1)
        log_probs = F.log_softmax(student_outputs / self.temperature, dim=1)
        distill_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        # Combine losses
        combined_loss = (1 - self.alpha) * task_loss + self.alpha * distill_loss
        
        return combined_loss