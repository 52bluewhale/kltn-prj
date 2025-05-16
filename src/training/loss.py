import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QATPenaltyLoss:
    """
    Custom loss with quantization penalty term.
    Combines task loss with a quantization error penalty.
    """
    def __init__(self, orig_loss_fn, model, alpha=0.01):
        self.orig_loss_fn = orig_loss_fn
        self.model = model
        self.alpha = alpha  # Weight for quantization penalty
    
    def __call__(self, preds, targets):
        # Original task loss
        task_loss = self.orig_loss_fn(preds, targets)
        
        # Quantization penalty: measure difference between quantized and non-quantized values
        quant_penalty = 0.0
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight_fake_quant') and hasattr(module, 'weight'):
                # Quantized weight
                w_q = module.weight_fake_quant(module.weight)
                # Original weight
                w = module.weight
                # Penalty is L2 norm of the difference
                quant_penalty += torch.norm(w_q - w) ** 2
        
        # Normalize by number of parameters
        quant_penalty /= sum(p.numel() for p in self.model.parameters())
        
        # Combined loss
        total_loss = task_loss + self.alpha * quant_penalty
        
        return total_loss


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