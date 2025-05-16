import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

# Setup logging
logger = logging.getLogger(__name__)

class Calibrator:
    """
    Calibrator for post-training quantization.
    """
    
    def __init__(self, model, calibration_loader, device='cuda', num_batches=100):
        """
        Initialize calibrator.
        
        Args:
            model: Model to calibrate
            calibration_loader: DataLoader for calibration
            device: Device to use for calibration
            num_batches: Number of batches to use for calibration
        """
        self.model = model
        self.calibration_loader = calibration_loader
        self.device = device
        self.num_batches = num_batches
    
    def calibrate(self, method='histogram', progress=True):
        """
        Calibrate model.
        
        Args:
            method: Calibration method
            progress: Whether to show progress bar
            
        Returns:
            Calibrated model
        """
        logger.info(f"Calibrating model using {method} method...")
        
        # Put model in eval mode for calibration
        self.model.eval()
        
        # Create iterator
        data_iter = iter(self.calibration_loader)
        
        # Create progress bar if requested
        if progress:
            pbar = tqdm(range(min(self.num_batches, len(self.calibration_loader))), 
                        desc=f"Calibrating ({method})")
        else:
            pbar = range(min(self.num_batches, len(self.calibration_loader)))
        
        # Run calibration
        with torch.no_grad():
            for _ in pbar:
                try:
                    # Get batch
                    batch = next(data_iter)
                    
                    # Handle different formats
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    
                    # Move to device
                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(self.device)
                    
                    # Forward pass to update observers
                    self.model(inputs)
                    
                except StopIteration:
                    # Restart iterator if we run out of data
                    data_iter = iter(self.calibration_loader)
                    batch = next(data_iter)
                    
                    # Handle different formats
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    
                    # Move to device
                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.to(self.device)
                    
                    # Forward pass to update observers
                    self.model(inputs)
        
        logger.info("Calibration complete")
        
        return self.model

def calibrate_model(model, dataloader, method='histogram', num_batches=100, device='cuda'):
    """
    Calibrate model using specified method.
    
    Args:
        model: Model to calibrate
        dataloader: DataLoader for calibration
        method: Calibration method (histogram, minmax, percentile)
        num_batches: Number of batches to use for calibration
        device: Device to use for calibration
        
    Returns:
        Calibrated model
    """
    calibrator = Calibrator(
        model=model,
        calibration_loader=dataloader,
        device=device,
        num_batches=num_batches
    )
    
    return calibrator.calibrate(method=method)

class PercentileCalibrator(Calibrator):
    """
    Calibrator using percentile method.
    Sets quantization parameters based on percentile of observed values.
    """
    
    def __init__(self, model, calibration_loader, device='cuda', num_batches=100, percentile=99.99):
        """
        Initialize percentile calibrator.
        
        Args:
            model: Model to calibrate
            calibration_loader: DataLoader for calibration
            device: Device to use for calibration
            num_batches: Number of batches to use for calibration
            percentile: Percentile to use for calibration
        """
        super().__init__(model, calibration_loader, device, num_batches)
        self.percentile = percentile
    
    def calibrate(self, progress=True):
        """
        Calibrate model using percentile method.
        
        Args:
            progress: Whether to show progress bar
            
        Returns:
            Calibrated model
        """
        logger.info(f"Calibrating model using percentile method (p={self.percentile})...")
        
        # Collect activations
        activations = {}
        
        # Register hooks to collect activations
        handles = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if name not in activations:
                    activations[name] = []
                # Collect sample of activations
                if len(activations[name]) < 1000:  # Limit samples to avoid memory issues
                    if isinstance(output, torch.Tensor):
                        activations[name].append(output.detach().cpu().view(-1))
                    elif isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
                        activations[name].append(output[0].detach().cpu().view(-1))
            return hook
        
        # Register hooks for modules with qconfig
        for name, module in self.model.named_modules():
            if hasattr(module, 'qconfig') and module.qconfig is not None:
                handles.append(module.register_forward_hook(hook_fn(name)))
        
        # Run forward passes to collect activations
        super().calibrate(method='percentile', progress=progress)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Calculate percentile for each module
        for name, acts in activations.items():
            if acts:
                # Concatenate activation samples
                acts_tensor = torch.cat(acts, dim=0)
                
                # Calculate percentile
                q_min = torch.quantile(acts_tensor, 1 - self.percentile/100, dim=0).item()
                q_max = torch.quantile(acts_tensor, self.percentile/100, dim=0).item()
                
                # Find module and update observers if available
                for n, m in self.model.named_modules():
                    if n == name and hasattr(m, 'activation_post_process'):
                        observer = m.activation_post_process
                        if hasattr(observer, 'min_val') and hasattr(observer, 'max_val'):
                            observer.min_val = torch.tensor(q_min)
                            observer.max_val = torch.tensor(q_max)
                            observer.initialized = torch.tensor(1, dtype=torch.bool)
        
        logger.info("Percentile calibration complete")
        
        return self.model

class EntropyCalibrator(Calibrator):
    """
    Calibrator using entropy method.
    Minimizes information loss during quantization.
    """
    
    def calibrate(self, progress=True):
        """
        Calibrate model using entropy method.
        
        Args:
            progress: Whether to show progress bar
            
        Returns:
            Calibrated model
        """
        logger.info("Calibrating model using entropy method...")
        
        # Collect activations
        activations = {}
        
        # Register hooks to collect activations
        handles = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if name not in activations:
                    activations[name] = []
                # Collect sample of activations
                if len(activations[name]) < 1000:  # Limit samples to avoid memory issues
                    if isinstance(output, torch.Tensor):
                        activations[name].append(output.detach().cpu())
                    elif isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
                        activations[name].append(output[0].detach().cpu())
            return hook
        
        # Register hooks for modules with qconfig
        for name, module in self.model.named_modules():
            if hasattr(module, 'qconfig') and module.qconfig is not None:
                handles.append(module.register_forward_hook(hook_fn(name)))
        
        # Run forward passes to collect activations
        super().calibrate(method='entropy', progress=progress)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Calculate optimal thresholds for each module
        for name, acts in activations.items():
            if acts:
                # Concatenate activation samples
                acts_tensor = torch.cat(acts, dim=0).view(-1)
                
                # Find optimal threshold using KL divergence
                threshold = self._find_optimal_threshold(acts_tensor)
                
                # Find module and update observers if available
                for n, m in self.model.named_modules():
                    if n == name and hasattr(m, 'activation_post_process'):
                        observer = m.activation_post_process
                        if hasattr(observer, 'min_val') and hasattr(observer, 'max_val'):
                            observer.min_val = torch.tensor(0.0)
                            observer.max_val = torch.tensor(threshold)
                            observer.initialized = torch.tensor(1, dtype=torch.bool)
        
        logger.info("Entropy calibration complete")
        
        return self.model
    
    def _find_optimal_threshold(self, tensor, bins=2048, quantile=0.9999):
        """
        Find optimal threshold using KL divergence.
        
        Args:
            tensor: Tensor of activations
            bins: Number of histogram bins
            quantile: Quantile to use for range
            
        Returns:
            Optimal threshold
        """
        # Trim outliers
        tensor = tensor[tensor < torch.quantile(tensor, quantile)]
        tensor = tensor[tensor > torch.quantile(tensor, 1 - quantile)]
        
        # Create histogram
        hist, bin_edges = torch.histogram(tensor, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Normalize histogram
        hist = hist / torch.sum(hist)
        
        # Add small constant to avoid zeros
        hist = hist + 1e-8
        hist = hist / torch.sum(hist)
        
        # Calculate KL divergence for different thresholds
        min_kl_div = float('inf')
        optimal_threshold = bin_centers[-1].item()
        
        # Try different thresholds
        for i in range(bins // 2, bins):
            threshold = bin_centers[i].item()
            
            # Create quantized distribution
            q_hist = torch.zeros_like(hist)
            for j in range(bins):
                if bin_centers[j] <= threshold:
                    # Quantize bin to closest quantized value
                    q_val = torch.round(bin_centers[j] / threshold * 255) * threshold / 255
                    # Find closest bin
                    closest_bin = torch.argmin(torch.abs(bin_centers - q_val))
                    # Add to quantized histogram
                    q_hist[closest_bin] += hist[j]
            
            # Normalize quantized histogram
            q_hist = q_hist + 1e-8
            q_hist = q_hist / torch.sum(q_hist)
            
            # Calculate KL divergence
            kl_div = torch.sum(hist * torch.log(hist / q_hist))
            
            # Update optimal threshold if KL divergence is lower
            if kl_div < min_kl_div:
                min_kl_div = kl_div
                optimal_threshold = threshold
        
        return optimal_threshold

def build_calibrator(model, dataloader, method='histogram', **kwargs):
    """
    Build calibrator based on method.
    
    Args:
        model: Model to calibrate
        dataloader: DataLoader for calibration
        method: Calibration method
        kwargs: Additional arguments for calibrator
        
    Returns:
        Calibrator instance
    """
    if method == 'percentile':
        percentile = kwargs.get('percentile', 99.99)
        return PercentileCalibrator(
            model=model,
            calibration_loader=dataloader,
            device=kwargs.get('device', 'cuda'),
            num_batches=kwargs.get('num_batches', 100),
            percentile=percentile
        )
    elif method == 'entropy':
        return EntropyCalibrator(
            model=model,
            calibration_loader=dataloader,
            device=kwargs.get('device', 'cuda'),
            num_batches=kwargs.get('num_batches', 100)
        )
    else:
        return Calibrator(
            model=model,
            calibration_loader=dataloader,
            device=kwargs.get('device', 'cuda'),
            num_batches=kwargs.get('num_batches', 100)
        )