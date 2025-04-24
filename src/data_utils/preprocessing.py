"""
Preprocessing utilities for YOLOv8 QAT training.

This module provides specialized preprocessing functions for preparing
data for quantization-aware training, ensuring that the input data
has appropriate statistical properties.
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any


def prepare_image(
    img_path: str,
    img_size: int = 640,
    normalize: bool = True,
    to_tensor: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Prepare an image for inference or training.
    
    Args:
        img_path: Path to the image file
        img_size: Target image size
        normalize: Whether to normalize pixel values
        to_tensor: Whether to convert to PyTorch tensor
        
    Returns:
        Prepared image as numpy array or PyTorch tensor
    """
    # Read image with OpenCV
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, (img_size, img_size))
    
    # Ensure correct datatype for quantization
    img = img.astype(np.float32)
    
    if normalize:
        # Standard normalization values for ImageNet pre-trained models
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = img / 255.0  # Scale to [0, 1]
        img = (img - mean) / std
    
    if to_tensor:
        # Convert to tensor and change channel order
        img = img.transpose(2, 0, 1)  # HWC to CHW format
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
    
    return img


def preprocess_batch(
    batch: List[Union[np.ndarray, torch.Tensor]],
    normalize: bool = True,
    pad_to_square: bool = True
) -> torch.Tensor:
    """
    Preprocess a batch of images for model input.
    
    Args:
        batch: List of images
        normalize: Whether to normalize pixel values
        pad_to_square: Whether to pad images to square
        
    Returns:
        Batch tensor ready for model input
    """
    if not batch:
        return None
    
    # Convert to numpy arrays if tensors
    if isinstance(batch[0], torch.Tensor):
        batch = [img.numpy() if img.is_cuda else img.cpu().numpy() for img in batch]
    
    # Determine target shape
    if pad_to_square:
        max_h = max(img.shape[0] for img in batch)
        max_w = max(img.shape[1] for img in batch)
        target_shape = (max(max_h, max_w), max(max_h, max_w))
    else:
        max_h = max(img.shape[0] for img in batch)
        max_w = max(img.shape[1] for img in batch)
        target_shape = (max_h, max_w)
    
    # Process each image
    processed_batch = []
    for img in batch:
        # Handle padding
        if pad_to_square or img.shape[0] != target_shape[0] or img.shape[1] != target_shape[1]:
            padded_img = np.zeros((target_shape[0], target_shape[1], 3), dtype=np.float32)
            padded_img[:img.shape[0], :img.shape[1], :] = img
            img = padded_img
        
        # Normalize if needed
        if normalize:
            img = img / 255.0  # Scale to [0, 1]
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
            img = (img - mean) / std
        
        # Convert to CHW format
        img = img.transpose(2, 0, 1)
        processed_batch.append(img)
    
    # Stack into batch tensor
    batch_tensor = torch.from_numpy(np.stack(processed_batch, axis=0))
    return batch_tensor


def quantization_preprocessing(img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply preprocessing specific to quantization-aware training.
    
    This function ensures that the input image has appropriate statistical
    properties for quantization, potentially including clipping outliers,
    special normalization, or other adjustments.
    
    Args:
        img: Input image as numpy array or PyTorch tensor
        
    Returns:
        Preprocessed image suitable for QAT
    """
    # Convert tensor to numpy if needed
    is_tensor = isinstance(img, torch.Tensor)
    if is_tensor:
        # Save original device
        device = img.device
        img = img.cpu().numpy()
    
    # Apply quantization-friendly preprocessing
    if img.ndim == 3:  # Single image
        # Ensure values are in good range for quantization
        # Slightly clip extreme values to avoid quantization issues
        if img.max() > 0:  # Only normalize non-empty images
            percentile_99 = np.percentile(img, 99)
            percentile_1 = np.percentile(img, 1)
            img = np.clip(img, percentile_1, percentile_99)
            
            # Rescale to utilize full range while avoiding extremes
            img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 0.9 + 0.05
    elif img.ndim == 4:  # Batch of images
        for i in range(img.shape[0]):
            if img[i].max() > 0:  # Only normalize non-empty images
                percentile_99 = np.percentile(img[i], 99)
                percentile_1 = np.percentile(img[i], 1)
                img[i] = np.clip(img[i], percentile_1, percentile_99)
                
                # Rescale to utilize full range while avoiding extremes
                img[i] = (img[i] - img[i].min()) / (img[i].max() - img[i].min() + 1e-6) * 0.9 + 0.05
    
    # Convert back to tensor if needed
    if is_tensor:
        img = torch.from_numpy(img).to(device)
    
    return img


def normalize_for_quantization(
    tensor: torch.Tensor,
    per_channel: bool = True,
    symmetric: bool = False,
    channel_dim: int = 1
) -> torch.Tensor:
    """
    Normalize tensor specifically for quantization.
    
    This function applies normalization that is designed to work well
    with quantization by optimizing the range of values to reduce
    quantization error.
    
    Args:
        tensor: Input tensor
        per_channel: Whether to normalize per channel
        symmetric: Whether to use symmetric normalization around zero
        channel_dim: Dimension for channels
        
    Returns:
        Normalized tensor suitable for quantization
    """
    if per_channel:
        # Get dimensions for reduction
        reduce_dims = list(range(tensor.dim()))
        reduce_dims.pop(channel_dim)
        
        # Calculate statistics per channel
        min_vals, _ = torch.min(tensor, dim=reduce_dims, keepdim=True)
        max_vals, _ = torch.max(tensor, dim=reduce_dims, keepdim=True)
        
        if symmetric:
            # For symmetric quantization, center around zero
            abs_max = torch.max(torch.abs(min_vals), torch.abs(max_vals))
            tensor = torch.clamp(tensor, -abs_max, abs_max)
            tensor = tensor / (abs_max + 1e-6)  # Scale to [-1, 1]
        else:
            # For asymmetric quantization, scale to [0, 1]
            tensor = (tensor - min_vals) / (max_vals - min_vals + 1e-6)
            # Slightly reduce the range to avoid edge cases
            tensor = tensor * 0.95 + 0.025
    else:
        # Global normalization
        min_val = tensor.min()
        max_val = tensor.max()
        
        if symmetric:
            # Center around zero for symmetric quantization
            abs_max = max(abs(min_val.item()), abs(max_val.item()))
            tensor = torch.clamp(tensor, -abs_max, abs_max)
            tensor = tensor / (abs_max + 1e-6)
        else:
            # Scale to [0, 1] for asymmetric quantization
            tensor = (tensor - min_val) / (max_val - min_val + 1e-6)
            # Slightly reduce the range to avoid edge cases
            tensor = tensor * 0.95 + 0.025
    
    return tensor


def apply_input_calibration(
    batch: torch.Tensor,
    calibration_stats: Dict[str, Any]
) -> torch.Tensor:
    """
    Apply calibration to input batch based on pre-computed statistics.
    
    Args:
        batch: Input batch of images
        calibration_stats: Dictionary of calibration statistics
        
    Returns:
        Calibrated batch
    """
    # Check if calibration stats are provided
    if not calibration_stats:
        return batch
    
    # Extract calibration statistics
    if 'min' in calibration_stats and 'max' in calibration_stats:
        min_val = calibration_stats['min']
        max_val = calibration_stats['max']
        
        # Apply calibration
        batch = torch.clamp(batch, min_val, max_val)
        batch = (batch - min_val) / (max_val - min_val + 1e-6)
    
    elif 'mean' in calibration_stats and 'std' in calibration_stats:
        mean = calibration_stats['mean']
        std = calibration_stats['std']
        
        # Apply normalization
        if mean.dim() == 1:  # Per-channel stats
            # Reshape for broadcasting
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        
        # Apply normalization
        batch = (batch - mean) / (std + 1e-6)
    
    return batch


def prepare_batch_for_qat(
    batch: torch.Tensor,
    device: torch.device,
    calibration_stats: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """
    Prepare a batch specifically for QAT training.
    
    Args:
        batch: Input batch of images
        device: Target device
        calibration_stats: Optional calibration statistics
        
    Returns:
        Batch prepared for QAT
    """
    # Move to device
    batch = batch.to(device)
    
    # Apply quantization-specific preprocessing
    batch = quantization_preprocessing(batch)
    
    # Apply calibration if provided
    if calibration_stats:
        batch = apply_input_calibration(batch, calibration_stats)
    
    return batch