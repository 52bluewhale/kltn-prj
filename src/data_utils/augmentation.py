"""
Augmentation utilities for YOLOv8 QAT training.

This module provides specialized augmentation techniques that are
compatible with quantization-aware training, ensuring that the
augmentations don't introduce unexpected statistical shifts.
"""

import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
from torch.nn import functional as F
from ultralytics.data.augment import Albumentations as YoloAlbumentations

class QATMosaic:
    """
    QAT-friendly mosaic augmentation.
    
    This implementation of mosaic augmentation is designed to maintain
    tensor statistics suitable for quantization by controlling value ranges.
    """
    
    def __init__(
        self,
        img_size: int = 640,
        mosaic_prob: float = 0.5,
        mosaic_scale: Tuple[float, float] = (0.5, 1.5)
    ):
        self.img_size = img_size
        self.mosaic_prob = mosaic_prob
        self.mosaic_scale = mosaic_scale
        
    def __call__(
        self, 
        images: List[np.ndarray], 
        labels: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply QAT-friendly mosaic augmentation to a batch of images.
        
        Args:
            images: List of input images as numpy arrays
            labels: List of labels corresponding to the images
            
        Returns:
            Tuple of (mosaic_image, mosaic_labels)
        """
        if np.random.rand() > self.mosaic_prob or len(images) < 4:
            # Return the first image if mosaic isn't applied
            return images[0], labels[0]
        
        # Select 4 images for mosaic
        indices = np.random.choice(len(images), 4, replace=False)
        selected_images = [images[i] for i in indices]
        selected_labels = [labels[i] for i in indices]
        
        # Create mosaic
        result_img, result_labels = self._create_mosaic(selected_images, selected_labels)
        
        return result_img, result_labels
    
    def _create_mosaic(
        self, 
        images: List[np.ndarray], 
        labels: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a mosaic from 4 images with QAT-friendly value distribution.
        
        Args:
            images: List of 4 input images
            labels: List of 4 corresponding label sets
            
        Returns:
            Tuple of (mosaic_image, combined_labels)
        """
        # Initialize output image and labels
        output_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        combined_labels = []
        
        # Center point of mosaic
        cx, cy = self.img_size // 2, self.img_size // 2
        
        # Random scaling factor
        scale = np.random.uniform(*self.mosaic_scale)
        
        # Apply for each position (top-left, top-right, bottom-left, bottom-right)
        for i, (img, img_labels) in enumerate(zip(images, labels)):
            # Original dimensions
            h, w = img.shape[:2]
            
            # Place images in 4 positions
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = 0, 0, cx, cy
                x1b, y1b, x2b, y2b = w - cx, h - cy, w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = cx, 0, self.img_size, cy
                x1b, y1b, x2b, y2b = 0, h - cy, cx, h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = 0, cy, cx, self.img_size
                x1b, y1b, x2b, y2b = w - cx, 0, w, cy
            elif i == 3:  # bottom-right
                x1a, y1a, x2a, y2a = cx, cy, self.img_size, self.img_size
                x1b, y1b, x2b, y2b = 0, 0, cx, cy
            
            # Copy part of the image
            output_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # Adjust labels
            if len(img_labels):
                # Clone labels
                adjusted_labels = img_labels.copy()
                
                # Adjust box coordinates
                if i == 0:  # top-left
                    adjusted_labels[:, 1] = (adjusted_labels[:, 1] * w - (w - cx)) / cx
                    adjusted_labels[:, 2] = (adjusted_labels[:, 2] * h - (h - cy)) / cy
                elif i == 1:  # top-right
                    adjusted_labels[:, 1] = adjusted_labels[:, 1] * w / cx
                    adjusted_labels[:, 2] = (adjusted_labels[:, 2] * h - (h - cy)) / cy
                elif i == 2:  # bottom-left
                    adjusted_labels[:, 1] = (adjusted_labels[:, 1] * w - (w - cx)) / cx
                    adjusted_labels[:, 2] = adjusted_labels[:, 2] * h / cy
                elif i == 3:  # bottom-right
                    adjusted_labels[:, 1] = adjusted_labels[:, 1] * w / cx
                    adjusted_labels[:, 2] = adjusted_labels[:, 2] * h / cy
                
                # Filter out boxes with invalid coordinates
                valid_indices = (
                    (0 <= adjusted_labels[:, 1]) & 
                    (adjusted_labels[:, 1] <= 1) & 
                    (0 <= adjusted_labels[:, 2]) & 
                    (adjusted_labels[:, 2] <= 1)
                )
                if valid_indices.any():
                    combined_labels.append(adjusted_labels[valid_indices])
        
        # Combine all valid labels
        if combined_labels:
            combined_labels = np.concatenate(combined_labels, axis=0)
        else:
            combined_labels = np.zeros((0, 5))  # Empty array with label format
        
        return output_img, combined_labels


class PreserveDetailAugmentation:
    """
    Augmentation that preserves details important for quantization.
    
    This class provides augmentations that avoid destroying fine details
    that are important for maintaining good quantization behavior.
    """
    
    def __init__(self, img_size: int = 640, preserve_prob: float = 0.5):
        self.img_size = img_size
        self.preserve_prob = preserve_prob
        self.transform = A.Compose([
            # Geometric transforms that preserve details
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=15, 
                border_mode=cv2.BORDER_CONSTANT,
                p=0.7
            ),
            # Color transforms that are quantization-friendly
            A.OneOf([
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            ], p=0.7),
            # Noise that mimics quantization noise
            A.OneOf([
                A.GaussNoise(var_limit=(5, 15), p=0.5),
                A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=0.5),
            ], p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3))
        
    def __call__(
        self, 
        image: np.ndarray, 
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply detail-preserving augmentation.
        
        Args:
            image: Input image
            labels: Optional bounding box labels
            
        Returns:
            Tuple of (augmented_image, augmented_labels)
        """
        if np.random.rand() > self.preserve_prob:
            return image, labels
        
        if labels is not None and len(labels):
            # Convert labels to the format expected by albumentations
            boxes = labels[:, 1:5].copy()  # YOLO format: class, x, y, w, h
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels[:, 0])
            
            # Recombine labels
            if transformed['bboxes']:
                transformed_boxes = np.array(transformed['bboxes'])
                transformed_classes = np.array(transformed['class_labels']).reshape(-1, 1)
                transformed_labels = np.hstack([transformed_classes, transformed_boxes])
            else:
                transformed_labels = np.zeros((0, 5))
                
            return transformed['image'], transformed_labels
        else:
            transformed = self.transform(image=image)
            return transformed['image'], labels


def get_training_transforms(img_size: int = 640) -> A.Compose:
    """
    Get standard training augmentations for YOLOv8.
    
    Args:
        img_size: Target image size
        
    Returns:
        Albumentations Compose object with training transforms
    """
    return A.Compose([
        A.RandomResizedCrop(
            height=img_size, 
            width=img_size, 
            scale=(0.8, 1.0), 
            ratio=(0.9, 1.1), 
            p=0.5
        ),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=0.2),
            A.MedianBlur(blur_limit=7, p=0.2),
            A.GaussianBlur(blur_limit=7, p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.3),
            A.PiecewiseAffine(p=0.3),
        ], p=0.2),
        # Additional augmentations
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1, label_fields=['class_labels']))


def get_validation_transforms(img_size: int = 640) -> A.Compose:
    """
    Get validation transforms for YOLOv8.
    
    Args:
        img_size: Target image size
        
    Returns:
        Albumentations Compose object with validation transforms
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1, label_fields=['class_labels']))


def get_qat_specific_transforms(img_size: int = 640) -> A.Compose:
    """
    Get transforms specifically designed for QAT training.
    
    These transforms avoid introducing extreme values that could affect
    quantization statistics, while still providing good augmentation.
    
    Args:
        img_size: Target image size
        
    Returns:
        Albumentations Compose object with QAT-friendly transforms
    """
    return A.Compose([
        # Moderate geometric transforms
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        # Conservative color transforms that don't alter statistics too much
        A.OneOf([
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        ], p=0.5),
        # Add some quantization-like noise to make model more robust
        A.OneOf([
            A.GaussNoise(var_limit=(5, 15), p=0.5),
            A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=0.5),
        ], p=0.3),
        # Normalize with standard ImageNet values
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1, label_fields=['class_labels']))


class YOLOv8QATAugmentations:
    """
    Collection of YOLOv8-specific augmentations adapted for QAT.
    
    This class wraps common YOLOv8 augmentations but modifies them to be
    more compatible with quantization-aware training.
    """
    
    def __init__(
        self,
        img_size: int = 640,
        mosaic_prob: float = 0.3,
        mixup_prob: float = 0.15,
        copy_paste_prob: float = 0.1
    ):
        self.img_size = img_size
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.copy_paste_prob = copy_paste_prob
        
        # Initialize QAT-friendly mosaic
        self.mosaic = QATMosaic(img_size=img_size, mosaic_prob=mosaic_prob)
        self.preserve_detail = PreserveDetailAugmentation(img_size=img_size)
        
        # Other base augmentations
        self.yolo_albumentation = YoloAlbumentations()
        
    def __call__(
        self, 
        images: Union[np.ndarray, List[np.ndarray]], 
        labels: Union[np.ndarray, List[np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a set of QAT-friendly augmentations to images and labels.
        
        Args:
            images: Input images or batch of images
            labels: Corresponding labels
            
        Returns:
            Tuple of (augmented_images, augmented_labels)
        """
        # Convert single image to list for consistent handling
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
            labels = [labels]
            is_single = True
        else:
            is_single = False
            
        # Apply mosaic augmentation with probability
        if len(images) >= 4 and np.random.rand() < self.mosaic_prob:
            img, label = self.mosaic(images, labels)
        else:
            # Use the first image if mosaic is not applied
            img, label = images[0], labels[0]
            
        # Apply additional detail-preserving augmentations
        img, label = self.preserve_detail(img, label)
        
        # Apply standard YOLOv8 augmentations but with controlled parameters
        if np.random.rand() < 0.5:
            img, label = self.yolo_albumentation(img, label)
        
        # Return single image or batch depending on input
        if is_single:
            return img, label
        else:
            # Create a batch with the single processed image
            return np.array([img]), np.array([label])