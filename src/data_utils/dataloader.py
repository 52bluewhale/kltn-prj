"""
Dataloader utilities for YOLOv8 QAT training.

This module provides specialized dataloaders for training and quantization-aware training 
that integrate with the YOLOv8 ecosystem.
"""

import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from ultralytics.yolo.data.dataset import YOLODataset
from typing import Dict, List, Optional, Tuple, Union, Callable

from .augmentation import get_training_transforms, get_validation_transforms, get_qat_specific_transforms
from .preprocessing import preprocess_batch, quantization_preprocessing


def get_dataset_from_yaml(yaml_path: str) -> Dict:
    """
    Load dataset configuration from YAML file.
    
    Args:
        yaml_path: Path to the dataset YAML file
        
    Returns:
        Dict containing dataset configuration
    """
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Convert relative paths to absolute if needed
    base_dir = os.path.dirname(yaml_path)
    for split in ['train', 'val', 'test']:
        if split in data_config and not os.path.isabs(data_config[split]):
            data_config[split] = os.path.normpath(
                os.path.join(base_dir, data_config[split])
            )
    
    return data_config


class QATDataset(YOLODataset):
    """
    Extended YOLODataset with quantization-specific features.
    
    This dataset adds specific functionality for quantization-aware training,
    including specialized preprocessing and transformations.
    """
    
    def __init__(
        self,
        img_path: str,
        label_path: Optional[str] = None,
        img_size: int = 640,
        batch_size: int = 16,
        augment: bool = False,
        hyp: Optional[Dict] = None,
        rect: bool = False,
        cache: bool = False,
        single_cls: bool = False,
        stride: int = 32,
        pad: float = 0.0,
        prefix: str = "",
        use_qat_transforms: bool = False,
        quantization_preprocessing: Optional[Callable] = None
    ):
        # Initialize with parent class
        super().__init__(
            img_path=img_path,
            label_path=label_path,
            img_size=img_size,
            batch_size=batch_size,
            augment=augment,
            hyp=hyp,
            rect=rect,
            cache=cache,
            single_cls=single_cls,
            stride=stride,
            pad=pad,
            prefix=prefix
        )
        
        # QAT-specific attributes
        self.use_qat_transforms = use_qat_transforms
        self.quantization_preprocessing = quantization_preprocessing
        
    def __getitem__(self, index):
        """Override getitem to add quantization-specific preprocessing."""
        # Get standard item from parent class
        img, labels, img_path, shapes = super().__getitem__(index)
        
        # Apply QAT-specific processing if needed
        if self.use_qat_transforms and self.quantization_preprocessing:
            img = self.quantization_preprocessing(img)
            
        return img, labels, img_path, shapes


def create_dataloader(
    dataset_yaml: str,
    batch_size: int = 16,
    img_size: int = 640,
    augment: bool = True,
    shuffle: bool = True,
    workers: int = 8,
    mode: str = 'train',
    rect: bool = False,
    cache: bool = False,
    single_cls: bool = False,
    hyp: Optional[Dict] = None
) -> Tuple[DataLoader, Dataset]:
    """
    Create a standard dataloader for YOLOv8 training.
    
    Args:
        dataset_yaml: Path to dataset YAML file
        batch_size: Batch size
        img_size: Image size (square)
        augment: Whether to apply augmentation
        shuffle: Whether to shuffle the dataset
        workers: Number of workers for dataloader
        mode: 'train', 'val', or 'test'
        rect: Use rectangular training
        cache: Cache images for faster training
        single_cls: Treat as single-class dataset
        hyp: Hyperparameters dictionary
        
    Returns:
        Tuple of (dataloader, dataset)
    """
    data_dict = get_dataset_from_yaml(dataset_yaml)
    
    # Determine the correct path based on mode
    if mode == 'train':
        img_path = data_dict.get('train')
        augment = augment  # Use augmentation for training
    elif mode == 'val':
        img_path = data_dict.get('val')
        augment = False  # No augmentation for validation
    else:  # test
        img_path = data_dict.get('test')
        augment = False  # No augmentation for testing
    
    # Get label path from images path
    img_dir = os.path.dirname(img_path)
    base_dir = os.path.dirname(img_dir)
    label_path = os.path.join(base_dir, 'labels')
    if not os.path.exists(label_path):
        label_path = None  # For inference without labels
        
    # Create dataset
    dataset = YOLODataset(
        img_path=img_path,
        label_path=label_path,
        img_size=img_size,
        batch_size=batch_size,
        augment=augment,
        hyp=hyp,
        rect=rect,
        cache=cache,
        single_cls=single_cls,
        stride=32,  # Standard stride for YOLOv8
        pad=0.5
    )
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not rect,  # Disable shuffle for rectangular training
        num_workers=workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    
    return loader, dataset


def create_qat_dataloader(
    dataset_yaml: str,
    batch_size: int = 16,
    img_size: int = 640,
    augment: bool = True,
    shuffle: bool = True,
    workers: int = 8,
    mode: str = 'train',
    rect: bool = False,
    cache: bool = False,
    single_cls: bool = False,
    hyp: Optional[Dict] = None,
    use_qat_transforms: bool = True
) -> Tuple[DataLoader, Dataset]:
    """
    Create a specialized dataloader for QAT training.
    
    This dataloader applies quantization-specific preprocessing and augmentations
    that are more suitable for QAT training.
    
    Args:
        dataset_yaml: Path to dataset YAML file
        batch_size: Batch size
        img_size: Image size (square)
        augment: Whether to apply augmentation
        shuffle: Whether to shuffle the dataset
        workers: Number of workers for dataloader
        mode: 'train', 'val', or 'test'
        rect: Use rectangular training
        cache: Cache images for faster training
        single_cls: Treat as single-class dataset
        hyp: Hyperparameters dictionary
        use_qat_transforms: Whether to use QAT-specific transforms
        
    Returns:
        Tuple of (dataloader, dataset)
    """
    data_dict = get_dataset_from_yaml(dataset_yaml)
    
    # Determine the correct path based on mode
    if mode == 'train':
        img_path = data_dict.get('train')
        augment = augment  # Use augmentation for training
    elif mode == 'val':
        img_path = data_dict.get('val')
        augment = False  # No augmentation for validation
    else:  # test
        img_path = data_dict.get('test')
        augment = False  # No augmentation for testing
    
    # Get label path from images path
    img_dir = os.path.dirname(img_path)
    base_dir = os.path.dirname(img_dir)
    label_path = os.path.join(base_dir, 'labels')
    if not os.path.exists(label_path):
        label_path = None  # For inference without labels
        
    # Create QAT-specific dataset
    dataset = QATDataset(
        img_path=img_path,
        label_path=label_path,
        img_size=img_size,
        batch_size=batch_size,
        augment=augment,
        hyp=hyp,
        rect=rect,
        cache=cache,
        single_cls=single_cls,
        stride=32,  # Standard stride for YOLOv8
        pad=0.5,
        use_qat_transforms=use_qat_transforms and mode == 'train',
        quantization_preprocessing=quantization_preprocessing if use_qat_transforms else None
    )
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not rect,  # Disable shuffle for rectangular training
        num_workers=workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    
    return loader, dataset


class CalibrationDataset(Dataset):
    """
    Dataset for calibrating quantization parameters.
    
    This dataset is specifically designed for the calibration phase of quantization,
    applying minimal augmentation to capture representative activation statistics.
    """
    
    def __init__(
        self,
        img_path: str,
        img_size: int = 640,
        cache: bool = False,
        transform: Optional[Callable] = None
    ):
        self.img_path = img_path
        self.img_size = img_size
        self.transform = transform
        self.cache = cache
        self.samples = []
        
        # Collect image paths
        if os.path.isdir(img_path):
            self.img_files = [
                os.path.join(img_path, f) 
                for f in os.listdir(img_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
        else:
            with open(img_path, 'r') as f:
                self.img_files = [line.strip() for line in f]
                
        # Cache images if requested
        self.cached_images = [None] * len(self.img_files) if cache else None
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        # Get image
        if self.cache and self.cached_images[index] is not None:
            img = self.cached_images[index]
        else:
            # Use OpenCV to maintain compatibility with YOLOv8
            import cv2
            img = cv2.imread(self.img_files[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            # Cache if needed
            if self.cache:
                self.cached_images[index] = img
        
        # Apply transforms if provided
        if self.transform:
            img = self.transform(image=img)['image']
        
        # Convert to tensor and normalize
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0
        
        return img, self.img_files[index]


def create_calibration_dataloader(
    img_path: str,
    batch_size: int = 8,
    img_size: int = 640,
    workers: int = 4,
    cache: bool = False
) -> DataLoader:
    """
    Create a dataloader specifically for calibration.
    
    Args:
        img_path: Path to calibration images
        batch_size: Batch size
        img_size: Image size (square)
        workers: Number of workers
        cache: Whether to cache images
        
    Returns:
        DataLoader for calibration
    """
    # Create dataset with minimal preprocessing
    dataset = CalibrationDataset(
        img_path=img_path,
        img_size=img_size,
        cache=cache,
        transform=get_validation_transforms(img_size)
    )
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle to get diverse samples for calibration
        num_workers=workers,
        pin_memory=True
    )
    
    return loader