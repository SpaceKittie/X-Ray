"""Data augmentation utilities for dental X-ray models."""
import albumentations as A
from detectron2.data import transforms as T
import numpy as np
import torch
from typing import Dict, List, Union, Any

def get_train_aug() -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Returns:
        Albumentations Compose object with training augmentations
    """
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(
            var_limit=(10.0, 50.0),
            mean=0,
            p=0.5
        ),
        A.GaussianBlur(
            blur_limit=(3, 7),
            p=0.3
        ),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=45,
            p=0.5
        ),
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['category_ids']
    ))

def get_val_aug() -> A.Compose:
    """
    Get validation augmentation pipeline.
    
    Returns:
        Albumentations Compose object with validation augmentations
    """
    return A.Compose([
        # Only normalize for validation
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['category_ids']
    ))

def augment_annotations(
    annotations: List[Dict[str, Any]], 
    image: np.ndarray,
    augmentation: A.Compose
) -> tuple:
    """
    Apply augmentations to image and annotations.
    
    Args:
        annotations: List of annotation dictionaries
        image: Input image as numpy array
        augmentation: Albumentations Compose object
        
    Returns:
        Tuple of (augmented image, augmented annotations)
    """
    # Extract bboxes and category IDs
    bboxes = [ann['bbox'] for ann in annotations]
    category_ids = [ann['category_id'] for ann in annotations]
    
    # Apply augmentations
    try:
        transformed = augmentation(
            image=image,
            bboxes=bboxes,
            category_ids=category_ids
        )
    except ValueError as e:
        print(f"Augmentation failed: {e}")
        return image, annotations
        
    # Update annotations with augmented bboxes
    for i, ann in enumerate(annotations):
        if i < len(transformed['bboxes']):
            ann['bbox'] = list(transformed['bboxes'][i])
            
    return transformed['image'], annotations

def mapper(dataset_dict: Dict[str, Any], is_train: bool = True) -> Dict[str, torch.Tensor]:
    """
    Map dataset dictionary to training format with augmentations.
    
    Args:
        dataset_dict: Input dataset dictionary
        is_train: Whether in training mode
        
    Returns:
        Mapped dictionary with augmented data
    """
    dataset_dict = dataset_dict.copy()
    image = dataset_dict.pop("image")
    
    if is_train:
        aug = get_train_aug()
    else:
        aug = get_val_aug()
        
    image, annotations = augment_annotations(
        dataset_dict["annotations"],
        image,
        aug
    )
    
    # Convert to tensor
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    
    return dataset_dict
