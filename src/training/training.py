"""Training functionality for dental X-ray models."""
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator
import os
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from torch.optim import AdamW

from .augmentation import mapper
from ..data.dataset import DentalDataset
from ..models.config import (
    get_quadrant_config,
    get_enumeration_config,
    get_disease_config
)

class DentalTrainer(DefaultTrainer):
    """Custom trainer for dental X-ray models."""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Build evaluator for the given dataset.
        
        Args:
            cfg: Configuration object
            dataset_name: Name of the dataset
            output_folder: Output directory for evaluation results
            
        Returns:
            COCOEvaluator instance
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(
            dataset_name, 
            tasks=("bbox", "segm"),
            distributed=False,
            output_dir=output_folder
        )
    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Build training loader with custom mapper.
        
        Args:
            cfg: Configuration object
            
        Returns:
            Training data loader
        """
        return build_detection_train_loader(
            cfg,
            mapper=lambda x: mapper(x, is_train=True)
        )
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Build test loader with custom mapper.
        
        Args:
            cfg: Configuration object
            dataset_name: Name of the dataset
            
        Returns:
            Test data loader
        """
        return build_detection_test_loader(
            cfg,
            dataset_name,
            mapper=lambda x: mapper(x, is_train=False)
        )
    
    def build_optimizer(self):
        """
        Build custom optimizer.
        
        Returns:
            AdamW optimizer
        """
        return setup_optimizer(self.cfg, self.model)

def setup_optimizer(cfg: Any, model: nn.Module) -> AdamW:
    """
    Create an AdamW optimizer with custom parameter groups.
    
    Args:
        cfg: Model configuration
        model: Model to optimize
        
    Returns:
        AdamW optimizer
    """
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue  # skip frozen weights
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr *= cfg.SOLVER.get('BIAS_LR_FACTOR', 1.0)
            weight_decay = 0.0
        params.append({
            "params": [value], 
            "lr": lr, 
            "weight_decay": weight_decay
        })

    return AdamW(
        params, 
        lr=cfg.SOLVER.BASE_LR, 
        betas=(0.9, 0.999), 
        weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )

def train_quadrant_phase(
    data_dir: str,
    json_file: str,
    output_dir: str,
    num_iterations: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None
) -> str:
    """
    Train the quadrant detection phase.
    
    Args:
        data_dir: Directory containing training images
        json_file: Path to COCO format annotations
        output_dir: Directory to save outputs
        num_iterations: Optional override for number of training iterations
        batch_size: Optional override for batch size
        learning_rate: Optional override for learning rate
        
    Returns:
        Path to the best model weights
    """
    # Register dataset
    dataset = DentalDataset(data_dir, json_file)
    
    # Get config
    cfg = get_quadrant_config()
    
    # Override settings if provided
    if num_iterations is not None:
        cfg.SOLVER.MAX_ITER = num_iterations
    if batch_size is not None:
        cfg.SOLVER.IMS_PER_BATCH = batch_size
    if learning_rate is not None:
        cfg.SOLVER.BASE_LR = learning_rate
    
    # Set output directory
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Train model
    trainer = DentalTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    return os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

def train_enumeration_phase(
    data_dir: str,
    json_file: str,
    output_dir: str,
    quadrant_weights: Optional[str] = None,
    num_iterations: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None
) -> str:
    """
    Train the tooth enumeration phase.
    
    Args:
        data_dir: Directory containing training images
        json_file: Path to COCO format annotations
        output_dir: Directory to save outputs
        quadrant_weights: Path to quadrant weights for transfer learning
        num_iterations: Optional override for number of training iterations
        batch_size: Optional override for batch size
        learning_rate: Optional override for learning rate
        
    Returns:
        Path to the best model weights
    """
    # Register dataset
    dataset = DentalDataset(data_dir, json_file)
    
    # Get config
    cfg = get_enumeration_config(quadrant_weights=quadrant_weights)
    
    # Override settings if provided
    if num_iterations is not None:
        cfg.SOLVER.MAX_ITER = num_iterations
    if batch_size is not None:
        cfg.SOLVER.IMS_PER_BATCH = batch_size
    if learning_rate is not None:
        cfg.SOLVER.BASE_LR = learning_rate
    
    # Set output directory
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Train model
    trainer = DentalTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    return os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

def train_disease_phase(
    data_dir: str,
    json_file: str,
    output_dir: str,
    enumeration_weights: Optional[str] = None,
    num_iterations: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None
) -> str:
    """
    Train the disease detection phase.
    
    Args:
        data_dir: Directory containing training images
        json_file: Path to COCO format annotations
        output_dir: Directory to save outputs
        enumeration_weights: Path to enumeration weights for transfer learning
        num_iterations: Optional override for number of training iterations
        batch_size: Optional override for batch size
        learning_rate: Optional override for learning rate
        
    Returns:
        Path to the best model weights
    """
    # Register dataset
    dataset = DentalDataset(data_dir, json_file)
    
    # Get config
    cfg = get_disease_config(enumeration_weights=enumeration_weights)
    
    # Override settings if provided
    if num_iterations is not None:
        cfg.SOLVER.MAX_ITER = num_iterations
    if batch_size is not None:
        cfg.SOLVER.IMS_PER_BATCH = batch_size
    if learning_rate is not None:
        cfg.SOLVER.BASE_LR = learning_rate
    
    # Set output directory
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Train model
    trainer = DentalTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    return os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
