"""Test suite for model training and evaluation."""
import pytest
import torch
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np

from src.models.detector import DentalDetector
from src.data.dataset import DentalDataset
from src.config.task_configs import QuadrantConfig, EnumerationConfig, DiseaseConfig
from src.utils.metrics import calculate_metrics


def test_model_initialization():
    """Test model initialization for all tasks."""
    configs = [
        QuadrantConfig(),
        EnumerationConfig(),
        DiseaseConfig()
    ]
    
    for config in configs:
        model = DentalDetector(config.model)
        assert isinstance(model, DentalDetector)
        assert model.num_classes == config.model.num_classes


@pytest.mark.slow
def test_cross_validation():
    """Test model performance with k-fold cross validation."""
    config = QuadrantConfig()  # Test with quadrant detection
    dataset = DentalDataset(
        data_dir=Path("data/processed/train"),
        annotation_file=Path("data/annotations/quadrant/train.json")
    )
    
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics_per_fold = []
    
    for fold, (train_idx, val_idx) in enumerate(k_fold.split(dataset)):
        # Create train/val splits
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config.data.batch_size,
            sampler=train_subsampler
        )
        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            sampler=val_subsampler
        )
        
        # Train model
        model = DentalDetector(config.model)
        model.train_model(train_loader, val_loader, config.training)
        
        # Evaluate
        metrics = calculate_metrics(model, val_loader)
        metrics_per_fold.append(metrics)
    
    # Calculate mean and std of metrics across folds
    mean_metrics = {
        k: np.mean([fold[k] for fold in metrics_per_fold])
        for k in metrics_per_fold[0].keys()
    }
    std_metrics = {
        k: np.std([fold[k] for fold in metrics_per_fold])
        for k in metrics_per_fold[0].keys()
    }
    
    # Assert performance thresholds
    assert mean_metrics["mAP"] > 0.75, "Mean mAP below threshold"
    assert mean_metrics["recall"] > 0.70, "Mean recall below threshold"


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_batch_processing(batch_size):
    """Test model with different batch sizes."""
    config = QuadrantConfig()
    config.data.batch_size = batch_size
    model = DentalDetector(config.model)
    
    # Create dummy batch
    dummy_batch = torch.randn(batch_size, 3, 1024, 1024)
    output = model(dummy_batch)
    
    assert output.shape[0] == batch_size
    assert output.shape[1] == config.model.num_classes 