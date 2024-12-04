"""Base configuration for all models and training."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import yaml
from omegaconf import OmegaConf


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: Path
    train_dir: Path
    val_dir: Path
    annotation_dir: Path
    image_size: tuple[int, int] = (1024, 1024)
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model configuration."""
    backbone: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 0  # Set by specific configs
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4


@dataclass
class TrainingConfig:
    """Training configuration."""
    max_epochs: int = 100
    early_stopping_patience: int = 10
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    device: str = "cuda"
    seed: int = 42


@dataclass
class BaseConfig:
    """Base configuration class."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "BaseConfig":
        """Load config from YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**OmegaConf.create(config_dict))
    
    def to_yaml(self, yaml_path: Path) -> None:
        """Save config to YAML file."""
        config_dict = OmegaConf.create(self.__dict__)
        with open(yaml_path, "w") as f:
            OmegaConf.save(config=config_dict, f=f) 