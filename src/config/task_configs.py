"""Task-specific configurations."""
from dataclasses import dataclass
from typing import Dict, Any

from .base_config import BaseConfig, DataConfig, ModelConfig, TrainingConfig


@dataclass
class QuadrantConfig(BaseConfig):
    """Configuration for quadrant detection."""
    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.model.num_classes = 4  # 4 quadrants
        self.model.backbone = "resnet50-fpn"


@dataclass
class EnumerationConfig(BaseConfig):
    """Configuration for tooth enumeration."""
    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.model.num_classes = 32  # 32 teeth
        self.model.backbone = "resnet101-fpn"
        self.model.learning_rate = 5e-5  # Slower learning for fine details


@dataclass
class DiseaseConfig(BaseConfig):
    """Configuration for disease detection."""
    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        self.model.num_classes = 4  # caries, deep_caries, periapical, impacted
        self.model.backbone = "resnet101-fpn"
        self.model.learning_rate = 1e-5  # Very careful learning
        self.training.max_epochs = 150  # Train longer for disease detection 