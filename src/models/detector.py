from typing import Dict, List, Optional
import os
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2 import model_zoo

class DiseaseDetector:
    """
    Dental disease detector using Detectron2's Mask R-CNN implementation.
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the disease detector.
        
        Args:
            config_path: Path to custom config file. If None, uses default config.
        """
        self.cfg = self._setup_config(config_path)
        self.predictor = None
        
    def _setup_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Set up the model configuration.
        
        Args:
            config_path: Path to custom config file
            
        Returns:
            Detectron2 CfgNode object
        """
        cfg = get_cfg()
        
        # Load base config
        base_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(base_config))
        
        # Load custom config if provided
        if config_path and os.path.exists(config_path):
            cfg.merge_from_file(config_path)
            
        # Set device
        cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        return cfg
        
    def train(
        self,
        train_dataset: str,
        val_dataset: str,
        output_dir: str,
        num_classes: int,
        iterations: int = 1000,
        batch_size: int = 4,
        learning_rate: float = 0.00025
    ) -> None:
        """
        Train the disease detector.
        
        Args:
            train_dataset: Name of registered training dataset
            val_dataset: Name of registered validation dataset
            output_dir: Directory to save model checkpoints
            num_classes: Number of disease classes
            iterations: Number of training iterations
            batch_size: Training batch size
            learning_rate: Base learning rate
        """
        self.cfg.OUTPUT_DIR = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Dataset config
        self.cfg.DATASETS.TRAIN = (train_dataset,)
        self.cfg.DATASETS.TEST = (val_dataset,)
        
        # Model config
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        
        # Training config
        self.cfg.SOLVER.IMS_PER_BATCH = batch_size
        self.cfg.SOLVER.BASE_LR = learning_rate
        self.cfg.SOLVER.MAX_ITER = iterations
        self.cfg.SOLVER.CHECKPOINT_PERIOD = iterations // 10
        
        # Create trainer and train
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
    def load_weights(self, weights_path: str) -> None:
        """
        Load trained weights.
        
        Args:
            weights_path: Path to model weights
        """
        self.cfg.MODEL.WEIGHTS = weights_path
        self.predictor = DefaultPredictor(self.cfg)
        
    def predict(self, image_path: str) -> Dict:
        """
        Make predictions on an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing predictions
        """
        if self.predictor is None:
            raise RuntimeError("Model weights not loaded. Call load_weights first.")
            
        import cv2
        image = cv2.imread(image_path)
        return self.predictor(image)
