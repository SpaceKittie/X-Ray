"""
Demo script for running inference with trained models.
"""
import os
import sys
import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectron2.engine import DefaultPredictor
from src.models.config import get_quadrant_config, get_enumeration_config, get_disease_config
from src.utils.visualization import DentalVisualizer

class DentalPredictor:
    """
    Predictor class that handles the three-phase inference process:
    1. Quadrant Detection
    2. Tooth Enumeration
    3. Disease Detection
    """
    def __init__(
        self,
        quadrant_weights: str,
        enumeration_weights: str,
        disease_weights: str
    ):
        """
        Initialize predictors for all phases.
        
        Args:
            quadrant_weights: Path to quadrant detection weights
            enumeration_weights: Path to tooth enumeration weights
            disease_weights: Path to disease detection weights
        """
        # Ensure weights exist
        for path in [quadrant_weights, enumeration_weights, disease_weights]:
            if not os.path.exists(path):
                raise ValueError(f"Model weights not found: {path}")
        
        # Quadrant predictor
        cfg_quad = get_quadrant_config()
        cfg_quad.MODEL.WEIGHTS = quadrant_weights
        self.quadrant_predictor = DefaultPredictor(cfg_quad)
        
        # Enumeration predictor
        cfg_enum = get_enumeration_config()
        cfg_enum.MODEL.WEIGHTS = enumeration_weights
        self.enum_predictor = DefaultPredictor(cfg_enum)
        
        # Disease predictor
        cfg_disease = get_disease_config()
        cfg_disease.MODEL.WEIGHTS = disease_weights
        self.disease_predictor = DefaultPredictor(cfg_disease)
    
    def predict(self, image_path: str):
        """
        Run full prediction pipeline on an image.
        
        Args:
            image_path: Path to input X-ray image
            
        Returns:
            Tuple of predictions from each phase
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run predictions
        quad_pred = self.quadrant_predictor(image)
        enum_pred = self.enum_predictor(image)
        disease_pred = self.disease_predictor(image)
        
        return quad_pred, enum_pred, disease_pred
