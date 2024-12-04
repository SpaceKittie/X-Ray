"""
Dental X-ray Analysis Pipeline Demo

This script demonstrates the three-stage dental analysis process:
1. Quadrant Detection: Divide the X-ray into four quadrants
2. Tooth Enumeration: Identify and number individual teeth in each quadrant
3. Disease Detection: Detect dental conditions on each tooth

The pipeline uses Mask R-CNN with transfer learning between stages:
- Quadrant detection learns basic dental X-ray features
- Tooth enumeration builds on quadrant features for precise tooth localization
- Disease detection uses tooth features to identify conditions
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes, Instances
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.config import get_quadrant_config, get_enumeration_config, get_disease_config
from src.utils.visualization import DentalVisualizer

class DentalPredictor:
    """
    Predictor class that handles the three-stage dental analysis pipeline.
    Each stage builds upon features learned in previous stages:
    1. Quadrant Detection: Basic dental X-ray features
    2. Tooth Enumeration: Precise tooth localization using quadrant context
    3. Disease Detection: Condition identification using tooth features
    """
    def __init__(
        self,
        quadrant_weights: str,
        enumeration_weights: str,
        disease_weights: str
    ):
        """
        Initialize predictors for all stages.
        
        Args:
            quadrant_weights: Path to quadrant detection weights
            enumeration_weights: Path to tooth enumeration weights (trained on quadrant features)
            disease_weights: Path to disease detection weights (trained on tooth features)
        """
        # Quadrant predictor (base features)
        cfg_quad = get_quadrant_config()
        cfg_quad.MODEL.WEIGHTS = quadrant_weights
        self.quadrant_predictor = DefaultPredictor(cfg_quad)
        
        # Enumeration predictor (uses quadrant features)
        cfg_enum = get_enumeration_config()
        cfg_enum.MODEL.WEIGHTS = enumeration_weights
        self.enum_predictor = DefaultPredictor(cfg_enum)
        
        # Disease predictor (uses tooth features)
        cfg_disease = get_disease_config()
        cfg_disease.MODEL.WEIGHTS = disease_weights
        self.disease_predictor = DefaultPredictor(cfg_disease)
    
    def predict(self, image: np.ndarray):
        """
        Run full prediction pipeline on an image.
        
        The pipeline processes the image in stages:
        1. Detect quadrants to establish dental regions
        2. Within each quadrant, enumerate individual teeth
        3. For each tooth, detect any dental conditions
        
        Args:
            image: Input X-ray image as numpy array
            
        Returns:
            Tuple of predictions from each stage:
            - quad_pred: Quadrant detections with confidence scores
            - enum_pred: Tooth detections with numbers and confidence scores
            - disease_pred: Disease detections with conditions and confidence scores
        """
        # Stage 1: Quadrant Detection
        quad_pred = self.quadrant_predictor(image)
        
        # Stage 2: Tooth Enumeration (using quadrant context)
        enum_pred = self.enum_predictor(image)
        
        # Stage 3: Disease Detection (using tooth context)
        disease_pred = self.disease_predictor(image)
        
        return quad_pred, enum_pred, disease_pred

def visualize_predictions(
    image: np.ndarray,
    quad_pred: dict,
    enum_pred: dict,
    disease_pred: dict,
    threshold: float = 0.5,
    save_path: str = None
):
    """
    Visualize predictions from all three stages.
    
    Creates a figure with four panels:
    1. Original X-ray
    2. Detected quadrants
    3. Enumerated teeth
    4. Identified conditions
    
    Args:
        image: Input X-ray image
        quad_pred: Quadrant detection predictions
        enum_pred: Tooth enumeration predictions
        disease_pred: Disease detection predictions
        threshold: Confidence threshold for visualization
        save_path: Optional path to save visualization
    """
    visualizers = {
        "quadrant": DentalVisualizer("dental_quadrant"),
        "enumeration": DentalVisualizer("dental_enumeration"),
        "disease": DentalVisualizer("dental_disease")
    }
    
    plt.figure(figsize=(20, 5))
    
    # Original image
    plt.subplot(141)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original X-ray")
    plt.axis("off")
    
    # Quadrant detections
    plt.subplot(142)
    vis_img = visualizers["quadrant"].visualize_prediction(
        image,
        quad_pred,
        confidence_threshold=threshold
    )
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("Dental Quadrants")
    plt.axis("off")
    
    # Tooth enumeration
    plt.subplot(143)
    vis_img = visualizers["enumeration"].visualize_prediction(
        image,
        enum_pred,
        confidence_threshold=threshold
    )
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("Tooth Enumeration")
    plt.axis("off")
    
    # Disease detection
    plt.subplot(144)
    vis_img = visualizers["disease"].visualize_prediction(
        image,
        disease_pred,
        confidence_threshold=threshold
    )
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("Dental Conditions")
    plt.axis("off")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def print_findings(
    quad_pred: dict,
    enum_pred: dict,
    disease_pred: dict,
    threshold: float = 0.5
):
    """
    Print detection results in a structured, human-readable format.
    
    Organizes findings by quadrant:
    1. Lists detected quadrants
    2. Within each quadrant, lists enumerated teeth
    3. For each tooth, lists any detected conditions
    
    Args:
        quad_pred: Quadrant detection predictions
        enum_pred: Tooth enumeration predictions
        disease_pred: Disease detection predictions
        threshold: Confidence threshold for reporting
    """
    visualizers = {
        "quadrant": DentalVisualizer("dental_quadrant"),
        "enumeration": DentalVisualizer("dental_enumeration"),
        "disease": DentalVisualizer("dental_disease")
    }
    
    print("\n=== Dental X-ray Analysis Results ===")
    
    # Get quadrant detections
    quadrants = {}
    for instance in quad_pred["instances"]:
        if instance.scores.item() > threshold:
            class_id = instance.pred_classes.item()
            class_name = visualizers["quadrant"].metadata.thing_classes[class_id]
            score = instance.scores.item()
            box = instance.pred_boxes.tensor[0].tolist()
            quadrants[class_name] = {
                "score": score,
                "box": box,
                "teeth": []
            }
    
    # Map teeth to quadrants
    for instance in enum_pred["instances"]:
        if instance.scores.item() > threshold:
            class_id = instance.pred_classes.item()
            tooth_name = visualizers["enumeration"].metadata.thing_classes[class_id]
            score = instance.scores.item()
            box = instance.pred_boxes.tensor[0].tolist()
            
            # Find which quadrant this tooth belongs to
            for quad_name, quad_info in quadrants.items():
                quad_box = quad_info["box"]
                # Check if tooth center is in quadrant
                tooth_center = [(box[0] + box[2])/2, (box[1] + box[3])/2]
                if (tooth_center[0] >= quad_box[0] and 
                    tooth_center[0] <= quad_box[2] and
                    tooth_center[1] >= quad_box[1] and 
                    tooth_center[1] <= quad_box[3]):
                    quad_info["teeth"].append({
                        "name": tooth_name,
                        "score": score,
                        "conditions": []
                    })
                    break
    
    # Map diseases to teeth
    for instance in disease_pred["instances"]:
        if instance.scores.item() > threshold:
            class_id = instance.pred_classes.item()
            condition = visualizers["disease"].metadata.thing_classes[class_id]
            score = instance.scores.item()
            box = instance.pred_boxes.tensor[0].tolist()
            
            # Find which tooth this condition belongs to
            for quad_info in quadrants.values():
                for tooth in quad_info["teeth"]:
                    tooth["conditions"].append({
                        "name": condition,
                        "score": score
                    })
    
    # Print organized results
    for quad_name, quad_info in quadrants.items():
        print(f"\n{quad_name}: {quad_info['score']:.2%} confidence")
        
        if not quad_info["teeth"]:
            print("  No teeth detected")
            continue
            
        for tooth in quad_info["teeth"]:
            print(f"  {tooth['name']}: {tooth['score']:.2%} confidence")
            
            if not tooth["conditions"]:
                print("    No conditions detected")
                continue
                
            for condition in tooth["conditions"]:
                print(f"    {condition['name']}: {condition['score']:.2%} confidence")

# Example usage:
if __name__ == "__main__":
    print("Dental X-ray Analysis Pipeline Demo")
    print("\nThis script demonstrates the three-stage dental analysis process:")
    print("1. Quadrant Detection: Divide the X-ray into four quadrants")
    print("2. Tooth Enumeration: Identify and number individual teeth")
    print("3. Disease Detection: Detect dental conditions")
    print("\nTo use this pipeline, you need:")
    print("1. Trained model weights (transfer learning between stages)")
    print("2. X-ray images in standard dental format")
    print("3. Proper preprocessing of input images")
    print("\nSee README.md for detailed setup and usage instructions.")
