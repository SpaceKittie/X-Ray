"""Visualization utilities for dental X-ray analysis."""
from typing import Dict, List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os
from pycocotools import mask

COLOR_MAP = {
    0: (0, 255, 0),    # Yellow for quadrant 2
    1: (255, 0, 0),    # Blue for quadrant 1
    2: (0, 0, 255),    # Green for quadrant 3
    3: (0, 255, 255)   # Red for quadrant 4
}

ID_TO_QUADRANT = {
    0: "2",  # COCO ID 0 is Quadrant 2
    1: "1",  # COCO ID 1 is Quadrant 1
    2: "3",  # COCO ID 2 is Quadrant 3
    3: "4"   # COCO ID 3 is Quadrant 4
}


class DentalVisualizer:
    """Visualization utilities for dental disease detection."""
    
    def __init__(self, metadata_name: str):
        """Initialize the visualizer.
        
        Args:
            metadata_name: Name of the registered dataset metadata
        """
        self.metadata = MetadataCatalog.get(metadata_name)
    
    def visualize_prediction(
        self,
        image: np.ndarray,
        predictions: Dict,
        confidence_threshold: float = 0.5
    ) -> np.ndarray:
        """Visualize model predictions on an image.
        
        Args:
            image: Input image
            predictions: Model predictions
            confidence_threshold: Minimum confidence score for visualization
            
        Returns:
            Image with visualized predictions
        """
        visualizer = Visualizer(image[:, :, ::-1], self.metadata, scale=1.2)
        
        if "instances" in predictions:
            instances = predictions["instances"]
            
            # Filter by confidence
            if confidence_threshold > 0:
                scores = instances.scores
                keep = scores > confidence_threshold
                instances = instances[keep]
            
            vis_output = visualizer.draw_instance_predictions(instances.to("cpu"))
            return vis_output.get_image()[:, :, ::-1]
        
        return image
    
    def plot_results(
        self,
        image_path: str,
        ground_truth: Dict,
        predictions: Dict,
        save_path: str = None
    ) -> None:
        """Plot ground truth and predictions side by side.
        
        Args:
            image_path: Path to input image
            ground_truth: Ground truth annotations
            predictions: Model predictions
            save_path: Path to save the visualization
        """
        image = cv2.imread(image_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Ground truth
        vis_gt = Visualizer(image[:, :, ::-1], self.metadata, scale=1.2)
        vis_gt = vis_gt.draw_dataset_dict(ground_truth)
        ax1.imshow(vis_gt.get_image()[:, :, ::-1])
        ax1.set_title('Ground Truth')
        ax1.axis('off')
        
        # Predictions
        vis_pred = self.visualize_prediction(image, predictions)
        ax2.imshow(vis_pred)
        ax2.set_title('Predictions')
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()


def analyze_results(
    coco_gt,
    coco_results,
    images_dir: str,
    num_samples: int = 10,
    confidence_threshold: float = 0.6
) -> None:
    """Analyze and visualize model results against ground truth.
    
    Args:
        coco_gt: COCO ground truth object
        coco_results: COCO results object
        images_dir: Directory containing images
        num_samples: Number of random samples to visualize
        confidence_threshold: Confidence threshold for predictions
    """
    # Get random sample of images
    image_ids = list(coco_gt.imgs.keys())
    random_image_ids = np.random.choice(image_ids, num_samples, replace=False)
    
    for img_id in random_image_ids:
        img_info = coco_gt.imgs[img_id]
        img_path = os.path.join(images_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Failed to load image at {img_path}")
            continue
            
        # Create visualization
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        
        # Ground Truth
        ann_ids_gt = coco_gt.getAnnIds(imgIds=[img_id])
        anns_gt = coco_gt.loadAnns(ann_ids_gt)
        image_with_gt = draw_annotations(
            image.copy(), 
            anns_gt, 
            coco_gt, 
            confidence_threshold
        )
        axs[0].imshow(cv2.cvtColor(image_with_gt, cv2.COLOR_BGR2RGB))
        axs[0].set_title('Ground Truth')
        axs[0].axis('off')
        
        # Predictions
        ann_ids_pred = coco_results.getAnnIds(imgIds=[img_id])
        anns_pred = coco_results.loadAnns(ann_ids_pred)
        image_with_pred = draw_annotations(
            image.copy(), 
            anns_pred, 
            coco_results, 
            confidence_threshold
        )
        axs[1].imshow(cv2.cvtColor(image_with_pred, cv2.COLOR_BGR2RGB))
        axs[1].set_title('Predictions')
        axs[1].axis('off')
        
        plt.tight_layout()
        plt.show()


def draw_annotations(
    image: np.ndarray,
    annotations: List[Dict],
    coco_obj,
    confidence_threshold: float = 0.6
) -> np.ndarray:
    """Draw bounding boxes, labels and segmentation masks on the image.
    
    Args:
        image: Input image
        annotations: List of COCO annotations
        coco_obj: COCO object (for category names)
        confidence_threshold: Minimum confidence score
        
    Returns:
        Image with drawn annotations
    """
    line_thickness = 2
    
    for ann in annotations:
        score = ann.get('score', 1)  # Default to 1 for ground truth
        if score >= confidence_threshold:
            x, y, width, height = ann['bbox']
            cat_id = ann['category_id']
            color = COLOR_MAP.get(cat_id, (255, 255, 255))
            
            quadrant = ID_TO_QUADRANT.get(cat_id)
            quadrant_label = f"Q{quadrant}"
            
            # Draw bounding box
            cv2.rectangle(
                image, 
                (int(x), int(y)), 
                (int(x + width), int(y + height)), 
                color, 
                line_thickness
            )
            
            # Draw label
            cv2.putText(
                image, 
                quadrant_label, 
                (int(x), int(y - 5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                color, 
                line_thickness
            )
            
            # Draw segmentation if present
            if 'segmentation' in ann:
                segmentation = ann['segmentation']
                if isinstance(segmentation, list):
                    mask_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    for seg in segmentation:
                        poly = np.array(seg).reshape((int(len(seg)/2), 2))
                        cv2.fillPoly(mask_img, [poly.astype(np.int32)], 1)
                else:
                    mask_img = mask.decode(segmentation)
                
                mask_img = np.stack([mask_img]*3, axis=-1) * color
                image[mask_img > 0] = image[mask_img > 0] * 0.5 + mask_img[mask_img > 0] * 0.5
            
            # Draw category name
            cat_name = coco_obj.loadCats(ids=cat_id)[0]['name']
            abbreviated_cat_name = ''.join(word[0] for word in cat_name.split()) + cat_name.split()[-1][1:]
            cv2.putText(
                image, 
                abbreviated_cat_name, 
                (int(x), int(y - 20)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                color, 
                line_thickness
            )
    
    return image