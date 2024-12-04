from typing import Dict, List, Union, Optional
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score
from torchvision.ops import box_iou

class DentalEvaluator:
    """
    Evaluation metrics for dental disease detection.
    """
    def __init__(self, dataset_name: str, output_dir: str):
        """
        Initialize the evaluator.
        
        Args:
            dataset_name: Name of the registered dataset
            output_dir: Directory to save evaluation results
        """
        self.evaluator = COCOEvaluator(
            dataset_name,
            output_dir=output_dir,
            tasks=("bbox", "segm"),
        )
        
    def evaluate(self, predictions: List[Dict]) -> Dict:
        """
        Evaluate model predictions.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.evaluator.reset()
        
        # Get dataset
        dataset_dicts = DatasetCatalog.get(self.evaluator._dataset_name)
        
        for input, output in zip(dataset_dicts, predictions):
            self.evaluator.process(input, output)
            
        return self.evaluator.evaluate()
    
    @staticmethod
    def calculate_metrics(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics for model evaluation.
        
        Args:
            model: The model to evaluate
            data_loader: DataLoader containing validation/test data
            iou_threshold: IoU threshold for considering a detection correct
        
        Returns:
            Dictionary containing various metrics
        """
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in data_loader:
                predictions = model(images)
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Convert to numpy for metric calculation
        pred_boxes = np.array([p["boxes"].cpu().numpy() for p in all_predictions])
        pred_scores = np.array([p["scores"].cpu().numpy() for p in all_predictions])
        pred_labels = np.array([p["labels"].cpu().numpy() for p in all_predictions])
        
        target_boxes = np.array([t["boxes"].cpu().numpy() for t in all_targets])
        target_labels = np.array([t["labels"].cpu().numpy() for t in all_targets])
        
        # Calculate metrics
        metrics = {}
        
        # Mean Average Precision
        metrics["mAP"] = calculate_map(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels,
            iou_threshold
        )
        
        # Per-class metrics
        class_metrics = calculate_per_class_metrics(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels,
            iou_threshold
        )
        metrics.update(class_metrics)
        
        # Overall detection metrics
        detection_metrics = calculate_detection_metrics(
            pred_boxes, pred_scores,
            target_boxes,
            iou_threshold
        )
        metrics.update(detection_metrics)
        
        return metrics
    
def calculate_map(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    target_boxes: np.ndarray,
    target_labels: np.ndarray,
    iou_threshold: float
) -> float:
    """Calculate mean Average Precision."""
    aps = []
    
    # Calculate AP for each class
    unique_classes = np.unique(np.concatenate([pred_labels, target_labels]))
    
    for cls in unique_classes:
        # Get predictions and targets for this class
        cls_pred_mask = pred_labels == cls
        cls_target_mask = target_labels == cls
        
        if not np.any(cls_target_mask):
            continue
        
        cls_pred_boxes = pred_boxes[cls_pred_mask]
        cls_pred_scores = pred_scores[cls_pred_mask]
        cls_target_boxes = target_boxes[cls_target_mask]
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(
            y_true=(cls_target_boxes.size > 0).astype(int),
            probas_pred=cls_pred_scores,
            pos_label=1
        )
        
        # Calculate AP
        ap = average_precision_score(
            y_true=(cls_target_boxes.size > 0).astype(int),
            y_score=cls_pred_scores,
            pos_label=1
        )
        aps.append(ap)
    
    return np.mean(aps)

def calculate_per_class_metrics(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    target_boxes: np.ndarray,
    target_labels: np.ndarray,
    iou_threshold: float
) -> Dict[str, float]:
    """Calculate precision and recall for each class."""
    metrics = {}
    unique_classes = np.unique(np.concatenate([pred_labels, target_labels]))
    
    for cls in unique_classes:
        cls_pred_mask = pred_labels == cls
        cls_target_mask = target_labels == cls
        
        if not np.any(cls_target_mask):
            continue
        
        # Calculate true positives, false positives, false negatives
        tp = np.sum(np.logical_and(cls_pred_mask, cls_target_mask))
        fp = np.sum(np.logical_and(cls_pred_mask, ~cls_target_mask))
        fn = np.sum(np.logical_and(~cls_pred_mask, cls_target_mask))
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f"class_{cls}_precision"] = precision
        metrics[f"class_{cls}_recall"] = recall
        metrics[f"class_{cls}_f1"] = f1
    
    return metrics

def calculate_detection_metrics(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    target_boxes: np.ndarray,
    iou_threshold: float
) -> Dict[str, float]:
    """Calculate overall detection metrics."""
    # Convert boxes to torch tensors for IoU calculation
    pred_boxes_t = torch.from_numpy(pred_boxes)
    target_boxes_t = torch.from_numpy(target_boxes)
    
    # Calculate IoU between all pred and target boxes
    iou_matrix = box_iou(pred_boxes_t, target_boxes_t)
    
    # Get matches above threshold
    matches = iou_matrix > iou_threshold
    
    # Calculate metrics
    true_positives = torch.sum(matches).item()
    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(target_boxes) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU score
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0
