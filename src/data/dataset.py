from typing import Dict, List, Optional, Tuple
import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

class DentalDataset:
    """
    Dataset class for dental X-ray images and annotations.
    """
    def __init__(self, data_dir: str, json_file: str):
        """
        Initialize the dental dataset.
        
        Args:
            data_dir: Directory containing the X-ray images
            json_file: Path to the COCO format annotation file
        """
        self.data_dir = data_dir
        self.json_file = json_file
        
    def get_dataset_dicts(self) -> List[Dict]:
        """
        Load and parse the dataset in Detectron2 format.
        """
        with open(self.json_file, 'r') as f:
            dataset = json.load(f)
            
        dataset_dicts = []
        for image in dataset['images']:
            record = {}
            
            # Image info
            file_name = os.path.join(self.data_dir, image['file_name'])
            if not os.path.exists(file_name):
                continue  # Skip if image file doesn't exist
                
            height, width = image['height'], image['width']
            image_id = image['id']
            
            record['file_name'] = file_name
            record['image_id'] = image_id
            record['height'] = height
            record['width'] = width
            
            # Annotations
            annos = [
                anno for anno in dataset['annotations']
                if anno['image_id'] == image_id
            ]
            
            objs = []
            for anno in annos:
                obj = {
                    'bbox': anno['bbox'],
                    'bbox_mode': BoxMode.XYWH_ABS,
                    'category_id': anno['category_id'],
                    'segmentation': anno['segmentation']
                }
                objs.append(obj)
            record['annotations'] = objs
            
            dataset_dicts.append(record)
            
        return dataset_dicts
    
    @staticmethod
    def register_dataset(
        name: str,
        data_dir: str,
        json_file: str,
        thing_classes: Optional[List[str]] = None
    ) -> None:
        """
        Register the dataset with Detectron2's DatasetCatalog.
        
        Args:
            name: Name to register the dataset under
            data_dir: Directory containing the X-ray images
            json_file: Path to the COCO format annotation file
            thing_classes: List of class names. If None, will be extracted from JSON.
        """
        # Load class names from JSON if not provided
        if thing_classes is None:
            with open(json_file, 'r') as f:
                data = json.load(f)
                thing_classes = [c['name'] for c in data['categories']]
        
        dataset = DentalDataset(data_dir, json_file)
        DatasetCatalog.register(
            name,
            lambda: dataset.get_dataset_dicts()
        )
        MetadataCatalog.get(name).set(thing_classes=thing_classes)
    
    @staticmethod
    def register_all_splits(
        base_dir: str,
        train_json: str,
        val_json: str,
        test_json: str,
        thing_classes: Optional[List[str]] = None
    ) -> None:
        """
        Register all dataset splits (train/val/test) with Detectron2.
        
        Args:
            base_dir: Base directory containing image directories
            train_json: Path to training annotations JSON
            val_json: Path to validation annotations JSON
            test_json: Path to test annotations JSON
            thing_classes: List of class names. If None, will be extracted from train JSON.
        """
        # Get class names from training JSON if not provided
        if thing_classes is None:
            with open(train_json, 'r') as f:
                data = json.load(f)
                thing_classes = [c['name'] for c in data['categories']]
        
        # Register each split
        DentalDataset.register_dataset(
            'dental_train',
            os.path.join(base_dir, 'train'),
            train_json,
            thing_classes
        )
        DentalDataset.register_dataset(
            'dental_val',
            os.path.join(base_dir, 'val'),
            val_json,
            thing_classes
        )
        DentalDataset.register_dataset(
            'dental_test',
            os.path.join(base_dir, 'test'),
            test_json,
            thing_classes
        )
    
    def verify_dataset(self) -> Tuple[bool, List[str]]:
        """
        Verify that all images referenced in JSON exist.
        
        Returns:
            Tuple of (is_valid, list of missing files)
        """
        missing_files = []
        with open(self.json_file, 'r') as f:
            data = json.load(f)
            for img in data['images']:
                file_path = os.path.join(self.data_dir, img['file_name'])
                if not os.path.exists(file_path):
                    missing_files.append(img['file_name'])
        
        return len(missing_files) == 0, missing_files
