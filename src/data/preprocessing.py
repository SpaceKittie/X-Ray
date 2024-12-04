"""Data preprocessing utilities for dental X-ray datasets."""
import json
import os
import shutil
from collections import defaultdict
import random
from typing import Dict, List, Any, Tuple

def filter_annotations(
    input_json_path: str,
    output_json_path: str,
    min_occurrences: int = 25
) -> None:
    """
    Filter annotations based on minimum category occurrences.
    
    Args:
        input_json_path: Path to input JSON file
        output_json_path: Path to save filtered JSON
        min_occurrences: Minimum number of occurrences required to keep a category
    """
    with open(input_json_path, 'r') as file:
        data = json.load(file)
    
    # Count category occurrences
    category_counts = defaultdict(int)
    for annotation in data['annotations']:
        category_counts[annotation['category_id']] += 1
    
    # Filter annotations
    filtered_annotations = [
        anno for anno in data['annotations'] 
        if category_counts[anno['category_id']] >= min_occurrences
    ]
    
    # Update data
    data['annotations'] = filtered_annotations
    
    # Save filtered data
    with open(output_json_path, 'w') as file:
        json.dump(data, file, indent=4)

def stratified_split_images(
    json_file_path: str,
    base_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2
) -> None:
    """
    Split dataset into train/val/test sets while maintaining category distribution.
    
    Args:
        json_file_path: Path to annotations JSON file
        base_dir: Base directory containing images
        train_ratio: Ratio of images for training
        val_ratio: Ratio of images for validation
        (test_ratio is 1 - train_ratio - val_ratio)
    """
    test_ratio = 1 - train_ratio - val_ratio
    
    # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Map image IDs to filenames
    image_id_to_file = {img['id']: img['file_name'] for img in data['images']}
    
    # Group annotations by category
    category_annotations = defaultdict(list)
    for anno in data['annotations']:
        category_annotations[anno['category_id']].append(anno['image_id'])
    
    # Create directories
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    def move_files(image_ids: List[int], target_dir: str) -> None:
        """Move files to target directory."""
        for image_id in image_ids:
            filename = image_id_to_file[image_id]
            src_path = os.path.join(base_dir, filename)
            dst_path = os.path.join(target_dir, filename)
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
            else:
                print(f"File not found: {src_path}")
    
    # Perform stratified split
    for category, ids in category_annotations.items():
        random.shuffle(ids)
        n_total = len(ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_ids = ids[:n_train]
        val_ids = ids[n_train:n_train + n_val]
        test_ids = ids[n_train + n_val:]
        
        move_files(train_ids, train_dir)
        move_files(val_ids, val_dir)
        move_files(test_ids, test_dir)

def remap_category_ids(
    dataset_path: str,
    output_path: str
) -> None:
    """
    Remap category IDs to be continuous starting from 0.
    
    Args:
        dataset_path: Path to input JSON file
        output_path: Path to save remapped JSON
    """
    with open(dataset_path, 'r') as file:
        data = json.load(file)
    
    # Create mapping from old to new IDs
    category_id_mapping = {
        cat['id']: idx for idx, cat in enumerate(data['categories'])
    }
    
    # Update categories
    for cat in data['categories']:
        cat['id'] = category_id_mapping[cat['id']]
    
    # Update annotations
    for ann in data['annotations']:
        ann['category_id'] = category_id_mapping[ann['category_id']]
    
    # Save updated data
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def get_category_counts(json_file_path: str) -> Dict[str, int]:
    """
    Get counts of each category in dataset.
    
    Args:
        json_file_path: Path to JSON file
        
    Returns:
        Dictionary mapping category names to counts
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Count occurrences
    category_counts = defaultdict(int)
    for annotation in data['annotations']:
        category_counts[annotation['category_id']] += 1
    
    # Map category IDs to names
    category_names = {
        category['id']: category['name'] 
        for category in data['categories']
    }
    
    # Return counts with category names
    return {
        category_names[cid]: count 
        for cid, count in category_counts.items()
    }
