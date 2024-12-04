# Data Directory Structure

This directory contains all the data used for training and validation. The structure is as follows:

```
data/
├── raw/                  # Original X-ray images
│   ├── train/           # Training images
│   └── val/             # Validation images
├── processed/           # Preprocessed images
│   ├── train/
│   └── val/
└── annotations/         # JSON annotation files
    ├── quadrant/
    ├── enumeration/
    └── disease/
```

## Data Organization

1. **Raw Data**
   - Original X-ray images in DICOM or JPG format
   - Split into training and validation sets

2. **Processed Data**
   - Preprocessed images (normalized, resized, etc.)
   - Ready for model training

3. **Annotations**
   - JSON files following COCO format
   - Separate files for each task:
     - Quadrant detection
     - Tooth enumeration
     - Disease detection

## Data Versioning

Use DVC (Data Version Control) for tracking data versions:
```bash
dvc add data/raw
dvc add data/processed
dvc add data/annotations
```

## Notes
- This directory is git-ignored except for this README
- Use the preprocessing scripts in `src/data/preprocessing.py` to process raw data
- Run `scripts/prepare_data.py` to set up the directory structure 