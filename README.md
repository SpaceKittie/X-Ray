```ascii
 _ 
 \`*-.    
  )  _`-. 
 .  : `. . 
 : _   '  \ 
 ; *` _.   `*-._
 `-.-'          `-.
   ;       `       `.
   :.       .        :
   . \  .   :   .-'   . 
   '  `+.;  ;  '      :
   :  '  |    ;       ;-.
   ; '   : :`-:     _.`* ;
 .*' /  .*' ; .*`- +'  `*'
*-*   `*-*   `*-*'
```

# Dental X-Ray Analysis with Detectron2

A three-phase deep learning system for dental X-ray analysis using Detectron2, developed using the DENTEX Challenge dataset from MICCAI 2023.

## Dataset: DENTEX Challenge

We utilize the DENTEX (Dental Enumeration and Diagnosis on Panoramic X-rays) dataset, a comprehensive collection designed for developing AI solutions in dental radiology. The dataset includes:

- **Size**: Over 3,900 panoramic X-rays total
- **Data Split**:
  - 693 X-rays for quadrant detection
  - 634 X-rays for tooth enumeration
  - 1,005 X-rays with full diagnosis labels
  - 1,571 unlabeled X-rays for pre-training

### Clinical Relevance

This project assists dental practitioners by:
1. Automatically detecting abnormal teeth
2. Providing precise dental enumeration (FDI system)
3. Diagnosing common conditions:
   - Caries
   - Deep caries
   - Periapical lesions
   - Impacted teeth

### Data Organization

The dataset follows the FDI (Fédération Dentaire Internationale) numbering system:
- Quadrants numbered 1-4 (clockwise from upper right)
- Each tooth numbered 1-8 (from midline)
- Complete notation: [Quadrant][Tooth Number]

Example: Tooth 48 = Lower right third molar (wisdom tooth)

## Project Structure
```
X-Ray/
├── config/                     # Model configurations
│   ├── paths.yaml             # Data and model paths
│   ├── disease.yaml           # Disease detection config
│   ├── enumeration.yaml       # Tooth enumeration config
│   └── quadrant.yaml          # Quadrant detection config
├── data/                      # Data directory (git-ignored)
│   ├── raw/                   # Original X-ray images
│   ├── processed/             # Preprocessed images
│   └── annotations/           # JSON annotation files
├── src/                       # Source code
│   ├── data/                  # Data handling
│   │   ├── __init__.py
│   │   ├── dataset.py        # Dataset class
│   │   └── preprocessing.py   # Data preprocessing
│   ├── models/               # Model definitions
│   │   ├── __init__.py
│   │   ├── config.py         # Model configurations
│   │   └── detector.py       # Detector implementations
│   ├── training/             # Training code
│   │   ├── __init__.py
│   │   ├── training.py       # Training loops
│   │   └── augmentation.py   # Data augmentation
│   └── utils/                # Utilities
│       ├── __init__.py
│       ├── metrics.py        # Evaluation metrics
│       └── visualization.py  # Visualization tools
├── demo/                     # Demo scripts
│   ├── demo.py              # Main demo script
│   └── inference.py         # Inference pipeline
├── notebooks/               # Jupyter notebooks
│   ├── Disease.ipynb       # Disease detection notebook
│   ├── Enumeration.ipynb   # Tooth enumeration notebook
│   └── Quadrant.ipynb      # Quadrant detection notebook
├── tests/                   # Unit tests
│   ├── __init__.py
│   └── test_models.py      # Model tests
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Results & Visualization

Our three-phase dental analysis system provides comprehensive results for each stage:

### 1. Quadrant Detection
![Quadrant Detection](demo/Quadrant%20Demo.png)
*Detection of four dental quadrants*

### 2. Tooth Enumeration
![Tooth Enumeration](demo/Enumeration%20Demo.png)
*Precise identification and numbering of individual teeth using FDI system*

### 3. Disease Detection
![Disease Detection](demo/Disease%20Demo.png)
*Detection of dental conditions and locations*

Each phase builds upon the previous one, using transfer learning to improve accuracy:
- Quadrant detection establishes the basic dental X-ray features
- Tooth enumeration uses quadrant features for precise tooth localization
- Disease detection leverages tooth features to identify conditions

## Usage

### Training Pipeline

The system is trained in three sequential phases:

```python
from src.training import train_quadrant_phase, train_enumeration_phase, train_disease_phase

# Phase 1: Quadrant Detection
quadrant_weights = train_quadrant_phase(
    data_dir="data/processed/quadrant/train",
    json_file="data/annotations/quadrant/train.json",
    output_dir="output/quadrant"
)

# Phase 2: Tooth Enumeration
enum_weights = train_enumeration_phase(
    data_dir="data/processed/enumeration/train",
    json_file="data/annotations/enumeration/train.json",
    output_dir="output/enumeration",
    quadrant_weights=quadrant_weights
)

# Phase 3: Disease Detection
disease_weights = train_disease_phase(
    data_dir="data/processed/disease/train",
    json_file="data/annotations/disease/train.json",
    output_dir="output/disease",
    enumeration_weights=enum_weights
)
```

### Running Inference

```python
from demo.inference import DentalPredictor

# Initialize predictor with trained weights
predictor = DentalPredictor(
    quadrant_weights="output/quadrant/model_final.pth",
    enumeration_weights="output/enumeration/model_final.pth",
    disease_weights="output/disease/model_final.pth"
)

# Run prediction on new X-ray
results = predictor.predict("path/to/xray.jpg")
```

### Demo Script

Try the demo script for quick visualization:

```bash
# Basic usage
python demo/demo.py --image path/to/xray.jpg --output results/

# Advanced options
python demo/demo.py --image path/to/xray.jpg \
                   --threshold 0.7 \
                   --output results/ \
                   --save_visualizations
```

Example output:
```
=== Dental X-ray Analysis Results ===

Quadrants Detected:
- Top Right (Q1): 98% confidence
- Top Left (Q2): 97% confidence
- Bottom Left (Q3): 99% confidence
- Bottom Right (Q4): 98% confidence

Teeth Detected:
- Tooth 11: 95% confidence
- Tooth 12: 93% confidence
[...]

Dental Conditions:
- Caries (Tooth 14): 87% confidence
- Deep Caries (Tooth 26): 92% confidence
- Periapical Lesion (Tooth 36): 89% confidence
```

## Model Architecture

The system uses Mask R-CNN with ResNet-50-FPN backbone:
- **Backbone**: ResNet-50 with Feature Pyramid Network
- **Head**: Region Proposal Network + RoI heads
- **Loss**: Combined loss for classification, box regression, and mask prediction

Key features:
- Transfer learning between phases
- Multi-scale feature detection
- Instance segmentation capability

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Important Note
This software is provided for research purposes only and should not be used for clinical diagnosis without proper validation and regulatory approval. While we strive for accuracy, dental diagnosis should always be performed by qualified dental professionals.

## Citation

If you use this project in your research, please cite:

```bibtex
@article{hamamci2023dentex,
    title    = {DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays},
    author   = {Hamamci, Ibrahim Ethem and Er, Sezgin and Simsar, Enis and others},
    journal  = {arXiv preprint arXiv:2305.19112},
    year     = {2023}
}
```

## Acknowledgments

- DENTEX Challenge organizers and dataset providers
- Detectron2 team at Facebook AI Research