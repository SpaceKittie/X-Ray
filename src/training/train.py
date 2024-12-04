"""
CLI entry point for dental X-ray model training.

This module provides a command-line interface for training the dental X-ray analysis models.
The training pipeline consists of three phases that should be run sequentially:

1. Quadrant Detection: Trains a model to detect dental quadrants
2. Tooth Enumeration: Uses transfer learning from quadrant model to detect individual teeth
3. Disease Detection: Uses transfer learning from enumeration model to detect dental conditions

Example Usage:
    # Train quadrant detection
    python -m src.training.train \\
        --phase quadrant \\
        --data-dir path/to/images \\
        --json-file path/to/annotations.json \\
        --output-dir output/quadrant

    # Train tooth enumeration (with transfer learning)
    python -m src.training.train \\
        --phase enumeration \\
        --data-dir path/to/images \\
        --json-file path/to/annotations.json \\
        --output-dir output/enumeration \\
        --quadrant-weights output/quadrant/model_final.pth

    # Train disease detection (with transfer learning)
    python -m src.training.train \\
        --phase disease \\
        --data-dir path/to/images \\
        --json-file path/to/annotations.json \\
        --output-dir output/disease \\
        --enumeration-weights output/enumeration/model_final.pth

Optional Arguments:
    --num-iterations: Number of training iterations
    --batch-size: Training batch size
    --learning-rate: Base learning rate
    --val-json: Path to validation annotations
    --config-file: Path to custom config file
"""
import argparse
import os
import logging
from typing import Optional

from .training import (
    train_quadrant_phase,
    train_enumeration_phase,
    train_disease_phase
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train dental X-ray models")
    
    # Required arguments
    parser.add_argument("--data-dir", required=True, help="Directory containing training images")
    parser.add_argument("--json-file", required=True, help="Path to COCO format annotations")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs")
    parser.add_argument("--phase", required=True, choices=["quadrant", "enumeration", "disease"],
                       help="Training phase to run")
    
    # Optional arguments
    parser.add_argument("--quadrant-weights", help="Path to quadrant weights for transfer learning")
    parser.add_argument("--enumeration-weights", help="Path to enumeration weights for transfer learning")
    parser.add_argument("--num-iterations", type=int, help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, help="Base learning rate")
    parser.add_argument("--val-json", help="Path to validation annotations (optional)")
    parser.add_argument("--config-file", help="Path to custom config file (optional)")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Common training arguments
    train_args = {
        "data_dir": args.data_dir,
        "json_file": args.json_file,
        "output_dir": args.output_dir,
        "num_iterations": args.num_iterations,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    
    # Run training for specified phase
    try:
        if args.phase == "quadrant":
            logger.info("Starting quadrant detection training...")
            weights_path = train_quadrant_phase(**train_args)
            
        elif args.phase == "enumeration":
            if args.quadrant_weights:
                train_args["quadrant_weights"] = args.quadrant_weights
            logger.info("Starting tooth enumeration training...")
            weights_path = train_enumeration_phase(**train_args)
            
        elif args.phase == "disease":
            if args.enumeration_weights:
                train_args["enumeration_weights"] = args.enumeration_weights
            logger.info("Starting disease detection training...")
            weights_path = train_disease_phase(**train_args)
            
        logger.info(f"Training completed. Model weights saved to: {weights_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
