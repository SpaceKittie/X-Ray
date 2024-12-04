from typing import Optional
import os
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class DentalTrainer:
    """
    Trainer class that handles the three-phase training process:
    1. Quadrant Detection
    2. Tooth Enumeration
    3. Disease Detection
    """
    def __init__(self, output_dir: str):
        """
        Initialize the trainer.
        
        Args:
            output_dir: Base directory for saving model outputs
        """
        self.output_dir = output_dir
        self.phases = ['quadrant', 'enumeration', 'disease']
        
    def _get_phase_dir(self, phase: str) -> str:
        """Get output directory for a specific phase."""
        phase_dir = os.path.join(self.output_dir, phase)
        os.makedirs(phase_dir, exist_ok=True)
        return phase_dir
        
    def _create_trainer(self, cfg, phase: str) -> DefaultTrainer:
        """Create a trainer for a specific phase."""
        cfg.OUTPUT_DIR = self._get_phase_dir(phase)
        
        class CustomTrainer(DefaultTrainer):
            @classmethod
            def build_evaluator(cls, cfg, dataset_name):
                return COCOEvaluator(
                    dataset_name,
                    output_dir=cfg.OUTPUT_DIR,
                    tasks=("bbox", "segm")
                )
                
        return CustomTrainer(cfg)
    
    def train_quadrant(self, cfg) -> str:
        """
        Train the quadrant detection model.
        
        Returns:
            Path to the best model weights
        """
        trainer = self._create_trainer(cfg, 'quadrant')
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        # Return path to best model
        return os.path.join(self._get_phase_dir('quadrant'), "model_final.pth")
    
    def train_enumeration(self, cfg, quadrant_weights: Optional[str] = None) -> str:
        """
        Train the tooth enumeration model.
        
        Args:
            cfg: Model configuration
            quadrant_weights: Path to quadrant detection weights for transfer learning
            
        Returns:
            Path to the best model weights
        """
        if quadrant_weights:
            cfg.MODEL.WEIGHTS = quadrant_weights
            
        trainer = self._create_trainer(cfg, 'enumeration')
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        return os.path.join(self._get_phase_dir('enumeration'), "model_final.pth")
    
    def train_disease(self, cfg, enumeration_weights: Optional[str] = None) -> str:
        """
        Train the disease detection model.
        
        Args:
            cfg: Model configuration
            enumeration_weights: Path to enumeration model weights for transfer learning
            
        Returns:
            Path to the best model weights
        """
        if enumeration_weights:
            cfg.MODEL.WEIGHTS = enumeration_weights
            
        trainer = self._create_trainer(cfg, 'disease')
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        return os.path.join(self._get_phase_dir('disease'), "model_final.pth")
