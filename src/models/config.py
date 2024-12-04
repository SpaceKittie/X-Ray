from detectron2.config import get_cfg
from detectron2 import model_zoo

def get_base_config():
    """
    Get base configuration for all models.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    
    # Common settings
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.TEST.EVAL_PERIOD = 500
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    
    return cfg

def get_quadrant_config(base_cfg=None):
    """
    Get configuration for quadrant detection.
    """
    cfg = base_cfg if base_cfg else get_base_config()
    
    # Model settings
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # 4 quadrants
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.MASK_ON = True
    
    # Training settings
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.BASE_LR = 0.00001
    cfg.SOLVER.WEIGHT_DECAY = 0.001
    cfg.SOLVER.OPTIMIZER_NAME = "ADAMW"
    cfg.SOLVER.MOMENTUM = 0
    
    # Gradient clipping
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    
    # Data augmentation
    cfg.INPUT.RANDOM_FLIP = "none"
    cfg.INPUT.MASK_FORMAT = "polygon"
    
    return cfg

def get_enumeration_config(quadrant_weights=None, base_cfg=None):
    """
    Get configuration for tooth enumeration.
    """
    cfg = base_cfg if base_cfg else get_base_config()
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 32  # 8 teeth per quadrant
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.BASE_LR = 0.001
    
    if quadrant_weights:
        cfg.MODEL.WEIGHTS = quadrant_weights
    
    return cfg

def get_disease_config(enumeration_weights=None, base_cfg=None):
    """
    Get configuration for disease detection.
    """
    cfg = base_cfg if base_cfg else get_base_config()
    
    # Adjust based on your disease classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    cfg.SOLVER.MAX_ITER = 1500
    cfg.SOLVER.BASE_LR = 0.0005  # Lower learning rate for fine-tuning
    
    if enumeration_weights:
        cfg.MODEL.WEIGHTS = enumeration_weights
    
    return cfg
