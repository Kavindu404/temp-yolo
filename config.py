import os
import yaml
from typing import Dict, Any

from utils import load_yaml, save_yaml


def get_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    # Check if config file exists
    if not os.path.exists(config_path):
        # Return default config
        return get_default_config()
    
    # Load config
    config = load_yaml(config_path)
    
    # Ensure all required keys are present
    default_config = get_default_config()
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_key not in config[key]:
                    config[key][sub_key] = sub_value
    
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration
    
    Returns:
        Default configuration dictionary
    """
    return {
        # Model parameters
        'model': {
            'input_size': 640,
            'num_classes': 80,  # COCO has 80 classes
        },
        
        # Dataloader parameters
        'dataloader': {
            'num_workers': 4,
            'pin_memory': True,
        },
        
        # Augmentation parameters
        'augmentation': {
            'mosaic_prob': 0.5,
            'copy_paste_prob': 0.3,
            'hsv_prob': 0.5,
            'flip_prob': 0.5,
            'scale': (0.5, 1.5),
            'translate': 0.1,
            'degrees': 10.0,
        },
        
        # Optimizer parameters
        'optimizer': {
            'lr': 0.001,
            'weight_decay': 0.0005,
            'momentum': 0.937,
        },
        
        # Scheduler parameters
        'scheduler': {
            'T_max': 100,
            'eta_min': 1e-5,
        },
        
        # Loss parameters
        'loss': {
            'box_gain': 0.05,
            'cls_gain': 0.5,
            'obj_gain': 1.0,
            'seg_gain': 1.0,
            'focal_loss_gamma': 2.0,
        },
    }


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    save_yaml(config, config_path)
