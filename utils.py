import os
import random
import numpy as np
import torch
import logging
import yaml
from datetime import datetime


def seed_everything(seed: int = 42) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def setup_logger(output_dir: str) -> logging.Logger:
    """
    Set up logger
    
    Args:
        output_dir: Output directory
    
    Returns:
        Logger object
    """
    logger = logging.getLogger("yolo")
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    log_file = os.path.join(output_dir, 'train.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def load_yaml(file_path: str) -> dict:
    """
    Load YAML file
    
    Args:
        file_path: Path to YAML file
    
    Returns:
        Dictionary with configuration
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, file_path: str) -> None:
    """
    Save dictionary to YAML file
    
    Args:
        data: Dictionary to save
        file_path: Path to save YAML file
    """
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
