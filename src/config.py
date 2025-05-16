# src/config.py
import os
import yaml
from pathlib import Path

# Default configuration settings
DATASET_YAML = "dataset/vietnam-traffic-sign-detection/dataset.yaml"
PRETRAINED_MODEL = "models/pretrained/yolov8n.pt"
IMG_SIZE = 640
BATCH_SIZE = 16
QAT_EPOCHS = 2
QAT_LEARNING_RATE = 0.0005
DEVICE = ""

def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file {config_path} not found. Using default settings.")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_default_config():
    """Get default configuration."""
    return {
        "dataset": DATASET_YAML,
        "model": PRETRAINED_MODEL,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": QAT_EPOCHS,
        "learning_rate": QAT_LEARNING_RATE,
        "device": DEVICE,
    }

def merge_configs(default_config, user_config):
    """Merge default configuration with user configuration."""
    merged_config = default_config.copy()
    
    # Deep merge the configurations
    def _merge(target, source):
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                _merge(target[key], value)
            else:
                target[key] = value
    
    _merge(merged_config, user_config)
    return merged_config