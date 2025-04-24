"""
Configuration utilities for YOLOv8 QAT training.

This module provides functions for loading, merging, and managing
configuration settings for training, quantization, and deployment.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union, List

# Setup logging
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            logger.info(f"Successfully loaded configuration from {config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_path}: {e}")
            raise

def get_default_config(config_type: str = "base") -> Dict[str, Any]:
    """
    Get default configuration dictionary.
    
    Args:
        config_type: Type of configuration ('base', 'qat', 'export')
        
    Returns:
        Default configuration dictionary
    """
    # Define base directory for configs
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(base_dir, "configs")
    
    # Map config type to file
    config_files = {
        "base": "base_config.yaml",
        "qat": "qat_config.yaml",
        "export": "export_config.yaml",
        "quantization": "quantization_config.yaml"
    }
    
    if config_type not in config_files:
        raise ValueError(f"Unknown config type '{config_type}'. "
                         f"Available types: {', '.join(config_files.keys())}")
    
    config_path = os.path.join(config_dir, config_files[config_type])
    return load_config(config_path)

def merge_configs(base_config: Dict[str, Any], 
                  override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge configurations with overrides.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    if base_config is None:
        return override_config.copy() if override_config else {}
    
    if override_config is None:
        return base_config.copy()
    
    merged_config = base_config.copy()
    
    # Recursively merge dictionaries
    for key, value in override_config.items():
        if (key in merged_config and 
            isinstance(merged_config[key], dict) and 
            isinstance(value, dict)):
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            merged_config[key] = value
    
    return merged_config

def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
        
    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Configuration saved to {output_path}")

def update_config_paths(config: Dict[str, Any], base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Update relative paths in config to absolute paths.
    
    Args:
        config: Configuration dictionary
        base_dir: Base directory for relative paths (default: project root)
        
    Returns:
        Updated configuration dictionary
    """
    if base_dir is None:
        # Use project root as default base directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    updated_config = config.copy()
    
    # Update paths in training section
    if 'save' in config:
        save_config = config['save']
        for key in ['dir', 'best', 'last']:
            if key in save_config and not os.path.isabs(save_config[key]):
                updated_config['save'][key] = os.path.normpath(
                    os.path.join(base_dir, save_config[key])
                )
    
    # Update paths in model section
    if 'model' in config and 'pretrained_weights' in config['model']:
        if not os.path.isabs(config['model']['pretrained_weights']):
            updated_config['model']['pretrained_weights'] = os.path.normpath(
                os.path.join(base_dir, config['model']['pretrained_weights'])
            )
    
    # Update paths in data section
    if 'data' in config and 'path' in config['data']:
        if not os.path.isabs(config['data']['path']):
            updated_config['data']['path'] = os.path.normpath(
                os.path.join(base_dir, config['data']['path'])
            )
    
    # Update paths in logging section
    if 'logging' in config and 'dir' in config['logging']:
        if not os.path.isabs(config['logging']['dir']):
            updated_config['logging']['dir'] = os.path.normpath(
                os.path.join(base_dir, config['logging']['dir'])
            )
    
    return updated_config

def get_quantization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract quantization configuration from main config.
    
    Args:
        config: Main configuration dictionary
        
    Returns:
        Quantization configuration dictionary
    """
    # Start with default quantization config
    default_quant_config = get_default_config("quantization")
    
    # Extract quantization config from main config
    quant_config = {}
    
    if 'quantization' in config:
        quant_config = config['quantization']
    elif 'qat' in config and 'quantization' in config['qat']:
        quant_config = config['qat']['quantization']
        
    # Merge with default
    return merge_configs(default_quant_config, quant_config)

def validate_config(config: Dict[str, Any], config_type: str = "base") -> bool:
    """
    Validate configuration against schema.
    
    Args:
        config: Configuration dictionary to validate
        config_type: Type of configuration ('base', 'qat', 'export')
        
    Returns:
        True if valid, False otherwise
    """
    # Get default config for comparison
    default_config = get_default_config(config_type)
    
    # Basic validation - check required top-level keys
    required_keys = set(default_config.keys())
    
    if config_type == "base":
        required_keys = {"training", "model", "data"}
    elif config_type == "qat":
        required_keys = {"training", "quantization"}
    elif config_type == "export":
        required_keys = {"format", "save"}
    
    missing_keys = required_keys - set(config.keys())
    
    if missing_keys:
        logger.warning(f"Missing required configuration keys: {missing_keys}")
        return False
    
    # More specific validation could be added here
    return True

def create_experiment_config(
    base_config_path: str,
    experiment_name: str,
    overrides: Dict[str, Any]
) -> str:
    """
    Create and save an experiment configuration.
    
    Args:
        base_config_path: Path to base configuration file
        experiment_name: Name of the experiment
        overrides: Configuration overrides for this experiment
        
    Returns:
        Path to the saved experiment configuration file
    """
    # Load base configuration
    base_config = load_config(base_config_path)
    
    # Apply overrides
    experiment_config = merge_configs(base_config, overrides)
    
    # Create experiment config directory if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    experiment_dir = os.path.join(base_dir, "configs", "experiments")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save experiment configuration
    experiment_path = os.path.join(experiment_dir, f"{experiment_name}.yaml")
    save_config(experiment_config, experiment_path)
    
    return experiment_path

def get_config_by_experiment(experiment_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Experiment configuration dictionary
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    experiment_path = os.path.join(base_dir, "configs", "experiments", f"{experiment_name}.yaml")
    
    if not os.path.exists(experiment_path):
        raise FileNotFoundError(f"Experiment configuration not found: {experiment_name}")
    
    return load_config(experiment_path)