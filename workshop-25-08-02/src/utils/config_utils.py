"""
Configuration utilities for Japanese Continual Learning with NeMo 2.0
"""

import yaml
import json
from typing import Any, Dict, Union
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load configuration file (YAML or JSON)
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration as OmegaConf DictConfig
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load based on file extension
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return OmegaConf.create(config_dict)

def save_config(config: Union[Dict, DictConfig], output_path: Union[str, Path]) -> None:
    """
    Save configuration to file
    
    Args:
        config: Configuration to save
        output_path: Path to save configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict if OmegaConf
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config
    
    # Save based on file extension
    if output_path.suffix.lower() in ['.yaml', '.yml']:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    elif output_path.suffix.lower() == '.json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported config file format: {output_path.suffix}")

def merge_configs(*configs: Union[Dict, DictConfig]) -> DictConfig:
    """
    Merge multiple configurations
    
    Args:
        *configs: Configurations to merge
        
    Returns:
        Merged configuration
    """
    merged = OmegaConf.create({})
    
    for config in configs:
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        merged = OmegaConf.merge(merged, config)
    
    return merged

def get_config_value(config: DictConfig, key: str, default: Any = None) -> Any:
    """
    Get configuration value with dot notation
    
    Args:
        config: Configuration object
        key: Key in dot notation (e.g., 'model.config.hidden_size')
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    try:
        return OmegaConf.select(config, key, default=default)
    except Exception:
        return default

def update_config_value(config: DictConfig, key: str, value: Any) -> None:
    """
    Update configuration value with dot notation
    
    Args:
        config: Configuration object
        key: Key in dot notation
        value: New value
    """
    OmegaConf.update(config, key, value, merge=True) 