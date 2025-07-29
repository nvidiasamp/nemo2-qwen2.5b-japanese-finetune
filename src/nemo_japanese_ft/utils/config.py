#!/usr/bin/env python3
"""
Configuration management utilities.
"""

import json
import yaml
import os
from typing import Dict, Any, Union, Optional
from pathlib import Path
from dataclasses import asdict

from .logging import setup_logging

logger = setup_logging(__name__)


class Config:
    """Configuration management class."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = config_dict or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
    
    def update(self, other_config: Union['Config', Dict[str, Any]]) -> None:
        """Update configuration with another config or dict."""
        if isinstance(other_config, Config):
            other_dict = other_config._config
        else:
            other_dict = other_config
            
        self._deep_update(self._config, other_dict)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if (
                key in base_dict 
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like assignment."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None


def load_config(config_path: str) -> Config:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file (.json, .yaml, .yml)
        
    Returns:
        Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is not supported
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    suffix = config_path.suffix.lower()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if suffix == '.json':
                config_dict = json.load(f)
            elif suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {suffix}")
        
        logger.info(f"Loaded configuration from {config_path}")
        return Config(config_dict)
        
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def save_config(config: Union[Config, Dict[str, Any]], config_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Config object or dictionary to save
        config_path: Path to save configuration file (.json, .yaml, .yml)
    """
    config_path = Path(config_path)
    
    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get config dict
    if isinstance(config, Config):
        config_dict = config.to_dict()
    else:
        config_dict = config
    
    suffix = config_path.suffix.lower()
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            if suffix == '.json':
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            elif suffix in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {suffix}")
        
        logger.info(f"Saved configuration to {config_path}")
        
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")
        raise


def merge_configs(*configs: Union[Config, Dict[str, Any]]) -> Config:
    """
    Merge multiple configurations.
    
    Args:
        *configs: Config objects or dictionaries to merge
        
    Returns:
        Merged Config object
    """
    merged = Config()
    
    for config in configs:
        merged.update(config)
    
    return merged


def config_from_dataclass(dataclass_instance: Any) -> Config:
    """
    Create Config from dataclass instance.
    
    Args:
        dataclass_instance: Dataclass instance
        
    Returns:
        Config object
    """
    try:
        config_dict = asdict(dataclass_instance)
        return Config(config_dict)
    except Exception as e:
        logger.error(f"Error converting dataclass to config: {e}")
        raise 