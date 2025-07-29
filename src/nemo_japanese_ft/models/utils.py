#!/usr/bin/env python3
"""
Utility functions for model operations.
"""

import os
from typing import Optional, Union
from pathlib import Path

from nemo.collections import llm

from ..utils import setup_logging

logger = setup_logging(__name__)


class ModelUtils:
    """Utility functions for model operations."""
    
    @staticmethod
    def import_model_from_hf(
        model_name: str,
        output_path: str,
        model_config: Optional[object] = None,
        overwrite: bool = True
    ) -> str:
        """
        Import a model from HuggingFace and convert to NeMo format.
        
        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-0.5B")
            output_path: Path to save the converted .nemo file
            model_config: NeMo model configuration (auto-detected if None)
            overwrite: Whether to overwrite existing file
            
        Returns:
            Path to the saved .nemo file
        """
        # Auto-detect model config if not provided
        if model_config is None:
            model_config = ModelUtils._get_config_for_model(model_name)
        
        # Create model instance
        model = llm.Qwen2Model(model_config)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        logger.info(f"Converting {model_name} to NeMo format...")
        logger.info(f"Output path: {output_path}")
        
        try:
            llm.import_ckpt(
                model=model,
                source=f'hf://{model_name}',
                output_path=output_path,
                overwrite=overwrite
            )
            logger.info(f"Model conversion successful. Saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Model conversion failed: {e}")
            raise
    
    @staticmethod
    def _get_config_for_model(model_name: str):
        """Get appropriate NeMo config for a HuggingFace model."""
        # Extract size from model name
        if "0.5B" in model_name:
            return llm.Qwen25Config500M()
        elif "1.5B" in model_name:
            return llm.Qwen25Config1_5B()
        elif "7B" in model_name:
            return llm.Qwen25Config7B()
        elif "14B" in model_name:
            return llm.Qwen25Config14B()
        elif "32B" in model_name:
            return llm.Qwen25Config32B()
        elif "72B" in model_name:
            return llm.Qwen25Config72B()
        else:
            logger.warning(f"Unknown model size in {model_name}, defaulting to 0.5B config")
            return llm.Qwen25Config500M()
    
    @staticmethod
    def validate_nemo_file(file_path: str) -> bool:
        """
        Validate that a .nemo file exists and is readable.
        
        Args:
            file_path: Path to the .nemo file
            
        Returns:
            True if file is valid, False otherwise
        """
        if not os.path.exists(file_path):
            logger.error(f"NeMo file not found: {file_path}")
            return False
        
        if not file_path.endswith('.nemo'):
            logger.error(f"File does not have .nemo extension: {file_path}")
            return False
        
        try:
            # Basic file read test
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logger.error(f"NeMo file is empty: {file_path}")
                return False
            
            logger.info(f"NeMo file validation passed: {file_path} ({file_size:,} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Error validating NeMo file {file_path}: {e}")
            return False
    
    @staticmethod
    def get_model_size_from_path(file_path: str) -> Optional[str]:
        """
        Extract model size from file path.
        
        Args:
            file_path: Path to model file
            
        Returns:
            Model size string (e.g., "0.5b") or None if not detected
        """
        path_lower = file_path.lower()
        
        size_patterns = {
            "500m": "0.5b",
            "0.5b": "0.5b",
            "1.5b": "1.5b", 
            "7b": "7b",
            "14b": "14b",
            "32b": "32b",
            "72b": "72b",
        }
        
        for pattern, size in size_patterns.items():
            if pattern in path_lower:
                return size
        
        return None


def import_model_from_hf(
    model_name: str,
    output_path: str,
    overwrite: bool = True
) -> str:
    """
    Convenience function to import a model from HuggingFace.
    
    Args:
        model_name: HuggingFace model name
        output_path: Path to save the converted .nemo file
        overwrite: Whether to overwrite existing file
        
    Returns:
        Path to the saved .nemo file
    """
    return ModelUtils.import_model_from_hf(
        model_name=model_name,
        output_path=output_path,
        overwrite=overwrite
    ) 