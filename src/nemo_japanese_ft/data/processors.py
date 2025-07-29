#!/usr/bin/env python3
"""
Data processors for NeMo-compatible data preparation and formatting.
"""

import os
import json
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule

from ..utils import setup_logging

logger = setup_logging(__name__)


class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    def prepare_data(self) -> None:
        """Prepare data for training."""
        pass


class QADataProcessor(FineTuningDataModule):
    """
    Custom data processor for question-answer datasets.
    
    Expects JSONL format: {"input": "question", "output": "answer"}
    """

    def __init__(self, dataset_root: str, **kwargs):
        self.dataset_root = dataset_root
        super().__init__(dataset_root=dataset_root, **kwargs)

    def prepare_data(self) -> None:
        """
        Prepare data for training - validates existing JSONL files
        and creates necessary symlinks if needed.
        """
        train_path = os.path.join(self.dataset_root, "training.jsonl")
        val_path = os.path.join(self.dataset_root, "validation.jsonl")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training file not found: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation file not found: {val_path}")

        logger.info(f"Found training data: {train_path}")
        logger.info(f"Found validation data: {val_path}")

        # Create symlinks if needed for NeMo expected format
        training_jsonl = os.path.join(self.dataset_root, "training.jsonl")
        validation_jsonl = os.path.join(self.dataset_root, "validation.jsonl")

        if not os.path.exists(training_jsonl):
            logger.info(f"Creating symlink: {training_jsonl} -> {train_path}")
            os.symlink(os.path.basename(train_path), training_jsonl)

        if not os.path.exists(validation_jsonl):
            logger.info(f"Creating symlink: {validation_jsonl} -> {val_path}")
            os.symlink(os.path.basename(val_path), validation_jsonl)

        super().prepare_data()

    def validate_data_format(self, file_path: str) -> bool:
        """
        Validate that the data file has the expected format.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            True if format is valid, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    if line_no > 10:  # Check first 10 lines only
                        break
                        
                    try:
                        data = json.loads(line.strip())
                        if not isinstance(data, dict):
                            logger.error(f"Line {line_no}: Expected dict, got {type(data)}")
                            return False
                        
                        if "input" not in data or "output" not in data:
                            logger.error(f"Line {line_no}: Missing 'input' or 'output' field")
                            return False
                            
                        if not isinstance(data["input"], str) or not isinstance(data["output"], str):
                            logger.error(f"Line {line_no}: 'input' and 'output' must be strings")
                            return False
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Line {line_no}: JSON decode error: {e}")
                        return False
                        
            logger.info(f"Data format validation passed for {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating {file_path}: {e}")
            return False

    def get_data_stats(self, file_path: str) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_samples": 0,
            "avg_input_length": 0,
            "avg_output_length": 0,
            "max_input_length": 0,
            "max_output_length": 0,
            "min_input_length": float('inf'),
            "min_output_length": float('inf'),
        }
        
        input_lengths = []
        output_lengths = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        input_len = len(data["input"])
                        output_len = len(data["output"])
                        
                        input_lengths.append(input_len)
                        output_lengths.append(output_len)
                        
                        stats["total_samples"] += 1
                        stats["max_input_length"] = max(stats["max_input_length"], input_len)
                        stats["max_output_length"] = max(stats["max_output_length"], output_len)
                        stats["min_input_length"] = min(stats["min_input_length"], input_len)
                        stats["min_output_length"] = min(stats["min_output_length"], output_len)
                        
                    except json.JSONDecodeError:
                        continue
                        
            if input_lengths:
                stats["avg_input_length"] = sum(input_lengths) / len(input_lengths)
                stats["avg_output_length"] = sum(output_lengths) / len(output_lengths)
            
            if stats["min_input_length"] == float('inf'):
                stats["min_input_length"] = 0
            if stats["min_output_length"] == float('inf'):
                stats["min_output_length"] = 0
                
        except Exception as e:
            logger.error(f"Error getting stats for {file_path}: {e}")
            
        return stats 