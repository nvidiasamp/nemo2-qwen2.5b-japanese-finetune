"""
NeMo Japanese Fine-tuning Package

Main package for Japanese language model fine-tuning with NeMo 2.0.
Supports PEFT (LoRA), SFT, and continual pre-training methods.
"""

from .data import DataConverter, DataProcessor
from .models import QwenModelConfig, ModelUtils
from .training import PEFTTrainer, SFTTrainer, ContinualTrainer
from .utils import Config, setup_logging

__version__ = "0.1.0"

__all__ = [
    "DataConverter",
    "DataProcessor", 
    "QwenModelConfig",
    "ModelUtils",
    "PEFTTrainer",
    "SFTTrainer", 
    "ContinualTrainer",
    "Config",
    "setup_logging",
] 