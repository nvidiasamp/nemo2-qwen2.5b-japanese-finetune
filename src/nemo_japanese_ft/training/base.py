#!/usr/bin/env python3
"""
Base trainer class for NeMo Japanese fine-tuning.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import nemo_run as run
from nemo import lightning as nl

from ..models import QwenModelConfig, QwenModelManager
from ..data import QADataProcessor
from ..utils import setup_logging

logger = setup_logging(__name__)


class BaseTrainer(ABC):
    """Abstract base class for all trainers."""
    
    def __init__(
        self,
        model_config: QwenModelConfig,
        dataset_root: str,
        checkpoint_dir: str,
        experiment_name: str,
        restore_from_path: Optional[str] = None,
    ):
        self.model_config = model_config
        self.dataset_root = dataset_root
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.restore_from_path = restore_from_path
        
        self.model_manager = QwenModelManager(model_config)
        self.recipe = None
        
    def setup_data_module(self) -> run.Config:
        """Setup data module configuration."""
        return run.Config(
            QADataProcessor,
            dataset_root=self.dataset_root,
            seq_length=self.model_config.seq_length,
            micro_batch_size=self.model_config.micro_batch_size,
            global_batch_size=self.model_config.global_batch_size,
            dataset_kwargs={}
        )
    
    def setup_restore_config(self) -> Optional[run.Config]:
        """Setup model restoration configuration."""
        if self.restore_from_path:
            return run.Config(
                nl.RestoreConfig,
                path=self.restore_from_path
            )
        return None
    
    @abstractmethod
    def get_training_type(self) -> str:
        """Get the type of training (e.g., 'PEFT', 'SFT', 'Continual')."""
        pass
    
    @abstractmethod
    def setup_recipe(self) -> None:
        """Setup the training recipe."""
        pass
    
    def configure_trainer(self) -> None:
        """Configure trainer settings."""
        if self.recipe is None:
            raise RuntimeError("Recipe must be setup before configuring trainer")
            
        # Basic trainer configuration
        self.recipe.trainer.max_steps = self.model_config.max_steps
        self.recipe.trainer.num_sanity_val_steps = self.model_config.num_sanity_val_steps
        self.recipe.trainer.val_check_interval = self.model_config.val_check_interval
        self.recipe.trainer.log_every_n_steps = self.model_config.log_every_n_steps
        self.recipe.trainer.enable_checkpointing = True
        
        # Precision setting
        if hasattr(self.recipe.trainer, 'precision'):
            self.recipe.trainer.precision = self.model_config.precision
    
    def configure_data(self) -> None:
        """Configure data module."""
        if self.recipe is None:
            raise RuntimeError("Recipe must be setup before configuring data")
            
        self.recipe.data = self.setup_data_module()
    
    def configure_restoration(self) -> None:
        """Configure model restoration."""
        if self.recipe is None:
            raise RuntimeError("Recipe must be setup before configuring restoration")
            
        restore_config = self.setup_restore_config()
        if restore_config:
            self.recipe.resume.restore_config = restore_config
    
    def validate_setup(self) -> None:
        """Validate training setup."""
        # Check data directory
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_root}")
        
        # Check training data
        train_file = os.path.join(self.dataset_root, "training.jsonl")
        val_file = os.path.join(self.dataset_root, "validation.jsonl")
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file not found: {train_file}")
        if not os.path.exists(val_file):
            raise FileNotFoundError(f"Validation file not found: {val_file}")
        
        # Check restore path if provided
        if self.restore_from_path and not os.path.exists(self.restore_from_path):
            raise FileNotFoundError(f"Restore path not found: {self.restore_from_path}")
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info("Training setup validation passed")
    
    def log_configuration(self) -> None:
        """Log training configuration."""
        logger.info("=== Training Configuration ===")
        logger.info(f"Training Type: {self.get_training_type()}")
        logger.info(f"Model: Qwen2.5-{self.model_config.model_size}")
        logger.info(f"Experiment: {self.experiment_name}")
        logger.info(f"Max Steps: {self.model_config.max_steps}")
        logger.info(f"Micro Batch Size: {self.model_config.micro_batch_size}")
        logger.info(f"Global Batch Size: {self.model_config.global_batch_size}")
        logger.info(f"Sequence Length: {self.model_config.seq_length}")
        logger.info(f"Learning Rate: {self.model_config.learning_rate}")
        logger.info(f"Precision: {self.model_config.precision}")
        logger.info(f"Checkpoint Dir: {self.checkpoint_dir}")
        logger.info(f"Data Root: {self.dataset_root}")
        if self.restore_from_path:
            logger.info(f"Restore From: {self.restore_from_path}")
    
    def prepare(self) -> None:
        """Prepare for training."""
        logger.info(f"Preparing {self.get_training_type()} training...")
        
        # Setup components
        self.setup_recipe()
        self.configure_trainer()
        self.configure_data()
        self.configure_restoration()
        
        # Validate setup
        self.validate_setup()
        
        # Log configuration
        self.log_configuration()
        
        logger.info("Training preparation completed")
    
    def train(self, executor: Optional[run.Executor] = None) -> None:
        """Execute training."""
        if self.recipe is None:
            raise RuntimeError("Must call prepare() before train()")
        
        if executor is None:
            executor = run.LocalExecutor()
        
        logger.info(f"Starting {self.get_training_type()} training...")
        
        try:
            run.run(self.recipe, executor=executor)
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def get_recipe_info(self) -> Dict[str, Any]:
        """Get information about the training recipe."""
        if self.recipe is None:
            return {}
        
        return {
            "training_type": self.get_training_type(),
            "model_config": self.model_manager.get_model_info(),
            "checkpoint_dir": self.checkpoint_dir,
            "experiment_name": self.experiment_name,
            "restore_from_path": self.restore_from_path,
        } 