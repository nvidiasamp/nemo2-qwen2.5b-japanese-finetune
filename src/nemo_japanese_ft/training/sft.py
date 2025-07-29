#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) trainer for NeMo Japanese fine-tuning.
"""

from typing import Optional

from .base import BaseTrainer
from ..utils import setup_logging

logger = setup_logging(__name__)


class SFTTrainer(BaseTrainer):
    """
    Trainer for traditional Supervised Fine-Tuning.
    
    SFT trains all model parameters and typically achieves the best performance
    but requires more memory and computational resources.
    """
    
    def __init__(
        self,
        model_config,
        dataset_root: str,
        checkpoint_dir: str,
        experiment_name: str,
        restore_from_path: Optional[str] = None,
    ):
        super().__init__(
            model_config=model_config,
            dataset_root=dataset_root,
            checkpoint_dir=checkpoint_dir,
            experiment_name=experiment_name,
            restore_from_path=restore_from_path,
        )
    
    def get_training_type(self) -> str:
        """Get the type of training."""
        return "SFT"
    
    def setup_recipe(self) -> None:
        """Setup the SFT training recipe."""
        logger.info("Setting up SFT training recipe...")
        
        # Get base recipe from model manager (no PEFT scheme)
        self.recipe = self.model_manager.get_finetune_recipe(
            checkpoint_dir=self.checkpoint_dir,
            name=self.experiment_name,
            peft_scheme="none",  # Full fine-tuning
            num_nodes=1,
            num_gpus_per_node=1,
        )
        
        logger.info("SFT recipe configured for full parameter training")
    
    def configure_sft_specific(self) -> None:
        """Configure SFT-specific settings."""
        if self.recipe is None:
            raise RuntimeError("Recipe must be setup before configuring SFT")
        
        # SFT-specific settings for full parameter training
        logger.info("Configuring SFT for full parameter optimization")
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.recipe.trainer, 'gradient_checkpointing'):
            self.recipe.trainer.gradient_checkpointing = True
        
        # Optimizer settings for full fine-tuning
        if hasattr(self.recipe, 'optim'):
            self.recipe.optim.lr = self.model_config.learning_rate
        
        # Mixed precision training
        if hasattr(self.recipe.trainer, 'precision'):
            self.recipe.trainer.precision = self.model_config.precision
            logger.info(f"Using precision: {self.model_config.precision}")
    
    def prepare(self) -> None:
        """Prepare for SFT training."""
        # Call base preparation
        super().prepare()
        
        # Configure SFT-specific settings
        self.configure_sft_specific()
        
        logger.info("SFT training preparation completed")
    
    def log_configuration(self) -> None:
        """Log SFT training configuration."""
        super().log_configuration()
        
        # Log SFT-specific settings
        logger.info("SFT Configuration:")
        logger.info("  - Full Parameter Training: All 494M parameters")
        logger.info("  - Maximum Performance: Best possible adaptation")
        logger.info("  - Memory Usage: High (~22.7GB peak)")
        logger.info("  - Training Time: Standard (baseline)")
        
        # Warn about resource requirements
        if self.model_config.model_size in ["7b", "14b", "32b", "72b"]:
            logger.warning("Large model detected! Ensure sufficient GPU memory.")
            logger.warning("Consider using PEFT for memory-constrained systems.") 