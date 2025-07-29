#!/usr/bin/env python3
"""
Continual Learning trainer for NeMo Japanese pre-training.
"""

from typing import Optional, Dict, Any

from .base import BaseTrainer
from ..utils import setup_logging

logger = setup_logging(__name__)


class ContinualTrainer(BaseTrainer):
    """
    Trainer for continual pre-training on Japanese corpus.
    
    Continual learning allows adapting a pre-trained model to new domains
    or languages while preserving existing knowledge.
    """
    
    def __init__(
        self,
        model_config,
        dataset_root: str,
        checkpoint_dir: str,
        experiment_name: str,
        restore_from_path: Optional[str] = None,
        learning_rate_schedule: str = "cosine",
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
    ):
        super().__init__(
            model_config=model_config,
            dataset_root=dataset_root,
            checkpoint_dir=checkpoint_dir,
            experiment_name=experiment_name,
            restore_from_path=restore_from_path,
        )
        
        self.learning_rate_schedule = learning_rate_schedule
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
    
    def get_training_type(self) -> str:
        """Get the type of training."""
        return "Continual Learning"
    
    def setup_recipe(self) -> None:
        """Setup the continual learning recipe."""
        logger.info("Setting up continual learning recipe...")
        
        # For continual learning, we typically use a pre-training recipe
        # This might need adjustment based on NeMo 2.0 API
        try:
            # Try to get a continual pre-training recipe
            self.recipe = self.model_manager.get_finetune_recipe(
                checkpoint_dir=self.checkpoint_dir,
                name=self.experiment_name,
                peft_scheme="none",  # Full parameter training for continual learning
                num_nodes=1,
                num_gpus_per_node=1,
            )
            
            logger.info("Continual learning recipe configured")
            
        except Exception as e:
            logger.error(f"Error setting up continual learning recipe: {e}")
            raise
    
    def configure_continual_specific(self) -> None:
        """Configure continual learning specific settings."""
        if self.recipe is None:
            raise RuntimeError("Recipe must be setup before configuring continual learning")
        
        logger.info("Configuring continual learning specific settings...")
        
        # Learning rate scheduler
        if hasattr(self.recipe, 'optim'):
            self.recipe.optim.lr = self.model_config.learning_rate
            self.recipe.optim.weight_decay = self.weight_decay
            
            # Set scheduler
            if self.learning_rate_schedule == "cosine":
                # Configure cosine annealing scheduler
                if hasattr(self.recipe.optim, 'sched'):
                    self.recipe.optim.sched.name = "CosineAnnealing"
                    self.recipe.optim.sched.warmup_steps = self.warmup_steps
                    self.recipe.optim.sched.max_steps = self.model_config.max_steps
                    
            logger.info(f"Learning rate schedule: {self.learning_rate_schedule}")
            logger.info(f"Warmup steps: {self.warmup_steps}")
            logger.info(f"Weight decay: {self.weight_decay}")
        
        # Continual learning specific trainer settings
        if hasattr(self.recipe.trainer, 'strategy'):
            # Enable distributed training optimizations
            self.recipe.trainer.strategy.ddp = "megatron"
            
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.recipe.trainer, 'gradient_checkpointing'):
            self.recipe.trainer.gradient_checkpointing = True
            
        # Set longer validation intervals for continual learning
        self.recipe.trainer.val_check_interval = max(500, self.model_config.val_check_interval)
    
    def setup_data_module(self):
        """Setup data module for continual learning."""
        # For continual learning, we might need a different data module
        # that handles raw text instead of Q&A pairs
        
        # This is a placeholder - actual implementation would depend on
        # the specific data format for continual pre-training
        return super().setup_data_module()
    
    def prepare(self) -> None:
        """Prepare for continual learning training."""
        # Call base preparation
        super().prepare()
        
        # Configure continual learning specific settings
        self.configure_continual_specific()
        
        logger.info("Continual learning training preparation completed")
    
    def log_configuration(self) -> None:
        """Log continual learning training configuration."""
        super().log_configuration()
        
        # Log continual learning specific settings
        logger.info("Continual Learning Configuration:")
        logger.info(f"  - Learning Rate Schedule: {self.learning_rate_schedule}")
        logger.info(f"  - Warmup Steps: {self.warmup_steps}")
        logger.info(f"  - Weight Decay: {self.weight_decay}")
        logger.info("  - Training Type: Full parameter continual pre-training")
        logger.info("  - Goal: Adapt to Japanese language while preserving existing knowledge")
    
    def get_recipe_info(self) -> Dict[str, Any]:
        """Get information about the continual learning recipe."""
        info = super().get_recipe_info()
        
        info.update({
            "learning_rate_schedule": self.learning_rate_schedule,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
        })
        
        return info 