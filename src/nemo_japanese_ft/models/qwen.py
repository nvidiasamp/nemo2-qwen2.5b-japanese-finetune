#!/usr/bin/env python3
"""
Qwen model configuration and management for NeMo fine-tuning.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass

from nemo.collections import llm

from ..utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class QwenModelConfig:
    """Configuration for Qwen models."""
    
    model_size: str = "0.5b"  # Model size: "0.5b", "1.5b", "7b", etc.
    seq_length: int = 2048
    micro_batch_size: int = 2
    global_batch_size: int = 16
    learning_rate: float = 3e-4
    max_steps: int = 1000
    val_check_interval: int = 100
    log_every_n_steps: int = 10
    num_sanity_val_steps: int = 2
    precision: str = "bf16-mixed"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_sizes = ["0.5b", "1.5b", "7b", "14b", "32b", "72b"]
        if self.model_size not in valid_sizes:
            raise ValueError(f"Invalid model size: {self.model_size}. Valid sizes: {valid_sizes}")
        
        if self.global_batch_size % self.micro_batch_size != 0:
            raise ValueError(
                f"Global batch size ({self.global_batch_size}) must be divisible by "
                f"micro batch size ({self.micro_batch_size})"
            )

    def get_nemo_config(self):
        """Get NeMo model configuration based on size."""
        config_map = {
            "0.5b": llm.Qwen25Config500M,
            "1.5b": llm.Qwen25Config1_5B,
            "7b": llm.Qwen25Config7B,
            "14b": llm.Qwen25Config14B,
            "32b": llm.Qwen25Config32B,
            "72b": llm.Qwen25Config72B,
        }
        
        if self.model_size not in config_map:
            raise ValueError(f"Unsupported model size: {self.model_size}")
            
        return config_map[self.model_size]()

    def get_hf_model_name(self) -> str:
        """Get HuggingFace model name based on size."""
        size_map = {
            "0.5b": "Qwen/Qwen2.5-0.5B",
            "1.5b": "Qwen/Qwen2.5-1.5B", 
            "7b": "Qwen/Qwen2.5-7B",
            "14b": "Qwen/Qwen2.5-14B",
            "32b": "Qwen/Qwen2.5-32B",
            "72b": "Qwen/Qwen2.5-72B",
        }
        
        return size_map.get(self.model_size, "Qwen/Qwen2.5-0.5B")


class QwenModelManager:
    """Manager for Qwen model operations."""
    
    def __init__(self, config: QwenModelConfig):
        self.config = config
        self._model = None
        self._model_config = None
        
    def get_model(self):
        """Get or create NeMo model instance."""
        if self._model is None:
            self._model_config = self.config.get_nemo_config()
            self._model = llm.Qwen2Model(self._model_config)
            logger.info(f"Created Qwen {self.config.model_size} model")
        
        return self._model
    
    def get_model_config(self):
        """Get NeMo model configuration."""
        if self._model_config is None:
            self._model_config = self.config.get_nemo_config()
            
        return self._model_config
    
    def import_from_hf(self, output_path: str, overwrite: bool = True) -> str:
        """
        Import model from HuggingFace and convert to NeMo format.
        
        Args:
            output_path: Path to save the converted .nemo file
            overwrite: Whether to overwrite existing file
            
        Returns:
            Path to the saved .nemo file
        """
        model = self.get_model()
        hf_model_name = self.config.get_hf_model_name()
        
        logger.info(f"Converting {hf_model_name} to NeMo format...")
        logger.info(f"Output path: {output_path}")
        
        try:
            llm.import_ckpt(
                model=model,
                source=f'hf://{hf_model_name}',
                output_path=output_path,
                overwrite=overwrite
            )
            logger.info(f"Model conversion successful. Saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Model conversion failed: {e}")
            raise
    
    def get_finetune_recipe(
        self,
        checkpoint_dir: str,
        name: str,
        peft_scheme: Optional[str] = None,
        num_nodes: int = 1,
        num_gpus_per_node: int = 1,
    ):
        """
        Get fine-tuning recipe based on model size.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            name: Recipe name
            peft_scheme: PEFT scheme ("lora" or None for full fine-tuning)
            num_nodes: Number of nodes
            num_gpus_per_node: Number of GPUs per node
            
        Returns:
            NeMo fine-tuning recipe
        """
        recipe_map = {
            "0.5b": llm.qwen25_500m.finetune_recipe,
            "1.5b": llm.qwen25_1_5b.finetune_recipe,
            "7b": llm.qwen25_7b.finetune_recipe,
            "14b": llm.qwen25_14b.finetune_recipe,
            "32b": llm.qwen25_32b.finetune_recipe,
            "72b": llm.qwen25_72b.finetune_recipe,
        }
        
        if self.config.model_size not in recipe_map:
            raise ValueError(f"No recipe available for model size: {self.config.model_size}")
            
        recipe_fn = recipe_map[self.config.model_size]
        
        recipe = recipe_fn(
            dir=checkpoint_dir,
            name=name,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            peft_scheme=peft_scheme,
        )
        
        return recipe
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_size": self.config.model_size,
            "hf_model_name": self.config.get_hf_model_name(),
            "seq_length": self.config.seq_length,
            "micro_batch_size": self.config.micro_batch_size,
            "global_batch_size": self.config.global_batch_size,
            "learning_rate": self.config.learning_rate,
            "max_steps": self.config.max_steps,
            "precision": self.config.precision,
        } 