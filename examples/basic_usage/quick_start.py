#!/usr/bin/env python3
"""
Quick start example for NeMo Japanese Fine-tuning.

This example demonstrates the basic usage of the package for fine-tuning
a Qwen model on Japanese data.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nemo_japanese_ft.models import QwenModelConfig
from nemo_japanese_ft.training import PEFTTrainer
from nemo_japanese_ft.data import JapaneseWikipediaConverter
from nemo_japanese_ft.utils import setup_logging

logger = setup_logging(__name__)


def main():
    """Quick start example for Japanese fine-tuning."""
    
    logger.info("=== NeMo Japanese Fine-tuning Quick Start ===")
    
    # Step 1: Convert data (if needed)
    logger.info("Step 1: Data conversion")
    converter = JapaneseWikipediaConverter()
    
    # This is just an example - replace with actual paths
    # converter.convert(
    #     input_dir="/path/to/japanese/wiki/data",
    #     output_dir="/path/to/converted/data"
    # )
    logger.info("Data conversion step (modify paths as needed)")
    
    # Step 2: Configure model
    logger.info("Step 2: Model configuration")
    model_config = QwenModelConfig(
        model_size="0.5b",
        seq_length=2048,
        micro_batch_size=2,
        global_batch_size=16,
        learning_rate=3e-4,
        max_steps=1000,
    )
    logger.info(f"Model configuration: {model_config.model_size}")
    
    # Step 3: Setup trainer
    logger.info("Step 3: Trainer setup")
    trainer = PEFTTrainer(
        model_config=model_config,
        dataset_root="/path/to/converted/data",  # Replace with actual path
        checkpoint_dir="/path/to/checkpoints",   # Replace with actual path
        experiment_name="japanese_quick_start",
        restore_from_path="/path/to/qwen2.5-0.5b.nemo",  # Replace with actual path
        peft_scheme="lora",
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    logger.info("Trainer configured with PEFT/LoRA")
    
    # Step 4: Prepare and train (commented out for demo)
    logger.info("Step 4: Training (modify paths and uncomment)")
    # trainer.prepare()
    # trainer.train()
    
    logger.info("Quick start example completed!")
    logger.info("To run actual training:")
    logger.info("1. Update the file paths in this script")
    logger.info("2. Uncomment the training lines")
    logger.info("3. Ensure you have NeMo 2.0 and GPU resources available")


if __name__ == "__main__":
    main() 