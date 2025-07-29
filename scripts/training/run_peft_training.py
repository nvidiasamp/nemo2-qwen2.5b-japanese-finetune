#!/usr/bin/env python3
"""
Run PEFT (Parameter-Efficient Fine-Tuning) training with NeMo.

This script performs LoRA-based fine-tuning of Qwen models on Japanese data.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nemo_japanese_ft.models import QwenModelConfig
from nemo_japanese_ft.training import PEFTTrainer
from nemo_japanese_ft.utils import setup_logging

logger = setup_logging(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run PEFT training with NeMo"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-size",
        type=str,
        default="0.5b",
        choices=["0.5b", "1.5b", "7b", "14b", "32b", "72b"],
        help="Qwen model size to use"
    )
    
    parser.add_argument(
        "--restore-from-path",
        type=str,
        help="Path to .nemo file to restore from"
    )
    
    # Data configuration
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root directory containing training.jsonl and validation.jsonl"
    )
    
    # Training configuration
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory to save checkpoints"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="qwen_peft_japanese",
        help="Name of the experiment"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of training steps"
    )
    
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=2,
        help="Micro batch size per GPU"
    )
    
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=16,
        help="Global batch size across all GPUs"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Sequence length"
    )
    
    parser.add_argument(
        "--val-check-interval",
        type=int,
        default=100,
        help="Validation check interval"
    )
    
    # PEFT configuration
    parser.add_argument(
        "--peft-scheme",
        type=str,
        default="lora",
        help="PEFT scheme to use"
    )
    
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout"
    )
    
    args = parser.parse_args()
    
    logger.info("=== PEFT Training Setup ===")
    
    # Create model configuration
    model_config = QwenModelConfig(
        model_size=args.model_size,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
    )
    
    # Create PEFT trainer
    trainer = PEFTTrainer(
        model_config=model_config,
        dataset_root=args.dataset_root,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name,
        restore_from_path=args.restore_from_path,
        peft_scheme=args.peft_scheme,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    try:
        # Prepare training
        trainer.prepare()
        
        # Start training
        trainer.train()
        
        logger.info("PEFT training completed successfully!")
        
    except Exception as e:
        logger.error(f"PEFT training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 