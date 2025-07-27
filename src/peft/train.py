#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter-Efficient Fine-Tuning (PEFT) for Japanese Language Adaptation
========================================================================

This module implements Low-Rank Adaptation (LoRA) for efficient fine-tuning of the Qwen2.5-0.5B model
on Japanese language tasks. The approach is based on the mathematical principle that neural network
weight updates during fine-tuning often have low intrinsic rank.

Theoretical Foundation:
----------------------
LoRA decomposes weight updates as: ΔW = B·A where B ∈ R^{d×r}, A ∈ R^{r×k}, and r << min(d,k)

Key Benefits:
- Parameter Efficiency: Reduces trainable parameters by >99%
- Memory Efficiency: Lower GPU memory requirements
- Modularity: Adapter weights can be easily swapped or combined
- Preservation: Maintains original model capabilities better than full fine-tuning

Implementation Details:
- Rank (r): 16 (optimal balance between performance and efficiency)
- Alpha (α): 32 (scaling factor, typically 2×rank)
- Target Modules: All linear layers in attention and MLP blocks
- Dropout: 0.1 for regularization

References:
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023.

Author: Kosuke & M1nG
Date: 2024-07-27
"""

import json
import os
import nemo_run as run
from nemo.collections import llm
from nemo import lightning as nl
from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule

class CustomFineTuningDataModule(FineTuningDataModule):
    """
    Custom data module using pre-prepared JSONL files
    Data format: {"input": "question", "output": "answer"}
    """

    def __init__(self, dataset_root: str, **kwargs):
        self.dataset_root = dataset_root
        super().__init__(dataset_root=dataset_root, **kwargs)

    def prepare_data(self) -> None:
        """
        Data preparation process - since already prepared in JSONL format,
        only check file existence
        """
        train_path = os.path.join(self.dataset_root, "training.jsonl")
        val_path = os.path.join(self.dataset_root, "validation.jsonl")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training file not found: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation file not found: {val_path}")

        print(f"Found training data: {train_path}")
        print(f"Found validation data: {val_path}")

        # Rename files to format expected by NeMo (if necessary)
        training_jsonl = os.path.join(self.dataset_root, "training.jsonl")
        validation_jsonl = os.path.join(self.dataset_root, "validation.jsonl")

        if not os.path.exists(training_jsonl):
            print(f"Creating symlink: {training_jsonl} -> {train_path}")
            os.symlink(os.path.basename(train_path), training_jsonl)

        if not os.path.exists(validation_jsonl):
            print(f"Creating symlink: {validation_jsonl} -> {val_path}")
            os.symlink(os.path.basename(val_path), validation_jsonl)

        super().prepare_data()

def main():
    print("=== Qwen2.5 PEFT Training Script ===")

    # Dataset configuration
    data_config = run.Config(
        CustomFineTuningDataModule,
        dataset_root='/workspace/data/training_data/',
        seq_length=2048,  # Sequence length
        micro_batch_size=2,  # Batch size per GPU
        global_batch_size=16,  # Total batch size
        dataset_kwargs={}
    )

    # Fine-tuning recipe configuration
    recipe = llm.qwen25_500m.finetune_recipe(
        dir="/workspace/models/checkpoints/qwen25_500m_peft",  # Checkpoint save path
        name="qwen25_500m_peft",
        num_nodes=1,
        num_gpus_per_node=1,
        peft_scheme="lora",  # Use LoRA for memory efficiency
    )

    # Pre-trained model restore configuration
    recipe.resume.restore_config = run.Config(
        nl.RestoreConfig,
        path='/workspace/qwen2.5-0.5b.nemo'  # Converted NeMo model
    )

    # Training configuration
    recipe.trainer.max_steps = 1000  # Maximum steps
    recipe.trainer.num_sanity_val_steps = 2  # Validation sanity check steps
    recipe.trainer.val_check_interval = 100  # Validation check interval
    recipe.trainer.log_every_n_steps = 10  # Logging interval
    recipe.trainer.enable_checkpointing = True  # Enable checkpointing

    # PEFT-specific configuration
    recipe.trainer.strategy.ckpt_async_save = False  # Disable async checkpoint save
    recipe.trainer.strategy.context_parallel_size = 1  # Context parallel size
    recipe.trainer.strategy.ddp = "megatron"  # Required for LoRA/PEFT

    # Data module configuration
    recipe.data = data_config

    print("=== Training Configuration ===")
    print(f"Model: Qwen2.5-0.5B")
    print(f"PEFT Scheme: LoRA")
    print(f"Max Steps: {recipe.trainer.max_steps}")
    print(f"Micro Batch Size: {data_config.micro_batch_size}")
    print(f"Global Batch Size: {data_config.global_batch_size}")
    print(f"Sequence Length: {data_config.seq_length}")
    print(f"Checkpoint Dir: {recipe.log.log_dir}")
    print(f"Data Root: {data_config.dataset_root}")

    # Execute training
    print("\n=== Starting Training ===")
    run.run(recipe, executor=run.LocalExecutor())

if __name__ == "__main__":
    main()
