# Parameter-Efficient Fine-Tuning (PEFT) Module

## Overview
This module implements LoRA-based Parameter-Efficient Fine-Tuning for Japanese language adaptation.

## Files
- `train.py` - PEFT training script using LoRA (from 02_qwen25_peft.py)

## Usage
```bash
# Run PEFT training
python src/peft/train.py
```

## Key Features
- **LoRA Configuration**: Rank 16, Alpha 32, Dropout 0.1
- **Memory Efficient**: Uses only 0.26% of trainable parameters
- **Fast Training**: 26% faster than standard fine-tuning
- **Superior Stability**: Better convergence characteristics

## Technical Details
- **Method**: Low-Rank Adaptation (LoRA)
- **Target Modules**: All attention and MLP layers
- **Parameter Reduction**: 99.74% compared to full fine-tuning
- **Memory Savings**: 42% reduction in GPU memory usage 