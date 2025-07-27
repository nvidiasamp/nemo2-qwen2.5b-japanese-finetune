# Continual Learning Module

## Overview
This module implements continual learning for Japanese language adaptation using NeMo 2.0 framework.

## Files
- `train.py` - Main training script for continual learning (from nemo_official_fixed_final.py)
- `preprocess.py` - Japanese text preprocessing utilities (from 00convert_ja.py)

## Usage
```bash
# Run continual learning training
python src/continual_learning/train.py

# Preprocess Japanese text
python src/continual_learning/preprocess.py --input input.txt --output processed.txt
```

## Features
- Optimized learning rate scheduling (3e-4)
- Checkpoint management and recovery
- Japanese Wikipedia training data support
- Mixed precision training with bf16 