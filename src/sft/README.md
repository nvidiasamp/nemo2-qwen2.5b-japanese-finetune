# Supervised Fine-Tuning (SFT) Module

## Overview
This module implements standard supervised fine-tuning for Japanese language adaptation.

## Files
- `train.py` - Standard fine-tuning script (from 03_qwen25_sft.py)

## Usage
```bash
# Run SFT training
python src/sft/train.py
```

## Key Features
- **Full Parameter Training**: Optimizes all 494M model parameters
- **Maximum Performance**: Achieves best possible adaptation results
- **Traditional Approach**: Standard supervised fine-tuning method
- **Comparison Baseline**: Reference point for PEFT performance

## Technical Details
- **Method**: Full model fine-tuning
- **Learning Rate**: 3e-4 (optimized)
- **Memory Usage**: ~22.7GB peak GPU memory
- **Training Time**: Slower but comprehensive adaptation
- **Use Case**: When maximum performance is critical 