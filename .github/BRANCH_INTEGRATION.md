# Branch Integration Guide

This document explains the integration of features from the M1nG and Kosuke branches into the main branch.

## 📊 Branch Overview

### M1nG Branch - Continual Pre-training
- **Purpose**: Large-scale Japanese language adaptation through continual pre-training
- **Data**: LLM-JP Wikipedia corpus (12GB+)
- **Key Script**: `workshop-25-08-02/scripts/training/nemo_official_fixed_final.py`

### Kosuke Branch - PEFT/SFT Fine-tuning
- **Purpose**: Efficient task-specific fine-tuning using PEFT (LoRA) and SFT
- **Data**: Custom Q&A datasets from Japanese Wikipedia
- **Key Scripts**: `src/00convert_ja.py`, `src/02_qwen25_peft.py`, `src/03_qwen25_sft.py`

### Main Branch - Integrated Solution
- **Purpose**: Production-ready, modular implementation combining both approaches
- **Structure**: Standard Python package with clear separation of concerns

## 🔄 Integration Architecture

```
Main Branch Structure:
├── src/nemo_japanese_ft/           # Core package (integrated features)
│   ├── data/                       # Data processing (from Kosuke)
│   ├── models/                     # Model management (unified)
│   └── training/                   # Training implementations
│       ├── peft.py                 # From Kosuke
│       ├── sft.py                  # From Kosuke
│       └── continual.py            # From M1nG
├── scripts/                        # Executable scripts
│   ├── data_processing/
│   │   ├── convert_japanese_data.py    # From Kosuke
│   │   └── process_llm_jp_data.sh      # From M1nG
│   └── training/
│       ├── run_peft_training.py         # From Kosuke
│       ├── run_sft_training.py          # From Kosuke
│       └── run_continual_pretraining.py # From M1nG
└── workflows/                      # Original implementations (preserved)
    ├── kosuke_peft_sft/           # Kosuke's original code
    └── ming_continual_learning/   # Reference to M1nG branch
```

## 🚀 Usage Guide

### Option 1: Use Integrated Main Branch (Recommended)

#### For PEFT/SFT Training (Kosuke's Features)
```bash
# Data preparation
python scripts/data_processing/convert_japanese_data.py \
    --input-dir /path/to/japanese/wiki \
    --output-dir /data/converted

# PEFT training
python scripts/training/run_peft_training.py \
    --dataset-root /data/converted \
    --checkpoint-dir /checkpoints/peft \
    --restore-from-path /models/qwen2.5-0.5b.nemo

# SFT training
python scripts/training/run_sft_training.py \
    --dataset-root /data/converted \
    --checkpoint-dir /checkpoints/sft \
    --restore-from-path /models/qwen2.5-0.5b.nemo
```

#### For Continual Pre-training (M1nG's Features)
```bash
# Process LLM-JP data
bash scripts/data_processing/process_llm_jp_data.sh

# Run continual pre-training
python scripts/training/run_continual_pretraining.py \
    --dataset-root /data/llm_jp_wiki/nemo_binary \
    --checkpoint-dir /checkpoints/continual \
    --restore-from-path /models/qwen2.5-0.5b.nemo
```

### Option 2: Use Original Branch Implementations

#### For M1nG Branch
```bash
# Switch to M1nG branch
git checkout origin/M1nG
cd workshop-25-08-02

# Follow M1nG workflow
bash scripts/data_processing/process_data_in_container_fixed.sh
python scripts/training/nemo_official_fixed_final.py
```

#### For Kosuke Branch
```bash
# Switch to Kosuke branch
git checkout origin/Kosuke

# Follow Kosuke workflow
python src/00convert_ja.py
python src/02_qwen25_peft.py  # or src/03_qwen25_sft.py
```

## 📋 Feature Mapping

| Feature | Original Branch | Main Branch Location |
|---------|----------------|---------------------|
| Japanese Q&A Conversion | Kosuke: `src/00convert_ja.py` | `src/nemo_japanese_ft/data/converters.py` |
| HF→NeMo Conversion | Kosuke: `src/01_convert_hf_to_nemo.py` | `src/nemo_japanese_ft/models/utils.py` |
| PEFT Training | Kosuke: `src/02_qwen25_peft.py` | `src/nemo_japanese_ft/training/peft.py` |
| SFT Training | Kosuke: `src/03_qwen25_sft.py` | `src/nemo_japanese_ft/training/sft.py` |
| LLM-JP Processing | M1nG: `process_data_in_container_fixed.sh` | `scripts/data_processing/process_llm_jp_data.sh` |
| Continual Learning | M1nG: `nemo_official_fixed_final.py` | `src/nemo_japanese_ft/training/continual.py` |

## 🔧 Technical Improvements

### Main Branch Advantages
1. **Modular Design**: Reusable components vs monolithic scripts
2. **Error Handling**: Comprehensive error handling and logging
3. **Configuration**: Flexible configuration management
4. **Testing**: Unit and integration tests
5. **Documentation**: Complete API documentation
6. **CLI**: User-friendly command-line interface

### Preserved Functionality
- All original functionality from both branches is preserved
- Original scripts remain in `workflows/` for reference
- Docker-based training still supported
- Same training parameters and configurations

## 📊 Performance Comparison

| Method | Memory Usage | Training Time | Use Case |
|--------|-------------|---------------|----------|
| **PEFT (Kosuke)** | Low (42% ↓) | Fast | Task-specific adaptation |
| **SFT (Kosuke)** | High (~22GB) | Standard | Maximum performance |
| **Continual (M1nG)** | High | Extended | Language adaptation |

## 🎯 Choosing the Right Approach

1. **PEFT**: Best for resource-constrained environments or quick experiments
2. **SFT**: Best for maximum task performance with sufficient resources
3. **Continual**: Best for adapting to Japanese language fundamentally

## 📝 Migration Notes

If migrating from branch-specific code:

### From Kosuke Branch
```python
# Old way
python src/02_qwen25_peft.py

# New way (main branch)
python scripts/training/run_peft_training.py \
    --dataset-root /data/training_data \
    --checkpoint-dir /models/checkpoints/peft
```

### From M1nG Branch
```bash
# Old way (in workshop-25-08-02/)
python scripts/training/nemo_official_fixed_final.py

# New way (main branch)
python scripts/training/run_continual_pretraining.py \
    --dataset-root /data/llm_jp_wiki/nemo_binary \
    --checkpoint-dir /models/checkpoints/continual
```

## 🤝 Contributing

When contributing new features:
1. Add modular components to `src/nemo_japanese_ft/`
2. Create user-friendly scripts in `scripts/`
3. Preserve original implementations in `workflows/` if significant
4. Update this integration guide
5. Add appropriate tests and documentation 