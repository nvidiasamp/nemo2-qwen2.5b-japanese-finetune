# LLM-JP Data Processing Guide - NeMo 2.0 Version

## Overview

Based on the successful methods validated by users in NeMo 1.0, adapted for NeMo 2.0 data processing workflow. Using Docker container environment to ensure consistency and reproducibility.

### Core Improvements
- **Model Adaptation**: Changed from `Llama-3.1-8B` to `Qwen/Qwen2.5-0.5B` model
- **Containerized Processing**: Based on NeMo 25.04 container environment
- **Smart Merging**: Automatically merge training files to avoid path issues
- **Real-time Monitoring**: Provides progress tracking tools

## File Description

### Core Scripts
- **`process_data_in_container_fixed.sh`** - Main data processing script
  - Executes within Docker container
  - Auto downloads LLM-JP data (if needed)
  - Merges training files and converts to NeMo format
  - Uses Qwen/Qwen2.5-0.5B tokenizer

### Monitoring Tools
- **`monitor_progress.sh`** - Progress monitoring script
  - Checks Docker container status
  - Shows file generation progress
  - Provides completion estimates

### Documentation
- **`README.md`** - This document

## Quick Usage

### Method 1: Manual Docker Execution (Recommended)
```bash
# Use existing NeMo 25.04 container
docker run \
    --gpus all -it --rm \
    --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "/home/cho/workspace/workshop-25-08-02:/workspace" \
    -w "/workspace" \
    nvcr.io/nvidia/nemo:25.04 \
    bash -c "chmod +x /workspace/scripts/data_processing/process_data_in_container_fixed.sh && /workspace/scripts/data_processing/process_data_in_container_fixed.sh"
```

### Method 2: Progress Monitoring
```bash
# One-time check
./scripts/data_processing/monitor_progress.sh

# Real-time monitoring (updates every 30 seconds)
watch -n 30 ./scripts/data_processing/monitor_progress.sh
```

## Expected Output

### File Structure
```
data/llm_jp_wiki/
├── raw/ja_wiki/                    # Raw data
│   ├── train_0.jsonl
│   ├── train_1.jsonl
│   ├── ...
│   ├── train_13.jsonl
│   ├── train_merged.jsonl          # Merged training file
│   └── validation_0.jsonl
└── nemo_binary/                    # NeMo format data
    ├── ja_wiki_train_text_document.bin
    ├── ja_wiki_train_text_document.idx
    ├── ja_wiki_val_text_document.bin
    └── ja_wiki_val_text_document.idx
```

### Processing Time
- **Data Download**: Already completed (30-60 minutes if first run)
- **Data Merging**: 5-10 minutes
- **NeMo Conversion**: 1-2 hours
- **Total**: 2-3 hours

### Technical Configuration
- **Container**: nvcr.io/nvidia/nemo:25.04
- **Tokenizer**: Qwen/Qwen2.5-0.5B (Model-specific tokenizer)
- **Output Format**: mmap (.bin/.idx)
- **Worker Threads**: 4 for training data, 2 for validation data

## Model Configuration Difference

This method adapts the proven LLM-JP data processing workflow for the Qwen2.5-0.5B model, requiring only one key adjustment:

```bash
# Model-specific tokenizer change (not NeMo version difference)
--tokenizer-type="meta-llama/Llama-3.1-8B"  # For Llama models
↓
--tokenizer-type="Qwen/Qwen2.5-0.5B"        # For Qwen2.5 models
```

**Important Note**: This is a model architecture difference, not a NeMo Framework version difference. Both configurations work in NeMo 2.0, depending on which model you're using.

## Troubleshooting

### Common Issues

**1. Container Startup Failed**
```bash
# Check Docker and GPU
docker --version
nvidia-smi
```

**2. Data Processing Interrupted**
```bash
# Check monitoring status
./scripts/data_processing/monitor_progress.sh

# Restart processing (will skip existing files)
# Simply re-run the Docker command
```

**3. Incomplete Output Files**
- Script will automatically validate all 4 required files
- If failed, will display detailed error information
- Re-running will continue from interruption point

### Log Levels
- 🟢 **[INFO]** - Normal progress information
- 🔵 **[STEP]** - Major processing steps
- 🟡 **[WARN]** - Warning information

## Completion Verification

After successful processing completion, you will see:
```
✅ Data processing completed!

📁 Generated files:
ja_wiki_train_text_document.bin
ja_wiki_train_text_document.idx
ja_wiki_val_text_document.bin
ja_wiki_val_text_document.idx

📊 File size statistics:
[Actual sizes will be displayed here]

🚀 Next step: You can start Task 7 - Implement PEFT-LoRA fine-tuning script
```

## Next Step Integration

The generated NeMo binary files can be directly used for:
- **Task 7**: PEFT-LoRA fine-tuning script
- **Task 4**: Japanese continual learning training
- **Task 6**: Continual learning execution

Data paths in configuration files should be set to:
```yaml
data:
  train_path: ./data/llm_jp_wiki/nemo_binary/ja_wiki_train_text_document
  validation_path: ./data/llm_jp_wiki/nemo_binary/ja_wiki_val_text_document
```

---

**Project Status**: This data processing workflow is based on methods validated by users in production environment, ensuring high reliability and stability.

**Reference**: Based on [NVIDIA Developer Blog - NeMo Framework で実践する継続事前学習](https://developer.nvidia.com/ja-jp/blog/how-to-use-continual-pre-training-with-japanese-language-on-nemo-framework/)