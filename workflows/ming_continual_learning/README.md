# M1nG Continual Learning Workflow

## Overview
This workflow implements Japanese continual pre-training using large-scale LLM-JP corpus data.

## Access Instructions

The complete M1nG continual learning implementation is located in the **M1nG branch**:

```bash
# Switch to M1nG branch
git checkout M1nG

# Navigate to the workshop directory
cd workshop-25-08-02

# Key components:
# - scripts/data_processing/     # LLM-JP data processing pipeline
# - scripts/training/            # Continual learning training scripts
# - src/                        # Core utilities and modules
# - docs/                       # Comprehensive documentation
```

## Data Processing Pipeline

### LLM-JP Wikipedia Corpus Processing
```bash
# Process LLM-JP data in Docker container
docker run --gpus all -it --rm \
    --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd):/workspace" \
    -w "/workspace" \
    nvcr.io/nvidia/nemo:25.04 \
    bash -c "chmod +x scripts/data_processing/process_data_in_container_fixed.sh && scripts/data_processing/process_data_in_container_fixed.sh"

# Monitor progress
./scripts/data_processing/monitor_progress.sh
```

### Expected Output Structure
```
data/llm_jp_wiki/
├── raw/ja_wiki/                 # Raw JSONL files
│   ├── train_0.jsonl to train_13.jsonl
│   ├── train_merged.jsonl       # Merged training data
│   └── validation_0.jsonl
└── nemo_binary/                 # NeMo format data
    ├── ja_wiki_train_text_document.bin
    ├── ja_wiki_train_text_document.idx
    ├── ja_wiki_val_text_document.bin
    └── ja_wiki_val_text_document.idx
```

## Continual Learning Training

```bash
# Run continual pre-training
docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd):/workspace" -w "/workspace" \
    nvcr.io/nvidia/nemo:25.04 \
    python scripts/training/nemo_official_fixed_final.py
```

## Key Features

- **Large-scale data processing**: Handles 12GB+ of LLM-JP Wikipedia corpus
- **Optimized training parameters**: Learning rate 3e-4, CosineAnnealing scheduler
- **Checkpoint management**: Automatic recovery and error handling
- **Production-ready**: Comprehensive logging and monitoring

## Documentation

Complete documentation is available in the M1nG branch:
- `docs/SETUP.md` - Environment setup
- `docs/TROUBLESHOOTING.md` - Problem solving
- `scripts/data_processing/README.md` - Data processing guide
- `scripts/training/README.md` - Training guide

## Reference

Based on [NVIDIA Developer Blog - NeMo Framework Japanese Continual Pre-training](https://developer.nvidia.com/ja-jp/blog/how-to-use-continual-pre-training-with-japanese-language-on-nemo-framework/) 