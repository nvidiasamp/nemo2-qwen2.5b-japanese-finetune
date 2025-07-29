# Migration Guide: From Branch-Specific to Main Branch

This guide helps users migrate from M1nG or Kosuke branch implementations to the integrated main branch.

## 🔄 Quick Migration Overview

| If you were using... | Now use... |
|---------------------|------------|
| Kosuke's `src/00convert_ja.py` | `scripts/data_processing/convert_japanese_data.py` |
| Kosuke's `src/02_qwen25_peft.py` | `scripts/training/run_peft_training.py` |
| Kosuke's `src/03_qwen25_sft.py` | `scripts/training/run_sft_training.py` |
| M1nG's data processing | `scripts/data_processing/process_llm_jp_data.sh` |
| M1nG's training script | `scripts/training/run_continual_pretraining.py` |

## 📋 Detailed Migration Instructions

### From Kosuke Branch to Main

#### 1. Data Conversion Migration

**Old Way (Kosuke Branch):**
```bash
cd src/
python 00convert_ja.py
```

**New Way (Main Branch):**
```bash
python scripts/data_processing/convert_japanese_data.py \
    --input-dir /workspace/data/ja_wiki \
    --output-dir /workspace/data/training_data \
    --max-output-length 1500 \
    --min-output-length 50 \
    --chunk-size 800
```

**Key Differences:**
- Command-line arguments instead of hardcoded paths
- More configuration options
- Better error handling and logging

#### 2. Model Conversion Migration

**Old Way (Kosuke Branch):**
```python
# In 01_convert_hf_to_nemo.py
output_path = "qwen2.5-0.5b.nemo"
llm.import_ckpt(model=model, source='hf://Qwen/Qwen2.5-0.5B', output_path=output_path)
```

**New Way (Main Branch):**
```bash
python scripts/model_conversion/hf_to_nemo.py \
    --model-name Qwen/Qwen2.5-0.5B \
    --output-path /models/qwen2.5-0.5b.nemo \
    --validate
```

#### 3. PEFT Training Migration

**Old Way (Kosuke Branch):**
```python
# In 02_qwen25_peft.py
recipe = llm.qwen25_500m.finetune_recipe(
    dir="/workspace/models/checkpoints/qwen25_500m_peft",
    name="qwen25_500m_peft",
    peft_scheme="lora",
)
```

**New Way (Main Branch):**
```bash
python scripts/training/run_peft_training.py \
    --model-size 0.5b \
    --dataset-root /workspace/data/training_data \
    --checkpoint-dir /workspace/models/checkpoints/peft \
    --experiment-name qwen25_500m_peft \
    --restore-from-path /workspace/qwen2.5-0.5b.nemo \
    --peft-scheme lora \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-dropout 0.1 \
    --max-steps 1000
```

**Python API (Alternative):**
```python
from nemo_japanese_ft import QwenModelConfig, PEFTTrainer

config = QwenModelConfig(
    model_size="0.5b",
    seq_length=2048,
    micro_batch_size=2,
    global_batch_size=16,
    learning_rate=3e-4,
    max_steps=1000
)

trainer = PEFTTrainer(
    model_config=config,
    dataset_root="/workspace/data/training_data",
    checkpoint_dir="/workspace/models/checkpoints/peft",
    experiment_name="qwen25_500m_peft",
    restore_from_path="/workspace/qwen2.5-0.5b.nemo",
    peft_scheme="lora",
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.1
)

trainer.prepare()
trainer.train()
```

### From M1nG Branch to Main

#### 1. Data Processing Migration

**Old Way (M1nG Branch):**
```bash
cd workshop-25-08-02
bash scripts/data_processing/process_data_in_container_fixed.sh
```

**New Way (Main Branch):**
```bash
# Option 1: Use the new standardized script
bash scripts/data_processing/process_llm_jp_data.sh

# Option 2: Use Docker directly with more control
docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd):/workspace" -w "/workspace" \
    nvcr.io/nvidia/nemo:25.04 \
    bash scripts/data_processing/process_llm_jp_data.sh
```

#### 2. Continual Training Migration

**Old Way (M1nG Branch):**
```python
# In nemo_official_fixed_final.py
recipe = llm.qwen25_500m.pretrain_recipe(
    name="qwen25_500m_fixed_lr",
    dir="experiments/qwen25_500m_fixed_lr",
    max_steps=1000,
)
recipe.optim.config.lr = 3e-4  # Fixed learning rate
```

**New Way (Main Branch):**
```bash
python scripts/training/run_continual_pretraining.py \
    --model-size 0.5b \
    --dataset-root /workspace/data/llm_jp_wiki/nemo_binary \
    --checkpoint-dir /workspace/experiments/continual \
    --experiment-name qwen25_500m_continual \
    --restore-from-path /workspace/qwen2.5-0.5b.nemo \
    --learning-rate 3e-4 \
    --learning-rate-schedule cosine \
    --warmup-steps 100 \
    --weight-decay 0.01 \
    --max-steps 5000
```

## 🔧 Configuration Migration

### Environment Variables

**Old Way:**
```bash
# M1nG branch
export TRAIN_DATA_PATH="{train:[1.0,data/llm_jp_wiki/nemo_binary/ja_wiki_train_text_document],...}"
```

**New Way:**
- Handled automatically by the scripts
- Can be overridden with command-line arguments

### Docker Commands

**Old Way (Both Branches):**
```bash
# Hardcoded paths and settings in scripts
docker run --rm --gpus all ... python specific_script.py
```

**New Way:**
```bash
# Flexible command-line interface
docker run --rm --gpus all ... python scripts/training/run_peft_training.py [OPTIONS]
```

## 📊 Path Mapping

| Old Path (Branch) | New Path (Main) | Purpose |
|-------------------|-----------------|---------|
| `src/` (Kosuke) | `scripts/` + `src/nemo_japanese_ft/` | Scripts and modules |
| `workshop-25-08-02/` (M1nG) | Project root | Flattened structure |
| `/workspace/data/training_data/` | Configurable via CLI | Data location |
| `/workspace/models/checkpoints/` | Configurable via CLI | Checkpoint location |

## 🎯 Best Practices for Migration

1. **Start Fresh**: Clean workspace before migration
   ```bash
   rm -rf experiments/ checkpoints/
   ```

2. **Update Paths**: Use absolute paths or environment variables
   ```bash
   export DATA_ROOT=/path/to/your/data
   export MODEL_ROOT=/path/to/your/models
   ```

3. **Test First**: Run with small datasets/steps
   ```bash
   python scripts/training/run_peft_training.py \
       --max-steps 10 \
       --val-check-interval 5
   ```

4. **Use Configuration Files**: For complex setups
   ```yaml
   # configs/training/my_config.yaml
   model_size: "0.5b"
   learning_rate: 3e-4
   max_steps: 1000
   ```

## ❓ Common Issues and Solutions

### Issue 1: Module Import Errors
```python
# Error: ImportError: No module named 'src'
```
**Solution**: Install the package in development mode
```bash
pip install -e .
```

### Issue 2: Path Not Found
```python
# Error: FileNotFoundError: training.jsonl not found
```
**Solution**: Use absolute paths or verify working directory
```bash
pwd  # Check current directory
ls data/training_data/  # Verify files exist
```

### Issue 3: Different Results
**Solution**: Ensure same hyperparameters
- Learning rate: 3e-4 (both branches)
- Batch sizes: micro=2, global=16
- LoRA config: rank=16, alpha=32, dropout=0.1

## 🚀 Advanced Migration

### Custom Workflows
If you have custom modifications:

1. **Identify Core Logic**: Extract your custom code
2. **Create Module**: Add to `src/nemo_japanese_ft/custom/`
3. **Create Script**: Add wrapper in `scripts/custom/`
4. **Test**: Add tests in `tests/custom/`

### Hybrid Approach
You can use both old and new approaches:
```bash
# Use old data processing
git checkout origin/M1nG -- workshop-25-08-02/scripts/data_processing/

# Use new training
python scripts/training/run_continual_pretraining.py
```

## 📞 Support

If you encounter issues during migration:
1. Check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
2. Review original implementations in `workflows/`
3. Open an issue with migration details 