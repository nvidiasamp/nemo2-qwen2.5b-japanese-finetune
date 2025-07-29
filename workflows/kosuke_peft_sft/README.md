# Kosuke PEFT/SFT Workflow

## Overview
This workflow implements Parameter-Efficient Fine-Tuning (PEFT) and Supervised Fine-Tuning (SFT) for Japanese language adaptation using custom question-answer datasets.

## Available Scripts

### 1. `00convert_ja.py` - Japanese Data Preprocessing
Converts Japanese Wikipedia data into question-answer format suitable for fine-tuning.

**Usage:**
```bash
python workflows/kosuke_peft_sft/00convert_ja.py
```

**Features:**
- Converts Wikipedia articles to Japanese Q&A pairs
- Generates multiple question types per article
- Optimized length control for training efficiency
- Automatic text cleaning and filtering

### 2. `01_convert_hf_to_nemo.py` - Model Conversion
Converts HuggingFace Qwen2.5-0.5B model to NeMo format.

**Usage:**
```bash
python workflows/kosuke_peft_sft/01_convert_hf_to_nemo.py
```

**Output:**
- Creates `qwen2.5-0.5b.nemo` model file
- Compatible with NeMo 2.0 training pipeline

### 3. `02_qwen25_peft.py` - PEFT Training
Implements LoRA-based parameter-efficient fine-tuning.

**Usage:**
```bash
docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd):/workspace" -w "/workspace" \
    nvcr.io/nvidia/nemo:25.04 \
    python workflows/kosuke_peft_sft/02_qwen25_peft.py
```

**Key Features:**
- **LoRA Configuration**: Rank 16, Alpha 32, Dropout 0.1
- **Memory Efficient**: 99.74% parameter reduction
- **Fast Training**: 26% faster than standard fine-tuning
- **Target Modules**: All attention and MLP layers

### 4. `03_qwen25_sft.py` - SFT Training
Implements traditional supervised fine-tuning for comparison.

**Usage:**
```bash
docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd):/workspace" -w "/workspace" \
    nvcr.io/nvidia/nemo:25.04 \
    python workflows/kosuke_peft_sft/03_qwen25_sft.py
```

**Key Features:**
- **Full Parameter Training**: Optimizes all 494M model parameters
- **Maximum Performance**: Best possible adaptation results
- **Baseline Comparison**: Reference point for PEFT performance

## Complete Workflow

### Step 1: Data Preparation
```bash
# Convert Japanese Wikipedia to Q&A format
python workflows/kosuke_peft_sft/00convert_ja.py
```

**Expected Output:**
```
data/training_data/
├── training.jsonl      # Q&A pairs for training
└── validation.jsonl    # Q&A pairs for validation
```

### Step 2: Model Conversion (Optional)
```bash
# Convert HF model to NeMo format (if using local model)
python workflows/kosuke_peft_sft/01_convert_hf_to_nemo.py
```

### Step 3: Choose Training Method

**Option A: PEFT Training (Recommended)**
```bash
# Memory-efficient LoRA training
docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd):/workspace" -w "/workspace" \
    nvcr.io/nvidia/nemo:25.04 \
    python workflows/kosuke_peft_sft/02_qwen25_peft.py
```

**Option B: SFT Training (Maximum Performance)**
```bash
# Full model fine-tuning
docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd):/workspace" -w "/workspace" \
    nvcr.io/nvidia/nemo:25.04 \
    python workflows/kosuke_peft_sft/03_qwen25_sft.py
```

## Technical Specifications

### PEFT Configuration
```yaml
LoRA:
  rank: 16
  alpha: 32
  dropout: 0.1
  target_modules: "all_linear"

Training:
  learning_rate: 3e-4
  batch_size: 16
  sequence_length: 2048
  max_steps: 1000
```

### SFT Configuration
```yaml
Training:
  learning_rate: 3e-4
  batch_size: 16
  sequence_length: 2048
  max_steps: 1000
  precision: "bf16-mixed"
```

## Performance Comparison

| Method | Memory Usage | Training Time | Parameter Efficiency | Performance |
|--------|-------------|---------------|---------------------|-------------|
| **PEFT** | Low (42% reduction) | Fast (26% faster) | 99.74% reduction | High |
| **SFT** | High (~22.7GB peak) | Standard | 0% reduction | Maximum |

## Data Format

### Input Format (Wikipedia Articles)
```json
{
  "text": "article content",
  "meta": {
    "id": "article_id",
    "title": "article_title",
    "url": "article_url"
  }
}
```

### Output Format (Q&A Pairs)
```json
{
  "input": "question in Japanese",
  "output": "answer in Japanese"
}
```

## Troubleshooting

### Common Issues
- **GPU Memory**: Use PEFT instead of SFT for memory-constrained systems
- **Data Processing**: Ensure Japanese text encoding is UTF-8
- **Model Loading**: Verify model path and format compatibility

### Resource Requirements
- **PEFT**: 12GB+ GPU memory
- **SFT**: 24GB+ GPU memory
- **Data Processing**: 8GB+ RAM

## References

- **LoRA Paper**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **NeMo Documentation**: [NVIDIA NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/)
- **Qwen Model**: [Qwen2.5 Model Series](https://huggingface.co/Qwen/Qwen2.5-0.5B) 