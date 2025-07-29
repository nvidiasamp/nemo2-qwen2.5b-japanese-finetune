# Japanese Language Adaptation with NeMo 2.0 - Dual Workflow Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![NeMo 2.0](https://img.shields.io/badge/NeMo-2.0-green.svg)](https://docs.nvidia.com/nemo-framework/user-guide/latest/)
[![Docker](https://img.shields.io/badge/Docker-supported-blue.svg)](https://www.docker.com/)

## 🎯 Project Overview

This repository implements **two distinct workflows** for Japanese language adaptation using **NVIDIA NeMo 2.0**:

### 🔄 Workflow 1: Continual Pre-training (M1nG Branch)
- **Purpose**: Adapt existing LLMs to Japanese using continual pre-training
- **Data Source**: LLM-JP Japanese Wikipedia corpus
- **Approach**: Continue pre-training on large-scale Japanese text
- **Output**: Foundation model adapted for Japanese

### ⚡ Workflow 2: Parameter-Efficient Fine-tuning (Kosuke Branch)  
- **Purpose**: Efficient Japanese adaptation using PEFT/SFT methods
- **Data Source**: Custom Japanese question-answer datasets
- **Approach**: LoRA-based fine-tuning and supervised fine-tuning
- **Output**: Task-specific Japanese models

## 🚀 Quick Start Guide

### Prerequisites
- **GPU**: NVIDIA GPU with 12GB+ VRAM
- **Docker**: Version 20.10+
- **CUDA**: Version 12.8+

### 📥 Repository Setup
```bash
# Clone repository
git clone https://github.com/nvidiasamp/nemo2-qwen2.5b-japanese-finetune.git
cd nemo2-qwen2.5b-japanese-finetune
```

## 🔄 Workflow 1: Continual Pre-training

### 📊 Data Preparation (LLM-JP Corpus)
```bash
# Step 1: Process LLM-JP Japanese Wikipedia data
docker run --gpus all -it --rm \
    --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd):/workspace" \
    -w "/workspace" \
    nvcr.io/nvidia/nemo:25.04 \
    bash -c "chmod +x scripts/data_processing/process_data_in_container_fixed.sh && scripts/data_processing/process_data_in_container_fixed.sh"

# Step 2: Monitor processing progress
./scripts/data_processing/monitor_progress.sh
```

**Expected Output**:
```
data/llm_jp_wiki/
├── raw/ja_wiki/                 # Raw JSONL files
│   ├── train_0.jsonl to train_13.jsonl
│   ├── train_merged.jsonl       # Merged training data
│   └── validation_0.jsonl
└── nemo_binary/                 # NeMo format
    ├── ja_wiki_train_text_document.bin
    ├── ja_wiki_train_text_document.idx
    ├── ja_wiki_val_text_document.bin
    └── ja_wiki_val_text_document.idx
```

### 🎓 Training Execution
```bash
# Run continual pre-training
docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd):/workspace" -w "/workspace" \
    nvcr.io/nvidia/nemo:25.04 \
    python src/continual_learning/train.py
```

## ⚡ Workflow 2: Parameter-Efficient Fine-tuning

### 📊 Data Preparation (Custom Japanese QA)
```bash
# Step 1: Convert Japanese Wikipedia to QA format
docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd):/workspace" -w "/workspace" \
    nvcr.io/nvidia/nemo:25.04 \
    python src/continual_learning/preprocess.py
```

**Expected Output**:
```
data/training_data/
├── training.jsonl               # QA pairs for training
└── validation.jsonl             # QA pairs for validation
```

### 🎓 Training Execution
```bash
# Option A: PEFT Training (Memory Efficient)
docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd):/workspace" -w "/workspace" \
    nvcr.io/nvidia/nemo:25.04 \
    python src/peft/train.py

# Option B: SFT Training (Maximum Performance)
docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd):/workspace" -w "/workspace" \
    nvcr.io/nvidia/nemo:25.04 \
    python src/sft/train.py
```

## 📁 Project Structure

```
nemo2-qwen2.5b-japanese-finetune/
├── README.md                    # This documentation
├── requirements.txt             # Python dependencies
├── setup.py                    # Package configuration
├── LICENSE                     # MIT License
│
├── src/                        # 🎯 Dual Workflow Source Code
│   ├── continual_learning/     # 🔄 Workflow 1: Continual Pre-training
│   │   ├── train.py            # Main continual learning script
│   │   ├── preprocess.py       # Japanese text preprocessing
│   │   └── README.md           # Module documentation
│   ├── peft/                   # ⚡ Workflow 2A: PEFT Training
│   │   ├── train.py            # LoRA-based fine-tuning
│   │   └── README.md           # Module documentation
│   ├── sft/                    # 🎯 Workflow 2B: SFT Training
│   │   ├── train.py            # Supervised fine-tuning
│   │   └── README.md           # Module documentation
│   └── utils/                  # 🛠️ Utility Functions
│       ├── convert_model.py    # Model format conversion
│       ├── check_gpu_config.py # GPU validation
│       └── validate_model.py   # Model validation
│
├── scripts/                    # 🔧 Execution Scripts
│   ├── data_processing/        # 📊 Data Processing Pipeline
│   │   ├── process_data_in_container_fixed.sh  # LLM-JP processing
│   │   ├── monitor_progress.sh # Progress monitoring
│   │   └── README.md           # Data processing guide
│   ├── training/               # 🎓 Training Pipeline
│   └── setup_environment.py   # Environment setup
│
├── configs/                    # ⚙️ Configuration Files
│   └── model_configs/          # Model configurations
│       └── qwen25_0.5b.yaml   # Qwen2.5-0.5B config
│
├── docs/                       # 📚 Documentation
│   ├── SETUP.md               # Environment setup guide
│   ├── TROUBLESHOOTING.md     # Problem solving guide
│   └── contributing.md        # Contribution guidelines
│
└── experiments/               # 📈 Training Outputs (Generated)
    ├── continual_learning/    # Workflow 1 results
    ├── peft/                  # Workflow 2A results
    └── sft/                   # Workflow 2B results
```

## 🔍 Workflow Comparison

| Aspect | Continual Pre-training | PEFT/SFT Fine-tuning |
|--------|----------------------|---------------------|
| **Data Size** | Large (LLM-JP corpus) | Medium (Custom QA) |
| **Training Time** | 2-3 hours | 1-2 hours |
| **Memory Usage** | High | Low (PEFT) / High (SFT) |
| **Output Quality** | Foundation adaptation | Task-specific |
| **Use Case** | General Japanese LLM | Specific applications |

## 🛠️ Branch-Specific Instructions

### 🔄 For Continual Pre-training (Based on M1nG Branch)
```bash
# Switch to M1nG branch for complete implementation
git checkout M1nG
cd workshop-25-08-02

# Follow the complete data processing pipeline
./scripts/data_processing/process_data_in_container_fixed.sh
```

### ⚡ For PEFT/SFT Fine-tuning (Based on Kosuke Branch)
```bash
# Switch to Kosuke branch for specialized implementations
git checkout Kosuke

# Use the streamlined scripts
python src/00convert_ja.py      # Data preprocessing
python src/01_convert_hf_to_nemo.py  # Model conversion
python src/02_qwen25_peft.py    # PEFT training
python src/03_qwen25_sft.py     # SFT training
```

## ⚙️ Configuration Details

### Model Setup
- **Base Model**: Qwen2.5-0.5B (500M parameters)
- **Framework**: NVIDIA NeMo 2.0
- **Tokenizer**: Qwen/Qwen2.5-0.5B (model-specific)

### Training Parameters
```yaml
# Continual Pre-training
learning_rate: 3e-4
warmup_steps: 200
scheduler: CosineAnnealing
precision: "bf16-mixed"

# PEFT Configuration
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.1
target_modules: "all_linear"
```

## 🔧 Troubleshooting

### Data Processing Issues
- **LLM-JP Download**: Ensure stable internet connection
- **Memory Errors**: Increase Docker memory allocation
- **Permission Errors**: Check file permissions in mounted volumes

### Training Issues
- **GPU Memory**: Reduce batch size or enable gradient checkpointing
- **Docker Issues**: Ensure `--gpus all` flag is used
- **CUDA Version**: Match NeMo container version with your CUDA

For detailed troubleshooting, see **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**

## 🤝 Contributors

This project represents a collaborative dual-approach implementation:
- **M1nG**: Continual pre-training workflow and infrastructure
- **Kosuke**: PEFT/SFT fine-tuning methods and optimization

## 📚 References

- **[NVIDIA Developer Blog - NeMo Framework Japanese Continual Pre-training](https://developer.nvidia.com/ja-jp/blog/how-to-use-continual-pre-training-with-japanese-language-on-nemo-framework/)**
- **[LLM-JP Corpus Documentation](https://huggingface.co/datasets/llm-jp/llm-jp-corpus)**
- **[NeMo 2.0 Framework Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/)**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

📖 **Choose Your Workflow**: [Continual Pre-training (M1nG)](../../tree/M1nG) | [PEFT/SFT Fine-tuning (Kosuke)](../../tree/Kosuke) | [Data Processing Guide](scripts/data_processing/README.md) 