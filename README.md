# Japanese Language Adaptation with Parameter-Efficient Fine-Tuning (PEFT) using NeMo 2.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![NeMo 2.0](https://img.shields.io/badge/NeMo-2.0-green.svg)](https://docs.nvidia.com/nemo-framework/user-guide/latest/)
[![Docker](https://img.shields.io/badge/Docker-supported-blue.svg)](https://www.docker.com/)

## Overview

This repository implements **Parameter-Efficient Fine-Tuning (PEFT)** and **Supervised Fine-Tuning (SFT)** for Japanese language adaptation using the **NVIDIA NeMo 2.0 framework**. Our implementation demonstrates efficient Japanese language model training with the Qwen2.5-0.5B model.

## ⚡ Key Features

- **Parameter-Efficient Training**: LoRA-based fine-tuning with significant memory savings
- **Dual Approach**: Both PEFT and traditional SFT implementations
- **Japanese Language Focus**: Specialized preprocessing and training for Japanese text
- **Production-Ready**: Optimized configurations with comprehensive troubleshooting
- **Easy Setup**: Docker-based environment with automated training scripts

## 🚀 Quick Start

### Prerequisites
- **GPU**: NVIDIA GPU with 12GB+ VRAM
- **Docker**: Version 20.10+
- **CUDA**: Version 12.8+

### Setup and Training
```bash
# 1. Clone repository
git clone https://github.com/nvidiasamp/nemo2-qwen2.5b-japanese-finetune.git
cd nemo2-qwen2.5b-japanese-finetune

# 2. Run PEFT training (recommended)
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/workspace -w /workspace \
    nvcr.io/nvidia/nemo:25.04 \
    python src/algorithms/02_qwen25_peft.py

# 3. Run SFT training (comparison)
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/workspace -w /workspace \
    nvcr.io/nvidia/nemo:25.04 \
    python src/algorithms/03_qwen25_sft.py
```

📖 **For detailed setup instructions, see [docs/SETUP.md](docs/SETUP.md)**

## 📁 Project Structure

```
nemo2-qwen2.5b-japanese-finetune/
├── README.md                          # Main documentation
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package configuration
├── LICENSE                           # MIT License
│
├── src/                              # Source code
│   ├── algorithms/                   # 🔥 Core algorithms 
│   │   ├── 00convert_ja.py          # Japanese text preprocessing
│   │   ├── 01_convert_hf_to_nemo.py # HuggingFace to NeMo conversion
│   │   ├── 02_qwen25_peft.py        # PEFT fine-tuning implementation  
│   │   └── 03_qwen25_sft.py         # Standard fine-tuning implementation
│   └── utils/                        # Utility functions
│       ├── check_gpu_config.py      # GPU validation
│       ├── validate_model.py        # Model validation
│       └── [other utilities]        # Additional helper functions
│
├── scripts/                          # Execution scripts
│   ├── training/                     # Training pipeline scripts
│   ├── data_processing/              # Data preprocessing scripts
│   ├── setup_environment.py         # Environment setup
│   └── start_container.sh           # Docker container launcher
│
├── configs/                          # Configuration files
│   └── model_configs/               # Model configurations
│       └── qwen25_0.5b.yaml        # Qwen2.5-0.5B config
│
├── docs/                            # Documentation (simplified)
│   ├── SETUP.md                     # Environment setup guide
│   ├── TROUBLESHOOTING.md           # Problem solving guide
│   └── contributing.md              # Contribution guidelines
│
└── experiments/                     # Training outputs (generated)
    └── [experiment_name]/           # Individual experiment results
```

## 🧬 Core Algorithms

### 📝 Japanese Text Processing (`00convert_ja.py`)
- Japanese text normalization and preprocessing
- Character encoding standardization
- Text cleaning and tokenization preparation

### 🔄 Model Conversion (`01_convert_hf_to_nemo.py`)
- HuggingFace to NeMo format conversion
- Model checkpoint transformation
- Configuration adaptation

### ⚡ PEFT Training (`02_qwen25_peft.py`)
- **Parameter-Efficient Fine-Tuning** with LoRA
- Memory-optimized training approach
- Adapter-based model customization

### 🎯 Standard Fine-tuning (`03_qwen25_sft.py`)
- Traditional supervised fine-tuning
- Full parameter optimization
- Comprehensive model adaptation

## ⚙️ Configuration

### Model Setup
- **Base Model**: Qwen2.5-0.5B (500M parameters)
- **Framework**: NVIDIA NeMo 2.0
- **Target Language**: Japanese

### Training Parameters
```python
# PEFT (LoRA) Configuration
rank: 16                   # Adaptation rank
alpha: 32                  # Scaling parameter  
dropout: 0.1               # Regularization
learning_rate: 3e-4        # Optimized learning rate

# Training Settings
warmup_steps: 200          # Extended warmup
scheduler: CosineAnnealing  # Stable convergence
mixed_precision: "bf16"    # Memory efficiency
```

## 🛠️ Usage Examples

### Algorithm Usage
```bash
# Japanese text preprocessing
python src/algorithms/00convert_ja.py --input input.txt --output processed.txt

# Model conversion
python src/algorithms/01_convert_hf_to_nemo.py --model_path qwen2.5-0.5b

# PEFT training (recommended)
python src/algorithms/02_qwen25_peft.py

# Standard fine-tuning (comparison)
python src/algorithms/03_qwen25_sft.py
```

### Validation
```bash
# Check GPU environment
python src/utils/check_gpu_config.py

# Validate trained model
python src/utils/validate_model.py --model_path experiments/peft_model/
```

## 🔧 Troubleshooting

For common issues and solutions, see **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**

### Quick Fixes
- **GPU Memory**: Reduce batch size or enable gradient checkpointing
- **Docker Issues**: Ensure Docker has access to GPUs with `--gpus all`
- **CUDA Version**: Use NeMo image matching your CUDA version
- **Permissions**: Add user to docker group: `sudo usermod -aG docker $USER`

## 🤝 Contributors

This project is a collaborative effort between:
- **M1nG**: Framework setup, documentation, and training optimization
- **Kosuke**: Core algorithms, preprocessing, and fine-tuning implementations

📖 **Want to contribute?** See [docs/contributing.md](docs/contributing.md)

## 📚 Citation

If you use this work, please cite:

```bibtex
@misc{ming_kosuke_2024_japanese_peft,
  title={Japanese Language Adaptation with Parameter-Efficient Fine-Tuning using NeMo 2.0},
  author={M1nG and Kosuke},
  year={2024},
  url={https://github.com/nvidiasamp/nemo2-qwen2.5b-japanese-finetune},
  note={PEFT and SFT implementations for Japanese language adaptation with NeMo 2.0}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NVIDIA NeMo Team** for the excellent NeMo 2.0 framework
- **Alibaba Qwen Team** for the Qwen2.5 base model
- **Japanese NLP Community** for preprocessing insights and datasets

---

📖 **Documentation**: [SETUP.md](docs/SETUP.md) | [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | [Contributing](docs/contributing.md) 