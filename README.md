# NeMo 2.0 Qwen2.5B Japanese Fine-tuning

A comprehensive toolkit for fine-tuning Qwen2.5 models on Japanese language data using NVIDIA NeMo 2.0. This project provides modular, production-ready implementations for PEFT (LoRA), SFT, and continual pre-training methods.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NeMo 2.0](https://img.shields.io/badge/NeMo-2.0+-green.svg)](https://github.com/NVIDIA/NeMo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **Multiple Training Methods**: PEFT (LoRA), SFT, and Continual Pre-training
- **Modular Design**: Clean, reusable components with clear interfaces
- **Japanese Language Optimized**: Specialized data processing for Japanese text
- **Memory Efficient**: PEFT reduces memory usage by 42% vs full fine-tuning
- **Production Ready**: Comprehensive logging, error handling, and configuration management
- **Easy to Use**: Simple command-line interface and Python API

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Methods](#training-methods)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Docker (optional, for containerized training)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-username/nemo2-qwen2.5b-japanese-finetune.git
cd nemo2-qwen2.5b-japanese-finetune

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Install with GPU Support

```bash
pip install -e ".[gpu]"
```

### Development Installation

```bash
pip install -e ".[dev,docs,notebooks]"
pre-commit install
```

## ⚡ Quick Start

### 1. Data Preparation

Convert Japanese Wikipedia data to training format:

```bash
python scripts/data_processing/convert_japanese_data.py \
    --input-dir /path/to/japanese/wiki \
    --output-dir /data/converted
```

### 2. Model Conversion

Convert HuggingFace model to NeMo format:

```bash
python scripts/model_conversion/hf_to_nemo.py \
    --model-name Qwen/Qwen2.5-0.5B \
    --output-path /models/qwen2.5-0.5b.nemo
```

### 3. Training

#### PEFT Training (Recommended)

```bash
python scripts/training/run_peft_training.py \
    --dataset-root /data/converted \
    --checkpoint-dir /checkpoints/peft \
    --restore-from-path /models/qwen2.5-0.5b.nemo \
    --max-steps 1000
```

#### SFT Training

```bash
python scripts/training/run_sft_training.py \
    --dataset-root /data/converted \
    --checkpoint-dir /checkpoints/sft \
    --restore-from-path /models/qwen2.5-0.5b.nemo \
    --max-steps 1000
```

### 4. Docker Usage

```bash
# Run PEFT training in Docker
docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd):/workspace" -w "/workspace" \
    nvcr.io/nvidia/nemo:25.04 \
    python scripts/training/run_peft_training.py \
        --dataset-root /workspace/data/converted \
        --checkpoint-dir /workspace/checkpoints/peft \
        --restore-from-path /workspace/models/qwen2.5-0.5b.nemo
```

## 🎯 Training Methods

### PEFT (Parameter-Efficient Fine-Tuning)

- **Memory Efficient**: 42% reduction vs full fine-tuning  
- **Fast Training**: 26% faster than SFT
- **Parameter Efficient**: 99.74% parameter reduction
- **Best for**: Resource-constrained environments

```python
from nemo_japanese_ft import QwenModelConfig, PEFTTrainer

config = QwenModelConfig(model_size="0.5b")
trainer = PEFTTrainer(
    model_config=config,
    dataset_root="/data/converted",
    checkpoint_dir="/checkpoints",
    lora_rank=16,
    lora_alpha=32
)
trainer.prepare()
trainer.train()
```

### SFT (Supervised Fine-Tuning)

- **Maximum Performance**: Full parameter optimization
- **High Memory**: ~22.7GB peak for 0.5B model
- **Baseline Quality**: Best possible adaptation
- **Best for**: Maximum performance requirements

### Continual Pre-training

- **Large-scale Data**: Optimized for corpus-level training
- **Knowledge Preservation**: Maintains existing capabilities
- **Cosine Scheduling**: Advanced learning rate management
- **Best for**: Domain adaptation and language learning

## 📁 Project Structure

```
nemo2-qwen2.5b-japanese-finetune/
├── src/nemo_japanese_ft/          # Core package
│   ├── data/                      # Data processing modules
│   ├── models/                    # Model management
│   ├── training/                  # Training implementations
│   └── utils/                     # Utilities and configuration
├── scripts/                       # Executable scripts
│   ├── data_processing/           # Data conversion scripts
│   ├── model_conversion/          # Model format conversion
│   └── training/                  # Training scripts
├── examples/                      # Usage examples
│   ├── basic_usage/              # Quick start examples
│   └── advanced_training/        # Advanced configurations
├── configs/                       # Configuration files
│   ├── models/                    # Model configurations
│   ├── training/                  # Training configurations
│   └── data/                      # Data configurations
├── tests/                         # Test suite
├── docs/                          # Documentation
└── docker/                       # Docker configurations
```

## 💡 Usage Examples

### Basic Python API

```python
from nemo_japanese_ft import (
    QwenModelConfig, 
    PEFTTrainer, 
    JapaneseWikipediaConverter
)

# Convert data
converter = JapaneseWikipediaConverter()
converter.convert(
    input_dir="/raw/data",
    output_dir="/converted/data"
)

# Configure model
config = QwenModelConfig(
    model_size="0.5b",
    max_steps=1000,
    learning_rate=3e-4
)

# Train with PEFT
trainer = PEFTTrainer(
    model_config=config,
    dataset_root="/converted/data",
    checkpoint_dir="/checkpoints"
)
trainer.prepare()
trainer.train()
```

### Command Line Interface

```bash
# Convert data
nemo-jp-convert-data --input-dir /raw --output-dir /converted

# Convert model  
nemo-jp-convert-model --model-name Qwen/Qwen2.5-0.5B --output-path model.nemo

# Train with PEFT
nemo-jp-train-peft --dataset-root /converted --checkpoint-dir /checkpoints

# Train with SFT
nemo-jp-train-sft --dataset-root /converted --checkpoint-dir /checkpoints
```

## ⚙️ Configuration

### Model Configurations

```yaml
# configs/models/qwen25_0.5b.yaml
model_size: "0.5b"
seq_length: 2048
micro_batch_size: 2
global_batch_size: 16
learning_rate: 3e-4
max_steps: 1000
precision: "bf16-mixed"
```

### Training Configurations

```yaml
# configs/training/peft_default.yaml
peft_scheme: "lora"
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.1
target_modules: "all_linear"
```

## 📊 Performance Comparison

| Method | Memory Usage | Training Time | Parameter Efficiency | Performance |
|--------|-------------|---------------|---------------------|-------------|
| **PEFT** | Low (42% ↓) | Fast (26% ↑) | 99.74% reduction | High |
| **SFT** | High (~22.7GB) | Standard | 0% reduction | Maximum |
| **Continual** | High | Extended | 0% reduction | Domain-Adaptive |

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=src tests/
```

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Quick Start Tutorial](docs/quick_start.md)
- [Training Guide](docs/user_guide/training_guide.md)
- [API Reference](docs/developer_guide/api_reference.md)
- [Troubleshooting](docs/user_guide/troubleshooting.md)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for the training framework
- [Qwen](https://huggingface.co/Qwen) for the base models
- [LLM-JP](https://llm-jp.nii.ac.jp/) for Japanese language resources

## 📞 Support

- **Documentation**: [Read the Docs](https://nemo2-qwen2.5b-japanese-finetune.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/your-username/nemo2-qwen2.5b-japanese-finetune/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/nemo2-qwen2.5b-japanese-finetune/discussions)

---

**Note**: This project is optimized for Japanese language fine-tuning. For other languages, consider adapting the data processing pipeline and configuration accordingly. 