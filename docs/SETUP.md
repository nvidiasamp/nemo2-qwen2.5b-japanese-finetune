# Setup Guide

## 🚀 Quick Start

### System Requirements
- **GPU**: NVIDIA GPU with 12GB+ VRAM
- **CUDA**: 12.8+
- **Docker**: 20.10+
- **Storage**: 20GB+ available space

### One-Click Launch

```bash
# 1. Clone the repository
git clone https://github.com/nvidiasamp/nemo2-qwen2.5b-japanese-finetune.git
cd nemo2-qwen2.5b-japanese-finetune

# 2. Start Docker container and run
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/workspace -w /workspace \
    nvcr.io/nvidia/nemo:25.04 \
    python src/algorithms/02_qwen25_peft.py
```

## 📋 Detailed Setup

### Docker Environment
```bash
# Start interactive container
docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/workspace -w /workspace \
    nvcr.io/nvidia/nemo:25.04 /bin/bash

# Verify environment
python -c "import nemo; print('NeMo version:', nemo.__version__)"
nvidia-smi
```

### Local Environment (Optional)
```bash
# Python 3.8+ virtual environment
python -m venv nemo_env
source nemo_env/bin/activate  # Linux/Mac
# nemo_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Configuration

### PEFT Training Configuration
```python
# LoRA parameters
rank = 16              # Balanced parameter efficiency
alpha = 32             # Scaling factor (2×rank)
dropout = 0.1          # Regularization
learning_rate = 3e-4   # Optimized learning rate
```

### SFT Training Configuration
```python
# Full fine-tuning parameters
learning_rate = 3e-4
batch_size = 4
max_steps = 1000
mixed_precision = "bf16"
```

## 🔧 Common Issues

### GPU Memory Insufficient
```bash
# Reduce batch size
export MICRO_BATCH_SIZE=2

# Enable gradient checkpointing
export GRADIENT_CHECKPOINTING=1
```

### Docker Permission Issues
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### CUDA Version Mismatch
```bash
# Check CUDA version
nvidia-smi

# Use corresponding NeMo image
# CUDA 11.8: nvcr.io/nvidia/nemo:24.07
# CUDA 12.1+: nvcr.io/nvidia/nemo:25.04
```

## 📚 Usage Examples

### PEFT Training
```bash
python src/algorithms/02_qwen25_peft.py
```

### SFT Training
```bash
python src/algorithms/03_qwen25_sft.py
```

### Model Inference
```bash
python src/utils/validate_model.py --model_path experiments/peft_model/
```

---

*For issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)* 