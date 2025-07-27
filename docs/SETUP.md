# 环境设置指南

## 🚀 快速开始

### 系统要求
- **GPU**: NVIDIA GPU with 12GB+ VRAM
- **CUDA**: 12.8+
- **Docker**: 20.10+
- **存储**: 20GB+ 可用空间

### 一键启动

```bash
# 1. 克隆项目
git clone https://github.com/nvidiasamp/nemo2-qwen2.5b-japanese-finetune.git
cd nemo2-qwen2.5b-japanese-finetune

# 2. 启动Docker容器并运行
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/workspace -w /workspace \
    nvcr.io/nvidia/nemo:25.04 \
    python src/algorithms/02_qwen25_peft.py
```

## 📋 详细设置

### Docker环境
```bash
# 启动交互式容器
docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/workspace -w /workspace \
    nvcr.io/nvidia/nemo:25.04 /bin/bash

# 验证环境
python -c "import nemo; print('NeMo version:', nemo.__version__)"
nvidia-smi
```

### 本地环境 (可选)
```bash
# Python 3.8+ 虚拟环境
python -m venv nemo_env
source nemo_env/bin/activate  # Linux/Mac
# nemo_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## ⚙️ 配置说明

### PEFT训练配置
```python
# LoRA参数
rank = 16              # 适中的参数效率
alpha = 32             # 缩放因子 (2×rank)
dropout = 0.1          # 正则化
learning_rate = 3e-4   # 优化的学习率
```

### SFT训练配置
```python
# 完整微调参数
learning_rate = 3e-4
batch_size = 4
max_steps = 1000
mixed_precision = "bf16"
```

## 🔧 常见问题

### GPU内存不足
```bash
# 减少批次大小
export MICRO_BATCH_SIZE=2

# 启用梯度检查点
export GRADIENT_CHECKPOINTING=1
```

### Docker权限问题
```bash
# 添加用户到docker组
sudo usermod -aG docker $USER
newgrp docker
```

### CUDA版本不匹配
```bash
# 检查CUDA版本
nvidia-smi

# 使用对应的NeMo镜像
# CUDA 11.8: nvcr.io/nvidia/nemo:24.07
# CUDA 12.1+: nvcr.io/nvidia/nemo:25.04
```

## 📚 使用示例

### PEFT训练
```bash
python src/algorithms/02_qwen25_peft.py
```

### SFT训练
```bash
python src/algorithms/03_qwen25_sft.py
```

### 模型推理
```bash
python src/utils/validate_model.py --model_path experiments/peft_model/
```

---

*如遇问题请查看 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)* 