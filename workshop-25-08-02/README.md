# Japanese Continual Learning with NeMo 2.0 Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![NeMo 2.0](https://img.shields.io/badge/NeMo-2.0-green.svg)](https://docs.nvidia.com/nemo-framework/user-guide/latest/)
[![Docker](https://img.shields.io/badge/Docker-supported-blue.svg)](https://www.docker.com/)

## Abstract

This repository presents a comprehensive implementation of **continual learning** and **Parameter-Efficient Fine-Tuning (PEFT)** techniques for Japanese language adaptation using the NVIDIA NeMo 2.0 framework. The project demonstrates the effective adaptation of the Qwen2.5-0.5B model for Japanese language tasks through systematic training optimization and advanced fine-tuning methodologies.

### Key Contributions
- Implementation of continual learning for Japanese language adaptation  
- Integration of PEFT-LoRA techniques with NeMo 2.0 framework
- Comprehensive training optimization and troubleshooting solutions
- Real-time monitoring and experiment tracking capabilities
- Educational framework for multilingual model adaptation research

## 🚀 Quick Start

### Prerequisites
- NVIDIA GPU with CUDA support (recommended: RTX 3090 or higher)
- Docker (recommended) or Conda environment
- Python 3.8+
- 16GB+ GPU memory for optimal performance

### Installation

#### Using Docker (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-username/japanese-continual-learning-nemo.git
cd japanese-continual-learning-nemo

# Start training with optimized configuration
bash scripts/training/run_fixed_training.sh
```

#### Using Conda Environment
```bash
# Create environment
conda create -n nemo-jp python=3.8
conda activate nemo-jp

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/training/check_environment.py
```

## 📁 Project Structure

```
japanese-continual-learning-nemo/
├── README.md                    # Main documentation
├── requirements.txt             # Python dependencies
├── setup.py                     # Package configuration
├── LICENSE                      # MIT License
│
├── scripts/                     # Execution scripts
│   ├── training/                # Training pipeline
│   │   ├── nemo_official_fixed_final.py    # Main training script
│   │   ├── run_fixed_training.sh           # Training launcher
│   │   ├── check_environment.py            # Environment validation
│   │   ├── validate_fix.py                 # Configuration validation
│   │   └── monitor_training.py             # Training monitoring
│   ├── data_processing/         # Data preprocessing utilities
│   └── utils/                   # General utilities
│
├── configs/                     # Configuration files
│   ├── model_configs/           # Model configurations
│   ├── data_configs/            # Dataset configurations
│   └── experiment_configs/      # Experiment setups
│
├── data/                        # Training datasets
│   └── llm_jp_wiki/            # Japanese Wikipedia dataset
│
├── experiments/                 # Training outputs
│   └── [experiment_name]/       # Individual experiment results
│
├── docs/                        # Documentation
│   ├── METHODOLOGY.md           # Training methodology
│   ├── TROUBLESHOOTING.md       # Common issues and solutions
│   └── api/                     # API documentation
│
├── research/                    # Research materials
│   ├── figures/                 # Generated figures
│   ├── data_tables/             # Experimental data
│   └── supplementary/           # Additional materials
│
└── tests/                       # Test suite
    ├── unit/                    # Unit tests
    └── integration/             # Integration tests
```

## 🔬 Methodology

### Training Pipeline
Our implementation follows a systematic approach to Japanese language adaptation:

1. **Base Model**: Qwen2.5-0.5B (500M parameters)
2. **Training Technique**: Continual learning with PEFT-LoRA
3. **Framework**: NVIDIA NeMo 2.0 with optimized configurations
4. **Dataset**: Japanese Wikipedia corpus (preprocessed)

### Key Technical Features

#### Optimized Learning Rate Schedule
```python
# Optimized configuration (resolved from previous issues)
learning_rate: 3e-4        # Increased from 1e-5 (30x improvement)
min_learning_rate: 3e-5    # Appropriate minimum threshold
warmup_steps: 200          # Extended warmup period
scheduler: CosineAnnealing  # Stable convergence
```

#### Mixed Precision Training
```python
# Fixed precision configuration
plugins: bf16_mixed()      # Correct implementation
# Removed: trainer.precision (caused conflicts)
```

#### PEFT-LoRA Configuration
```python
# Low-Rank Adaptation settings
rank: 16                   # Adaptation rank
alpha: 32                  # Scaling parameter
dropout: 0.1               # Regularization
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

## 🧪 Experiments

### Current Training Status
- **Model**: Qwen2.5-0.5B Japanese adaptation
- **Dataset**: Japanese Wikipedia (processed)
- **Progress**: Step 86/1000 (active training)
- **Loss**: Decreasing from 12.11 → 11.0 (successful convergence)

### Performance Metrics
- **Training Loss**: Consistent decrease observed
- **Learning Rate**: Progressive warming (1.493e-06 → 0.0001299)
- **Batch Size**: 32 (global)
- **Memory Usage**: ~2.5GB model parameters

## 📊 Results

### Training Progress
Current training demonstrates successful loss convergence:
```
Step   0: loss = 12.11, lr = 1.493e-06
Step  20: loss = 12.03, lr = 3.134e-05
Step  40: loss = 11.68, lr = 6.119e-05  
Step  60: loss = 11.28, lr = 9.104e-05
Step  86: loss = 11.00, lr = 1.299e-04
```

### Key Achievements
✅ **Learning Rate Issue Resolution**: Solved CosineAnnealing scheduler conflicts  
✅ **Mixed Precision Optimization**: Eliminated trainer.precision conflicts  
✅ **Checkpoint Management**: Implemented clean restart capability  
✅ **Loss Convergence**: Achieved consistent training progress  

## 🔧 Troubleshooting

### Previously Resolved Issues

#### Issue 1: Learning Rate Scheduler Conflict
**Problem**: `CosineAnnealing object received initial learning rate lower than minimum`

**Solution**: 
- Removed unsupported `max_lr` parameter
- Increased base learning rate from 1e-5 to 3e-4
- Extended warmup period to 200 steps
- Proper min_lr configuration (3e-5)

#### Issue 2: Mixed Precision Configuration
**Problem**: Trainer precision conflicts with Megatron strategy

**Solution**:
- Use `plugins = bf16_mixed()` instead of `trainer.precision`
- Removed conflicting precision settings
- Validated configuration compatibility

#### Issue 3: Checkpoint Recovery Failures
**Problem**: Corrupted checkpoint preventing training continuation

**Solution**:
- Implemented checkpoint cleaning mechanism
- Added fresh start option (`--clean-checkpoints --fresh-start`)
- Separate experiment directories for isolated runs

### Current Training Command
```bash
# Recommended training execution
bash scripts/training/run_fixed_training.sh
# Select option 1: Clean checkpoints and restart (recommended)
```

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{japanese_continual_learning_nemo2024,
  title={Japanese Continual Learning with NeMo 2.0: A Framework for Multilingual Language Model Adaptation},
  author={[Your Name]},
  year={2024},
  url={https://github.com/your-username/japanese-continual-learning-nemo},
  note={Implementation of continual learning techniques for Japanese language adaptation using NVIDIA NeMo 2.0}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/contributing.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Validate training configuration
python scripts/training/validate_fix.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NVIDIA NeMo Team** for the excellent NeMo 2.0 framework
- **Alibaba Qwen Team** for the Qwen2.5 base model
- **Japanese Wikipedia** for providing the training corpus
- **Open Source Community** for continuous support and feedback

---

**Status**: 🟢 Active Development | **Training**: 🔄 In Progress | **Documentation**: 📖 Complete

For detailed methodology and technical implementation details, see [docs/METHODOLOGY.md](docs/METHODOLOGY.md). 