# Contributing

Thanks for your interest in contributing! This project focuses on Japanese language adaptation using Parameter-Efficient Fine-Tuning (PEFT) with NeMo 2.0.

## Quick Start for Contributors

### Prerequisites
- NVIDIA GPU with CUDA support
- Docker
- Git

### Setup
```bash
# 1. Fork and clone
git clone https://github.com/your-username/nemo2-qwen2.5b-japanese-finetune.git
cd nemo2-qwen2.5b-japanese-finetune

# 2. Create feature branch
git checkout -b feature/your-feature

# 3. Test your changes
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/workspace -w /workspace \
    nvcr.io/nvidia/nemo:25.04 \
    python src/algorithms/02_qwen25_peft.py
```

## Types of Contributions

- **Bug fixes**: Fix issues in training or inference
- **New features**: Add PEFT methods, evaluation metrics, datasets
- **Documentation**: Improve setup guides or examples
- **Examples**: Add Japanese language use cases

## Code Style

- Follow PEP 8 for Python code
- Add docstrings for new functions
- Keep functions focused and small
- Test your changes before submitting

## Pull Request Process

1. **Create an issue first** for major changes
2. **Write clear commit messages**:
   ```
   feat: add QLoRA support for memory efficiency
   fix: resolve CUDA out of memory in PEFT training
   docs: update setup guide for Docker
   ```
3. **Test thoroughly** on your GPU setup
4. **Update documentation** if needed

## Issue Reporting

When reporting bugs, include:
- GPU model and VRAM
- Docker image version
- Full error message
- Steps to reproduce

### Example Bug Report
```
**Environment:**
- GPU: RTX 4090 (24GB)
- Docker: nvcr.io/nvidia/nemo:25.04
- CUDA: 12.1

**Error:**
CUDA out of memory during PEFT training at step 50

**Steps to reproduce:**
1. Run `python src/algorithms/02_qwen25_peft.py`
2. Error occurs after 50 training steps
```

## Questions?

- **Check** [TROUBLESHOOTING.md](TROUBLESHOOTING.md) first
- **Create an issue** for questions
- **Be specific** about your setup and use case

Thanks for contributing! 🎉 