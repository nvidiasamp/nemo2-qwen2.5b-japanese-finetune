# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-07-29

### Added
- Initial public release
- Integrated features from M1nG and Kosuke branches
- Modular Python package structure (`src/nemo_japanese_ft/`)
- PEFT (LoRA) training implementation
- SFT (Supervised Fine-Tuning) implementation
- Continual pre-training implementation
- Command-line interface for all training methods
- Comprehensive documentation and examples
- Unit test framework with pytest
- CI/CD pipeline with GitHub Actions
- Docker support for containerized training
- Branch integration guide
- Migration guide from original branches

### Features
- **Data Processing**
  - Japanese Wikipedia to Q&A conversion
  - LLM-JP corpus processing for continual learning
  - Support for custom JSONL datasets

- **Model Support**
  - Qwen2.5 models (0.5B to 72B)
  - HuggingFace to NeMo format conversion
  - Flexible model configuration

- **Training Methods**
  - PEFT: Memory-efficient LoRA training (42% memory reduction)
  - SFT: Full parameter fine-tuning for maximum performance
  - Continual: Large-scale Japanese language adaptation

### Technical Specifications
- NeMo Framework 2.0 compatibility
- PyTorch Lightning backend
- Mixed precision training (bf16)
- Multi-GPU support
- Checkpoint management

### Performance
- PEFT: 99.74% parameter reduction, 26% faster training
- SFT: Baseline quality with full parameter optimization
- Continual: Optimized for 12GB+ corpus processing

### Contributors
- M1nG team: Continual pre-training implementation
- Kosuke team: PEFT/SFT fine-tuning methods
- Integration team: Modular restructuring and standardization

## [0.9.0] - 2024-01-XX (Pre-release)

### Added
- Original branch implementations
- Basic project structure
- Initial documentation

### Notes
- M1nG branch: Focus on continual learning workflow
- Kosuke branch: Focus on PEFT/SFT implementations
- Separate codebases before integration

---

## Upgrade Guide

### From Branch-Specific Code to v1.0.0

#### From Kosuke Branch
```bash
# Old: python src/02_qwen25_peft.py
# New:
python scripts/training/run_peft_training.py \
    --dataset-root /data/training_data \
    --checkpoint-dir /checkpoints/peft
```

#### From M1nG Branch
```bash
# Old: python workshop-25-08-02/scripts/training/nemo_official_fixed_final.py
# New:
python scripts/training/run_continual_pretraining.py \
    --dataset-root /data/llm_jp_wiki/nemo_binary \
    --checkpoint-dir /checkpoints/continual
```

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed instructions. 