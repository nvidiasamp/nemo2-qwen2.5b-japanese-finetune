# Experimental Design

## Abstract

This document outlines the comprehensive experimental methodology for Japanese language adaptation using Parameter-Efficient Fine-Tuning (PEFT) and Supervised Fine-Tuning (SFT) approaches with the Qwen2.5-0.5B model in the NeMo 2.0 framework.

## 1. Research Questions

### Primary Research Questions
1. **RQ1**: How does PEFT (LoRA) compare to SFT in terms of convergence speed and final performance for Japanese language adaptation?
2. **RQ2**: What is the optimal balance between parameter efficiency and performance for Japanese continual learning?
3. **RQ3**: How do different learning rate schedules affect the stability of Japanese language adaptation?

### Secondary Research Questions
1. **RQ4**: What is the memory efficiency gain of PEFT vs. SFT for the Qwen2.5-0.5B model?
2. **RQ5**: How does the choice of rank parameter in LoRA affect Japanese language performance?
3. **RQ6**: What preprocessing strategies are most effective for Japanese text in the NeMo framework?

## 2. Experimental Framework

### 2.1 Base Model Configuration
```yaml
Base Model: Qwen2.5-0.5B
Framework: NVIDIA NeMo 2.0
Model Size: ~500M parameters
Architecture: Transformer Decoder
Context Length: 2048 tokens
Vocabulary Size: 151,936 tokens
```

### 2.2 Hardware Setup
```
GPU: NVIDIA RTX 4090 (24GB VRAM)
CPU: Multi-core (8+ threads recommended)
Memory: 32GB+ RAM
Storage: SSD for fast data loading
Docker: NVIDIA Container Runtime
```

## 3. Experimental Conditions

### 3.1 PEFT (LoRA) Configuration
```python
# LoRA Hyperparameters
rank: 16                    # Low-rank decomposition dimension
alpha: 32                   # Scaling parameter (α = 2 × rank)
dropout: 0.1               # Regularization
target_modules:            # Applied to attention layers
  - q_proj
  - k_proj  
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Training Parameters
trainable_params: ~1.3M    # <1% of base model
memory_usage: ~12GB        # Significantly reduced
```

### 3.2 SFT (Standard Fine-tuning) Configuration
```python
# Full Model Fine-tuning
trainable_params: ~500M    # All model parameters
target_modules: "all"      # Complete model adaptation
memory_usage: ~20GB        # Full VRAM utilization
```

### 3.3 Shared Training Configuration
```python
# Optimization Settings
learning_rate: 3e-4        # Optimized from grid search
min_learning_rate: 3e-5    # Lower bound for scheduler
warmup_steps: 200          # Extended warmup period
scheduler: "CosineAnnealing"
weight_decay: 0.01

# Training Dynamics
batch_size: 4              # Per-GPU micro-batch size
global_batch_size: 4       # Effective batch size
accumulate_grad_batches: 1
max_steps: 1000           # Training duration
save_interval: 100         # Checkpoint frequency

# Mixed Precision
precision: "bf16-mixed"    # Brain float16 for stability
```

## 4. Data Preparation

### 4.1 Japanese Text Preprocessing Pipeline

#### Stage 1: Text Normalization (`00convert_ja.py`)
```python
def japanese_preprocessing(text):
    """
    Comprehensive Japanese text preprocessing
    """
    # 1. Unicode normalization (NFKC)
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Remove unnecessary characters
    text = re.sub(r'[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\w\s]', '', text)
    
    # 3. Standardize punctuation
    text = text.replace('。', '.')
    text = text.replace('、', ',')
    
    # 4. Handle special cases
    text = handle_katakana_variations(text)
    text = normalize_kanji_variants(text)
    
    return text
```

#### Stage 2: Tokenization Strategy
```python
# SentencePiece Configuration
tokenizer_config = {
    'model_type': 'BPE',
    'vocab_size': 32000,
    'character_coverage': 0.9995,
    'split_by_unicode_script': True,
    'byte_fallback': True
}
```

### 4.2 Dataset Composition
```yaml
Training Data:
  - Japanese Wikipedia: ~200k sentences
  - Japanese News Articles: ~100k sentences  
  - Japanese Literature: ~50k sentences
  - Total Training Samples: ~350k sentences

Validation Data:
  - Held-out Japanese Wikipedia: ~10k sentences
  - Japanese Comprehension Tasks: ~5k samples
  - Total Validation Samples: ~15k samples
```

## 5. Training Procedures

### 5.1 Continual Learning Protocol

#### Phase 1: Base Model Preparation
```bash
# Convert HuggingFace to NeMo format
python src/algorithms/01_convert_hf_to_nemo.py \
  --model_path "Qwen/Qwen2.5-0.5B" \
  --output_path "models/qwen25_base.nemo"
```

#### Phase 2: Japanese Text Processing
```bash
# Preprocess Japanese training data
python src/algorithms/00convert_ja.py \
  --input_file "data/raw/japanese_corpus.txt" \
  --output_file "data/processed/japanese_tokenized.jsonl" \
  --preprocessing_config "configs/japanese_preprocessing.yaml"
```

#### Phase 3: PEFT Training
```bash
# Execute PEFT training
python src/algorithms/02_qwen25_peft.py \
  --config "configs/model_configs/qwen25_0.5b.yaml" \
  --data_path "data/processed/japanese_tokenized.jsonl" \
  --output_dir "experiments/peft_japanese_adaptation"
```

#### Phase 4: SFT Training (Comparison)
```bash
# Execute SFT training
python src/algorithms/03_qwen25_sft.py \
  --config "configs/model_configs/qwen25_0.5b.yaml" \
  --data_path "data/processed/japanese_tokenized.jsonl" \
  --output_dir "experiments/sft_japanese_adaptation"
```

### 5.2 Training Monitoring

#### Real-time Metrics
```python
monitored_metrics = {
    'loss': 'training_loss',
    'perplexity': 'exp(loss)',
    'learning_rate': 'current_lr',
    'memory_usage': 'gpu_memory_mb',
    'throughput': 'tokens_per_second',
    'gradient_norm': 'grad_norm'
}
```

#### Convergence Criteria
```python
early_stopping_config = {
    'monitor': 'validation_loss',
    'patience': 50,
    'min_delta': 0.01,
    'mode': 'min'
}
```

## 6. Evaluation Protocol

### 6.1 Intrinsic Evaluation Metrics

#### Language Modeling Metrics
```python
evaluation_metrics = {
    'perplexity': 'Primary language modeling metric',
    'cross_entropy_loss': 'Training optimization objective', 
    'bits_per_character': 'Information-theoretic measure',
    'convergence_speed': 'Steps to reach 95% of final performance'
}
```

#### Efficiency Metrics
```python
efficiency_metrics = {
    'parameter_efficiency': 'trainable_params / total_params',
    'memory_efficiency': 'peak_memory_usage_gb',
    'time_efficiency': 'training_time_hours',
    'inference_latency': 'average_inference_time_ms'
}
```

### 6.2 Extrinsic Evaluation Tasks

#### Japanese Language Understanding
```yaml
Tasks:
  - Japanese Reading Comprehension
  - Japanese Question Answering  
  - Japanese Sentiment Analysis
  - Japanese Text Classification
  
Datasets:
  - JGLUE: Japanese General Language Understanding
  - JSQuAD: Japanese Reading Comprehension
  - JNLI: Japanese Natural Language Inference
```

## 7. Experimental Variations

### 7.1 Hyperparameter Grid Search

#### LoRA Rank Sensitivity
```python
lora_rank_experiments = [4, 8, 16, 32, 64]
for rank in lora_rank_experiments:
    run_peft_experiment(
        rank=rank,
        alpha=2*rank,  # Standard α = 2r heuristic
        experiment_name=f"peft_rank_{rank}"
    )
```

#### Learning Rate Schedule Comparison
```python
scheduler_experiments = {
    'cosine': 'CosineAnnealingLR',
    'linear': 'LinearLR', 
    'exponential': 'ExponentialLR',
    'constant': 'ConstantLR'
}
```

### 7.2 Ablation Studies

#### Preprocessing Impact
```python
preprocessing_ablations = {
    'baseline': 'No preprocessing',
    'normalization': 'Unicode normalization only',
    'full_pipeline': 'Complete preprocessing pipeline'
}
```

#### Mixed Precision Impact
```python
precision_experiments = {
    'fp32': 'Full precision training',
    'fp16': 'Half precision training', 
    'bf16': 'Brain float16 training'
}
```

## 8. Reproducibility Guidelines

### 8.1 Environment Specification
```dockerfile
# Docker Environment
FROM nvcr.io/nvidia/nemo:24.01.framework
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt
```

### 8.2 Random Seed Management
```python
# Ensure reproducible results
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
```

### 8.3 Configuration Management
```yaml
# All experiments use versioned configuration files
experiment_config:
  version: "1.0"
  timestamp: "2024-07-27T17:00:00Z"
  git_commit: "18a5a10"
  environment: "nemo_2.0_japanese"
```

## 9. Expected Outcomes

### 9.1 Quantitative Results
```python
expected_results = {
    'peft_convergence': 'Faster initial convergence, stable final loss',
    'sft_performance': 'Lower final perplexity, higher resource usage',
    'memory_savings': '40-60% reduction with PEFT',
    'parameter_efficiency': '99%+ parameter reduction with PEFT'
}
```

### 9.2 Qualitative Analysis
- **Training Stability**: PEFT expected to show smoother loss curves
- **Japanese Fluency**: Both methods should improve Japanese generation
- **Knowledge Retention**: PEFT should better preserve original capabilities

## 10. Statistical Analysis Plan

### 10.1 Significance Testing
```python
# Compare PEFT vs SFT performance
from scipy import stats

def statistical_comparison(peft_scores, sft_scores):
    """
    Perform paired t-test for performance comparison
    """
    statistic, p_value = stats.ttest_rel(peft_scores, sft_scores)
    effect_size = (np.mean(peft_scores) - np.mean(sft_scores)) / np.std(peft_scores)
    return statistic, p_value, effect_size
```

### 10.2 Confidence Intervals
```python
# Report 95% confidence intervals for all metrics
confidence_level = 0.95
alpha = 1 - confidence_level
```

---

*This experimental design ensures rigorous, reproducible evaluation of Japanese language adaptation approaches. For detailed results, see [Results Analysis](results_analysis.md).* 