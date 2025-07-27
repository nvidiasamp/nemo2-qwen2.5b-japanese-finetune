# Evaluation Metrics

## Overview

This document defines the comprehensive evaluation framework for assessing Japanese language adaptation performance using Parameter-Efficient Fine-Tuning (PEFT) and Supervised Fine-Tuning (SFT) approaches. Our evaluation adopts a multi-dimensional approach covering **performance**, **efficiency**, **stability**, and **generalization**.

## 1. Intrinsic Language Modeling Metrics

### 1.1 Perplexity (Primary Metric)
**Definition**: Perplexity measures how well a probability model predicts a sample.

```python
def calculate_perplexity(model, test_data):
    """
    Calculate perplexity on Japanese test data
    """
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in test_data:
            outputs = model(batch['input_ids'], labels=batch['labels'])
            loss = outputs.loss
            total_loss += loss.item() * batch['input_ids'].size(1)
            total_tokens += batch['input_ids'].size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()
```

**Interpretation**:
- **Lower is Better**: Lower perplexity indicates better language modeling
- **Baseline**: Qwen2.5-0.5B baseline perplexity on Japanese text: ~15-20
- **Target**: Post-adaptation perplexity: <12

### 1.2 Cross-Entropy Loss
**Definition**: The optimization objective during training.

```python
def cross_entropy_loss(logits, targets, reduction='mean'):
    """
    Standard cross-entropy loss for language modeling
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), 
        targets.view(-1), 
        ignore_index=-100,
        reduction=reduction
    )
```

**Tracking**:
- **Training Loss**: Monitored every step
- **Validation Loss**: Computed every 100 steps
- **Convergence**: Loss plateauing indicates convergence

### 1.3 Bits Per Character (BPC)
**Definition**: Information-theoretic measure of compression efficiency.

```python
def bits_per_character(cross_entropy_loss, tokenizer, text):
    """
    Calculate bits per character for Japanese text
    """
    # Convert cross-entropy to bits
    bits_per_token = cross_entropy_loss / math.log(2)
    
    # Account for tokenization ratio
    num_tokens = len(tokenizer.encode(text))
    num_chars = len(text)
    tokenization_ratio = num_tokens / num_chars
    
    bpc = bits_per_token * tokenization_ratio
    return bpc
```

**Japanese-Specific Considerations**:
- Japanese characters have higher information density
- Typical BPC for Japanese: 2.5-4.0 bits/character
- Kanji characters contribute more information than Hiragana/Katakana

## 2. Efficiency Metrics

### 2.1 Parameter Efficiency
**Definition**: Ratio of trainable parameters to total model parameters.

```python
def parameter_efficiency(model, peft_config=None):
    """
    Calculate parameter efficiency for PEFT vs SFT
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    if peft_config:  # PEFT case
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    else:  # SFT case
        trainable_params = total_params
    
    efficiency = trainable_params / total_params
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'efficiency_ratio': efficiency,
        'parameter_reduction': 1 - efficiency
    }
```

**Expected Results**:
- **PEFT (LoRA r=16)**: ~1.3M trainable params (~0.26% of total)
- **SFT**: ~500M trainable params (100% of total)
- **Efficiency Gain**: 99.74% parameter reduction with PEFT

### 2.2 Memory Efficiency
**Definition**: Peak GPU memory usage during training and inference.

```python
def memory_efficiency_monitor():
    """
    Monitor GPU memory usage throughout training
    """
    if torch.cuda.is_available():
        memory_stats = {
            'allocated': torch.cuda.memory_allocated() / 1e9,  # GB
            'reserved': torch.cuda.memory_reserved() / 1e9,    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
            'max_reserved': torch.cuda.max_memory_reserved() / 1e9
        }
        return memory_stats
    return None
```

**Tracking Categories**:
- **Model Parameters**: Memory for storing model weights
- **Optimizer States**: Adam optimizer state memory
- **Gradients**: Gradient computation memory
- **Activations**: Forward pass activation memory

### 2.3 Training Time Efficiency
**Definition**: Wall-clock time required for training completion.

```python
def training_efficiency_metrics(start_time, end_time, total_steps):
    """
    Calculate training efficiency metrics
    """
    total_time = end_time - start_time
    time_per_step = total_time / total_steps
    
    return {
        'total_training_time': total_time,
        'time_per_step': time_per_step,
        'steps_per_hour': 3600 / time_per_step
    }
```

**Benchmark Targets**:
- **PEFT**: Expected ~20% faster training than SFT
- **Throughput**: Tokens processed per second
- **Convergence Speed**: Steps to reach 95% of final performance

## 3. Convergence and Stability Metrics

### 3.1 Loss Curve Analysis
**Definition**: Analysis of training loss trajectory for stability assessment.

```python
def convergence_analysis(loss_history, window_size=50):
    """
    Analyze convergence characteristics
    """
    # Smooth loss curve
    smoothed_loss = np.convolve(loss_history, 
                               np.ones(window_size)/window_size, 
                               mode='valid')
    
    # Calculate metrics
    final_loss = np.mean(loss_history[-window_size:])
    loss_variance = np.var(loss_history[-window_size:])
    convergence_step = find_convergence_point(smoothed_loss)
    
    return {
        'final_loss': final_loss,
        'loss_stability': 1 / (1 + loss_variance),  # Higher is more stable
        'convergence_step': convergence_step,
        'overfitting_indicator': detect_overfitting(loss_history)
    }
```

### 3.2 Gradient Norm Monitoring
**Definition**: Track gradient norms to detect training instabilities.

```python
def gradient_norm_analysis(model):
    """
    Calculate gradient norms for stability monitoring
    """
    total_norm = 0
    param_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    
    return {
        'gradient_norm': total_norm,
        'gradient_variance': calculate_gradient_variance(model),
        'gradient_stability': assess_gradient_stability(total_norm)
    }
```

### 3.3 Learning Rate Sensitivity
**Definition**: Assess robustness to learning rate changes.

```python
def lr_sensitivity_analysis(model, test_data, lr_range):
    """
    Evaluate model sensitivity to learning rate changes
    """
    sensitivity_scores = []
    
    for lr in lr_range:
        # Test model with different learning rates
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        test_loss = evaluate_model(model, test_data, optimizer)
        sensitivity_scores.append(test_loss)
    
    # Calculate sensitivity coefficient
    lr_sensitivity = np.std(sensitivity_scores) / np.mean(sensitivity_scores)
    return lr_sensitivity
```

## 4. Japanese Language-Specific Metrics

### 4.1 Character-Level Accuracy
**Definition**: Accuracy of character prediction for Japanese text.

```python
def japanese_character_accuracy(model, tokenizer, test_sentences):
    """
    Measure character-level accuracy for Japanese
    """
    correct_chars = 0
    total_chars = 0
    
    for sentence in test_sentences:
        predicted = model.generate(sentence[:-1])
        target = sentence
        
        # Convert to character level
        pred_chars = tokenizer.decode(predicted)
        target_chars = tokenizer.decode(target)
        
        for p_char, t_char in zip(pred_chars, target_chars):
            if p_char == t_char:
                correct_chars += 1
            total_chars += 1
    
    accuracy = correct_chars / total_chars
    return accuracy
```

### 4.2 Script-Specific Performance
**Definition**: Separate evaluation for different Japanese writing systems.

```python
def script_specific_evaluation(model, tokenizer, test_data):
    """
    Evaluate performance on different Japanese scripts
    """
    import unicodedata
    
    def classify_character(char):
        if '\u3040' <= char <= '\u309F':
            return 'hiragana'
        elif '\u30A0' <= char <= '\u30FF':
            return 'katakana'
        elif '\u4E00' <= char <= '\u9FAF':
            return 'kanji'
        else:
            return 'other'
    
    script_performance = {
        'hiragana': {'correct': 0, 'total': 0},
        'katakana': {'correct': 0, 'total': 0},
        'kanji': {'correct': 0, 'total': 0},
        'other': {'correct': 0, 'total': 0}
    }
    
    # Evaluate per script type
    for sentence in test_data:
        predicted = model.generate(sentence[:-1])
        target = sentence
        
        pred_text = tokenizer.decode(predicted)
        target_text = tokenizer.decode(target)
        
        for p_char, t_char in zip(pred_text, target_text):
            script_type = classify_character(t_char)
            script_performance[script_type]['total'] += 1
            if p_char == t_char:
                script_performance[script_type]['correct'] += 1
    
    # Calculate accuracy per script
    script_accuracies = {}
    for script, stats in script_performance.items():
        if stats['total'] > 0:
            script_accuracies[script] = stats['correct'] / stats['total']
        else:
            script_accuracies[script] = 0.0
    
    return script_accuracies
```

### 4.3 Semantic Coherence (BLEU Score)
**Definition**: Evaluate semantic coherence using BLEU score for Japanese generation.

```python
from sacrebleu import sentence_bleu

def japanese_bleu_evaluation(model, tokenizer, test_pairs):
    """
    Calculate BLEU scores for Japanese text generation
    """
    bleu_scores = []
    
    for input_text, reference_text in test_pairs:
        # Generate prediction
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        predicted_ids = model.generate(input_ids, max_length=100)
        predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        
        # Calculate BLEU score
        bleu_score = sentence_bleu(predicted_text, [reference_text])
        bleu_scores.append(bleu_score.score)
    
    return {
        'mean_bleu': np.mean(bleu_scores),
        'std_bleu': np.std(bleu_scores),
        'bleu_scores': bleu_scores
    }
```

## 5. Comparative Analysis Metrics

### 5.1 PEFT vs SFT Performance Ratio
**Definition**: Direct comparison between PEFT and SFT approaches.

```python
def peft_sft_comparison(peft_metrics, sft_metrics):
    """
    Compare PEFT and SFT performance across all metrics
    """
    comparison = {}
    
    # Performance comparison (lower is better for loss/perplexity)
    comparison['performance_ratio'] = peft_metrics['perplexity'] / sft_metrics['perplexity']
    
    # Efficiency comparison (higher is better)
    comparison['parameter_efficiency_gain'] = (
        sft_metrics['trainable_params'] / peft_metrics['trainable_params']
    )
    
    comparison['memory_efficiency_gain'] = (
        sft_metrics['memory_usage'] / peft_metrics['memory_usage']
    )
    
    comparison['time_efficiency_gain'] = (
        sft_metrics['training_time'] / peft_metrics['training_time']
    )
    
    return comparison
```

### 5.2 Statistical Significance Testing
**Definition**: Determine if performance differences are statistically significant.

```python
from scipy import stats

def statistical_significance_test(peft_results, sft_results, alpha=0.05):
    """
    Perform statistical tests for significance
    """
    # Paired t-test for performance comparison
    t_stat, p_value = stats.ttest_rel(peft_results, sft_results)
    
    # Effect size calculation (Cohen's d)
    pooled_std = np.sqrt(
        (np.var(peft_results) + np.var(sft_results)) / 2
    )
    cohens_d = (np.mean(peft_results) - np.mean(sft_results)) / pooled_std
    
    # Confidence interval
    diff_mean = np.mean(peft_results) - np.mean(sft_results)
    diff_std = np.sqrt(np.var(peft_results) + np.var(sft_results))
    
    confidence_interval = stats.norm.interval(
        1 - alpha, 
        loc=diff_mean, 
        scale=diff_std
    )
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': cohens_d,
        'confidence_interval': confidence_interval,
        'significant': p_value < alpha
    }
```

## 6. Evaluation Protocol

### 6.1 Evaluation Schedule
```python
evaluation_schedule = {
    'every_step': ['training_loss', 'learning_rate', 'gradient_norm'],
    'every_100_steps': ['validation_loss', 'memory_usage'],
    'every_500_steps': ['perplexity', 'japanese_character_accuracy'],
    'end_of_training': ['full_evaluation_suite', 'statistical_analysis']
}
```

### 6.2 Reporting Format
```python
def generate_evaluation_report(metrics_dict):
    """
    Generate comprehensive evaluation report
    """
    report = {
        'executive_summary': {
            'final_perplexity': metrics_dict['perplexity'],
            'parameter_efficiency': metrics_dict['param_efficiency'],
            'memory_savings': metrics_dict['memory_efficiency'],
            'convergence_achieved': metrics_dict['converged']
        },
        'detailed_metrics': metrics_dict,
        'statistical_analysis': metrics_dict['statistical_tests'],
        'japanese_specific': metrics_dict['japanese_metrics'],
        'recommendations': generate_recommendations(metrics_dict)
    }
    return report
```

## 7. Benchmark Targets

### 7.1 Performance Targets
```python
benchmark_targets = {
    'perplexity': {
        'excellent': '< 10',
        'good': '10-12', 
        'acceptable': '12-15',
        'poor': '> 15'
    },
    'parameter_efficiency': {
        'peft_target': '< 1% trainable params',
        'memory_reduction': '> 40%',
        'time_reduction': '> 20%'
    },
    'japanese_accuracy': {
        'character_level': '> 85%',
        'hiragana': '> 90%',
        'katakana': '> 88%',
        'kanji': '> 80%'
    }
}
```

### 7.2 Success Criteria
```python
success_criteria = {
    'primary': [
        'PEFT achieves <10% performance loss vs SFT',
        'Parameter reduction >99%',
        'Memory savings >40%'
    ],
    'secondary': [
        'Training time reduction >20%',
        'Stable convergence achieved',
        'Japanese fluency maintained'
    ]
}
```

---

*This evaluation framework ensures comprehensive, rigorous assessment of Japanese language adaptation approaches. For experimental setup details, see [Experimental Design](experimental_design.md).* 