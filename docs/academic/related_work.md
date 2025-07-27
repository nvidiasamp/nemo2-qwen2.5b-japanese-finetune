# Related Work

## Overview

This section provides a comprehensive review of the relevant literature in **Parameter-Efficient Fine-Tuning (PEFT)**, **Continual Learning**, and **Japanese Natural Language Processing** that forms the theoretical foundation of our approach.

## 1. Parameter-Efficient Fine-Tuning (PEFT)

### 1.1 Low-Rank Adaptation (LoRA)
**Hu et al. (2021)** introduced Low-Rank Adaptation (LoRA) as a parameter-efficient approach for adapting large pre-trained models. The key innovation lies in decomposing weight updates into low-rank matrices:

```
ΔW = BA
```

Where:
- `B ∈ R^{d×r}` and `A ∈ R^{r×k}` with rank `r << min(d,k)`
- This reduces trainable parameters by orders of magnitude
- **Theoretical Foundation**: Based on the intrinsic rank hypothesis of neural networks

### 1.2 Adapter Methods
**Houlsby et al. (2019)** proposed adapter layers that insert small bottleneck layers between transformer blocks:
- **Computational Overhead**: Minimal increase in inference time
- **Parameter Efficiency**: Typically <5% of original model parameters
- **Performance**: Maintains 95%+ of full fine-tuning performance

### 1.3 Prefix Tuning and Prompt Engineering
**Li & Liang (2021)** developed prefix tuning methods:
- **Concept**: Optimize continuous prompts rather than discrete tokens
- **Advantages**: No architectural changes required
- **Applications**: Particularly effective for generation tasks

## 2. Continual Learning

### 2.1 Catastrophic Forgetting Problem
**McCloskey & Cohen (1989)** first identified catastrophic forgetting in neural networks:
- **Definition**: Loss of previously learned knowledge when learning new tasks
- **Measurement**: Performance degradation on earlier tasks
- **Relevance**: Critical challenge in language model adaptation

### 2.2 Continual Learning Strategies

#### Regularization-Based Approaches
**Kirkpatrick et al. (2017)** - Elastic Weight Consolidation (EWC):
```
L(θ) = L_B(θ) + λ/2 ∑_i F_i(θ_i - θ*_A,i)²
```
Where `F_i` represents Fisher information importance weights.

#### Memory-Based Approaches
**Lopez-Paz & Ranzato (2017)** - Gradient Episodic Memory:
- Store representative examples from previous tasks
- Use stored examples to constrain learning on new tasks

#### Architecture-Based Approaches
**Yoon et al. (2018)** - Lifelong Learning with Dynamically Expandable Networks:
- Dynamically add network capacity for new tasks
- Maintain dedicated parameters for each task

## 3. Japanese Natural Language Processing

### 3.1 Japanese Language Characteristics
**Unique Challenges**:
- **Multiple Writing Systems**: Hiragana, Katakana, Kanji, Romaji
- **Agglutinative Morphology**: Complex word formation patterns
- **Context Dependency**: High reliance on context for disambiguation
- **Word Segmentation**: No explicit word boundaries

### 3.2 Tokenization Approaches
**Kudo & Richardson (2018)** - SentencePiece:
- **Subword Regularization**: Improves robustness and generalization
- **Language Agnostic**: Effective for Japanese without explicit segmentation

**Nakagawa & Uchimoto (2007)** - Japanese Word Segmentation:
- **Statistical Approaches**: CRF-based methods for boundary detection
- **Neural Approaches**: BiLSTM-CRF architectures

### 3.3 Pre-trained Models for Japanese

#### BERT-based Models
**Tohoku University BERT** (2019):
- First Japanese BERT model
- Trained on Japanese Wikipedia and web text
- Established baseline for Japanese NLP tasks

#### GPT-based Models
**OpenCALM** (2023):
- Japanese-focused language model
- Demonstrates effectiveness of dedicated Japanese training

**Qwen2.5** (2024):
- Multilingual model with strong Japanese capabilities
- Base model for our adaptation experiments

## 4. NeMo Framework and Model Optimization

### 4.1 NVIDIA NeMo Framework
**Kuchaiev et al. (2019)**:
- **Modular Design**: Separates data, model, and training components
- **Scalability**: Supports distributed training across multiple GPUs
- **Reproducibility**: Standardized configuration management

### 4.2 Mixed Precision Training
**Micikevicius et al. (2018)**:
- **FP16 Training**: Reduces memory usage and increases throughput
- **Loss Scaling**: Prevents gradient underflow in low precision
- **Hardware Acceleration**: Optimized for modern GPU architectures

## 5. Research Gap and Our Contribution

### 5.1 Identified Gaps
1. **Limited PEFT Studies on Japanese**: Most PEFT research focuses on English
2. **Continual Learning for Language Adaptation**: Insufficient exploration of continual learning for cross-lingual adaptation
3. **Framework Integration**: Lack of comprehensive frameworks combining PEFT + Continual Learning + Japanese NLP

### 5.2 Our Contribution
1. **Systematic Comparison**: PEFT vs. SFT for Japanese adaptation
2. **Continual Learning Integration**: Novel application to Japanese language adaptation
3. **Reproducible Framework**: Complete implementation using NeMo 2.0
4. **Performance Analysis**: Detailed evaluation of memory efficiency and convergence

## 6. References

1. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.
2. Houlsby, N., et al. (2019). Parameter-Efficient Transfer Learning for NLP. *ICML 2019*.
3. Li, X. L., & Liang, P. (2021). Prefix-tuning: Optimizing continuous prompts for generation. *ACL 2021*.
4. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*.
5. Kudo, T., & Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer. *EMNLP 2018*.
6. Kuchaiev, O., et al. (2019). NeMo: a toolkit for building AI applications using Neural Modules. *arXiv preprint*.
7. Micikevicius, P., et al. (2018). Mixed precision training. *ICLR 2018*.

---

*This document provides the theoretical foundation for our Japanese language adaptation approach. For implementation details, see [Experimental Design](experimental_design.md).* 