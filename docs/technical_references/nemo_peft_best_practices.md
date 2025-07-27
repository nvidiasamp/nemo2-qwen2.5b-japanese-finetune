# NeMo PEFT 最佳实践与核心代码经验

> 创建时间：2025-07-14
> 适用于：NeMo 2.0 Framework
> 项目：LLM-JP 日语持续学习项目

## 概述

本文档整合了 NVIDIA NeMo Framework 中 PEFT（Parameter Efficient Fine-Tuning）的核心代码经验和最佳实践，旨在帮助项目团队避免常见陷阱，建立高效的 PEFT 训练流程。

## 1. 基础架构模式

### 1.1 推荐的项目结构

```python
# 使用 NeMo Run 进行实验管理
from nemo.run import Experiment, Runner
from nemo.collections.llm import peft, pretrain

# 配置管理模式
class PEFTConfig:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-0.5B"  # 项目使用的模型
        self.peft_config = peft.LoRAConfig(
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            rank=32,  # 适配0.5B模型
            alpha=16,
            dropout=0.1
        )
        self.training_config = self.get_training_config()
```

### 1.2 关键设计原则

- ✅ **使用 NeMo Run 框架**：更好的实验管理和资源调度
- ✅ **模块化配置**：避免硬编码，便于调试和复现
- ✅ **明确目标模块**：不要盲目对所有层进行 PEFT

## 2. 训练配置最佳实践

### 2.1 核心训练参数

```python
def get_training_config():
    return {
        # 基础训练设置
        "max_steps": 1000,
        "val_check_interval": 100,
        "limit_val_batches": 10,
        
        # 优化器设置
        "optimizer": {
            "class_path": "megatron.core.optimizer.OptimizerConfig",
            "init_args": {
                "lr": 1e-4,
                "weight_decay": 0.01,
                "beta1": 0.9,
                "beta2": 0.95
            }
        },
        
        # 学习率调度
        "lr_scheduler": {
            "class_path": "megatron.core.optimizer.lr_scheduler.CosineAnnealingScheduler",
            "init_args": {
                "warmup_steps": 100,
                "constant_steps": 0,
                "min_lr": 1e-6
            }
        }
    }
```

### 2.2 性能优化策略

```python
# 内存和计算优化
def calculate_gradient_accumulation_steps(
    global_batch_size, 
    micro_batch_size, 
    num_gpus
):
    return global_batch_size // (micro_batch_size * num_gpus)

# 训练器配置
trainer_config = {
    "precision": "bf16-mixed",  # 混合精度训练
    "gradient_clip_val": 1.0,   # 梯度裁剪
    "accumulate_grad_batches": calculate_gradient_accumulation_steps(64, 4, 1)
}
```

## 3. PEFT 特定优化

### 3.1 LoRA 参数调优指南

```python
# 不同模型大小的 LoRA 配置建议
def get_lora_config(model_size):
    configs = {
        "500m": {"rank": 32, "alpha": 16, "dropout": 0.1},  # 适配Qwen2.5-0.5B
        "7b": {"rank": 64, "alpha": 16, "dropout": 0.1},
        "13b": {"rank": 32, "alpha": 8, "dropout": 0.1},
        "30b": {"rank": 16, "alpha": 4, "dropout": 0.05},
        "70b": {"rank": 8, "alpha": 2, "dropout": 0.05}
    }
    return configs.get(model_size, configs["500m"])
```

### 3.2 目标模块选择策略

```python
def get_target_modules(model_type):
    if "qwen" in model_type.lower():
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "llama" in model_type.lower():
        return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "mistral" in model_type.lower():
        return ["q_proj", "v_proj", "k_proj", "o_proj"]
    else:
        return ["attention", "mlp"]  # 通用回退
```

## 4. 数据处理标准流程

### 4.1 数据预处理模板

```python
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.llm.data import SquadDataModule

def prepare_training_data(dataset_path, model_name):
    """标准化数据准备流程"""
    
    # 1. 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. 数据模块配置
    data_module = SquadDataModule(
        file_path=dataset_path,
        tokenizer=tokenizer,
        max_seq_length=2048,
        micro_batch_size=4,
        global_batch_size=16,
        num_workers=4
    )
    
    return data_module
```

### 4.2 数据质量检查

```python
def validate_dataset_format(dataset_path):
    """验证数据集格式"""
    # 检查文件存在性
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # 检查数据格式
    with open(dataset_path, 'r') as f:
        first_line = f.readline()
        # 验证JSON格式或其他格式
        
    return True
```

## 5. 训练监控和调试

### 5.1 WandB 集成

```python
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def setup_monitoring():
    # 权重跟踪
    wandb.init(project="nemo-peft-training")
    
    # 检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="best-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3
    )
    
    # 早停回调
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        verbose=True,
        mode="min"
    )
    
    return [checkpoint_callback, early_stop_callback]
```

### 5.2 关键指标监控

```python
# 训练过程监控
def log_training_metrics(trainer, model):
    wandb.log({
        "train_loss": trainer.callback_metrics.get("train_loss", 0),
        "val_loss": trainer.callback_metrics.get("val_loss", 0),
        "learning_rate": trainer.optimizers[0].param_groups[0]["lr"],
        "gpu_memory": torch.cuda.memory_allocated() / 1024**3,  # GB
        "peft_params": count_peft_parameters(model)
    })
```

### 5.3 梯度检查

```python
def check_gradients(model):
    """检查梯度爆炸/消失问题"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            if grad_norm > 1.0:  # 梯度爆炸检测
                print(f"Warning: Large gradient in {name}: {grad_norm}")
            elif grad_norm < 1e-8:  # 梯度消失检测
                print(f"Warning: Small gradient in {name}: {grad_norm}")
```

## 6. 模型验证和评估

### 6.1 验证流程

```python
def validate_peft_model(model, validation_dataloader):
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in validation_dataloader:
            outputs = model(batch)
            loss = outputs.loss
            total_loss += loss.item() * batch.size(0)
            total_samples += batch.size(0)
    
    avg_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return {
        "val_loss": avg_loss,
        "perplexity": perplexity.item()
    }
```

### 6.2 质量评估指标

```python
def calculate_peft_efficiency(model):
    """计算PEFT效率指标"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": (trainable_params / total_params) * 100
    }
```

## 7. 模型保存和导出

### 7.1 标准化保存流程

```python
def save_peft_model(model, output_dir):
    """完整的模型保存流程"""
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 NeMo 格式
    model.save_to(f"{output_dir}/model.nemo")
    
    # 保存 PEFT 适配器
    model.save_pretrained(f"{output_dir}/peft_adapter")
    
    # 保存训练配置
    with open(f"{output_dir}/training_config.json", "w") as f:
        json.dump(model.cfg, f, indent=2)
    
    # 保存模型信息
    model_info = {
        "model_type": "peft_lora",
        "base_model": model.cfg.get("pretrained_model_path", "unknown"),
        "peft_config": model.cfg.get("peft", {}),
        "training_steps": model.cfg.get("max_steps", 0),
        "save_timestamp": datetime.now().isoformat()
    }
    
    with open(f"{output_dir}/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model saved to {output_dir}")
```

### 7.2 模型加载

```python
def load_peft_model(model_path):
    """加载PEFT模型"""
    from nemo.collections.llm import api
    
    # 加载基础模型
    model = api.load_model(model_path)
    
    # 如果有PEFT适配器，加载它
    adapter_path = f"{model_path}/peft_adapter"
    if os.path.exists(adapter_path):
        model.load_adapter(adapter_path)
    
    return model
```

## 8. 故障排除和调试

### 8.1 常见问题解决

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| 内存不足 | CUDA OOM | 减少batch_size, 启用gradient_checkpointing |
| 梯度爆炸 | Loss突然变大 | 减少学习率, 调整gradient_clip_val |
| 收敛慢 | Loss下降缓慢 | 增加学习率, 调整warmup_steps |
| 过拟合 | 训练loss低但验证loss高 | 增加dropout, 减少训练步数 |

### 8.2 调试检查清单

- [ ] 数据格式和分词器匹配
- [ ] LoRA配置参数合理
- [ ] 目标模块选择正确
- [ ] 学习率设置适当
- [ ] 批处理大小适配硬件
- [ ] 检查点保存路径正确
- [ ] WandB配置正常

## 9. 关键避坑指南

### 9.1 配置管理

❌ **错误做法**：
```python
# 硬编码配置
model.peft.lora_rank = 64
model.peft.lora_alpha = 16
```

✅ **正确做法**：
```python
# 使用配置类
config = PEFTConfig()
model.peft.lora_rank = config.lora_config["rank"]
model.peft.lora_alpha = config.lora_config["alpha"]
```

### 9.2 数据处理

❌ **错误做法**：
```python
# 跳过数据验证
data = load_data(path)
model.train(data)
```

✅ **正确做法**：
```python
# 先验证再训练
validate_dataset_format(path)
data = prepare_training_data(path, model_name)
model.train(data)
```

### 9.3 实验跟踪

❌ **错误做法**：
```python
# 不记录实验配置
model.train()
```

✅ **正确做法**：
```python
# 完整记录实验
wandb.init(project="peft-experiments")
wandb.log({"config": config.to_dict()})
model.train()
```

## 10. 项目特定配置

### 10.1 Qwen2.5-0.5B 优化配置

```python
# 针对本项目的优化配置
QWEN_PEFT_CONFIG = {
    "model_name": "Qwen/Qwen2.5-0.5B",
    "lora_config": {
        "rank": 32,
        "alpha": 16,
        "dropout": 0.1,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    },
    "training_config": {
        "max_steps": 1000,
        "micro_batch_size": 4,
        "global_batch_size": 16,
        "lr": 1e-4,
        "warmup_steps": 100,
        "val_check_interval": 100
    }
}
```

### 10.2 Japanese 数据处理

```python
# 日语数据特定处理
def prepare_japanese_data(dataset_path):
    # 确保正确的分词器
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    # 日语特定的预处理
    def japanese_preprocessing(text):
        # 添加必要的日语文本处理
        return text
    
    return processed_data
```

## 总结

本文档提供了基于官方 NeMo PEFT notebook 的核心经验，旨在帮助项目团队高效实现 PEFT-LoRA 微调。遵循这些最佳实践可以显著减少试错时间，提高训练效率和模型质量。

---

**参考资料**：
- [NeMo PEFT官方教程](https://github.com/NVIDIA/NeMo/blob/main/tutorials/llm/automodel/peft.ipynb)
- [NeMo Framework用户指南](https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html)
- [WandB集成指南](https://wandb.ai/site/integrations/nemo)