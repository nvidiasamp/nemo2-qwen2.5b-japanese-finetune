# 模型目录

## 目录结构
- `qwen25_0.5b.nemo` - 导入的基础模型
- `checkpoints/` - 训练checkpoint
  - `qwen25_continual/` - 继续学习checkpoint
  - `qwen25_peft/` - PEFT-LoRA微调checkpoint

## 模型流程
1. HuggingFace → NeMo格式 (`qwen25_0.5b.nemo`)
2. 继续学习训练 → `qwen25_continual/`
3. PEFT微调 → `qwen25_peft/`
