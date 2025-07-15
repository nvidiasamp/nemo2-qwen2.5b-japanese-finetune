# LLM-JP 数据处理指南 - NeMo 2.0 版本

## 概述

基于用户在 NeMo 1.0 中验证的成功方法，适配 NeMo 2.0 的数据处理流程。使用Docker容器环境确保一致性和可重复性。

### 核心改进
- **分词器适配**：从 `Llama-3.1-8B` 改为 `Qwen/Qwen2.5-0.5B`
- **容器化处理**：基于NeMo 25.04容器环境
- **智能合并**：自动合并训练文件避免路径问题
- **实时监控**：提供进度跟踪工具

## 文件说明

### 核心脚本
- **`process_data_in_container_fixed.sh`** - 主要数据处理脚本
  - 在Docker容器内执行
  - 自动下载LLM-JP数据（如需要）
  - 合并训练文件并转换为NeMo格式
  - 使用Qwen/Qwen2.5-0.5B分词器

### 监控工具
- **`monitor_progress.sh`** - 进度监控脚本
  - 检查Docker容器状态
  - 显示文件生成进度
  - 提供完成度估算

### 文档
- **`README.md`** - 本文档

## 快速使用

### 方法1：手动Docker执行（推荐）
```bash
# 使用现有NeMo 25.04容器
docker run \
    --gpus all -it --rm \
    --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "/home/cho/workspace/workshop-25-08-02:/workspace" \
    -w "/workspace" \
    nvcr.io/nvidia/nemo:25.04 \
    bash -c "chmod +x /workspace/scripts/data_processing/process_data_in_container_fixed.sh && /workspace/scripts/data_processing/process_data_in_container_fixed.sh"
```

### 方法2：进度监控
```bash
# 一次性检查
./scripts/data_processing/monitor_progress.sh

# 实时监控（每30秒更新）
watch -n 30 ./scripts/data_processing/monitor_progress.sh
```

## 预期输出

### 文件结构
```
data/llm_jp_wiki/
├── raw/ja_wiki/                    # 原始数据（12GB）
│   ├── train_0.jsonl
│   ├── train_1.jsonl
│   ├── ...
│   ├── train_13.jsonl
│   ├── train_merged.jsonl          # 合并的训练文件（6GB）
│   └── validation_0.jsonl
└── nemo_binary/                    # NeMo 格式数据
    ├── ja_wiki_train_text_document.bin    (~500-800MB)
    ├── ja_wiki_train_text_document.idx
    ├── ja_wiki_val_text_document.bin      (~10-20MB)
    └── ja_wiki_val_text_document.idx
```

### 处理时间
- **数据下载**：已完成（如果初次运行需30-60分钟）
- **数据合并**：5-10分钟
- **NeMo转换**：1-2小时
- **总计**：2-3小时

### 技术配置
- **容器**：nvcr.io/nvidia/nemo:25.04
- **分词器**：Qwen/Qwen2.5-0.5B
- **输出格式**：mmap (.bin/.idx)
- **工作线程**：训练数据4个，验证数据2个

## 与NeMo 1.0的兼容性

此方法100%基于用户在NeMo 1.0中验证的成功流程，只需要一个关键调整：

```bash
# 唯一改动
--tokenizer-type="meta-llama/Llama-3.1-8B"  # NeMo 1.0
↓
--tokenizer-type="Qwen/Qwen2.5-0.5B"        # NeMo 2.0适配
```

## 故障排除

### 常见问题

**1. 容器启动失败**
```bash
# 检查Docker和GPU
docker --version
nvidia-smi
```

**2. 数据处理中断**
```bash
# 查看监控状态
./scripts/data_processing/monitor_progress.sh

# 重新启动处理（会跳过已存在的文件）
# 重新运行Docker命令即可
```

**3. 输出文件不完整**
- 脚本会自动验证所有4个必需文件
- 如果失败，会显示详细错误信息
- 重新运行会从中断点继续

### 日志级别
- 🟢 **[INFO]** - 正常进度信息
- 🔵 **[STEP]** - 主要处理步骤
- 🟡 **[WARN]** - 警告信息

## 完成验证

处理成功完成后，您将看到：
```
✅ 数据处理完成！

📁 生成的文件：
ja_wiki_train_text_document.bin - 512M
ja_wiki_train_text_document.idx - 4.0K
ja_wiki_val_text_document.bin - 15M
ja_wiki_val_text_document.idx - 1.0K

📊 文件大小统计：
训练数据: 512M
验证数据: 15M
总计: 527M

🚀 下一步：您可以开始任务7 - 实现PEFT-LoRA微调脚本
```

## 下一步集成

生成的NeMo二进制文件可直接用于：
- **任务7**：PEFT-LoRA微调脚本
- **任务4**：日语持续学习训练
- **任务6**：持续学习执行

配置文件中的数据路径应该设置为：
```yaml
data:
  train_path: ./data/llm_jp_wiki/nemo_binary/ja_wiki_train_text_document
  validation_path: ./data/llm_jp_wiki/nemo_binary/ja_wiki_val_text_document
```

---

**项目状态**：此数据处理流程基于用户在生产环境中验证的成功方法，具有很高的可靠性和稳定性。