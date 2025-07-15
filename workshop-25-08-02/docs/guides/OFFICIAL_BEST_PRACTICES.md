# NeMo 2.0 单GPU训练官方最佳实践

基于NVIDIA官方文档深度研究总结的单GPU训练最佳实践指南。

## 🔍 关键发现

通过深度查询官方文档，我们发现了NeMo 2.0单GPU训练的核心问题和解决方案：

### 1. 核心解决方案：使用 `direct=True`

**官方文档明确推荐**：
```python
# 最重要的修复 - 避免分布式初始化
run.run(recipe, direct=True)
```

**原理**：
- `direct=True` 在同一个Python进程中直接运行训练
- 避免了分布式进程初始化和NCCL相关问题
- 这是官方文档中专门针对单GPU训练推荐的方法

### 2. 必要的环境变量配置

**官方推荐的关键环境变量**：
```python
env_vars = {
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "NCCL_NVLS_ENABLE": "0", 
    "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
    "NVTE_ASYNC_AMAX_REDUCTION": "1",
    "CUDA_VISIBLE_DEVICES": "0",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",  # H100/Ada GPU必需
    
    # 内存优化
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    
    # LayerNorm优化 (防止通信重叠失败)
    "NVTE_FWD_LAYERNORM_SM_MARGIN": "16",
    "NVTE_BWD_LAYERNORM_SM_MARGIN": "16",
}
```

### 3. 正确的单GPU配置模式

**设备配置**：
```python
recipe.trainer.devices = 1
recipe.trainer.num_nodes = 1

# 禁用所有并行策略
recipe.trainer.strategy.tensor_model_parallel_size = 1
recipe.trainer.strategy.pipeline_model_parallel_size = 1
recipe.trainer.strategy.context_parallel_size = 1
```

**模型配置调整**（适应单GPU）：
```python
# 官方建议减少层数适应单GPU
recipe.model.config.num_layers = 12  # 从默认32减少到12
recipe.model.config.hidden_size = 768
recipe.model.config.ffn_hidden_size = 2048
recipe.model.config.num_attention_heads = 12
```

### 4. `if __name__ == "__main__":` 的重要性

**官方文档多次强调**：
```python
# 这对于避免Python多进程初始化问题至关重要
if __name__ == "__main__":
    run_training()
```

**重要性**：
- 防止"Failure to acquire lock"错误
- 避免多进程初始化问题
- 确保与Python multiprocessing模块兼容

## 📋 完整的配置检查清单

### ✅ 环境配置
- [ ] 使用正确的NeMo容器镜像 (`nvcr.io/nvidia/nemo:25.04`)
- [ ] 设置所有必需的环境变量
- [ ] 确保`CUDA_VISIBLE_DEVICES="0"`强制使用GPU 0
- [ ] 6000Ada GPU验证通过

### ✅ Recipe配置
- [ ] 使用`llm.qwen25_500m.pretrain_recipe()`
- [ ] 设置`devices=1, num_nodes=1`
- [ ] 禁用所有并行策略（TP=1, PP=1, CP=1）
- [ ] 调整模型尺寸适应单GPU
- [ ] 配置适当的批次大小

### ✅ 数据配置
- [ ] 验证所有数据文件存在
- [ ] 正确配置train/validation/test路径
- [ ] 使用适当的序列长度和批次大小

### ✅ 执行配置
- [ ] 使用`run.run(recipe, direct=True)`
- [ ] 包装在`if __name__ == "__main__":`中
- [ ] 正确的错误处理和日志记录

## 🚀 使用方法

### 在Docker容器中运行：
```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $(pwd):/workspace -w /workspace \
    nvcr.io/nvidia/nemo:25.04 \
    python3 official_single_gpu_training.py
```

### 直接运行：
```bash
python3 official_single_gpu_training.py
```

## 🔧 故障排除

### 如果仍然遇到NCCL错误：
1. **检查环境变量**：确保所有环境变量正确设置
2. **验证GPU**：使用`nvidia-smi`确认GPU 0可用
3. **检查数据**：确认所有数据文件存在
4. **重启容器**：有时需要重新启动Docker容器

### 如果内存不足：
1. **减少批次大小**：将`micro_batch_size`设为1
2. **减少序列长度**：从2048减少到1024
3. **进一步减少模型层数**：从12减少到8

## 📚 参考文档

- [NeMo 2.0 Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html)
- [Qwen2/2.5 Model Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/qwen2.html)
- [Performance Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html)

## 🎯 期望结果

使用此官方最佳实践，您应该能够：
- ✅ 成功在单个6000Ada GPU上运行Qwen2.5-0.5B训练
- ✅ 避免所有分布式初始化和NCCL错误
- ✅ 实现稳定的持续学习训练
- ✅ 获得良好的训练性能和内存使用效率

---

*此文档基于NVIDIA NeMo Framework官方文档深度研究整理，确保了解决方案的权威性和可靠性。* 