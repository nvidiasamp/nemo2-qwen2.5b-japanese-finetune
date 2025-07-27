# NeMo Framework Docker 配置指南

## 重要配置更新

根据NVIDIA官方推荐，运行NeMo Framework时需要特定的Docker参数以确保最佳性能和稳定性。

## 问题背景

NeMo 25.04容器会显示以下警告：
```
NOTE: The SHMEM allocation limit is set to the default of 64MB. This may be
   insufficient for NeMo Framework. NVIDIA recommends the use of the following flags:
   docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ...
```

## 推荐的Docker运行命令

### 配置验证
```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/nemo:25.04 python scripts/test_training_config.py
```

### 训练执行
```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/nemo:25.04 python continual_learning.py
```

## 参数说明

| 参数 | 作用 | 重要性 |
|------|------|--------|
| `--gpus all` | 访问所有GPU | 必需 |
| `--ipc=host` | 使用主机IPC命名空间 | **关键** - 提高共享内存性能 |
| `--ulimit memlock=-1` | 取消内存锁定限制 | **关键** - 防止CUDA内存分配失败 |
| `--ulimit stack=67108864` | 设置堆栈大小为64MB | **重要** - 支持大型神经网络计算图 |
| `-v $(pwd):/workspace` | 挂载当前目录 | 必需 |
| `-w /workspace` | 设置工作目录 | 必需 |

## 技术原理

### IPC Host模式
- 允许容器与主机共享进程间通信
- 对多进程数据加载和分布式训练至关重要
- 提高大型模型训练时的内存访问效率

### 内存锁定限制移除
- 防止CUDA运行时内存分配失败
- 特别重要对于GPU密集型的深度学习工作负载
- 避免"failed to allocate memory"错误

### 堆栈大小增加
- 支持深度神经网络的大型计算图
- 防止递归调用栈溢出
- 64MB堆栈大小足以处理大多数NeMo模型

## 验证配置

运行以下命令验证配置是否正确：

```bash
# 测试配置
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/nemo:25.04 python scripts/test_training_config.py

# 期望输出：
# 🎉 所有测试通过！配置验证成功
# ✅ continual_learning.py 可以安全运行
```

## 故障排除

如果仍遇到内存相关错误，可以考虑：

1. **增加共享内存大小**（如果主机允许）：
   ```bash
   --shm-size=2g
   ```

2. **检查主机资源**：
   ```bash
   # 检查可用内存
   free -h
   
   # 检查GPU状态
   nvidia-smi
   ```

3. **减少批次大小**：
   在训练脚本中调整 `micro_batch_size` 或 `global_batch_size`

## 更新历史

- **2025-07-14**: 根据NeMo 25.04容器警告更新Docker参数
- **修复前**: 基本参数 `docker run --rm --gpus all`
- **修复后**: 完整参数包含IPC和内存限制配置 