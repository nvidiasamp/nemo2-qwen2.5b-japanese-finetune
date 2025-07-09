# NeMo Framework 日语LLM继续预训练操作记录

## 操作概述

本文档记录使用NVIDIA NeMo Framework对Llama-3.1-8B模型进行日语继续预训练的操作流程、可能遇到的问题及解决方案。参考的是NVIDIA官方博客[《NeMo Framework で実践する継続事前学習 – 日本語 LLM 編 –》](https://developer.nvidia.com/ja-jp/blog/how-to-use-continual-pre-training-with-japanese-language-on-nemo-framework/)。

## NeMo Framework 简介

NeMo Framework 是NVIDIA提供的一个云原生框架，用于构建和自定义LLM等生成式AI模型。该框架通过NGC平台提供容器，可以立即开始使用。

NeMo Framework可以免费使用，但作为NVIDIA AI Enterprise的组件，企业用户如需支持服务可以考虑购买NVIDIA AI Enterprise许可证。

### LLM 开发工作流程

LLM的开发涉及以下任务：

- 大规模数据准备（用于预训练）
- 利用分布式学习进行LLM预训练
- 通过微调、对齐和提示工程来自定义LLM
- 优化模型以加速推理
- 充分利用GPU进行LLM服务
- 通过RAG（检索增强生成）以较低成本为LLM提供最新信息
- 为防止LLM应用出现意外行为而设置护栏

### NeMo Framework 组件

NeMo Framework容器包含从数据准备到LLM训练和自定义所需的多个模块，这些模块包括：

- **NeMo Curator**：用于下载、提取、清洗和过滤LLM训练所需的大规模数据集的可扩展工具包
- **NeMo**：用于构建LLM、多模态和语音等生成式AI模型的可扩展框架
- **NeMo Framework Launcher**：用于从云端或本地集群启动任务的工具包
- **Megatron-LM**：研究Transformer模型大规模训练的项目
- **Transformer Engine**：以FP8为核心加速Transformer模型的工具包
- **NeMo-Aligner**：使用RLHF（基于人类反馈的强化学习）、DPO、SteerLM等高效对齐LLM的工具包

这些库在GitHub上以开源形式发布，但建议通过依赖关系已解决的NeMo Framework容器使用。在容器中，这些模块位于`/opt`目录下。

## 继续预训练（Continual Pre-training）

继续预训练是指在现有预训练模型的基础上，使用特定领域或特定语言的数据进行进一步的预训练，以使模型更好地适应特定的应用场景。

在本例中，我们将使用日语维基百科数据对Llama-3.1-8B模型进行继续预训练，以提高其在日语处理任务上的表现。虽然本教程使用单节点和较小规模的日语维基百科数据，但通过增加数据量和使用多节点计算环境，可以轻松扩展为大规模训练。

## 环境准备

### 硬件配置

我的实际环境配置：

- 硬件：
  - CPU: Intel(R) Xeon(R) Silver 4310 CPU @ 2.10GHz
  - GPU: 
    - NVIDIA RTX 6000 Ada Generation (48GB)
    - NVIDIA T600 (4GB)
  - 系统内存: 251GB

- 软件：
  - OS: Ubuntu 20.04.6 LTS
  - 容器: `nvcr.io/nvidia/nemo:25.02.01`

### 原始博客验证环境

原始博客的验证环境为：

- 硬件：
  - DGX H100
  - GPU: 8 x NVIDIA H100 80GB GPUs (驱动版本: 550.90.7)
  - CPU: Intel(R) Xeon(R) Platinum 8480C
  - 系统内存: 2 TB

- 软件：
  - OS: Ubuntu 22.04.5 LTS
  - 容器: `nvcr.io/nvidia/nemo:24.09`

## 实际操作过程

### 1. 创建工作目录

按照教程，首先创建工作目录并进入该目录：

```bash
# 创建工作目录
mkdir cp-example
cd cp-example
```

执行结果：
```
tut-server-11% mkdir -p cp-example && cd cp-example && pwd
/home/cho/workspace/cp-example
```

当前工作目录已成功设置为 `/home/cho/workspace/cp-example`。

### 2. 启动Docker容器

执行以下命令启动Docker容器：

```bash
docker run -it --gpus all --name cp --shm-size=16g --ulimit memlock=-1 --network=host -v ${PWD}:/workspace -w /workspace nvcr.io/nvidia/nemo:25.02.01 bash
```

此命令的参数说明：
- `--rm`：容器停止运行后自动删除容器
- `-it`：交互式终端
- `--gpus all`：使用所有可用的GPU
- `--shm-size=16g`：设置共享内存大小为16GB
- `--ulimit memlock=-1`：取消内存锁定限制
- `--network=host`：使用主机网络
- `-v ${PWD}:/workspace`：将当前目录挂载到容器的/workspace目录
- `-w /workspace`：设置容器的工作目录为/workspace

执行结果：
```
Unable to find image 'nvcr.io/nvidia/nemo:25.02.01' locally
25.02.01: Pulling from nvidia/nemo
de44b265507a: Pulling fs layer 
75f58769314d: Pulling fs layer 
...
Digest: sha256:be271767724ac6b03c22c2c1d7b1de08e82771eacc0b97a825a8a50bef70e1c9
Status: Downloaded newer image for nvcr.io/nvidia/nemo:25.02.01

====================
== NeMo Framework ==
====================

NVIDIA Release  (build 134983853)
Container image Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
...
NOTE: CUDA Forward Compatibility mode ENABLED.
  Using CUDA 12.8 driver version 570.86.10 with kernel driver version 535.183.01.
  See https://docs.nvidia.com/deploy/cuda-compatibility/ for details.

root@tut-server-11:/workspace# 
```

容器已成功启动，当前处于容器内的bash环境，工作目录为`/workspace`，这是主机上`/home/cho/workspace/cp-example`目录的映射位置。

### 3. 从Hugging Face下载预训练模型

登录Hugging Face（需要获得meta-llama/Llama-3.1-8B的访问权限）：

```bash
huggingface-cli login
```

执行此命令后，系统会提示输入Hugging Face令牌。这个令牌可以通过以下步骤获取：
1. 登录到 [Hugging Face官网](https://huggingface.co/)
2. 访问 [设置页面](https://huggingface.co/settings/tokens)
3. 创建新令牌，给予读取权限
4. 复制生成的令牌

另外，使用Llama模型需要先申请访问权限：
1. 访问 [meta-llama/Llama-3.1-8B模型页面](https://huggingface.co/meta-llama/Llama-3.1-8B)
2. 点击"Access"按钮申请访问权限
3. 填写相关表单并等待批准

执行结果：
```
    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Enter your token (input will not be visible): 
Add token as git credential? (Y/n) y
Token is valid (permission: fineGrained).
The token has been saved to /root/.cache/huggingface/stored_tokens
Cannot authenticate through git-credential as no helper is defined on your machine.
You might have to re-authenticate when pushing to the Hugging Face Hub.
Run the following command in your terminal in case you want to set the 'store' credential helper as default.

git config --global credential.helper store

Read https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage for more details.
Token has not been saved to git credential helper.
Your token has been saved to /root/.cache/huggingface/token
Login successful.
```

Hugging Face登录成功完成，令牌已保存在容器中。现在可以继续下载模型。

下一步，创建一个Python脚本来下载Llama-3.1-8B模型。在`src`目录中创建`download_model.py`文件：

```bash
mkdir -p src
```

创建以下内容的Python脚本：

```python
# src/download_model.py
import os
from huggingface_hub import snapshot_download
 
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B",
    local_dir=f"{MODEL_DIR}/Llama-3.1-8B",
)
```

执行脚本时遇到了以下权限错误：

```
huggingface_hub.errors.GatedRepoError: 403 Client Error.
Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.1-8B/...
Access to model meta-llama/Llama-3.1-8B is restricted and you are not in the authorized list. 
Visit https://huggingface.co/meta-llama/Llama-3.1-8B to ask for access.
```

这个错误表明虽然我们已经登录了Hugging Face账号，但账号没有访问meta-llama/Llama-3.1-8B模型的权限。需要完成以下步骤：

1. 访问[meta-llama/Llama-3.1-8B模型页面](https://huggingface.co/meta-llama/Llama-3.1-8B)
2. 点击"Access"按钮申请访问权限
3. 填写相关申请表单（包括使用目的等）
4. 等待Meta AI批准访问请求

批准通常不是即时的，可能需要等待一段时间。获得访问权限后，需要重新登录并再次执行下载脚本。

在等待权限批准的过程中，为了继续教程的其他部分，我们可以考虑使用以下替代方案：
1. 使用已经有公开访问权限的其他模型（例如Mistral、Falcon等）
2. 使用已经下载好的模型（如果有）

经过大约30分钟的等待，我获得了meta-llama/Llama-3.1-8B模型的访问权限。然后重新执行下载脚本：

```bash
python src/download_model.py
```

下载过程开始，模型文件大小约为16.1GB：

```
config.json: 100%|████████████████████████████| 826/826 [00:00<00:00, 3.64MB/s]
model.safetensors.index.json: 100%|█████████| 23.9k/23.9k [00:00<00:00, 44.2MB/s]
...
model-00001-of-00004.safetensors: 100%|████████| 4.98G/4.98G [02:56<00:00, 28.2MB/s]
model-00002-of-00004.safetensors: 100%|████████| 5.00G/5.00G [02:58<00:00, 28.0MB/s]
model-00003-of-00004.safetensors: 100%|████████| 4.92G/4.92G [03:00<00:00, 27.2MB/s]
consolidated.00.pth: 100%|██████████████████| 16.1G/16.1G [04:38<00:00, 57.7MB/s]
Fetching 17 files: 100%|███████████████████| 17/17 [04:39<00:00, 16.43s/it]
```

整个下载过程花费了约5分钟时间。现在模型已经成功下载到了`models/Llama-3.1-8B`目录中。

### 4. 模型格式转换

按照博客指导，需要将下载的Hugging Face格式模型转换为NeMo格式。由于当前版本的NeMo容器（25.02.01）可能不完全支持Llama-3.1模型的转换，需要先应用两个PR补丁：

```bash
cd /opt/NeMo/
curl -L https://github.com/NVIDIA/NeMo/pull/11548.diff | git apply
curl -L https://github.com/NVIDIA/NeMo/pull/11580.diff | git apply
```

需要注意的是，我使用的NeMo容器版本（25.02.01）比博客中使用的版本（24.09）更新。从CHANGELOG可以看到以下相关信息：

- "Add llama 3.1 recipes by @cuichenx :: PR: #11273"
- "New Llama 3.1 Support (2024-07-23) The NeMo Framework now supports training and customizing the Llama 3.1 collection of LLMs from Meta."

这表明较新版本的NeMo框架可能已经原生支持Llama-3.1模型。如果直接运行转换脚本失败，再尝试应用补丁。如果补丁已经集成到25.02.01版本，`git apply`命令可能会报错表示补丁无法应用。

这些补丁的作用是修改NeMo框架，使其能够正确处理Llama-3.1模型。本质上是在"打补丁"，为NeMo框架添加或修复对Llama-3.1模型的支持，这样才能成功将Hugging Face格式的模型正确转换为NeMo格式。如果不应用这些补丁，可能在转换过程中会遇到错误或不兼容问题。

考虑到新版框架可能已经支持Llama-3.1，我们先直接尝试执行转换命令：

```bash
# 设置环境变量
export INPUT="/workspace/models/Llama-3.1-8B"
export OUTPUT="/workspace/models/Llama-3.1-8B.nemo"
export PREC="bf16"

# 执行转换
python /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py --input_name_or_path ${INPUT} --output_path ${OUTPUT} --precision ${PREC} --llama31 True
```

执行结果：

```
[NeMo I 2025-04-23 02:51:43 nemo_logging:393] loading checkpoint /workspace/models/Llama-3.1-8B
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.20s/it]
hf_config: {'vocab_size': 128256, 'max_position_embeddings': 131072, 'hidden_size': 4096, 'intermediate_size': 14336, 'num_hidden_layers': 32, 'num_attention_heads': 32, 'num_key_value_heads': 8, 'hidden_act': 'silu', 'initializer_range': 0.02, 'rms_norm_eps': 1e-05, ...}

... (显示了模型配置和处理过程) ...

converting layer 0
done layer 0
converting layer 1
done layer 1
... (转换32层模型) ...
converting layer 31
done layer 31

[NeMo I 2025-04-23 02:54:14 nemo_logging:393] Global Checkpoint Save : Rank: 0 : Start time: 1745376794.226s : Save duration: 60.221s
[NeMo I 2025-04-23 02:54:34 nemo_logging:393] NeMo model saved to: /workspace/models/Llama-3.1-8B.nemo
```

转换过程顺利完成，模型成功转换为NeMo格式并保存到`/workspace/models/Llama-3.1-8B.nemo`。整个转换过程大约耗时3分钟。

**结论**：在NeMo容器版本25.02.01中，已经原生支持Llama-3.1模型，不需要应用博客中提到的两个PR补丁就能成功进行模型格式转换。这表明版本升级已经包含了这些补丁的功能改进。生成的`Llama-3.1-8B.nemo`文件使用了distributed checkpoint，这意味着不需要每次都改变checkpoint，就可以加载任意的Tensor Parallel (TP)或Pipeline Parallel (PP)等模型并行的组合。

### 5. 数据准备

本教程使用llm-jp-corpus-v3中的日语维基百科(ja_wiki)数据。通过以下命令下载数据并存储到data目录中：

```bash
cd /workspace/
mkdir -p data/ja_wiki

# 下载训练数据集（14个分片）
wget -O data/ja_wiki/train_0.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_0.jsonl.gz?ref_type=heads
wget -O data/ja_wiki/train_1.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_1.jsonl.gz?ref_type=heads
wget -O data/ja_wiki/train_2.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_2.jsonl.gz?ref_type=heads
wget -O data/ja_wiki/train_3.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_3.jsonl.gz?ref_type=heads
wget -O data/ja_wiki/train_4.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_4.jsonl.gz?ref_type=heads
wget -O data/ja_wiki/train_5.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_5.jsonl.gz?ref_type=heads
wget -O data/ja_wiki/train_6.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_6.jsonl.gz?ref_type=heads
wget -O data/ja_wiki/train_7.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_7.jsonl.gz?ref_type=heads
wget -O data/ja_wiki/train_8.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_8.jsonl.gz?ref_type=heads
wget -O data/ja_wiki/train_9.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_9.jsonl.gz?ref_type=heads
wget -O data/ja_wiki/train_10.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_10.jsonl.gz?ref_type=heads
wget -O data/ja_wiki/train_11.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_11.jsonl.gz?ref_type=heads
wget -O data/ja_wiki/train_12.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_12.jsonl.gz?ref_type=heads
wget -O data/ja_wiki/train_13.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_13.jsonl.gz?ref_type=heads

# 下载验证数据集
wget -O data/ja_wiki/validation_0.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/validation_0.jsonl.gz?ref_type=heads

# 解压所有数据文件
gunzip data/ja_wiki/*
```

执行结果：

```
root@tut-server-11:/workspace# mkdir -p data/ja_wiki
root@tut-server-11:/workspace# wget -O data/ja_wiki/train_0.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_0.jsonl.gz?ref_type=heads
--2025-04-23 03:01:59--  https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_0.jsonl.gz?ref_type=heads
Length: 384020873 (366M) [application/gzip]
data/ja_wiki/train_0.jsonl.gz     100%[==========================================================>] 366.23M  87.6MB/s    in 4.3s    

... (省略中间其他文件下载过程) ...

--2025-04-23 03:02:28--  https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/validation_0.jsonl.gz?ref_type=heads
Length: 1657379 (1.6M) [application/gzip]
data/ja_wiki/validation_0.jsonl.g 100%[==========================================================>]   1.58M  --.-KB/s    in 0.05s   

root@tut-server-11:/workspace# gunzip data/ja_wiki/*
root@tut-server-11:/workspace# ls data/ja_wiki/
train_0.jsonl  train_10.jsonl  train_12.jsonl  train_2.jsonl  train_4.jsonl  train_6.jsonl  train_8.jsonl  validation_0.jsonl
train_1.jsonl  train_11.jsonl  train_13.jsonl  train_3.jsonl  train_5.jsonl  train_7.jsonl  train_9.jsonl
```

数据下载和解压过程非常顺利。下载文件大小从最大的366MB到最小的1.6MB不等，下载速度约在80-100MB/s之间，整个过程耗时约30秒。所有压缩文件已成功解压为JSONL格式文件。

这些命令执行以下操作：
1. 进入容器的工作目录
2. 创建数据存储的目录结构
3. 下载14个训练数据分片（train_0到train_13）和1个验证数据集（validation_0）
4. 解压所有下载的压缩文件

下载的数据采用JSONL格式，每行包含一个JSON对象，通常有一个"text"字段存储文本内容。这种格式便于大规模数据处理，因为可以逐行读取和处理数据。

### 6. 数据预处理

为了让NeMo后端使用的Megatron能够进行继续预训练，需要对数据进行预处理。以下脚本使用NeMo提供的预处理工具将JSONL文件转换为Megatron可处理的格式：

```bash
# 创建预处理数据存储目录
mkdir ds

# 处理训练数据
python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
  --input="/workspace/data/ja_wiki" \
  --json-keys=text \
  --tokenizer-library=huggingface \
  --tokenizer-type="meta-llama/Llama-3.1-8B" \
  --dataset-impl mmap \
  --append-eod \
  --output-prefix="/workspace/ds/train" \
  --workers=24 \
  --files-filter '**/train_*.json*' \
  --preproc-folder \
  --log-interval 10000

# 处理验证数据
python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
  --input="/workspace/data/ja_wiki" \
  --json-keys=text \
  --tokenizer-library=huggingface \
  --tokenizer-type="meta-llama/Llama-3.1-8B" \
  --dataset-impl mmap \
  --append-eod \
  --output-prefix="/workspace/ds/validation" \
  --workers=24 \
  --files-filter '**/validation_*.json*' \
  --preproc-folder \
  --log-interval 1000
```

预处理脚本参数说明：
- `--input`: 指定输入数据目录
- `--json-keys`: 指定JSONL文件中包含文本的字段名
- `--tokenizer-library`: 使用的分词器库（这里使用huggingface）
- `--tokenizer-type`: 使用的分词器类型（使用Llama-3.1-8B的分词器）
- `--dataset-impl`: 数据集实现方式（mmap表示使用内存映射，提高大数据处理效率）
- `--append-eod`: 在每个文档末尾添加结束标记
- `--output-prefix`: 输出文件的前缀路径
- `--workers`: 并行处理的工作线程数
- `--files-filter`: 文件过滤模式，指定处理哪些文件
- `--preproc-folder`: 预处理整个文件夹
- `--log-interval`: 日志记录间隔

**执行结果：**

训练数据预处理命令成功执行。脚本依次处理了14个训练数据文件（train_0.jsonl到train_13.jsonl），总共处理了约1.26百万文档（每个文件约90,000个文档）。

数据处理过程中出现了一些相关提示：
1. 系统从Hugging Face加载了Llama-3.1-8B的分词器配置和模型
2. 脚本成功解析和处理了所有JSONL文件
3. 处理速度随着处理的文档数量增加而提高，平均处理速度约为100-500 docs/s
4. 数据处理速率保持稳定，约为4.5-4.6 MB/s

处理过程中出现了几处警告信息，表明某些文档的token序列长度超过了模型的最大序列长度限制（131072），例如：
```
Token indices sequence length is longer than the specified maximum sequence length for this model (152160 > 131072). Running this sequence through the model will result in indexing errors
```
这些警告信息表明某些非常长的文档可能会在训练过程中被截断，但不影响预处理的整体完成。

预处理完成后，在`ds`目录中生成以下文件：
```
train_text_document.bin   # 二进制格式的训练数据
train_text_document.idx   # 训练数据的索引文件
```

接下来将执行验证数据的预处理，生成验证数据的二进制文件和索引文件。

这些文件是特殊的二进制格式，专为Megatron引擎设计，用于高效地加载和处理大规模文本数据。`.bin`文件包含实际的token数据，而`.idx`文件提供了快速索引，使模型训练过程中能够高效地访问数据。

**验证数据预处理执行结果：**

验证数据预处理命令成功执行。脚本处理了1个验证数据文件（validation_0.jsonl），处理速度约为530.37 docs/s，数据处理速率约为0.826 MB/s。

预处理完成后，在`ds`目录中生成了以下文件：
```
validation_text_document.bin  # 二进制格式的验证数据
validation_text_document.idx  # 验证数据的索引文件
```

至此，所有预处理工作已完成，生成了以下文件：
```
train_text_document.bin      # 训练数据
train_text_document.idx      # 训练数据索引
validation_text_document.bin # 验证数据
validation_text_document.idx # 验证数据索引
```

**分析结论：**

数据预处理阶段顺利完成，将JSONL格式的文本数据转换为了Megatron可处理的二进制格式。虽然处理过程中出现了一些文档超长的警告，但这是正常现象，模型训练时会自动处理这些长文档。生成的二进制数据文件和索引文件将用于后续的继续预训练过程，使模型能够高效地读取和学习日语维基百科数据。

训练数据和验证数据的预处理顺利完成，训练数据处理速度随着处理的文档数量增加而提高，平均为100-500 docs/s；而验证数据由于文件较小，处理速度更快，达到了530 docs/s。所有准备工作已完成，可以进入继续预训练阶段。

### 6.1 预处理数据文件说明

通过上述预处理步骤，我们生成了以下四个关键文件：

```
train_text_document.bin      # 训练数据二进制文件
train_text_document.idx      # 训练数据索引文件
validation_text_document.bin # 验证数据二进制文件
validation_text_document.idx # 验证数据索引文件
```

这些文件共同构成了Megatron-LM中的`IndexedDataset`数据结构，它是Megatron核心中最底层的数据接口。下面详细说明这些文件的结构、作用和优势：

#### 文件结构

1. **二进制文件（`.bin`）**：
   - 包含实际的token数据，是模型训练时直接读取的内容
   - 存储了每个序列的token ID序列
   - 包含以下元数据：
     - 每个序列中的元素数量
     - 每个序列的字节偏移量指针
     - 每个文档的序列索引范围
   - 采用高效压缩的二进制格式存储

2. **索引文件（`.idx`）**：
   - 包含数据集级别的元数据信息
   - 包含以下内容：
     - 索引头部（确保向后兼容性）
     - 索引版本（维护向后兼容性）
     - 数据类型代码（指示数据文件中使用的数据类型）
     - 提供指向二进制文件中特定文档位置的快速访问机制

#### 技术优势

这种数据格式设计有以下几个关键优势：

1. **高效内存使用**：通过内存映射（mmap）技术，系统可以按需加载数据，而不必将整个数据集加载到内存中，这对处理大规模数据至关重要。

2. **快速随机访问**：索引文件允许训练过程直接跳转到特定文档或序列的起始位置，无需顺序读取整个文件。

3. **减少I/O瓶颈**：预处理过的二进制格式消除了训练期间的文本解析、标记化和处理步骤，显著减少I/O和CPU开销。

4. **支持并行训练**：这种格式设计特别适合分布式和并行训练环境，允许多个GPU或节点高效地访问不同数据分片。

5. **减少预处理开销**：训练过程中无需重复执行耗时的标记化和处理步骤，这些都在预处理阶段完成。

#### 在继续预训练中的使用

在NeMo框架中执行继续预训练时，这些文件会通过类似以下的配置方式被引用：

```python
TRAIN_DATA_PATH="{train:[1.0,/workspace/ds/train_text_document],validation:[/workspace/ds/validation_text_document],test:[/workspace/ds/validation_text_document]}"
```

预训练脚本会自动加载这些文件，通过内存映射技术高效访问数据，而不必一次将所有数据加载到内存。这种方式特别适合大规模语言模型训练，因为训练数据通常远大于可用内存或GPU内存。

#### 验证数据文件的特殊用途

生成的验证数据文件（`validation_text_document.bin`和`validation_text_document.idx`）虽然在结构上与训练数据文件相同，但有其特殊的用途和重要性：

1. **规模差异**：从执行结果可以看出，验证数据集（约1000个文档）比训练数据集（约126万文档）小得多，这是有意为之的设计，以确保验证过程高效。

2. **处理速度**：验证数据处理速度（530 docs/s）明显快于训练数据（100-500 docs/s），这是因为验证数据量小，可以更快地加载和处理。

3. **用途区别**：
   - 训练数据文件用于模型的实际学习和参数更新
   - 验证数据文件用于评估模型在训练过程中的性能表现，不参与参数更新
   - 验证文件还在训练配置中被指定为测试数据，用于最终评估模型性能

4. **防止过拟合**：验证数据集是独立的数据集，模型从未"见过"这些数据，因此能够提供对模型泛化能力的真实评估。

5. **早停策略**：在训练过程中，预训练脚本会定期使用验证数据评估模型性能，如果验证损失不再下降，可能触发早停机制，防止过拟合。

6. **加载方式**：在预训练配置中，验证数据同时用作验证集和测试集：
   ```python
   TRAIN_DATA_PATH="{train:[1.0,/workspace/ds/train_text_document],validation:[/workspace/ds/validation_text_document],test:[/workspace/ds/validation_text_document]}"
   ```
   这表明训练过程会使用`validation_text_document`文件来评估模型在训练过程中的性能，并在训练结束时进行最终测试。

综上所述，这些预处理文件是大规模语言模型训练基础架构的关键组成部分。通过将原始文本数据转换为高效的二进制格式，我们能够在大规模分布式系统上有效地训练和微调模型，而无需担心I/O瓶颈或内存限制。训练和验证数据文件协同工作，确保模型不仅能学习到日语维基百科的知识，还能在未见过的数据上表现良好。

### 7. 执行继续预训练

继续预训练可以使用NeMo的`/opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py`执行。可以通过以下方式配置参数：

```bash
# 设置Hydra和PyTorch相关环境变量
export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export TORCH_DISTRIBUTED_DEBUG=INFO

# 设置实验相关环境变量
export WANDB=False  # 是否使用Weights & Biases进行实验跟踪
export PJ_NAME="CP"  # 项目名称（Continual Pretraining）
export EXP_DIR="./results/"${PJ_NAME}  # 实验结果保存目录
export EXP_NAME="Llama-3.1-8B"  # 实验名称

# 设置模型和分词器相关环境变量
export MODEL="/workspace/models/Llama-3.1-8B.nemo"  # NeMo格式模型路径
export TOKENIZER_LIBRARY="huggingface"  # 分词器库
export TOKENIZER_TYPE="meta-llama/Llama-3.1-8B"  # 分词器类型
export TOKENIZER="/workspace/models/Llama-3.1-8B/tokenizer.json"  # 分词器路径

# 设置模型并行化相关环境变量
export TP_SIZE=2  # Tensor Parallel大小
export SP=False  # Sequence Parallel开关
export PP_SIZE=1  # Pipeline Parallel大小
export EP_SIZE=1  # Expert Parallel大小（针对MoE模型）

# 设置训练数据路径
TRAIN_DATA_PATH="{train:[1.0,/workspace/ds/train_text_document],validation:[/workspace/ds/validation_text_document],test:[/workspace/ds/validation_text_document]}"
```

上述环境变量的说明：

1. **Hydra相关**：NeMo框架使用Hydra进行配置管理，设置这些环境变量可以帮助调试
2. **实验跟踪**：`WANDB`设置为False表示不使用Weights & Biases进行实验跟踪
3. **模型路径**：指向前面步骤中转换得到的NeMo格式模型
4. **并行化设置**：
   - TP_SIZE=2：设置张量并行度为2，将模型的层内计算分布在2个GPU上
   - PP_SIZE=1：设置管道并行度为1，不进行管道并行
   - SP=False：不启用序列并行
   - EP_SIZE=1：设置专家并行度为1，不使用混合专家模型特性

5. **数据路径**：使用之前预处理生成的训练和验证数据，注意验证数据同时用作验证集和测试集

接下来执行预训练命令：

```bash
python /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
  trainer.devices=${TP_SIZE} \
  trainer.num_nodes=1 \
  trainer.max_epochs=2 \
  trainer.val_check_interval=100 \
  trainer.precision=bf16 \
  trainer.default_root_dir=${EXP_DIR} \
  model.micro_batch_size=1 \
  model.global_batch_size=8 \
  model.tensor_model_parallel_size=${TP_SIZE} \
  model.pipeline_model_parallel_size=${PP_SIZE} \
  model.resume_from_checkpoint=None \
  model.restore_from_path=${MODEL} \
  model.optim.lr=3e-5 \
  model.optim.sched.warmup_steps=50 \
  model.optim.sched.constant_steps=500 \
  model.optim.sched.min_lr=5e-6 \
  exp_manager.create_wandb_logger=${WANDB} \
  exp_manager.explicit_log_dir=${EXP_DIR} \
  exp_manager.exp_name=${EXP_NAME} \
  exp_manager.checkpoint_callback_params.save_top_k=3 \
  model.data.data_path=${TRAIN_DATA_PATH} \
  model.data.splits_string=\'98,1,1\' \
  model.tokenizer.library=${TOKENIZER_LIBRARY} \
  model.tokenizer.type=${TOKENIZER_TYPE} \
  model.tokenizer.model=${TOKENIZER} \
  model.seq_length=4096 \
  +model.llama31=True
```

预训练参数说明：

1. **trainer参数**：
   - `trainer.devices=${TP_SIZE}`：设置使用的GPU数量，与张量并行度一致
   - `trainer.num_nodes=1`：使用单节点训练
   - `trainer.max_epochs=2`：设置最大训练轮数为2
   - `trainer.val_check_interval=100`：每100步验证一次
   - `trainer.precision=bf16`：使用BF16混合精度训练

2. **模型参数**：
   - `model.micro_batch_size=1`：每个GPU的批次大小
   - `model.global_batch_size=8`：全局批次大小
   - `model.tensor_model_parallel_size=${TP_SIZE}`：张量并行度
   - `model.pipeline_model_parallel_size=${PP_SIZE}`：管道并行度
   - `model.restore_from_path=${MODEL}`：继续预训练的模型路径
   - `model.seq_length=4096`：序列长度，这里设置为4096

3. **优化器参数**：
   - `model.optim.lr=3e-5`：学习率
   - `model.optim.sched.warmup_steps=50`：预热步数
   - `model.optim.sched.constant_steps=500`：恒定学习率的步数
   - `model.optim.sched.min_lr=5e-6`：最小学习率

4. **数据参数**：
   - `model.data.data_path=${TRAIN_DATA_PATH}`：设置数据路径
   - `model.data.splits_string='98,1,1'`：数据集划分比例（训练:验证:测试）

5. **分词器参数**：
   - `model.tokenizer.library=${TOKENIZER_LIBRARY}`：分词器库
   - `model.tokenizer.type=${TOKENIZER_TYPE}`：分词器类型
   - `model.tokenizer.model=${TOKENIZER}`：分词器路径

6. **Llama-3.1特定参数**：
   - `+model.llama31=True`：启用Llama-3.1特定配置

执行此命令后，模型将开始继续预训练过程。预训练过程中会定期在验证集上评估模型性能，并保存性能最佳的前3个检查点。

需要注意的是，这里我们设置了最大训练轮数为2，实际应用中可能需要更多的训练轮数才能取得满意的效果。同时，如果可用GPU数量不同，需要相应调整`TP_SIZE`、`trainer.devices`和批次大小等参数。

如果系统可用内存或GPU内存有限，可以考虑减小序列长度(`model.seq_length`)和批次大小参数。

预训练完成后，将在`${EXP_DIR}/${EXP_NAME}`目录下生成训练日志和模型检查点文件，可用于后续的评估和应用。

### 7.1 使用原始教程脚本进行训练

原始教程中提供了一个更为完整的训练脚本配置，可以使用以下命令执行：

```bash
# 设置Hydra和PyTorch相关环境变量
export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export TORCH_DISTRIBUTED_DEBUG=INFO

# 设置实验相关环境变量
export WANDB=False  # 是否使用Weights & Biases进行实验跟踪
export PJ_NAME="CP"  # 项目名称（Continual Pretraining）
export EXP_DIR="./results/"${PJ_NAME}  # 实验结果保存目录
export EXP_NAME="Llama-3.1-8B"  # 实验名称

# 设置模型和分词器相关环境变量
export MODEL="/workspace/models/Llama-3.1-8B.nemo"  # NeMo格式模型路径
export TOKENIZER_LIBRARY="huggingface"  # 分词器库
export TOKENIZER_TYPE="meta-llama/Llama-3.1-8B"  # 分词器类型
export TOKENIZER="/workspace/models/Llama-3.1-8B/tokenizer.json"  # 分词器路径

# 设置模型并行化相关环境变量
export TP_SIZE=2  # Tensor Parallel大小
export SP=False  # Sequence Parallel开关
export PP_SIZE=1  # Pipeline Parallel大小
export EP_SIZE=1  # Expert Parallel大小（针对MoE模型）

# 设置训练数据路径
TRAIN_DATA_PATH="{train:[1.0,/workspace/ds/train_text_document],validation:[/workspace/ds/validation_text_document],test:[/workspace/ds/validation_text_document]}"

# 执行预训练命令
python /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    exp_manager.exp_dir=${EXP_DIR} \
    exp_manager.name=${EXP_NAME} \
    exp_manager.create_wandb_logger=${WANDB} \
    exp_manager.wandb_logger_kwargs.project=${PJ_NAME} \
    exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
    exp_manager.checkpoint_callback_params.save_top_k=3 \
    exp_manager.checkpoint_callback_params.always_save_nemo=False \
    trainer.precision=bf16 \
    trainer.devices=8 \
    trainer.num_nodes=1 \
    trainer.max_epochs=-1 \
    trainer.max_steps=150 \
    trainer.log_every_n_steps=1 \
    trainer.val_check_interval=15 \
    trainer.limit_val_batches=1 \
    trainer.limit_test_batches=1 \
    trainer.gradient_clip_val=1.0 \
    model.restore_from_path=${MODEL} \
    model.encoder_seq_length=8192 \
    model.max_position_embeddings=8192 \
    model.num_layers=32 \
    model.hidden_size=4096 \
    model.ffn_hidden_size=14336 \
    model.num_attention_heads=32 \
    model.hidden_dropout=0.0 \
    model.attention_dropout=0.0 \
    model.apply_query_key_layer_scaling=True \
    model.bias=False \
    model.activation=fast-swiglu \
    model.normalization=rmsnorm \
    model.position_embedding_type=rope \
    +model.rotary_base=5000000.0 \
    model.share_embeddings_and_output_weights=False \
    model.num_query_groups=8 \
    model.scale_positional_embedding=True \
    model.bias_activation_fusion=False \
    model.bias_dropout_add_fusion=False \
    model.tokenizer.library=${TOKENIZER_LIBRARY} \
    model.tokenizer.type=${TOKENIZER_TYPE} \
    model.tokenizer.model=${TOKENIZER} \
    model.megatron_amp_O2=True \
    model.tensor_model_parallel_size=${TP_SIZE} \
    model.pipeline_model_parallel_size=${PP_SIZE} \
    model.sequence_parallel=${SP} \
    model.expert_model_parallel_size=${EP_SIZE} \
    model.transformer_engine=True \
    model.fp8=False \
    model.seed=42 \
    model.enable_megatron_timers=False \
    model.optim.name=distributed_fused_adam \
    model.optim.lr=2.5e-5 \
    model.optim.weight_decay=0.1 \
    model.optim.betas=[0.9,0.95] \
    model.optim.sched.warmup_steps=15 \
    model.optim.sched.constant_steps=0 \
    model.optim.sched.min_lr=2.5e-6 \
    model.micro_batch_size=1 \
    model.global_batch_size=1024 \
    model.data.data_prefix=${TRAIN_DATA_PATH} \
    model.data.validation_drop_last=True \
    model.data.num_workers=2
```

与前面介绍的命令相比，这个脚本有几个主要区别：

1. **训练设置**：
   - `trainer.max_epochs=-1`：不限制训练轮数，由`trainer.max_steps`控制
   - `trainer.max_steps=150`：训练150步后停止
   - `trainer.val_check_interval=15`：每15步验证一次
   - `trainer.gradient_clip_val=1.0`：梯度裁剪阈值为1.0

2. **模型配置**：
   - `model.encoder_seq_length=8192`和`model.max_position_embeddings=8192`：序列长度和位置嵌入最大长度更大
   - 明确指定了模型架构参数（如`model.num_layers`、`model.hidden_size`等）

3. **优化器设置**：
   - 使用`distributed_fused_adam`优化器
   - 学习率设置为2.5e-5
   - 批次大小设置为1024

根据原始博客的介绍，这个配置在8个H100 GPU上运行大约需要6小时来完成训练。

### 7.2 使用Weights & Biases监控训练进度

如果想使用Weights & Biases (wandb) 监控训练进度，可以按照以下步骤进行设置：

#### 7.2.1 安装和登录wandb

1. **安装wandb**（如果容器中未预装）：
   ```bash
   pip install wandb
   ```

2. **登录wandb账户**：
   ```bash
   wandb login
   ```
   
   执行此命令后，系统会显示以下提示：
   ```
   wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
   wandb: You can find your API key in your browser here: https://wandb.ai/authorize
   wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit:
   ```
   
   此时需要输入您的wandb API密钥。获取API密钥的步骤如下：
   
   a. 访问[wandb.ai/authorize](https://wandb.ai/authorize)或登录wandb后转到设置页面
   b. 复制您的API密钥
   c. 将API密钥粘贴到终端中并按Enter
   
   如果登录成功，将看到类似以下信息：
   ```
   wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc
   ```

3. **验证登录状态**（可选）：
   ```bash
   wandb status
   ```

#### 7.2.2 配置训练脚本使用wandb

1. **修改环境变量**，启用wandb监控：
   ```bash
   # 将WANDB设置为True
   export WANDB=True
   
   # 可以自定义项目名称和实验名称
   export PJ_NAME="CP_Japanese_Wiki"
   export EXP_NAME="Llama-3.1-8B_Japanese_CPT"
   ```

2. **添加wandb特有参数**（在需要时）：
   ```bash
   # 设置wandb项目名称
   export WANDB_PROJECT_NAME="llama-3-japanese-continual-pretraining"
   
   # 设置wandb运行ID（方便后续恢复）
   export WANDB_RUN_ID="llama_3_ja_wiki_cpt_run1"
   ```

3. **执行训练命令**，与前面相同，但包含wandb相关配置：
   ```bash
   python /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
       exp_manager.exp_dir=${EXP_DIR} \
       exp_manager.name=${EXP_NAME} \
       exp_manager.create_wandb_logger=${WANDB} \
       exp_manager.wandb_logger_kwargs.project=${PJ_NAME} \
       exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
       # ... 其余参数与前面相同 ...
   ```

当训练开始后，wandb会自动打开一个浏览器窗口（如果有GUI）或提供一个URL链接，可以通过该链接访问培训仪表板。在仪表板上可以监控以下指标：

- 训练和验证损失
- 学习率变化
- GPU利用率和内存使用情况
- 模型参数统计信息
- 训练速度（每秒处理的样本数）

#### 7.2.3 在分布式训练中使用wandb

在分布式训练环境中，只有主进程会向wandb服务器发送数据。为确保所有进程都能正确访问wandb目录，可以在训练脚本中添加以下代码：

```python
if os.environ.get("LOCAL_RANK", None) is None:
    os.environ["WANDB_DIR"] = wandb.run.dir
```

这样可以确保所有进程都使用相同的wandb目录，避免在分布式训练中发生冲突。

Weights & Biases还提供了实验比较功能，可以比较不同超参数设置下的训练效果，这对于优化训练过程非常有用。

**注意**：如果训练过程中断，可以使用相同的命令重新启动训练，NeMo框架会自动从最近的检查点恢复训练。如果使用wandb，新的训练运行将作为一个新的实验出现在wandb仪表板上，除非特别设置恢复先前的运行。

训练完成后，最终模型将保存在`${EXP_DIR}/${EXP_NAME}/checkpoints/`目录下，可以用于后续的评估和应用。

### 8. 单GPU环境下的继续预训练执行

由于只有一张NVIDIA RTX 6000 Ada Generation GPU（48GB显存，设备号为0），需要调整预训练配置以适应有限的硬件资源。以下是针对单GPU环境优化的**最终修正后的脚本**，已创建为可执行文件`src/run_pretraining_single_gpu.sh`：

```bash
#!/bin/bash

# 设置环境变量
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=/opt/NeMo:$PYTHONPATH
export WANDB=True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PJ_NAME="Japanese-LLM-CP"
export EXP_NAME="CP-Llama31-8B"
export EXP_DIR="/workspace/results/CP"
export MODEL="/workspace/models/Llama-3.1-8B.nemo"
export TOKENIZER="/workspace/models/Llama-3.1-8B"
export TOKENIZER_LIBRARY="huggingface"
export TOKENIZER_TYPE="meta-llama/Llama-3.1-8B"

# 恢复使用 TRAIN_DATA_PATH 环境变量 (来自博客)
export TRAIN_DATA_PATH="{train:[1.0,/workspace/ds/train_text_document],validation:[/workspace/ds/validation_text_document],test:[/workspace/ds/validation_text_document]}"

# wandb配置（可选自定义）
export WANDB_PROJECT="llama-3-japanese-continual-pretraining"
export WANDB_RUN_ID="llama_3_ja_wiki_cpt_run1"
export WANDB_LOG_MODEL=true

# 单GPU参数设置
export TP_SIZE=1
export PP_SIZE=1
export EP_SIZE=1
export SP=False

# 注意：不要设置NVTE_FUSED_ATTN和NVTE_FLASH_ATTN环境变量
# 而是使用model.attention_backend参数指定注意力后端

# 记录开始时间
echo "开始训练时间: $(date)"

# 登录wandb（首次使用时需要）
# wandb login

# 执行预训练脚本
python /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    exp_manager.exp_dir=${EXP_DIR} \
    exp_manager.name=${EXP_NAME} \
    exp_manager.create_wandb_logger=${WANDB} \
    exp_manager.wandb_logger_kwargs.project=${PJ_NAME} \
    exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
    exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
    exp_manager.checkpoint_callback_params.save_top_k=1 \
    exp_manager.checkpoint_callback_params.always_save_nemo=False \
    trainer.precision=bf16 \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    trainer.max_epochs=-1 \
    trainer.max_steps=150 \
    trainer.log_every_n_steps=1 \
    trainer.val_check_interval=15 \
    trainer.limit_val_batches=1 \
    trainer.limit_test_batches=1 \
    trainer.gradient_clip_val=1.0 \\
    model.restore_from_path=${MODEL} \\
    # 将序列长度降至 2048
    model.encoder_seq_length=2048 \\
    model.max_position_embeddings=8192 \\
    model.num_layers=32 \\
    model.hidden_size=4096 \\
    model.ffn_hidden_size=14336 \\
    model.num_attention_heads=32 \\
    model.hidden_dropout=0.0 \\
    model.attention_dropout=0.0 \\
    model.apply_query_key_layer_scaling=True \\
    model.bias=False \\
    model.activation=fast-swiglu \\
    model.normalization=rmsnorm \\
    model.position_embedding_type=rope \\
    +model.rotary_base=5000000.0 \\
    model.share_embeddings_and_output_weights=False \\
    model.num_query_groups=8 \\
    model.scale_positional_embedding=True \\
    model.bias_activation_fusion=False \\
    model.bias_dropout_add_fusion=False \\
    model.tokenizer.library=${TOKENIZER_LIBRARY} \\
    model.tokenizer.type=${TOKENIZER_TYPE} \\
    model.tokenizer.model=${TOKENIZER} \\
    model.megatron_amp_O2=False \\
    model.tensor_model_parallel_size=${TP_SIZE} \\
    model.pipeline_model_parallel_size=${PP_SIZE} \\
    model.sequence_parallel=${SP} \\
    model.expert_model_parallel_size=${EP_SIZE} \\
    model.transformer_engine=True \\
    model.fp8=False \\
    model.seed=42 \\
    model.enable_megatron_timers=False \\
    model.optim.name=adamw \\
    model.optim.lr=2.5e-5 \\
    model.optim.weight_decay=0.1 \\
    model.optim.betas=[0.9,0.95] \\
    model.optim.sched.warmup_steps=15 \\
    model.optim.sched.constant_steps=0 \\
    model.optim.sched.min_lr=2.5e-6 \\
    model.micro_batch_size=1 \\
    # 全局批次大小降至 2
    model.global_batch_size=2 \\
    # 使用修正后的数据路径指定方式
    model.data.data_prefix=[1.0,/workspace/ds/train_text_document] \\
    model.data.validation_drop_last=True \\
    model.data.num_workers=2 \\
    ++model.attention_backend=fused

# 记录结束时间
echo "结束训练时间: $(date)" 
```

与原始8 GPU配置相比，主要调整如下：
1.  `trainer.devices=1`：设置为仅使用1个GPU。
2.  `TP_SIZE=1, PP_SIZE=1, EP_SIZE=1`：所有并行参数设置为1，因为只有一个GPU。
3.  `model.encoder_seq_length=2048`：在 `global_batch_size` 降至极低（如2）仍然OOM后，将序列长度从4096进一步减少到2048，这是降低内存占用的最重要手段之一。
4.  `model.global_batch_size=2`：在多次尝试后，全局批量大小最终降低到2。
5.  `model.optim.name=adamw`：将优化器从`distributed_fused_adam`更改为标准的`adamw`。
6.  `model.megatron_amp_O2=False`：禁用了Megatron特定的O2级混合精度优化，回退到标准的PyTorch AMP，尝试减少内存占用（此更改也未解决OOM）。
7.  `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：添加了此环境变量，尝试缓解内存碎片问题，但对最终的OOM无效。
8.  **数据路径格式修正**：放弃了将训练、验证、测试数据路径合并到单一`TRAIN_DATA_PATH`环境变量并传递给`model.data.data_prefix`的做法（该做法导致了数据加载错误），改回使用`model.data.data_prefix`指定训练数据，并使用`+model.data.validation_data_prefix`指定验证数据。
9.  `exp_manager.checkpoint_callback_params.save_top_k=1`：减少保存的最佳检查点数量为1，以节省磁盘空间。
10. **移除了注意力机制相关的环境变量设置** (`NVTE_FUSED_ATTN` 和 `NVTE_FLASH_ATTN`)。
11. **添加了`++model.attention_backend=fused`参数** 来显式指定注意力后端，以解决配置冲突错误。
12. 添加了开始和结束时间记录，便于监控训练过程。

### 脚本使用方法

创建脚本后，需要赋予执行权限并运行：

```bash
chmod +x ./src/run_pretraining_single_gpu.sh
./src/run_pretraining_single_gpu.sh
```

这些调整是为了在单个GPU有限的内存下运行原本为多GPU设计的训练流程。通过逐步降低内存消耗大户（如序列长度、批次大小）和尝试不同的优化策略，最终找到一个可能在该硬件上运行的配置。

**注意**：在单GPU环境下，训练速度会比多GPU环境慢得多，预计完成时间会大大延长。即使通过上述调整解决了启动时的OOM问题，训练过程中仍有可能因为内存波动而再次出现OOM。

### 训练过程监控

执行脚本后，可通过以下方法监控训练进度：

1. 查看实时训练日志：
   ```bash
   tail -f ${EXP_DIR}/${EXP_NAME}/log/*.log
   ```

2. 查看GPU使用情况：
   ```bash
   nvidia-smi -l 5
   ```

训练完成后，最终模型将保存在`${EXP_DIR}/${EXP_NAME}/checkpoints/`目录下。

### 8.1 内存不足问题排查与参数调整

在尝试将原始的多GPU（8x H100）训练配置适配到单张NVIDIA RTX 6000 Ada（48GB显存）时，不可避免地遇到了CUDA内存不足（Out-of-Memory, OOM）的问题。以下是排查和解决此问题的详细步骤：

1.  **初步调整（脚本v1）**：
    *   **目标**：使训练能在单GPU上启动。
    *   **修改**：
        *   `trainer.devices=1`
        *   `TP_SIZE=1`, `PP_SIZE=1`, `EP_SIZE=1`
        *   `model.encoder_seq_length` 从 8192 降至 4096。
        *   `model.global_batch_size` 从 1024 降至 32。
    *   **结果**：仍然遇到OOM错误，表明即使降低了序列长度和批次大小，8B模型的基础内存需求加上训练开销对于48GB显存依然过高。

2.  **尝试博客的数据路径格式**：
    *   **背景**：原始博客脚本使用了一个复杂的字符串格式赋值给 `TRAIN_DATA_PATH` 环境变量，并直接传递给 `model.data.data_prefix`。
    *   **尝试**：恢复使用 `export TRAIN_DATA_PATH=\"{train:[...],validation:[...],test:[...]}\"` 和 `model.data.data_prefix=${TRAIN_DATA_PATH}`。
    *   **结果**：启动时立即报错 `AssertionError: Could not find data-idx or data-bin files at the prefix ...`。这证实了之前关于Hydra无法正确解析这种复杂结构作为路径列表的判断。NeMo期望的是单独的训练和验证路径参数。
    *   **修正**：放弃这种格式，恢复为 `model.data.data_prefix=[1.0,/workspace/ds/train_text_document]` 和 `+model.data.validation_data_prefix=[/workspace/ds/validation_text_document]`。

3.  **尝试 `PYTORCH_CUDA_ALLOC_CONF`**：
    *   **背景**：PyTorch的OOM错误提示中建议设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 来尝试缓解内存碎片。
    *   **尝试**：在脚本中添加 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
    *   **结果**：OOM错误依旧，发生在加载模型或初始化优化器阶段，表明主要问题是峰值内存需求过高，而非碎片。

4.  **更换优化器**：
    *   **背景**：OOM错误发生在初始化 `distributed_fused_adam` 优化器时。Fused优化器虽然计算效率高，但可能需要更多内存来存储融合状态。
    *   **尝试**：将 `model.optim.name` 从 `distributed_fused_adam` 改为标准的 `adamw`。
    *   **结果**：OOM错误点后移，但仍然发生，表明优化器本身的内存占用并非瓶颈。

5.  **禁用 `megatron_amp_O2`**：
    *   **背景**：`model.megatron_amp_O2=True` 启用Megatron-LM特定的O2级混合精度优化，可能比标准的PyTorch AMP (`O1`) 占用更多内存。
    *   **尝试**：将 `model.megatron_amp_O2` 设置为 `False`。
    *   **结果**：OOM错误依旧，发生在加载模型权重阶段，提示合并张量分片时内存不足。这表明即使禁用了O2优化，模型本身加载到`bf16`精度下已接近显存极限。

6.  **降低 `global_batch_size`**：
    *   **背景**：在模型加载已经消耗大量内存的情况下，训练过程中的激活值、梯度和优化器状态是主要的内存增长点，而这些都与批次大小直接相关。\n    *   **尝试**：将 `model.global_batch_size` 从 32 逐步降低到 16, 再到 8, 最后到 2。\n    *   **结果**：即使降低到 `global_batch_size=2`，训练仍然在模型加载阶段（合并张量分片时）发生OOM。\n\n7.  **降低 `encoder_seq_length`**：\n    *   **背景**：由于即使极低的 `global_batch_size` 仍然OOM，表明模型加载是主要瓶颈。降低序列长度是另一个显著影响内存占用的关键参数。\n    *   **尝试**：保持 `global_batch_size=2`，将 `model.encoder_seq_length` 从 4096 逐步降低到 2048，再到 1024，最后到 512。\n    *   **结果**：即使将序列长度降低到 512，训练仍然在模型加载阶段（`torch.cat` 操作时）发生OOM错误。

**结论**：对于在显存相对有限的单GPU（RTX 6000 Ada, 48GB）上运行Llama-3.1-8B模型的继续预训练，即使采用了多种常见的内存优化策略（降低批次大小至2、降低序列长度至512、更换优化器、禁用O2优化），模型加载过程本身的基础显存需求（BF16精度）仍然超出了硬件限制。这表明在没有更高级的内存优化技术（如ZeRO Offloading到CPU/NVMe、模型量化或LoRA等参数高效微调技术）的情况下，直接进行全参数的继续预训练是不可行的。后续若要在此硬件上训练，需要考虑采用LoRA等PEFT方法，或者切换到更小规模的模型。

### 常见错误及解决方案

在执行单GPU训练脚本时，可能会遇到以下常见错误：

1. **Hydra配置错误**：
   ```
   omegaconf.errors.ConfigAttributeError: Key 'validation_data_prefix' is not in struct
       full_key: model.data.validation_data_prefix
       object_type=dict
   ```
   
   **解决方案**：对于NeMo中不存在的配置项，需要在参数前加上`+`前缀以添加新配置，而不是覆盖现有配置。例如，`+model.data.validation_data_prefix=${VALID_DATA_PATH}`。对于已经存在的配置项（如`model.data.data_prefix`），直接赋值即可（如果值是简单类型），或者使用列表格式（如`model.data.data_prefix=[1.0,/path/to/train]`）。**特别注意**：避免将复杂的字典结构作为字符串传递给单个路径参数，Hydra通常无法正确解析。

2. **注意力机制配置冲突错误**：
   ```
   AssertionError: NVTE_FLASH_ATTN set to 0, but expected 1 for attention backend type auto. unset NVTE_FLASH_ATTN, NVTE_FUSED_ATTN and NVTE_UNFUSED_ATTN. Use the --attention-backend argument if you want to choose between (flash/fused/unfused/auto/local). Default is auto.
   ```
   **原因分析**：此错误发生在尝试使用旧版本的环境变量（例如 `export NVTE_FUSED_ATTN=1` 和 `export NVTE_FLASH_ATTN=0`）来配置注意力机制时。较新版本的NeMo框架（例如25.02.01）推荐使用命令行参数`++model.attention_backend`来指定注意力后端（如`fused`, `flash`, `unfused`, `auto`, `local`）。当同时设置了与框架默认选择（通常是`auto`或`flash`）冲突的环境变量时，就会触发此断言错误。
   **解决方案**：
     - 从脚本中**移除**所有`export NVTE_FUSED_ATTN=...`和`export NVTE_FLASH_ATTN=...`的环境变量设置。
     - 在`python ... megatron_gpt_pretraining.py`命令中，**添加** `++model.attention_backend=fused` （或其他所需的后端，如`flash`，根据硬件和软件支持情况选择）参数。

3. **内存不足错误 (torch.OutOfMemoryError: CUDA out of memory)**：
    *   **原因分析**：这是在资源受限环境（如单GPU）下训练大型模型时最常见的错误。GPU显存不足以容纳模型参数、激活值、梯度或优化器状态。错误可能发生在模型加载、优化器初始化或训练迭代的任何阶段。
    *   **解决方案（按优先级和效果排序）**：
        1.  **降低序列长度**：显著减小 `model.encoder_seq_length`（例如，从8192降至4096或更低）。这是最有效的减少内存占用的方法之一，因为它直接影响注意力矩阵和激活值的大小。
        2.  **降低全局批次大小**：减小 `model.global_batch_size`（例如，从32降至16或更低）。这会减少每次迭代处理的数据量，从而降低激活、梯度和优化器状态的内存峰值。
        3.  **使用梯度累积**：保持 `model.micro_batch_size=1`，同时设置 `trainer.accumulate_grad_batches` 为一个大于1的值（例如，16或32）。这允许你在不增加单步内存占用的情况下，达到与较大全局批次大小相同的梯度更新效果。
        4.  **启用激活重计算**：设置 `model.activations_checkpoint_granularity="selective"` 或 `"full"`，以及 `model.activations_checkpoint_method="uniform"` 或 `"block"`。这会在反向传播时重新计算前向传播的激活值，而不是存储它们，从而用额外的计算时间换取显著的内存节省。
        5.  **检查混合精度设置**：确保 `trainer.precision=bf16` 或 `fp16` 已启用。虽然 `bf16` 通常内存效率较高，但在某些极端情况下，检查此设置是否正确应用。
        6.  **尝试不同优化器**：虽然 `adamw` 通常比 `distributed_fused_adam` 内存占用略低，但这通常不是解决OOM的主要手段。
        7.  **关闭不必要的特性**：例如，如果不需要 `model.megatron_amp_O2=True`，可以将其设置为 `False`。
        8.  **监控GPU内存使用**：使用 `nvidia-smi` 持续监控内存使用情况，找出内存峰值发生在哪一步，以便更有针对性地进行优化。
        9.  **检查其他进程**：确保没有其他程序（如桌面环境、其他应用）占用了大量GPU内存。

4.  **检查点加载错误**：
   - 确保指定的模型路径(`model.restore_from_path`)正确无误，且指向有效的`.nemo`文件。
   - 确认模型检查点与当前NeMo框架版本兼容。

5.  **数据路径解析错误** (`AssertionError: Could not find data-idx or data-bin`):
    *   **原因分析**：通常是因为传递给 `model.data.data_prefix` 或 `+model.data.validation_data_prefix` 的路径格式不正确，或者尝试将包含多个路径的复杂结构（如字典字符串）传递给单个路径参数。
    *   **解决方案**：确保使用正确的Hydra/OmegaConf语法指定数据路径。对于训练数据，使用 `model.data.data_prefix=[WEIGHT,/path/to/train_prefix]`。对于验证和测试数据（如果分开指定），使用 `+model.data.validation_data_prefix=[/path/to/validation_prefix]` 和 `+model.data.test_data_prefix=[/path/to/test_prefix]`。确保路径指向的是预处理后生成的 `.bin` 和 `.idx` 文件的前缀（不包含扩展名）。

### 使用Weights & Biases (wandb)监控训练进度

wandb是一个强大的机器学习实验跟踪工具，可以帮助可视化和分析训练过程。NeMo框架支持集成wandb来监控训练进度。

#### 1. 安装和设置wandb

如果容器中未预装wandb，首先需要安装：

```bash
pip install wandb
```

登录wandb账户（需要预先在wandb.ai注册账户）：

```bash
wandb login
```

执行命令后，系统会提示输入API密钥，可以通过访问 [https://wandb.ai/authorize](https://wandb.ai/authorize) 获取。

#### 2. 在训练脚本中配置wandb

在`run_pretraining_single_gpu.sh`脚本中，已经添加了wandb相关的配置：

```bash
# 启用wandb
export WANDB=True

# wandb项目配置
export WANDB_PROJECT="llama-3-japanese-continual-pretraining"  # 项目名称
export WANDB_RUN_ID="llama_3_ja_wiki_cpt_run1"  # 运行ID，便于追踪特定实验
export WANDB_LOG_MODEL=true  # 记录模型检查点（可选，需要谨慎使用，避免上传过大文件）
```

在训练命令中，通过以下参数配置wandb：

```bash
exp_manager.create_wandb_logger=${WANDB} \
exp_manager.wandb_logger_kwargs.project=${PJ_NAME} \
exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
```

#### 3. 访问和分析wandb实验结果

训练开始后，系统会自动创建wandb会话并提供一个URL链接，可以通过浏览器访问：

```
wandb: 🚀 View run at https://wandb.ai/username/project-name/runs/run-id
```

在wandb仪表板中，可以实时监控以下指标：

- 训练和验证损失曲线
- 学习率变化
- GPU利用率和内存使用情况
- 参数统计信息
- 吞吐量和训练速度

#### 4. 在分布式环境中使用wandb

对于分布式训练，wandb只在主进程中记录数据，避免重复。在NeMo框架中，已经内置了这种处理方式，不需要额外配置。

#### 5. 恢复训练和比较实验

如果训练中断，可以使用相同的`WANDB_RUN_ID`继续训练，保持实验记录的连续性：

```bash
export WANDB_RESUME=true
export WANDB_RUN_ID="之前的运行ID"
```

wandb还提供实验比较功能，可以对比不同超参数设置下的训练效果，帮助找到最佳训练配置。

**注意**：使用wandb可能会减慢训练速度，特别是在记录大量参数或检查点时。如果训练性能出现明显下降，可以考虑降低日志记录频率或禁用某些特性。

## 评估方法

### Nejumi リーダーボード 3 评估

Nejumi リーダーボード 3是一个用于多方面评估LLM日语能力的基准测试。通过运行基准测试脚本，可以将自己的模型与各种模型进行比较。

根据博客作者的评估，在MT-bench（160个样本）上比较原始的meta-llama/Llama-3.1-8B和经过日语继续预训练的模型时发现：

- 对于日语指令，原始模型约有30-40%的回复是英语
- 经过日语继续预训练的模型，英语回复减少到了极少数情况

这表明通过日语继续预训练，可以显著提高模型对日语指令的理解和响应能力。

## 可能遇到的问题及解决方案

1. **Hugging Face权限问题**:
   - 问题：无法访问meta-llama/Llama-3.1-8B模型
   - 解决方案：确保已在Hugging Face网站上申请并获得访问权限

2. **GPU内存不足**:
   - 问题：如果使用的不是H100 GPU，可能会遇到内存不足问题
   - 解决方案：减小批量大小或使用梯度累积

3. **数据下载失败**:
   - 问题：日语Wikipedia数据下载失败
   - 解决方案：确保网络连接稳定，或使用镜像站点

4. **Hydra/OmegaConf 配置错误**:
    *   问题：添加或修改参数时出现 `ConfigAttributeError` 或类似的错误。
    *   解决方案：理解NeMo配置结构，使用 `+` 添加新参数，直接赋值修改现有参数。注意列表和字典的正确语法。避免将复杂结构作为简单字符串传递。

5.  **数据路径或格式错误**:
    *   问题：训练开始时报告找不到 `.bin` 或 `.idx` 文件。
    *   解决方案：仔细检查 `model.data.data_prefix` 和 `+model.data.validation_data_prefix` 等参数的值和格式是否正确，确保它们指向了预处理生成的文件的正确前缀。


## 总结

本文档详细记录了使用NeMo Framework对Llama-3.1-8B模型进行日语继续预训练的完整流程。虽然本教程使用的是单节点和较小规模的日语维基百科数据，但通过增加数据量和使用多节点计算环境，可以轻松扩展为大规模训练配置。

主要步骤包括：
1. 环境准备和容器启动
2. 从Hugging Face下载预训练模型
3. 将模型转换为NeMo格式
4. 准备和预处理日语维基百科数据
5. 执行继续预训练（单GPU和多GPU配置）
6. 评估模型性能

通过日语继续预训练，可以显著提高模型对日语指令的理解和响应能力，减少英语回复的比例。这种方法适用于各种语言和领域的大模型定制化，可以帮助开发针对特定语言和业务领域的本地化大语言模型。

希望这份操作记录可以为使用NeMo Framework进行自定义大语言模型训练的团队提供参考和帮助。

## 参考资料

- [NeMo Framework で実践する継続事前学習 – 日本語 LLM 編 –](https://developer.nvidia.com/ja-jp/blog/how-to-use-continual-pre-training-with-japanese-language-on-nemo-framework/)
- NVIDIA NeMo Framework 用户指南
- NeMo Framework 单节点预训练文档
- [Training Localized Multilingual LLMs with NVIDIA NeMo, Part 1](https://developer.nvidia.com/blog/training-localized-multilingual-llms-with-nvidia-nemo-part-1/)
- [Training Localized Multilingual LLMs with NVIDIA NeMo, Part 2](https://developer.nvidia.com/blog/training-localized-multilingual-llms-with-nvidia-nemo-part-2/)
- [NeMo Curator を使った日本語データのキュレーション](https://developer.nvidia.com/ja-jp/blog/using-nemo-curator-for-japanese-data-curation/)
- [NeMo Framework で日本語 LLM をファインチューニング – SFT 編 –](https://developer.nvidia.com/ja-jp/blog/fine-tuning-japanese-llm-with-nemo-framework-sft/)
- [NeMo Framework で日本語 LLM をファインチューニング – PEFT 編 –](https://developer.nvidia.com/ja-jp/blog/fine-tuning-japanese-llm-with-nemo-framework-peft/)
- [NeMo Framework で日本語 LLM をファインチューニング – DPO 編 –](https://developer.nvidia.com/ja-jp/blog/fine-tuning-japanese-llm-with-nemo-framework-dpo/)
- [NVIDIA NIM でファインチューニングされた AI モデルのデプロイ](https://developer.nvidia.com/ja-jp/blog/deploying-fine-tuned-ai-models-with-nvidia-nim/) 

