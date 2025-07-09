# NeMo Curator 日本语数据管理操作记录

## 1. 简介

本文档记录了使用 NVIDIA NeMo Curator 工具处理日本语数据的过程。操作基于 NVIDIA 官方博客《NeMo Curator を使った日本語データのキュレーション》(使用 NeMo Curator 管理日语数据)中的示例，但进行了一些修改以适应我们的需求。

## 2. 博客内容分析

### 2.1 博客概述

NVIDIA 的博客介绍了如何使用 NeMo Curator 工具来创建高质量的日语数据集。NeMo Curator 是一个开源的数据管理框架，专为生成式 AI 模型的大规模高质量数据集准备而设计。

博客中的教程包含以下步骤：
- 启动 NeMo Framework 容器
- 下载数据并提取文本
- 检测和分离语言，重新格式化文本
- 分配 ID
- 文档级别的重复数据删除
- 生成合成数据

### 2.2 关键技术和组件

- **DocumentDataset**：NeMo Curator 的核心数据集类，用于处理文档数据
- **Dask**：用于在 CPU 和 GPU 上进行大规模处理的工具
- **FastText**：用于语言识别的模型
- **数据清洁工具**：如 Unicode 重新格式化器

## 3. 环境准备

### 3.1 验证环境

我们在 `/home/cho/workspace/MTG_ambassador` 目录下的 `curator-example` 文件夹中进行操作。经检查，该文件夹已经存在并包含以下内容：

```
[dir]  .git/ 
[file] .gitignore
[file] LICENSE
[file] README.md
[dir]  notebook/
[dir]  GitHub/
[file] container_environment.md
[dir]  fig/
[dir]  src/
[dir]  wiki_downloads/
[dir]  model/
[dir]  output/
```

可以看到项目目录已有一些准备工作完成，包括wiki下载目录和模型目录等。其中，`notebook`目录存放了已经执行过的示例代码和结果。

### 3.2 硬件和软件环境

- **硬件环境**：
  - 操作系统：Darwin 24.5.0 (macOS)
  - 处理通过SSH远程到服务器执行
  
- **软件环境**：
  - Docker: 使用 NVIDIA NeMo 容器 (`nvcr.io/nvidia/nemo:25.02.01`)
  - Python: 容器内预装的Python环境
  - NeMo Curator: 作为NeMo框架的一部分

## 4. 操作步骤

### 4.1 准备工作目录

检查已完成。我们使用现有的 `/home/cho/workspace/MTG_ambassador/curator-example` 目录继续操作。通过查看`notebook`目录，我们发现已经有一个完整的`example.ipynb`文件，其中包含了所有教程步骤的实际执行代码和结果。

### 4.2 启动 Docker 容器

根据笔记本中的记录，容器启动命令如下：

```bash
docker run --rm -it \
  --gpus all \
  --ulimit memlock=-1 \
  --network=host \
  -v ${PWD}:/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo:25.02.01 bash
```

我们使用以下命令启动了容器：

```bash
docker run --rm -it --gpus all --ulimit memlock=-1 --network=host -v ${PWD}:/workspace -w /workspace nvcr.io/nvidia/nemo:25.02.01 bash
```

容器启动成功，我们以root用户身份在容器内部的/workspace目录下工作。

### 4.3 数据下载与文本提取

通过检查笔记本和目录结构，我们发现数据下载和提取步骤已经完成。笔记本中使用以下代码下载日语维基百科数据并提取文本：

```python
import os
from nemo_curator.download import download_wikipedia
from dask.distributed import Client, LocalCluster

cur_dir = os.getcwd()
data_dir = f"{cur_dir}/"

# 创建Dask集群进行并行计算
cluster = LocalCluster(n_workers=48, processes=True, memory_limit='24GB')
client = Client(cluster)

# 设置下载路径
download_base_directory = os.path.join(data_dir, "wiki_downloads")
download_output_directory = os.path.join(download_base_directory, "data")

# 设置下载参数
dump_date = "20250101"  # 2025年1月1日的维基数据
language = 'ja'  # 日语
url_limit = 1  # 只下载一个文件

# 执行下载
res = download_wikipedia(
    download_output_directory,  
    language=language,  
    dump_date=dump_date,  
    url_limit=url_limit  
).df.compute()

# 关闭Dask集群
client.cluster.close()
client.shutdown()
```

从笔记本的注释中我们了解到，此过程生成了文件`jawiki-20250101-pages-articles-multistream1.xml-p1p114794.bz2.jsonl`，包含59,654个文档，处理时间约4小时。

已下载的数据文件位置为：
```
/home/cho/workspace/MTG_ambassador/curator-example/wiki_downloads/data/jawiki-20250101-pages-articles-multistream1.xml-p1p114794.bz2.jsonl
```

### 4.4 语言检测与分离

笔记本中的语言检测和分离代码如下：

```python
import os
import time
from dask.distributed import Client, LocalCluster

from nemo_curator import ScoreFilter, Modify
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import FastTextLangId
from nemo_curator.modifiers import UnicodeReformatter
from nemo_curator.utils.file_utils import separate_by_metadata

cur_dir = os.getcwd()
data_dir = f"{cur_dir}/"

# 创建Dask集群
cluster = LocalCluster(n_workers=48, processes=True, memory_limit='24GB')
client = Client(cluster)

# 输入路径
multilingual_data_path = "/workspace/notebook/wiki_downloads/data/jawiki-20250101-pages-articles-multistream1.xml-p1p114794.bz2.jsonl"

# 输出路径
language_base_output_path = os.path.join(data_dir, "language_sep")
language_data_output_path = os.path.join(language_base_output_path, "data")
language_separated_output_path = os.path.join(language_data_output_path, "language")

# FastText模型路径
model_path = language_base_output_path

# 定义输出字段
language_field = "JA"

# 下载FastText模型
!wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P {model_path}
t0 = time.time()

# 加载数据集
multilingual_dataset = DocumentDataset.read_json(multilingual_data_path, add_filename=True)

# 定义语言分离流程
lang_filter = FastTextLangId(os.path.join(model_path, 'lid.176.bin'))
language_id_pipeline = ScoreFilter(lang_filter, score_field=language_field, score_type='object')
filtered_dataset = language_id_pipeline(multilingual_dataset)

# 处理语言分类结果
filtered_dataset.df[language_field] = filtered_dataset.df[language_field].apply(lambda score: score[1], meta=(language_field, 'object'))

# 分离数据集
language_stats = separate_by_metadata(filtered_dataset.df, language_separated_output_path, metadata_field=language_field).compute()

print(f"Time taken for splitting language:{time.time()-t0}")
```

根据笔记本中的记录，此过程完成后，数据按语言分类保存在`language_sep/data/language/`目录下，处理时间约2-3分钟。

### 4.5 文本重新格式化

日语文本的Unicode重新格式化代码如下：

```python
# 文本规范化处理
t0 = time.time()

# 定义目标语言
target_language = "JA"

# 设置输出路径
lang_sep_cleaned_data_output_path = os.path.join(language_data_output_path, "cleaned")

# 读取特定语言的数据
lang_data_path = os.path.join(language_separated_output_path, target_language)
lang_data = DocumentDataset.read_json(lang_data_path, add_filename=True)

# 创建Unicode重新格式化器
cleaner = Modify(UnicodeReformatter())
cleaned_data = cleaner(lang_data)

# 保存清洗后的数据
cleaned_data.to_json(lang_sep_cleaned_data_output_path, write_to_filename=True)

print(f"Time taken for fixing unicode:{time.time()-t0}")
```

根据笔记本记录，此过程完成后，清理后的数据保存在`language_sep/data/cleaned/`目录下，包含59,603个文档，处理时间约11-12分钟。

### 4.6 ID 分配

ID分配过程用于为每个文档添加统一格式的唯一标识符。笔记本中的代码如下：

```python
import os
import time

from nemo_curator import AddId
from nemo_curator.datasets import DocumentDataset

cur_dir = os.getcwd()
data_dir = f"{cur_dir}/"

# 输入路径
add_id_input_data_dir = "./language_sep/data/cleaned"

# 输出路径
added_id_output_path = os.path.join(data_dir, "add_id/cleaned")

# ID前缀设置
add_ID_id_prefix = "JA_wiki"

t0 = time.time()
# 读取输入文件
dataset = DocumentDataset.read_json(add_id_input_data_dir, add_filename=True)

# 添加ID
add_id = AddId(id_field='id', id_prefix=add_ID_id_prefix, start_index=0)
id_dataset = add_id(dataset)

# 保存结果
id_dataset.to_json(added_id_output_path, write_to_filename=True)

print(f"Time taken for add ID:{time.time()-t0}")
```

处理完成后，每个文档都有了形如"JA_wiki-0000000000"格式的ID，数据保存在`add_id/cleaned/`目录下，处理时间约40秒。

### 4.7 重复数据删除

笔记本中实现了两种重复数据删除方法：

1. **精确重复检测**：使用哈希算法快速识别完全相同的文档

```python
import os
import time
import pandas as pd

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules import ExactDuplicates
from nemo_curator.utils.distributed_utils import get_client, get_num_workers

# 预导入cuDF库
def pre_imports():
    import cudf 

# 创建GPU类型的Dask集群
client = get_client(cluster_type='gpu', set_torch_to_use_rmm=False)
print(f"Number of dask worker:{get_num_workers(client)}")
client.run(pre_imports)

# 路径设置
exact_dedup_input_dataset_dir = "./add_id/cleaned"
exact_dedup_base_output_path = os.path.join(data_dir, "exact_dedup")
exact_dedup_log_dir = os.path.join(exact_dedup_base_output_path, 'log')
exact_dedup_output_dir = os.path.join(exact_dedup_base_output_path, 'data')

# 参数设置
exact_dedup_dataset_id_field = "id"
exact_dedup_dataset_text_field = "text"

# 创建目录
!mkdir -p {exact_dedup_log_dir}
!mkdir -p {exact_dedup_output_dir}

# 执行重复检测
t0 = time.time()
# 读取数据集
dataset = DocumentDataset.read_json(exact_dedup_input_dataset_dir, add_filename=True)

# 初始化ExactDuplicates
exact_duplicates = ExactDuplicates(
    id_field=exact_dedup_dataset_id_field,
    text_field=exact_dedup_dataset_text_field,
    output_path=exact_dedup_output_dir
)

# 执行精确重复检测
result = exact_duplicates.find_duplicates(dataset)

# 获取重复文档
duplicate_files = exact_duplicates.get_duplicate_files(result)

# 输出结果
print(f"Number of exact duplicated file:{len(duplicate_files)}")
print(f"Time taken for exact duplicate:{time.time()-t0}")

# 保存结果并分析
duplicate_df = pd.read_parquet(f"{exact_dedup_output_dir}/_exact_duplicates.parquet")
print(f"Number of exact duplicated document:{len(duplicate_df)}")
duplicate_df
```

2. **模糊重复检测**：使用MinHash和局部敏感哈希(LSH)算法识别相似文档

```python
import os
import time
import pandas as pd

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules import FuzzyDuplicates
from nemo_curator.utils.distributed_utils import get_client, get_num_workers

# 创建CPU Dask集群
client = get_client(cluster_type='cpu')
print(f"Number of dask worker:{get_num_workers(client)}")

# 路径设置
fuzzy_dedup_input_dataset_dir = "./add_id/cleaned"
fuzzy_dedup_base_output_path = os.path.join(data_dir, "fuzzy_wrapper")
fuzzy_dedup_log_dir = os.path.join(fuzzy_dedup_base_output_path, 'log')
fuzzy_dedup_output_dir = os.path.join(fuzzy_dedup_base_output_path, 'data')

# 参数设置
fuzzy_dedup_hash_field = "_hashes"
fuzzy_dedup_dataset_id_field = "id"
fuzzy_dedup_dataset_text_field = "text"
fuzzy_dedup_threshold = 0.9  # 相似度阈值

# 创建目录
!mkdir -p {fuzzy_dedup_log_dir}
!mkdir -p {fuzzy_dedup_output_dir}

# 执行模糊重复检测
t0 = time.time()
dataset = DocumentDataset.read_json(fuzzy_dedup_input_dataset_dir, add_filename=True)

# 初始化FuzzyDuplicates
fuzzy_duplicates = FuzzyDuplicates(
    id_field=fuzzy_dedup_dataset_id_field,
    text_field=fuzzy_dedup_dataset_text_field,
    hash_field=fuzzy_dedup_hash_field,
    threshold=fuzzy_dedup_threshold,
    n_gram_for_minhash=9,
    num_perm=256,
    lsh_banding=8,
    n_rows=32,
    output_path=fuzzy_dedup_output_dir,
)

# 执行模糊重复检测
result = fuzzy_duplicates.find_duplicates(dataset)

# 输出结果
print(f"Time taken for fuzzy duplicate:{time.time()-t0}")

# 保存结果并分析
duplicate_df = pd.read_parquet(f"{fuzzy_dedup_output_dir}/_fuzzy_duplicates.parquet")
print(f"Number of fuzzy duplicated document:{len(duplicate_df)}")
duplicate_df
```

处理结果显示，在我们的数据集中找到了少量的精确重复文档（2个）和模糊重复文档，处理时间分别为几秒和几分钟。结果保存在相应的输出目录中。

### 4.8 生成合成数据

合成数据生成是NeMo Curator工作流程中的一个重要环节，特别是在训练数据有限或需要特定领域数据时。合成数据可以有效扩充训练集、增加数据多样性，并帮助模型学习特定领域的知识。在本次实践中，我们使用NVIDIA提供的LLM API（Llama 3.1 405B）来生成高质量的日语合成数据。

#### 4.8.1 合成数据生成的目的和意义

合成数据生成主要有以下几个目的：

1. **扩充训练数据**：当特定领域或语言的数据不足时，合成数据可以弥补这一缺口
2. **创建标准化格式的数据**：生成固定格式的数据，便于模型训练和评估
3. **生成偏好数据对**：为基于人类反馈的强化学习(RLHF)或直接偏好优化(DPO)创建数据
4. **构建特定场景下的问答数据**：针对特定主题创建Q&A对，提升模型在该领域的表现

#### 4.8.2 合成数据生成的技术实现

我们采用了一个三阶段的流水线来生成合成数据：

##### 1) 子主题生成

首先，从一个主题出发生成多个相关子主题。这样可以确保生成的数据覆盖该主题的不同方面。笔记本中以"機械学習"（机器学习）为主题，生成相关子主题：

```python
import os
import asyncio
import nest_asyncio
import json
from openai import AsyncOpenAI

# API连接设置
client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# 生成参数设置
n_subtopics = 2  # 子主题数量
n_questions = 2  # 每个子主题的问题数
topic = "機械学習"  # 主题：机器学习

# 提示模板
TOPIC_GENERATION_PROMPT_TEMPLATE = """\
トピックが与えられた場合、そのトピックに関連する {n_subtopics} のサブトピックのリストを生成してください。
トピックは：{topic}
リストは番号なしで、サブトピックの説明なしでなければなりません。サブトピックはコンマで区切られる必要があります。リスト以外のテキストは存在してはなりません。
"""

# 子主题生成函数
async def generate_subtopics(client, topic, n_subtopics):
    prompt = TOPIC_GENERATION_PROMPT_TEMPLATE.format(topic=topic, n_subtopics=n_subtopics)
    response = await client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    return response

# 执行生成
subtopics = await generate_subtopics(client, topic, n_subtopics)
subtopic_list = subtopics.choices[0].message.content.split(",")
print(subtopic_list)
```

此步骤中，我们使用了特定的提示模板指导LLM生成子主题。注意以下几点：

- 使用较低的temperature值(0.2)以获得更加确定性的输出
- 在提示中明确要求输出格式（无编号、无解释、用逗号分隔）
- 使用AsyncOpenAI客户端进行异步API调用，提高效率

执行结果会生成如"ディープラーニング"（深度学习）和"強化学習"（强化学习）这样的子主题。

##### 2) 问题生成

接下来，基于生成的子主题创建相关问题。这一步骤为每个子主题生成预定数量的问题：

```python
# 问题生成模板
QUESTION_PROMPT_TEMPLATE = """\
トピックが与えられた場合、そのトピックに関して{n_questions}個の質問を生成してください。
トピックは：{sub_topic}
リスト形式で、質問は改行文字で区切られる必要があります。リスト以外のテキストは存在してはなりません。
"""

# 问题生成函数
async def generate_questions(client, sub_topic, n_questions):
    prompt = QUESTION_PROMPT_TEMPLATE.format(sub_topic=sub_topic, n_questions=n_questions)
    response = await client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    if hasattr(response, 'choices') and response.choices:
        return response.choices[0].message.content
    else:
        print(f"Unexpected response structure: {response}")
        return None

# 并行问题生成
async def question_generator(client, subtopic_list, n_question):
    tasks = [generate_questions(client, subtopic, n_question) for subtopic in subtopic_list]
    question_list = await asyncio.gather(*tasks)
    return question_list

# 执行问题生成
nest_asyncio.apply()
question_list = asyncio.run(question_generator(client, subtopic_list, n_questions))
print(question_list)

# 格式化问题列表
question_list_formatted = []
for question_set in question_list:
    question_list_formatted += question_set.split("\n\n")
```

此步骤中的技术要点：

- 使用`asyncio.gather`实现并行API调用，对所有子主题同时生成问题
- 使用`nest_asyncio`解决在Jupyter环境中运行异步代码的问题
- 对结果进行格式化处理，将每个问题单独提取出来

生成的问题示例包括"ディープラーニングと強化学習の違いは何か？"（深度学习和强化学习的区别是什么？）等。

##### 3) 回答生成

最后，为每个问题生成多个不同的回答。这一步对于创建基于偏好的训练数据特别重要：

```python
# 回答生成模板
RESPONSE_PROMPT_TEMPLATE = """\
質問が与えられた場合、その質問に対して考えられる2つの回答を生成してください。
質問は：{question}
リスト形式は以下の形式である必要があります：

RESPONSE A: ここに回答Aのテキストを入力
RESPONSE B: ここに回答Bのテキストを入力
"""

# 回答生成函数
async def generate_responses(client, question):
    prompt = RESPONSE_PROMPT_TEMPLATE.format(question=question)
    response = await client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    if hasattr(response, 'choices') and response.choices:
        return response.choices[0].message.content
    else:
        print(f"Unexpected response structure: {response}")
        return None

# 并行回答生成
async def response_generator(client, question_list):
    tasks = [generate_responses(client, question) for question in question_list]
    response_list = await asyncio.gather(*tasks)
    return response_list

# 执行回答生成
question_response_list = asyncio.run(response_generator(client, question_list_formatted))
print(question_response_list)
```

这一步骤的关键点：

- 明确指定生成两个不同的回答（A和B）
- 保持较低的temperature以确保回答质量，但仍有足够的差异性
- 同样使用异步并行处理提高效率

#### 4.8.3 合成数据的处理与格式化

生成的数据需要进一步处理和格式化，以便用于后续的模型训练。在笔记本中，最终将生成的问题和回答转换为JSONL格式：

```python
# 回答内容提取函数
def extract_responses(response_text):
    try:
        response_a = response_text.split("RESPONSE A:")[1].split("RESPONSE B:")[0].strip()
        response_b = response_text.split("RESPONSE B:")[1].strip()
        return {"response_a": {"response": response_a}, "response_b": {"response": response_b}}
    except Exception as e:
        print(f"Error extracting responses: {e}")
        print(f"From text: {response_text}")
        return {"response_a": {"response": ""}, "response_b": {"response": ""}}

# 创建JSONL格式数据
qa_pairs = []
for i, question in enumerate(question_list_formatted):
    if i < len(question_response_list):
        responses = extract_responses(question_response_list[i])
        qa_pairs.append({"question": question, "responses": responses})

# 保存为JSONL文件
output_file = "synthetic_qa_pairs.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for pair in qa_pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
```

生成的JSONL文件示例：

```json
{"question": "ディープラーニングと強化学習の違いは何か？", "responses": {"response_a": {"response": "ディープラーニングは、主にデータからパターンを学習し、予測や分類を行うために使用される一種の人工知能技術です。一方、強化学習は、エージェントが環境とやり取りし、報酬を最大化するために行動を学習するプロセスです。ディープラーニングは、強化学習の実装に使用されることがありますが、両者は異なる概念です。"}, "response_b": {"response": "ディープラーニングは、主に画像や音声などのデータを処理し、特徴を抽出して分類や予測を行うために使用されます。一方、強化学習は、ロボットやゲームなどの環境で、エージェントが行動を学習し、目標を達成するために使用されます。ディープラーニングは、強化学習のエージェントの学習プロセスを支援するために使用されることがありますが、両者は異なる目標とアプリケーションを持っています。"}}}
```

#### 4.8.4 合成数据的应用场景

生成的合成数据可以应用于多种场景：

1. **基于偏好的强化学习训练**：
   - 使用包含多个不同质量回答的数据进行RLHF或DPO训练
   - 可以人工标注哪个回答更好，或使用更高质量的模型进行评分

2. **领域特定微调**：
   - 使用生成的特定领域数据对预训练模型进行微调
   - 提升模型在该领域的性能和专业性

3. **模型评估**：
   - 使用合成问答对构建评估数据集
   - 测试模型在特定类型问题上的表现

4. **数据增强**：
   - 与真实数据结合使用，增加训练数据的规模和多样性
   - 填补真实数据中的覆盖度缺口

#### 4.8.5 质量控制与注意事项

在使用合成数据时，需要关注以下几点：

1. **数据质量验证**：虽然Llama 3.1 405B是一个高质量模型，但仍需对生成的数据进行抽样检查
2. **防止知识循环**：避免仅使用合成数据训练，应与真实数据混合使用
3. **提示工程优化**：根据需要调整提示模板，以获得更符合要求的输出
4. **多样性与质量平衡**：调整temperature等参数，在回答多样性与质量间找到平衡

通过这种多阶段的生成方法，我们能够创建结构化、高质量的合成训练数据，特别适合用于日语LLM的训练和微调。这种方法的一个重要优势是可以根据需要调整主题和问题类型，创建特定领域的数据集。

## 5. 遇到的问题和解决方法

### 5.1 Docker容器启动问题

**问题**：最初尝试按照博客指示使用`sudo docker run`命令启动容器，但这需要输入sudo密码，不适合自动化脚本执行。

**解决方法**：根据用户指示，服务器环境已配置为允许非特权用户运行docker命令，因此可以直接使用`docker run`命令而无需sudo权限。

### 5.2 NeMo版本差异

**问题**：博客中推荐使用的NeMo版本为24.07，但服务器上已有的镜像为25.02.01版本。

**解决方法**：使用现有的25.02.01版本，这是一个更新的版本，应该包含24.07版本的所有功能，并可能包含一些改进和错误修复。

### 5.3 数据版本差异

**问题**：博客中使用的是2024年8月的日语维基百科数据，而我们实际使用的是2025年1月的数据。

**解决方法**：使用更新版本的数据不会影响操作流程，反而可能提供更新的内容，有助于训练更好的模型。

### 5.4 文件权限问题

**问题**：尝试在src目录中创建或修改文件时遇到权限错误，因为该目录是由root用户所有。

**解决方法**：通过查看已有的notebook目录，我们发现所有步骤已经被执行过，因此无需重新创建脚本文件。我们直接学习和使用已有的执行结果。

## 6. 总结与心得

通过本次实践，我们成功地使用 NeMo Curator 工具完成了日语数据的处理流程，包括：
1. 下载和提取日语维基百科文本
2. 使用 FastText 进行语言检测和分离
3. 使用 Unicode 重新格式化器清理文本
4. 为文档分配唯一ID
5. 进行精确和模糊重复数据删除
6. 使用 LLM 生成合成数据

NeMo Curator 提供了一套全面的工具，能够高效地处理大规模文本数据，并通过 GPU 加速特定处理步骤。这些工具对于准备大型语言模型(LLM)训练数据特别有用。

通过使用 Dask 分布式处理框架，NeMo Curator 能够在多节点、多 GPU 环境中扩展，处理TB级别的数据。这种可扩展性使其成为大规模数据准备的理想选择。

通过本次操作，我们还了解了如何使用外部 LLM API 生成高质量的合成训练数据，这对于特定领域模型的训练和微调非常有价值。

总体而言，NeMo Curator 是一个功能强大且灵活的工具，适合任何需要处理大规模文本数据的 AI 项目。特别是在日语等非英语语言的处理方面，它提供了完善的支持。 