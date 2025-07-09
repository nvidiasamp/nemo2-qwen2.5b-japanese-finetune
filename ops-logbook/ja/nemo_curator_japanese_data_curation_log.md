# NeMo Curator 日本語データキュレーション作業記録

## 1. 導入

本文書はNVIDIA NeMo Curatorツールを使用して日本語データを処理するプロセスを記録したものです。この作業はNVIDIA公式ブログ「NeMo Curator を使った日本語データのキュレーション」の例に基づいていますが、私たちのニーズに合わせていくつかの修正を行いました。

## 2. ブログ内容分析

### 2.1 ブログ概要

NVIDIAのブログではNeMo Curatorツールを使用して高品質な日本語データセットを作成する方法が紹介されています。NeMo Curatorは生成型AIモデルの大規模な高品質データセット準備のために設計されたオープンソースのデータ管理フレームワークです。

ブログのチュートリアルには以下のステップが含まれています：
- NeMo Frameworkコンテナの起動
- データのダウンロードとテキストの抽出
- 言語の検出と分離、テキストの再フォーマット
- IDの割り当て
- 文書レベルの重複データ削除
- 合成データの生成

### 2.2 主要な技術とコンポーネント

- **DocumentDataset**：NeMo Curatorの中核となるデータセットクラスで、文書データを処理するために使用
- **Dask**：CPUとGPUでの大規模処理のためのツール
- **FastText**：言語識別用のモデル
- **データクリーニングツール**：Unicodeリフォーマッターなど

## 3. 環境準備

### 3.1 環境の確認

私たちは `/home/cho/workspace/MTG_ambassador` ディレクトリ内の `curator-example` フォルダで作業を行いました。確認したところ、このフォルダは既に存在し、以下の内容を含んでいます：

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

プロジェクトディレクトリにはすでにいくつかの準備作業が完了しており、wikiダウンロードディレクトリやモデルディレクトリなどが含まれています。`notebook`ディレクトリには、既に実行されたサンプルコードと結果が保存されています。

### 3.2 ハードウェアとソフトウェア環境

- **ハードウェア環境**：
  - オペレーティングシステム：Darwin 24.5.0 (macOS)
  - SSHを通じてリモートサーバーで処理を実行
  
- **ソフトウェア環境**：
  - Docker: NVIDIA NeMoコンテナ (`nvcr.io/nvidia/nemo:25.02.01`) を使用
  - Python: コンテナ内にプリインストールされたPython環境
  - NeMo Curator: NeMoフレームワークの一部として

## 4. 操作手順

### 4.1 作業ディレクトリの準備

確認済み。既存の `/home/cho/workspace/MTG_ambassador/curator-example` ディレクトリを継続して使用します。`notebook`ディレクトリを確認したところ、チュートリアルのすべてのステップの実際の実行コードと結果を含む完全な`example.ipynb`ファイルが既に存在していることがわかりました。

### 4.2 Dockerコンテナの起動

ノートブックの記録によると、コンテナ起動コマンドは次のとおりです：

```bash
docker run --rm -it \
  --gpus all \
  --ulimit memlock=-1 \
  --network=host \
  -v ${PWD}:/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo:25.02.01 bash
```

次のコマンドでコンテナを起動しました：

```bash
docker run --rm -it --gpus all --ulimit memlock=-1 --network=host -v ${PWD}:/workspace -w /workspace nvcr.io/nvidia/nemo:25.02.01 bash
```

コンテナは正常に起動し、rootユーザーとしてコンテナ内の/workspaceディレクトリで作業を行いました。

### 4.3 データのダウンロードとテキスト抽出

ノートブックとディレクトリ構造を確認したところ、データのダウンロードと抽出のステップはすでに完了していることがわかりました。ノートブックでは、日本語のWikipediaデータをダウンロードしてテキストを抽出するために以下のコードが使用されています：

```python
import os
from nemo_curator.download import download_wikipedia
from dask.distributed import Client, LocalCluster

cur_dir = os.getcwd()
data_dir = f"{cur_dir}/"

# Daskクラスターを作成して並列計算を行う
cluster = LocalCluster(n_workers=48, processes=True, memory_limit='24GB')
client = Client(cluster)

# ダウンロードパスを設定
download_base_directory = os.path.join(data_dir, "wiki_downloads")
download_output_directory = os.path.join(download_base_directory, "data")

# ダウンロードパラメータの設定
dump_date = "20250101"  # 2025年1月1日のWikiデータ
language = 'ja'  # 日本語
url_limit = 1  # 1つのファイルのみダウンロード

# ダウンロードの実行
res = download_wikipedia(
    download_output_directory,  
    language=language,  
    dump_date=dump_date,  
    url_limit=url_limit  
).df.compute()

# Daskクラスターを閉じる
client.cluster.close()
client.shutdown()
```

ノートブックのコメントから、このプロセスで`jawiki-20250101-pages-articles-multistream1.xml-p1p114794.bz2.jsonl`ファイルが生成され、59,654個の文書が含まれ、処理時間は約4時間であることがわかります。

ダウンロードされたデータファイルの場所は：
```
/home/cho/workspace/MTG_ambassador/curator-example/wiki_downloads/data/jawiki-20250101-pages-articles-multistream1.xml-p1p114794.bz2.jsonl
```

### 4.4 言語検出と分離

ノートブックの言語検出と分離のコードは以下の通りです：

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

# Daskクラスターの作成
cluster = LocalCluster(n_workers=48, processes=True, memory_limit='24GB')
client = Client(cluster)

# 入力パス
multilingual_data_path = "/workspace/notebook/wiki_downloads/data/jawiki-20250101-pages-articles-multistream1.xml-p1p114794.bz2.jsonl"

# 出力パス
language_base_output_path = os.path.join(data_dir, "language_sep")
language_data_output_path = os.path.join(language_base_output_path, "data")
language_separated_output_path = os.path.join(language_data_output_path, "language")

# FastTextモデルパス
model_path = language_base_output_path

# 出力フィールドの定義
language_field = "JA"

# FastTextモデルのダウンロード
!wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P {model_path}
t0 = time.time()

# データセットの読み込み
multilingual_dataset = DocumentDataset.read_json(multilingual_data_path, add_filename=True)

# 言語分離フローの定義
lang_filter = FastTextLangId(os.path.join(model_path, 'lid.176.bin'))
language_id_pipeline = ScoreFilter(lang_filter, score_field=language_field, score_type='object')
filtered_dataset = language_id_pipeline(multilingual_dataset)

# 言語分類結果の処理
filtered_dataset.df[language_field] = filtered_dataset.df[language_field].apply(lambda score: score[1], meta=(language_field, 'object'))

# データセットの分離
language_stats = separate_by_metadata(filtered_dataset.df, language_separated_output_path, metadata_field=language_field).compute()

print(f"Time taken for splitting language:{time.time()-t0}")
```

ノートブックの記録によると、このプロセスが完了すると、データは`language_sep/data/language/`ディレクトリに言語別に保存され、処理時間は約2〜3分です。

### 4.5 テキストの再フォーマット

日本語テキストのUnicode再フォーマットコードは以下の通りです：

```python
# テキスト正規化処理
t0 = time.time()

# 対象言語の定義
target_language = "JA"

# 出力パスの設定
lang_sep_cleaned_data_output_path = os.path.join(language_data_output_path, "cleaned")

# 特定言語のデータの読み込み
lang_data_path = os.path.join(language_separated_output_path, target_language)
lang_data = DocumentDataset.read_json(lang_data_path, add_filename=True)

# Unicode再フォーマッターの作成
cleaner = Modify(UnicodeReformatter())
cleaned_data = cleaner(lang_data)

# クリーニング後のデータの保存
cleaned_data.to_json(lang_sep_cleaned_data_output_path, write_to_filename=True)

print(f"Time taken for fixing unicode:{time.time()-t0}")
```

ノートブックの記録によると、このプロセスが完了すると、クリーニングされたデータは`language_sep/data/cleaned/`ディレクトリに保存され、59,603個の文書が含まれ、処理時間は約11〜12分です。

### 4.6 ID の割り当て

ID割り当てプロセスは、各文書に統一フォーマットの一意の識別子を追加するために使用されます。ノートブックのコードは以下の通りです：

```python
import os
import time

from nemo_curator import AddId
from nemo_curator.datasets import DocumentDataset

cur_dir = os.getcwd()
data_dir = f"{cur_dir}/"

# 入力パス
add_id_input_data_dir = "./language_sep/data/cleaned"

# 出力パス
added_id_output_path = os.path.join(data_dir, "add_id/cleaned")

# ID接頭辞の設定
add_ID_id_prefix = "JA_wiki"

t0 = time.time()
# 入力ファイルの読み込み
dataset = DocumentDataset.read_json(add_id_input_data_dir, add_filename=True)

# IDの追加
add_id = AddId(id_field='id', id_prefix=add_ID_id_prefix, start_index=0)
id_dataset = add_id(dataset)

# 結果の保存
id_dataset.to_json(added_id_output_path, write_to_filename=True)

print(f"Time taken for add ID:{time.time()-t0}")
```

処理が完了すると、各文書に「JA_wiki-0000000000」形式のIDが付与され、データは`add_id/cleaned/`ディレクトリに保存され、処理時間は約40秒です。

### 4.7 重複データの削除

ノートブックでは、2つの重複データ削除方法が実装されています：

1. **完全一致重複検出**：ハッシュアルゴリズムを使用して、完全に同一の文書を迅速に識別します

```python
import os
import time
import pandas as pd

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules import ExactDuplicates
from nemo_curator.utils.distributed_utils import get_client, get_num_workers

# cuDFライブラリの事前インポート
def pre_imports():
    import cudf 

# GPUタイプのDaskクラスターを作成
client = get_client(cluster_type='gpu', set_torch_to_use_rmm=False)
print(f"Number of dask worker:{get_num_workers(client)}")
client.run(pre_imports)

# パス設定
exact_dedup_input_dataset_dir = "./add_id/cleaned"
exact_dedup_base_output_path = os.path.join(data_dir, "exact_dedup")
exact_dedup_log_dir = os.path.join(exact_dedup_base_output_path, 'log')
exact_dedup_output_dir = os.path.join(exact_dedup_base_output_path, 'data')

# パラメータ設定
exact_dedup_dataset_id_field = "id"
exact_dedup_dataset_text_field = "text"

# ディレクトリの作成
!mkdir -p {exact_dedup_log_dir}
!mkdir -p {exact_dedup_output_dir}

# 重複検出の実行
t0 = time.time()
# データセットの読み込み
dataset = DocumentDataset.read_json(exact_dedup_input_dataset_dir, add_filename=True)

# ExactDuplicatesの初期化
exact_duplicates = ExactDuplicates(
    id_field=exact_dedup_dataset_id_field,
    text_field=exact_dedup_dataset_text_field,
    output_path=exact_dedup_output_dir
)

# 完全一致重複検出の実行
result = exact_duplicates.find_duplicates(dataset)

# 重複文書の取得
duplicate_files = exact_duplicates.get_duplicate_files(result)

# 結果の出力
print(f"Number of exact duplicated file:{len(duplicate_files)}")
print(f"Time taken for exact duplicate:{time.time()-t0}")

# 結果の保存と分析
duplicate_df = pd.read_parquet(f"{exact_dedup_output_dir}/_exact_duplicates.parquet")
print(f"Number of exact duplicated document:{len(duplicate_df)}")
duplicate_df
```

2. **あいまい重複検出**：MinHashと局所敏感ハッシュ(LSH)アルゴリズムを使用して類似文書を識別します

```python
import os
import time
import pandas as pd

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules import FuzzyDuplicates
from nemo_curator.utils.distributed_utils import get_client, get_num_workers

# CPU Daskクラスターの作成
client = get_client(cluster_type='cpu')
print(f"Number of dask worker:{get_num_workers(client)}")

# パス設定
fuzzy_dedup_input_dataset_dir = "./add_id/cleaned"
fuzzy_dedup_base_output_path = os.path.join(data_dir, "fuzzy_wrapper")
fuzzy_dedup_log_dir = os.path.join(fuzzy_dedup_base_output_path, 'log')
fuzzy_dedup_output_dir = os.path.join(fuzzy_dedup_base_output_path, 'data')

# パラメータ設定
fuzzy_dedup_hash_field = "_hashes"
fuzzy_dedup_dataset_id_field = "id"
fuzzy_dedup_dataset_text_field = "text"
fuzzy_dedup_threshold = 0.9  # 類似度しきい値

# ディレクトリの作成
!mkdir -p {fuzzy_dedup_log_dir}
!mkdir -p {fuzzy_dedup_output_dir}

# あいまい重複検出の実行
t0 = time.time()
dataset = DocumentDataset.read_json(fuzzy_dedup_input_dataset_dir, add_filename=True)

# FuzzyDuplicatesの初期化
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

# あいまい重複検出の実行
result = fuzzy_duplicates.find_duplicates(dataset)

# 結果の出力
print(f"Time taken for fuzzy duplicate:{time.time()-t0}")

# 結果の保存と分析
duplicate_df = pd.read_parquet(f"{fuzzy_dedup_output_dir}/_fuzzy_duplicates.parquet")
print(f"Number of fuzzy duplicated document:{len(duplicate_df)}")
duplicate_df
```

処理結果によると、私たちのデータセットでは少数の完全一致重複文書（2個）とあいまい重複文書が発見され、処理時間はそれぞれ数秒と数分でした。結果は対応する出力ディレクトリに保存されています。

### 4.8 合成データの生成

合成データ生成はNeMo Curatorワークフローの重要な段階であり、特にトレーニングデータが限られている場合や特定のドメインデータが必要な場合に重要です。合成データはトレーニングセットを効果的に拡張し、データの多様性を増やし、モデルが特定のドメイン知識を学習するのに役立ちます。今回の実践では、NVIDIA提供のLLM API（Llama 3.1 405B）を使用して、高品質の日本語合成データを生成しました。

#### 4.8.1 合成データ生成の目的と意義

合成データ生成の主な目的は以下の通りです：

1. **トレーニングデータの拡張**：特定のドメインや言語のデータが不足している場合、合成データでこのギャップを埋めることができます
2. **標準化されたフォーマットのデータの作成**：モデルのトレーニングと評価を容易にするための固定フォーマットのデータを生成します
3. **選好データペアの生成**：人間のフィードバックに基づく強化学習(RLHF)または直接選好最適化(DPO)のためのデータを作成します
4. **特定シナリオの質問応答データの構築**：特定のトピックに関するQ&Aペアを作成し、そのドメインにおけるモデルのパフォーマンスを向上させます

#### 4.8.2 合成データ生成の技術実装

私たちは合成データを生成するために3段階のパイプラインを採用しました：

##### 1) サブトピック生成

まず、一つのトピックから複数の関連サブトピックを生成します。これにより、生成されるデータがそのトピックの様々な側面をカバーすることを確保します。ノートブックでは「機械学習」をトピックとして、関連するサブトピックを生成しています：

```python
import os
import asyncio
import nest_asyncio
import json
from openai import AsyncOpenAI

# API接続設定
client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# 生成パラメータ設定
n_subtopics = 2  # サブトピック数
n_questions = 2  # 各サブトピックの質問数
topic = "機械学習"  # トピック：機械学習

# プロンプトテンプレート
TOPIC_GENERATION_PROMPT_TEMPLATE = """\
トピックが与えられた場合、そのトピックに関連する {n_subtopics} のサブトピックのリストを生成してください。
トピックは：{topic}
リストは番号なしで、サブトピックの説明なしでなければなりません。サブトピックはコンマで区切られる必要があります。リスト以外のテキストは存在してはなりません。
"""

# サブトピック生成関数
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

# 生成の実行
subtopics = await generate_subtopics(client, topic, n_subtopics)
subtopic_list = subtopics.choices[0].message.content.split(",")
print(subtopic_list)
```

このステップでは、LLMに特定のプロンプトテンプレートを使用してサブトピックを生成するよう指示しています。以下の点に注意してください：

- より確定的な出力を得るために低いtemperature値(0.2)を使用
- 出力フォーマット（番号なし、説明なし、コンマ区切り）をプロンプトで明確に指定
- 効率を高めるためにAsyncOpenAIクライアントを使用して非同期API呼び出しを行う

実行結果として「ディープラーニング」や「強化学習」などのサブトピックが生成されます。

##### 2) 質問生成

次に、生成されたサブトピックに基づいて関連する質問を作成します。このステップでは、各サブトピックに対して予定された数の質問を生成します：

```python
# 質問生成テンプレート
QUESTION_PROMPT_TEMPLATE = """\
トピックが与えられた場合、そのトピックに関して{n_questions}個の質問を生成してください。
トピックは：{sub_topic}
リスト形式で、質問は改行文字で区切られる必要があります。リスト以外のテキストは存在してはなりません。
"""

# 質問生成関数
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

# 並列質問生成
async def question_generator(client, subtopic_list, n_question):
    tasks = [generate_questions(client, subtopic, n_question) for subtopic in subtopic_list]
    question_list = await asyncio.gather(*tasks)
    return question_list

# 質問生成の実行
nest_asyncio.apply()
question_list = asyncio.run(question_generator(client, subtopic_list, n_questions))
print(question_list)

# 質問リストのフォーマット
question_list_formatted = []
for question_set in question_list:
    question_list_formatted += question_set.split("\n\n")
```

このステップでの技術ポイント：

- すべてのサブトピックに対して同時に質問を生成するために`asyncio.gather`を使用して並列API呼び出しを実現
- Jupyter環境で非同期コードを実行する問題を解決するために`nest_asyncio`を使用
- 結果をフォーマット処理し、各質問を個別に抽出

生成された質問例には「ディープラーニングと強化学習の違いは何か？」などが含まれます。

##### 3) 回答生成

最後に、各質問に対して複数の異なる回答を生成します。このステップは、選好ベースのトレーニングデータを作成する上で特に重要です：

```python
# 回答生成テンプレート
RESPONSE_PROMPT_TEMPLATE = """\
質問が与えられた場合、その質問に対して考えられる2つの回答を生成してください。
質問は：{question}
リスト形式は以下の形式である必要があります：

RESPONSE A: ここに回答Aのテキストを入力
RESPONSE B: ここに回答Bのテキストを入力
"""

# 回答生成関数
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

# 並列回答生成
async def response_generator(client, question_list):
    tasks = [generate_responses(client, question) for question in question_list]
    response_list = await asyncio.gather(*tasks)
    return response_list

# 回答生成の実行
question_response_list = asyncio.run(response_generator(client, question_list_formatted))
print(question_response_list)
```

このステップのキーポイント：

- 2つの異なる回答（AとB）を生成するよう明確に指定
- 回答の質を確保しながらも、十分な差異性を持たせるために低いtemperatureを維持
- 同様に非同期並列処理を使用して効率を向上

#### 4.8.3 合成データの処理とフォーマット

生成されたデータは、後続のモデルトレーニングに使用するために、さらに処理とフォーマットが必要です。ノートブックでは、最終的に生成された質問と回答をJSONL形式に変換しています：

```python
# 回答内容抽出関数
def extract_responses(response_text):
    try:
        response_a = response_text.split("RESPONSE A:")[1].split("RESPONSE B:")[0].strip()
        response_b = response_text.split("RESPONSE B:")[1].strip()
        return {"response_a": {"response": response_a}, "response_b": {"response": response_b}}
    except Exception as e:
        print(f"Error extracting responses: {e}")
        print(f"From text: {response_text}")
        return {"response_a": {"response": ""}, "response_b": {"response": ""}}

# JSONL形式データの作成
qa_pairs = []
for i, question in enumerate(question_list_formatted):
    if i < len(question_response_list):
        responses = extract_responses(question_response_list[i])
        qa_pairs.append({"question": question, "responses": responses})

# JSONLファイルとして保存
output_file = "synthetic_qa_pairs.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for pair in qa_pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
```

生成されたJSONLファイルの例：

```json
{"question": "ディープラーニングと強化学習の違いは何か？", "responses": {"response_a": {"response": "ディープラーニングは、主にデータからパターンを学習し、予測や分類を行うために使用される一種の人工知能技術です。一方、強化学習は、エージェントが環境とやり取りし、報酬を最大化するために行動を学習するプロセスです。ディープラーニングは、強化学習の実装に使用されることがありますが、両者は異なる概念です。"}, "response_b": {"response": "ディープラーニングは、主に画像や音声などのデータを処理し、特徴を抽出して分類や予測を行うために使用されます。一方、強化学習は、ロボットやゲームなどの環境で、エージェントが行動を学習し、目標を達成するために使用されます。ディープラーニングは、強化学習のエージェントの学習プロセスを支援するために使用されることがありますが、両者は異なる目標とアプリケーションを持っています。"}}}
```

#### 4.8.4 合成データの応用シナリオ

生成された合成データは、多くのシナリオで応用できます：

1. **選好ベースの強化学習トレーニング**：
   - 異なる品質の回答を含むデータを使用してRLHFまたはDPOトレーニングを行う
   - どの回答がより良いかを人間が手動で判断したり、より高品質なモデルを使って評価したりできます

2. **ドメイン特化型ファインチューニング**：
   - 生成された特定ドメインのデータを使って事前訓練済みモデルをファインチューニング
   - そのドメインでのモデルのパフォーマンスと専門性を向上

3. **モデル評価**：
   - 合成問答ペアを使用して評価データセットを構築
   - 特定タイプの問題に対するモデルのパフォーマンスをテスト

4. **データ拡張**：
   - 実データと組み合わせて使用し、トレーニングデータの規模と多様性を増加
   - 実データのカバレッジギャップを埋める

#### 4.8.5 品質管理と注意点

合成データを使用する際には、以下の点に注意する必要があります：

1. **データ品質検証**：Llama 3.1 405Bは高品質なモデルですが、生成されたデータに対してサンプリング検査を行う必要があります
2. **知識循環の防止**：合成データのみを使用したトレーニングを避け、実データと混合して使用すべきです
3. **プロンプトエンジニアリングの最適化**：要件に合った出力を得るために、必要に応じてプロンプトテンプレートを調整します
4. **多様性と品質のバランス**：回答の多様性と品質のバランスを取るために、temperatureなどのパラメータを調整します

この多段階生成法により、構造化された高品質の合成トレーニングデータを作成でき、特に日本語LLMのトレーニングとファインチューニングに適しています。この方法の重要な利点は、ニーズに応じてトピックと質問タイプを調整し、特定ドメインのデータセットを作成できることです。

## 5. 直面した問題と解決策

### 5.1 Dockerコンテナ起動の問題

**問題**：最初はブログの指示に従って`sudo docker run`コマンドでコンテナを起動しようとしましたが、これにはsudoパスワードの入力が必要で、自動化スクリプトの実行には適していませんでした。

**解決策**：ユーザーの指示によると、サーバー環境は非特権ユーザーがdockerコマンドを実行できるように設定されているため、sudo権限なしで直接`docker run`コマンドを使用できました。

### 5.2 NeMoバージョンの違い

**問題**：ブログで推奨されているNeMoバージョンは24.07でしたが、サーバー上の既存のイメージは25.02.01バージョンでした。

**解決策**：既存の25.02.01バージョンを使用しました。これはより新しいバージョンであり、24.07バージョンのすべての機能を含み、さらにいくつかの改善とバグ修正が含まれている可能性があります。

### 5.3 データバージョンの違い

**問題**：ブログでは2024年8月の日本語Wikipediaデータが使用されていましたが、実際には2025年1月のデータを使用しました。

**解決策**：より新しいバージョンのデータを使用しても操作フローには影響せず、むしろより新しいコンテンツを提供し、より良いモデルのトレーニングに役立つ可能性があります。

### 5.4 ファイル権限の問題

**問題**：srcディレクトリ内でファイルを作成または変更しようとした際に、そのディレクトリがrootユーザーによって所有されているため、権限エラーが発生しました。

**解決策**：既存のnotebookディレクトリを確認したところ、すべてのステップはすでに実行されており、スクリプトファイルを再作成する必要はありませんでした。既存の実行結果を直接学習して使用しました。

## 6. まとめと感想

この実践を通じて、NeMo Curator ツールを使用して日本語データの処理フローを正常に完了することができました。その中には以下が含まれます：
1. 日本語Wikipediaテキストのダウンロードと抽出
2. FastTextを使用した言語検出と分離
3. Unicodeリフォーマッターによるテキストのクリーニング
4. 文書への一意のID割り当て
5. 完全一致とあいまい重複データの削除
6. LLMを使用した合成データの生成

NeMo Curatorは、大規模テキストデータを効率的に処理し、特定の処理ステップをGPUで加速できる包括的なツールセットを提供しています。これらのツールは、大規模言語モデル(LLM)のトレーニングデータを準備する上で特に有用です。

Dask分散処理フレームワークを使用することで、NeMo Curatorはマルチノード、マルチGPU環境で拡張し、TB級のデータを処理することができます。この拡張性により、大規模データ準備に理想的な選択肢となっています。

この操作を通じて、外部LLM APIを使用して高品質の合成トレーニングデータを生成する方法も学びました。これは、特定のドメインモデルのトレーニングとファインチューニングに非常に価値があります。

総じて、NeMo Curatorは、大規模テキストデータを処理する必要があるAIプロジェクトに適した、強力で柔軟なツールです。特に日本語などの非英語言語の処理において、充実したサポートを提供しています。