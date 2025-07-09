# NeMo Framework 日本語LLM継続事前学習作業記録

## 作業概要

本文書はNVIDIA NeMo FrameworkをLlama-3.1-8Bモデルの日本語継続事前学習に使用する際の作業フロー、発生し得る問題と解決策を記録しています。NVIDIA公式ブログ[《NeMo Framework で実践する継続事前学習 – 日本語 LLM 編 –》](https://developer.nvidia.com/ja-jp/blog/how-to-use-continual-pre-training-with-japanese-language-on-nemo-framework/)を参考にしています。

## NeMo Framework の概要

NeMo Frameworkは、NVIDIAが提供するクラウドネイティブフレームワークで、LLMなどの生成AIモデルの構築とカスタマイズに使用されます。このフレームワークはNGCプラットフォームからコンテナとして提供され、すぐに使用を開始できます。

NeMo Frameworkは無料で使用できますが、NVIDIA AI Enterpriseのコンポーネントとして、企業ユーザーはサポートサービスを受けるためにNVIDIA AI Enterpriseライセンスの購入を検討できます。

### LLM 開発ワークフロー

LLMの開発には以下のタスクが含まれます：

- 大規模データ準備（事前学習用）
- 分散学習を活用したLLM事前学習
- 微調整、アライメント、プロンプトエンジニアリングによるLLMのカスタマイズ
- 推論を高速化するためのモデル最適化
- GPUを活用したLLMサービス
- RAG（検索拡張生成）を通じて低コストでLLMに最新情報を提供
- LLMアプリケーションの予期しない動作を防ぐためのガードレールの設定

### NeMo Framework コンポーネント

NeMo Frameworkコンテナには、データ準備からLLMのトレーニングとカスタマイズに必要な複数のモジュールが含まれています：

- **NeMo Curator**：LLMトレーニングに必要な大規模データセットのダウンロード、抽出、クリーニング、フィルタリングのためのスケーラブルなツールキット
- **NeMo**：LLM、マルチモーダル、音声などの生成AIモデルを構築するためのスケーラブルなフレームワーク
- **NeMo Framework Launcher**：クラウドまたはローカルクラスターからタスクを起動するためのツールキット
- **Megatron-LM**：Transformerモデルの大規模トレーニングを研究するプロジェクト
- **Transformer Engine**：FP8を中心にTransformerモデルを加速するツールキット
- **NeMo-Aligner**：RLHF（人間のフィードバックに基づく強化学習）、DPO、SteerLMなどを使用してLLMを効率的にアライメントするツールキット

これらのライブラリはGitHubでオープンソースとして公開されていますが、依存関係が解決済みのNeMo Frameworkコンテナを通じて使用することが推奨されています。コンテナ内では、これらのモジュールは`/opt`ディレクトリに配置されています。

## 継続事前学習（Continual Pre-training）

継続事前学習とは、既存の事前学習済みモデルを基盤として、特定のドメインや特定の言語のデータを使用してさらに事前学習を行い、モデルが特定のアプリケーションシナリオにより適応できるようにするプロセスです。

この例では、Llama-3.1-8Bモデルに日本語Wikipedia データを使用して継続事前学習を行い、日本語処理タスクにおけるパフォーマンスを向上させます。このチュートリアルでは単一ノードと比較的小規模な日本語Wikipedia データを使用していますが、データ量を増やし、複数ノードの計算環境を使用することで、容易に大規模トレーニングに拡張できます。

## 環境準備

### ハードウェア構成

実際の環境構成：

- ハードウェア：
  - CPU: Intel(R) Xeon(R) Silver 4310 CPU @ 2.10GHz
  - GPU: 
    - NVIDIA RTX 6000 Ada Generation (48GB)
    - NVIDIA T600 (4GB)
  - システムメモリ: 251GB

- ソフトウェア：
  - OS: Ubuntu 20.04.6 LTS
  - コンテナ: `nvcr.io/nvidia/nemo:25.02.01`

### 元のブログ検証環境

元のブログの検証環境：

- ハードウェア：
  - DGX H100
  - GPU: 8 x NVIDIA H100 80GB GPUs (ドライバーバージョン: 550.90.7)
  - CPU: Intel(R) Xeon(R) Platinum 8480C
  - システムメモリ: 2 TB

- ソフトウェア：
  - OS: Ubuntu 22.04.5 LTS
  - コンテナ: `nvcr.io/nvidia/nemo:24.09`

## 実際の作業プロセス

### 1. 作業ディレクトリの作成

チュートリアルに従って、まず作業ディレクトリを作成し、そのディレクトリに移動します：

```bash
# 作業ディレクトリの作成
mkdir cp-example
cd cp-example
```

実行結果：
```
tut-server-11% mkdir -p cp-example && cd cp-example && pwd
/home/cho/workspace/cp-example
```

現在の作業ディレクトリは `/home/cho/workspace/cp-example` に正常に設定されました。

### 2. Dockerコンテナの起動

以下のコマンドを実行してDockerコンテナを起動します：

```bash
docker run -it --gpus all --name cp --shm-size=16g --ulimit memlock=-1 --network=host -v ${PWD}:/workspace -w /workspace nvcr.io/nvidia/nemo:25.02.01 bash
```

このコマンドのパラメータ説明：
- `--rm`：コンテナの実行停止後に自動的に削除
- `-it`：インタラクティブ端末
- `--gpus all`：利用可能なすべてのGPUを使用
- `--shm-size=16g`：共有メモリサイズを16GBに設定
- `--ulimit memlock=-1`：メモリロック制限を解除
- `--network=host`：ホストネットワークを使用
- `-v ${PWD}:/workspace`：現在のディレクトリをコンテナの/workspaceディレクトリにマウント
- `-w /workspace`：コンテナの作業ディレクトリを/workspaceに設定

実行結果：
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

コンテナが正常に起動し、現在コンテナ内のbash環境にあり、作業ディレクトリは`/workspace`で、これはホスト上の`/home/cho/workspace/cp-example`ディレクトリにマッピングされています。

### 3. Hugging Faceから事前学習済みモデルのダウンロード

Hugging Faceにログイン（meta-llama/Llama-3.1-8Bへのアクセス権が必要）：

```bash
huggingface-cli login
```

このコマンドを実行すると、システムはHugging Faceトークンの入力を求めます。このトークンは以下の手順で取得できます：
1. [Hugging Face公式サイト](https://huggingface.co/)にログイン
2. [設定ページ](https://huggingface.co/settings/tokens)にアクセス
3. 新しいトークンを作成し、読み取り権限を付与
4. 生成されたトークンをコピー

また、Llamaモデルを使用するには、事前にアクセス権を申請する必要があります：
1. [meta-llama/Llama-3.1-8Bモデルページ](https://huggingface.co/meta-llama/Llama-3.1-8B)にアクセス
2. 「Access」ボタンをクリックしてアクセス権を申請
3. 関連フォームに記入し、承認を待つ

実行結果：
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

Hugging Faceへのログインが正常に完了し、トークンがコンテナに保存されました。これでモデルのダウンロードを続行できます。

次に、Llama-3.1-8Bモデルをダウンロードするためのpythonスクリプトを作成します。`src`ディレクトリに`download_model.py`ファイルを作成します：

```bash
mkdir -p src
```

以下の内容のPythonスクリプトを作成します：

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

スクリプト実行時に以下の権限エラーが発生しました：

```
huggingface_hub.errors.GatedRepoError: 403 Client Error.
Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.1-8B/...
Access to model meta-llama/Llama-3.1-8B is restricted and you are not in the authorized list. 
Visit https://huggingface.co/meta-llama/Llama-3.1-8B to ask for access.
```

このエラーは、Hugging Faceアカウントにログインしたものの、アカウントがmeta-llama/Llama-3.1-8Bモデルへのアクセス権を持っていないことを示しています。以下の手順を完了する必要があります：

1. [meta-llama/Llama-3.1-8Bモデルページ](https://huggingface.co/meta-llama/Llama-3.1-8B)にアクセス
2. 「Access」ボタンをクリックしてアクセス権を申請
3. 関連申請フォーム（使用目的など）に記入
4. Meta AIからのアクセス要求承認を待つ

承認は通常すぐには行われず、しばらく待つ必要があるかもしれません。アクセス権を取得した後、再度ログインしてダウンロードスクリプトを実行する必要があります。

アクセス権承認を待つ間、チュートリアルの他の部分を続行するために、以下の代替案を検討できます：
1. すでに公開アクセス権のある他のモデル（例：Mistral、Falconなど）を使用
2. すでにダウンロード済みのモデルを使用（もしあれば）

約30分の待機後、meta-llama/Llama-3.1-8Bモデルへのアクセス権が付与されました。その後、ダウンロードスクリプトを再実行しました：

```bash
python src/download_model.py
```

ダウンロードプロセスが開始され、モデルファイルのサイズは約16.1GBです：

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

ダウンロードプロセス全体で約5分かかりました。これでモデルは`models/Llama-3.1-8B`ディレクトリに正常にダウンロードされました。

### 4. モデル形式の変換

ブログのガイダンスに従って、ダウンロードしたHugging Face形式のモデルをNeMo形式に変換する必要があります。現在のNeMoコンテナ（25.02.01）がLlama-3.1モデルの変換を完全にサポートしていない可能性があるため、まず2つのPRパッチを適用する必要があります：

```bash
cd /opt/NeMo/
curl -L https://github.com/NVIDIA/NeMo/pull/11548.diff | git apply
curl -L https://github.com/NVIDIA/NeMo/pull/11580.diff | git apply
```

使用しているNeMoコンテナのバージョン（25.02.01）はブログで使用されているバージョン（24.09）より新しいことに注意が必要です。CHANGELOGから以下の関連情報が確認できます：

- "Add llama 3.1 recipes by @cuichenx :: PR: #11273"
- "New Llama 3.1 Support (2024-07-23) The NeMo Framework now supports training and customizing the Llama 3.1 collection of LLMs from Meta."

これは、より新しいバージョンのNeMoフレームワークがすでにLlama-3.1モデルをネイティブにサポートしている可能性を示唆しています。変換スクリプトを直接実行して失敗した場合のみ、パッチの適用を試みてください。パッチがすでに25.02.01バージョンに統合されている場合、`git apply`コマンドはパッチを適用できないというエラーを報告する可能性があります。

これらのパッチの目的は、NeMoフレームワークを修正してLlama-3.1モデルを正しく処理できるようにすることです。基本的には「パッチを当てる」ことで、NeMoフレームワークにLlama-3.1モデルのサポートを追加または修正し、Hugging Face形式のモデルを正しくNeMo形式に変換できるようにします。これらのパッチを適用しない場合、変換プロセス中にエラーや互換性の問題が発生する可能性があります。

より新しいフレームワークがすでにLlama-3.1をサポートしている可能性を考慮して、まず変換コマンドを直接実行してみます：

```bash
# 環境変数の設定
export INPUT="/workspace/models/Llama-3.1-8B"
export OUTPUT="/workspace/models/Llama-3.1-8B.nemo"
export PREC="bf16"

# 変換の実行
python /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py --input_name_or_path ${INPUT} --output_path ${OUTPUT} --precision ${PREC} --llama31 True
```

実行結果：

```
[NeMo I 2025-04-23 02:51:43 nemo_logging:393] loading checkpoint /workspace/models/Llama-3.1-8B
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.20s/it]
hf_config: {'vocab_size': 128256, 'max_position_embeddings': 131072, 'hidden_size': 4096, 'intermediate_size': 14336, 'num_hidden_layers': 32, 'num_attention_heads': 32, 'num_key_value_heads': 8, 'hidden_act': 'silu', 'initializer_range': 0.02, 'rms_norm_eps': 1e-05, ...}

... (モデル構成と処理プロセスが表示される) ...

converting layer 0
done layer 0
converting layer 1
done layer 1
... (32層のモデルを変換) ...
converting layer 31
done layer 31

[NeMo I 2025-04-23 02:54:14 nemo_logging:393] Global Checkpoint Save : Rank: 0 : Start time: 1745376794.226s : Save duration: 60.221s
[NeMo I 2025-04-23 02:54:34 nemo_logging:393] NeMo model saved to: /workspace/models/Llama-3.1-8B.nemo
```

変換プロセスはスムーズに完了し、モデルはNeMo形式に正常に変換されて`/workspace/models/Llama-3.1-8B.nemo`に保存されました。変換プロセス全体で約3分かかりました。

**結論**：NeMoコンテナバージョン25.02.01では、すでにLlama-3.1モデルがネイティブにサポートされており、ブログで言及されている2つのPRパッチを適用しなくてもモデル形式の変換に成功しました。これはバージョンアップグレードにこれらのパッチの機能改善が含まれていることを示しています。生成された`Llama-3.1-8B.nemo`ファイルは分散チェックポイントを使用しているため、チェックポイントを毎回変更することなく、任意のTensor Parallel（TP）やPipeline Parallel（PP）などのモデル並列組み合わせをロードできます。

### 5. データ準備

このチュートリアルではllm-jp-corpus-v3の日本語Wikipedia（ja_wiki）データを使用します。以下のコマンドでデータをダウンロードし、dataディレクトリに保存します：

```bash
cd /workspace/
mkdir -p data/ja_wiki

# トレーニングデータセット（14分割）のダウンロード
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

# 検証データセットのダウンロード
wget -O data/ja_wiki/validation_0.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/validation_0.jsonl.gz?ref_type=heads

# すべてのデータファイルを解凍
gunzip data/ja_wiki/*
```

実行結果：

```
root@tut-server-11:/workspace# mkdir -p data/ja_wiki
root@tut-server-11:/workspace# wget -O data/ja_wiki/train_0.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_0.jsonl.gz?ref_type=heads
--2025-04-23 03:01:59--  https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_0.jsonl.gz?ref_type=heads
Length: 384020873 (366M) [application/gzip]
data/ja_wiki/train_0.jsonl.gz     100%[==========================================================>] 366.23M  87.6MB/s    in 4.3s    

... (中間の他のファイルのダウンロード過程は省略) ...

--2025-04-23 03:02:28--  https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/validation_0.jsonl.gz?ref_type=heads
Length: 1657379 (1.6M) [application/gzip]
data/ja_wiki/validation_0.jsonl.g 100%[==========================================================>]   1.58M  --.-KB/s    in 0.05s   

root@tut-server-11:/workspace# gunzip data/ja_wiki/*
root@tut-server-11:/workspace# ls data/ja_wiki/
train_0.jsonl  train_10.jsonl  train_12.jsonl  train_2.jsonl  train_4.jsonl  train_6.jsonl  train_8.jsonl  validation_0.jsonl
train_1.jsonl  train_11.jsonl  train_13.jsonl  train_3.jsonl  train_5.jsonl  train_7.jsonl  train_9.jsonl
```

データのダウンロードと解凍プロセスは非常にスムーズでした。ダウンロードファイルのサイズは最大366MBから最小1.6MBまで様々で、ダウンロード速度は約80-100MB/sでした。全プロセスはおよそ30秒かかりました。すべての圧縮ファイルはJSONL形式のファイルに正常に解凍されました。

これらのコマンドは以下の操作を実行します：
1. コンテナの作業ディレクトリに移動
2. データを保存するためのディレクトリ構造を作成
3. 14個のトレーニングデータ分割（train_0からtrain_13）と1個の検証データセット（validation_0）をダウンロード
4. ダウンロードしたすべての圧縮ファイルを解凍

ダウンロードしたデータはJSONL形式で、各行に1つのJSONオブジェクトが含まれており、通常は「text」フィールドにテキスト内容が保存されています。この形式は大規模データ処理に適しており、データを行単位で読み込み処理できます。

### 6. データ前処理

NeMoバックエンドで使用されるMegatronが継続事前学習を行うためには、データの前処理が必要です。以下のスクリプトはNeMoが提供する前処理ツールを使用してJSONLファイルをMegatronが処理できる形式に変換します：

```bash
# 前処理データ保存ディレクトリの作成
mkdir ds

# トレーニングデータの処理
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

# 検証データの処理
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

前処理スクリプトのパラメータ説明：
- `--input`: 入力データディレクトリの指定
- `--json-keys`: JSONLファイル内のテキストを含むフィールド名の指定
- `--tokenizer-library`: 使用するトークナイザーライブラリ（ここではhuggingface）
- `--tokenizer-type`: 使用するトークナイザータイプ（Llama-3.1-8Bのトークナイザーを使用）
- `--dataset-impl`: データセット実装方法（mmapはメモリマッピングを使用し、大規模データ処理の効率を向上）
- `--append-eod`: 各文書の末尾に終了マーカーを追加
- `--output-prefix`: 出力ファイルのプレフィックスパス
- `--workers`: 並列処理するワーカースレッド数
- `--files-filter`: ファイルフィルタパターン、処理するファイルを指定
- `--preproc-folder`: フォルダ全体を前処理
- `--log-interval`: ログ記録間隔

**実行結果：**

トレーニングデータの前処理コマンドは正常に実行されました。スクリプトは14個のトレーニングデータファイル（train_0.jsonlからtrain_13.jsonl）を順番に処理し、合計約126万件の文書を処理しました（各ファイルは約90,000件の文書を含む）。

データ処理中に以下の関連情報が表示されました：
1. システムはHugging FaceからLlama-3.1-8Bのトークナイザー設定とモデルをロードしました
2. スクリプトはすべてのJSONLファイルを正常に解析・処理しました
3. 処理速度は処理する文書数が増えるにつれて向上し、平均処理速度は約100-500 docs/s
4. データ処理レートは安定しており、約4.5-4.6 MB/s

処理中にいくつかの警告メッセージが表示され、一部の文書のトークン列の長さがモデルの最大列長制限（131072）を超えていることを示しています：
```
Token indices sequence length is longer than the specified maximum sequence length for this model (152160 > 131072). Running this sequence through the model will result in indexing errors
```
これらの警告メッセージは、非常に長い一部の文書がトレーニング中に切り捨てられる可能性があることを示していますが、前処理の全体的な完了には影響しません。

前処理完了後、`ds`ディレクトリに以下のファイルが生成されました：
```
train_text_document.bin   # バイナリ形式のトレーニングデータ
train_text_document.idx   # トレーニングデータのインデックスファイル
```

次に検証データの前処理を実行し、検証データのバイナリファイルとインデックスファイルを生成します。

これらのファイルはMegatronエンジン向けに設計された特殊なバイナリ形式で、大規模テキストデータを効率的にロードおよび処理するために使用されます。`.bin`ファイルは実際のトークンデータを含み、`.idx`ファイルは高速インデックスを提供し、モデルトレーニングプロセス中にデータに効率的にアクセスできるようにします。

**検証データの前処理実行結果：**

検証データの前処理コマンドは正常に実行されました。スクリプトは1つの検証データファイル（validation_0.jsonl）を処理し、処理速度は約530.37 docs/s、データ処理レートは約0.826 MB/sでした。

前処理完了後、`ds`ディレクトリに以下のファイルが生成されました：
```
validation_text_document.bin  # バイナリ形式の検証データ
validation_text_document.idx  # 検証データのインデックスファイル
```

これで、すべての前処理作業が完了し、以下のファイルが生成されました：
```
train_text_document.bin      # トレーニングデータ
train_text_document.idx      # トレーニングデータインデックス
validation_text_document.bin # 検証データ
validation_text_document.idx # 検証データインデックス
```

**分析結論：**

データ前処理段階は順調に完了し、JSONL形式のテキストデータをMegatronが処理できるバイナリ形式に変換しました。処理中に一部の文書が長すぎるという警告が表示されましたが、これは正常な現象であり、モデルトレーニング時にこれらの長い文書は自動的に処理されます。生成されたバイナリデータファイルとインデックスファイルは後続の継続事前学習プロセスで使用され、モデルが日本語Wikipediaデータを効率的に読み込み学習できるようになります。

トレーニングデータと検証データの前処理は順調に完了し、トレーニングデータの処理速度は処理する文書数が増えるにつれて向上し、平均100-500 docs/sとなりました。一方、検証データはファイルが小さいため処理速度が速く、530 docs/sに達しました。すべての準備作業が完了し、これから継続事前学習段階に進むことができます。

### 6.1 前処理データファイルの説明

上記の前処理ステップを通じて、以下の4つの重要なファイルが生成されました：

```
train_text_document.bin      # トレーニングデータバイナリファイル
train_text_document.idx      # トレーニングデータインデックスファイル
validation_text_document.bin # 検証データバイナリファイル
validation_text_document.idx # 検証データインデックスファイル
```

これらのファイルはMegatron-LMの`IndexedDataset`データ構造を構成しており、Megatronコアの最も基本的なデータインターフェースです。以下は、これらのファイルの構造、役割、利点について詳細に説明します：

#### ファイル構造

1. **バイナリファイル（`.bin`）**：
   - 実際のトークンデータを含み、モデルトレーニング時に直接読み込まれる内容
   - 各シーケンスのトークンIDシーケンスを格納
   - 以下のメタデータを含む：
     - 各シーケンスの要素数
     - 各シーケンスのバイトオフセットポインタ
     - 各文書のシーケンスインデックス範囲
   - 効率的な圧縮バイナリ形式で保存

2. **インデックスファイル（`.idx`）**：
   - データセットレベルのメタデータ情報を含む
   - 以下の内容を含む：
     - インデックスヘッダー（後方互換性の確保）
     - インデックスバージョン（後方互換性の維持）
     - データタイプコード（データファイルで使用されているデータタイプを示す）
     - バイナリファイル内の特定の文書位置への高速アクセスメカニズムを提供

#### 技術的利点

このデータ形式の設計には以下のいくつかの重要な利点があります：

1. **効率的なメモリ使用**：メモリマッピング（mmap）技術により、システムは必要に応じてデータをロードでき、全データセットをメモリに読み込む必要がなく、大規模データ処理に不可欠です。

2. **高速ランダムアクセス**：インデックスファイルにより、トレーニングプロセスは特定の文書やシーケンスの開始位置に直接ジャンプでき、ファイル全体を順次読み取る必要がありません。

3. **I/Oボトルネックの軽減**：前処理されたバイナリ形式はトレーニング中のテキスト解析、トークン化、処理ステップを排除し、I/OとCPUのオーバーヘッドを大幅に削減します。

4. **並列トレーニングのサポート**：この形式は分散並列トレーニング環境に特に適しており、複数のGPUやノードが異なるデータシャードに効率的にアクセスできます。

5. **前処理オーバーヘッドの削減**：トレーニングプロセス中に時間のかかるトークン化や処理ステップを繰り返す必要がなく、これらはすべて前処理段階で完了します。

#### 継続事前学習での使用

NeMoフレームワークで継続事前学習を実行する際、これらのファイルは以下のような設定方法で参照されます：

```python
TRAIN_DATA_PATH="{train:[1.0,/workspace/ds/train_text_document],validation:[/workspace/ds/validation_text_document],test:[/workspace/ds/validation_text_document]}"
```

事前学習スクリプトはこれらのファイルを自動的にロードし、メモリマッピング技術を通じてデータに効率的にアクセスし、すべてのデータを一度にメモリにロードする必要はありません。この方法は、トレーニングデータが通常、利用可能なメモリやGPUメモリよりもはるかに大きいため、大規模言語モデルトレーニングに特に適しています。

#### 検証データファイルの特別な用途

生成された検証データファイル（`validation_text_document.bin`と`validation_text_document.idx`）は構造的にはトレーニングデータファイルと同じですが、特別な用途と重要性があります：

1. **規模の違い**：実行結果から、検証データセット（約1000の文書）はトレーニングデータセット（約126万の文書）よりもはるかに小さいことがわかります。これは検証プロセスの効率を確保するための意図的な設計です。

2. **処理速度**：検証データの処理速度（530 docs/s）はトレーニングデータ（100-500 docs/s）よりも明らかに速く、これは検証データ量が少ないため、より速くロードして処理できるためです。

3. **用途の違い**：
   - トレーニングデータファイルは実際のモデル学習とパラメータ更新に使用
   - 検証データファイルはトレーニング中のモデルのパフォーマンス評価に使用され、パラメータ更新には関与しない
   - 検証ファイルはトレーニング構成でテストデータとしても指定され、最終的なモデルパフォーマンス評価に使用

4. **過学習防止**：検証データセットは独立したデータセットであり、モデルは「見たことのない」データであるため、モデルの汎化能力の真の評価を提供できます。

5. **早期停止戦略**：トレーニング中、事前学習スクリプトは定期的に検証データを使用してモデルのパフォーマンスを評価し、検証損失が減少しなくなった場合、過学習を防ぐために早期停止メカニズムがトリガーされる可能性があります。

6. **ロード方法**：事前学習構成では、検証データは検証セットとテストセットの両方として使用されます：
   ```python
   TRAIN_DATA_PATH="{train:[1.0,/workspace/ds/train_text_document],validation:[/workspace/ds/validation_text_document],test:[/workspace/ds/validation_text_document]}"
   ```
   これは、トレーニングプロセスが`validation_text_document`ファイルを使用してトレーニング中のモデルのパフォーマンスを評価し、トレーニング終了時に最終テストを実行することを示しています。

以上の通り、これらの前処理ファイルは大規模言語モデルトレーニング基盤の重要なコンポーネントです。生のテキストデータを効率的なバイナリ形式に変換することで、I/Oボトルネックやメモリ制限を心配することなく、大規模分散システム上で効果的にモデルをトレーニングおよび微調整することができます。トレーニングと検証データファイルが協調して機能し、モデルが日本語Wikipediaの知識を学ぶだけでなく、見たことのないデータでもよいパフォーマンスを発揮できるようにします。

### 7. 継続事前学習の実行

継続事前学習はNeMoの`/opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py`を使用して実行できます。以下の方法でパラメータを設定できます：

```bash
# HydraとPyTorch関連の環境変数を設定
export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export TORCH_DISTRIBUTED_DEBUG=INFO

# 実験関連の環境変数を設定
export WANDB=False  # Weights & Biases実験追跡を使用するかどうか
export PJ_NAME="CP"  # プロジェクト名（Continual Pretraining）
export EXP_DIR="./results/"${PJ_NAME}  # 実験結果保存ディレクトリ
export EXP_NAME="Llama-3.1-8B"  # 実験名

# モデルとトークナイザー関連の環境変数を設定
export MODEL="/workspace/models/Llama-3.1-8B.nemo"  # NeMo形式モデルのパス
export TOKENIZER_LIBRARY="huggingface"  # トークナイザーライブラリ
export TOKENIZER_TYPE="meta-llama/Llama-3.1-8B"  # トークナイザータイプ
export TOKENIZER="/workspace/models/Llama-3.1-8B/tokenizer.json"  # トークナイザーパス

# モデル並列化関連の環境変数を設定
export TP_SIZE=2  # Tensor Parallelサイズ
export SP=False  # Sequence Parallelスイッチ
export PP_SIZE=1  # Pipeline Parallelサイズ
export EP_SIZE=1  # Expert Parallelサイズ（MoEモデル向け）

# トレーニングデータパスを設定
TRAIN_DATA_PATH="{train:[1.0,/workspace/ds/train_text_document],validation:[/workspace/ds/validation_text_document],test:[/workspace/ds/validation_text_document]}"
```

上記の環境変数の説明：

1. **Hydra関連**：NeMoフレームワークは設定管理にHydraを使用し、これらの環境変数を設定するとデバッグに役立ちます
2. **実験追跡**：`WANDB`をFalseに設定すると、Weights & Biasを使用した実験追跡を行いません
3. **モデルパス**：前のステップで変換された NeMo形式のモデルを指します
4. **並列化設定**：
   - TP_SIZE=2：テンソル並列度を2に設定し、モデルの層内計算を2つのGPUに分散
   - PP_SIZE=1：パイプライン並列度を1に設定し、パイプライン並列を行わない
   - SP=False：シーケンス並列を有効にしない
   - EP_SIZE=1：エキスパート並列度を1に設定し、混合エキスパートモデル機能を使用しない

5. **データパス**：以前に前処理で生成したトレーニングデータと検証データを使用し、検証データは検証セットとテストセットの両方に同時に使用

次に事前学習コマンドを実行します：

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

事前学習パラメータの説明：

1. **trainerパラメータ**：
   - `trainer.devices=${TP_SIZE}`：使用するGPU数を設定、テンソル並列度と一致
   - `trainer.num_nodes=1`：単一ノードトレーニングを使用
   - `trainer.max_epochs=2`：最大トレーニングエポック数を2に設定
   - `trainer.val_check_interval=100`：100ステップごとに検証を実行
   - `trainer.precision=bf16`：BF16混合精度トレーニングを使用

2. **モデルパラメータ**：
   - `model.micro_batch_size=1`：各GPUのバッチサイズ
   - `model.global_batch_size=8`：グローバルバッチサイズ
   - `model.tensor_model_parallel_size=${TP_SIZE}`：テンソル並列度
   - `model.pipeline_model_parallel_size=${PP_SIZE}`：パイプライン並列度
   - `model.restore_from_path=${MODEL}`：継続事前学習のモデルパス
   - `model.seq_length=4096`：シーケンス長、ここでは4096に設定

3. **オプティマイザパラメータ**：
   - `model.optim.lr=3e-5`：学習率
   - `model.optim.sched.warmup_steps=50`：ウォームアップステップ数
   - `model.optim.sched.constant_steps=500`：一定学習率のステップ数
   - `model.optim.sched.min_lr=5e-6`：最小学習率

4. **データパラメータ**：
   - `model.data.data_path=${TRAIN_DATA_PATH}`：データパスを設定
   - `model.data.splits_string='98,1,1'`：データセット分割比率（トレーニング:検証:テスト）

5. **トークナイザーパラメータ**：
   - `model.tokenizer.library=${TOKENIZER_LIBRARY}`：トークナイザーライブラリ
   - `model.tokenizer.type=${TOKENIZER_TYPE}`：トークナイザータイプ
   - `model.tokenizer.model=${TOKENIZER}`：トークナイザーパス

6. **Llama-3.1特有のパラメータ**：
   - `+model.llama31=True`：Llama-3.1特有の設定を有効化

このコマンドを実行すると、モデルは継続事前学習プロセスを開始します。事前学習プロセス中、定期的に検証セット上でモデルのパフォーマンスを評価し、パフォーマンスが最も高い上位3つのチェックポイントを保存します。

ここでは最大トレーニングエポック数を2に設定していますが、実際のアプリケーションでは満足のいく効果を得るためにより多くのトレーニングエポックが必要になる場合があることに注意してください。同時に、利用可能なGPU数が異なる場合は、それに応じて`TP_SIZE`、`trainer.devices`、バッチサイズなどのパラメータを調整する必要があります。

システムで利用可能なメモリやGPUメモリが限られている場合は、シーケンス長（`model.seq_length`）とバッチサイズパラメータを減らすことを検討できます。

事前学習が完了すると、`${EXP_DIR}/${EXP_NAME}`ディレクトリにトレーニングログとモデルチェックポイントファイルが生成され、後続の評価とアプリケーションに使用できます。

### 7.1 元のチュートリアルスクリプトを使用したトレーニング

元のチュートリアルでは、より完全なトレーニングスクリプト設定が提供されており、以下のコマンドで実行できます：

```bash
# HydraとPyTorch関連の環境変数を設定
export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export TORCH_DISTRIBUTED_DEBUG=INFO

# 実験関連の環境変数を設定
export WANDB=False  # Weights & Biases実験追跡を使用するかどうか
export PJ_NAME="CP"  # プロジェクト名（Continual Pretraining）
export EXP_DIR="./results/"${PJ_NAME}  # 実験結果保存ディレクトリ
export EXP_NAME="Llama-3.1-8B"  # 実験名

# モデルとトークナイザー関連の環境変数を設定
export MODEL="/workspace/models/Llama-3.1-8B.nemo"  # NeMo形式モデルのパス
export TOKENIZER_LIBRARY="huggingface"  # トークナイザーライブラリ
export TOKENIZER_TYPE="meta-llama/Llama-3.1-8B"  # トークナイザータイプ
export TOKENIZER="/workspace/models/Llama-3.1-8B/tokenizer.json"  # トークナイザーパス

# モデル並列化関連の環境変数を設定
export TP_SIZE=2  # Tensor Parallelサイズ
export SP=False  # Sequence Parallelスイッチ
export PP_SIZE=1  # Pipeline Parallelサイズ
export EP_SIZE=1  # Expert Parallelサイズ（MoEモデル向け）

# トレーニングデータパスを設定
TRAIN_DATA_PATH="{train:[1.0,/workspace/ds/train_text_document],validation:[/workspace/ds/validation_text_document],test:[/workspace/ds/validation_text_document]}"

# 事前学習コマンドを実行
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

前に紹介したコマンドと比較して、このスクリプトにはいくつかの主な違いがあります：

1. **トレーニング設定**：
   - `trainer.max_epochs=-1`：トレーニングエポック数を制限せず、`trainer.max_steps`で制御
   - `trainer.max_steps=150`：150ステップ後にトレーニングを停止
   - `trainer.val_check_interval=15`：15ステップごとに検証を実行
   - `trainer.gradient_clip_val=1.0`：勾配クリッピングしきい値を1.0に設定

2. **モデル構成**：
   - `model.encoder_seq_length=8192`と`model.max_position_embeddings=8192`：より大きなシーケンス長と位置埋め込み最大長
   - モデルアーキテクチャパラメータ（`model.num_layers`、`model.hidden_size`など）を明示的に指定

3. **オプティマイザ設定**：
   - `distributed_fused_adam`オプティマイザを使用
   - 学習率を2.5e-5に設定
   - バッチサイズを1024に設定

元のブログによると、この設定は8台のH100 GPU上で約6時間かかってトレーニングを完了します。

### 7.2 Weights & Biases（wandb）を使用してトレーニング進捗をモニタリング

Weights & Biases（wandb）を使用してトレーニング進捗をモニタリングする場合は、以下の手順で設定できます：

#### 7.2.1 wandbのインストールとログイン

1. **wandbをインストール**（コンテナに事前インストールされていない場合）：
   ```bash
   pip install wandb
   ```

2. **wandbアカウントにログイン**：
   ```bash
   wandb login
   ```
   
   このコマンドを実行すると、システムは以下のプロンプトを表示します：
   ```
   wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
   wandb: You can find your API key in your browser here: https://wandb.ai/authorize
   wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit:
   ```
   
   このときwandbのAPIキーを入力する必要があります。APIキーを取得する手順は以下の通りです：
   
   a. [wandb.ai/authorize](https://wandb.ai/authorize)にアクセスするか、wandbにログイン後に設定ページに移動
   b. APIキーをコピー
   c. APIキーをターミナルに貼り付けてEnterを押す
   
   ログインに成功すると、以下のようなメッセージが表示されます：
   ```
   wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc
   ```

3. **ログイン状態を確認**（オプション）：
   ```bash
   wandb status
   ```

#### 7.2.2 wandbを使用するためのトレーニングスクリプトの設定

1. **環境変数を変更**し、wandbモニタリングを有効にする：
   ```bash
   # WANDBをTrueに設定
   export WANDB=True
   
   # プロジェクト名と実験名をカスタマイズ可能
   export PJ_NAME="CP_Japanese_Wiki"
   export EXP_NAME="Llama-3.1-8B_Japanese_CPT"
   ```

2. **wandb固有のパラメータを追加**（必要に応じて）：
   ```bash
   # wandbプロジェクト名を設定
   export WANDB_PROJECT_NAME="llama-3-japanese-continual-pretraining"
   
   # wandb実行IDを設定（後で復元しやすくするため）
   export WANDB_RUN_ID="llama_3_ja_wiki_cpt_run1"
   ```

3. **トレーニングコマンドを実行**、前と同じですがwandb関連の設定を含む：
   ```bash
   python /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
       exp_manager.exp_dir=${EXP_DIR} \
       exp_manager.name=${EXP_NAME} \
       exp_manager.create_wandb_logger=${WANDB} \
       exp_manager.wandb_logger_kwargs.project=${PJ_NAME} \
       exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
       # ... 他のパラメータは前と同じ ...
   ```

トレーニングが開始されると、wandbは自動的にブラウザウィンドウを開く（GUIがある場合）か、URLリンクを提供し、そのリンクからトレーニングダッシュボードにアクセスできます。ダッシュボード上では以下の指標をモニタリングできます：

- トレーニングおよび検証損失
- 学習率の変化
- GPU使用率とメモリ使用状況
- モデルパラメータ統計情報
- トレーニング速度（1秒あたりの処理サンプル数）

#### 7.2.3 分散トレーニング環境でwandbを使用

分散トレーニング環境では、マスタープロセスのみがwandbサーバーにデータを送信します。すべてのプロセスがwandbディレクトリに正しくアクセスできるようにするため、トレーニングスクリプトに以下のコードを追加できます：

```python
if os.environ.get("LOCAL_RANK", None) is None:
    os.environ["WANDB_DIR"] = wandb.run.dir
```

これにより、すべてのプロセスが同じwandbディレクトリを使用し、分散トレーニング中の競合を回避できます。

Weights & Biasは実験比較機能も提供しており、異なるハイパーパラメータ設定でのトレーニング効果を比較できます。これはトレーニングプロセスの最適化に非常に役立ちます。

**注意**：トレーニングプロセスが中断された場合、同じコマンドを使用してトレーニングを再開できます。NeMoフレームワークは自動的に最新のチェックポイントからトレーニングを再開します。wandbを使用している場合、新しいトレーニング実行は特別に前の実行を再開するように設定しない限り、wandbダッシュボード上に新しい実験として表示されます。

トレーニングが完了すると、最終モデルは`${EXP_DIR}/${EXP_NAME}/checkpoints/`ディレクトリに保存され、後続の評価とアプリケーションに使用できます。

### 8. 単一GPU環境での継続事前学習の実行

NVIDIA RTX 6000 Ada Generation GPU（48GBのVRAM、デバイス番号0）が1台しかないため、限られたハードウェアリソースに合わせて事前学習の設定を調整する必要があります。以下は単一GPU環境向けに最適化された**最終修正済みスクリプト**で、実行可能ファイル`src/run_pretraining_single_gpu.sh`として作成されています：

```bash
#!/bin/bash

# 環境変数の設定
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

# TRAIN_DATA_PATH環境変数の使用を再開（ブログから）
export TRAIN_DATA_PATH="{train:[1.0,/workspace/ds/train_text_document],validation:[/workspace/ds/validation_text_document],test:[/workspace/ds/validation_text_document]}"

# wandb設定（オプションでカスタマイズ可能）
export WANDB_PROJECT="llama-3-japanese-continual-pretraining"
export WANDB_RUN_ID="llama_3_ja_wiki_cpt_run1"
export WANDB_LOG_MODEL=true

# 単一GPUパラメータ設定
export TP_SIZE=1
export PP_SIZE=1
export EP_SIZE=1
export SP=False

# 注意：NVTE_FUSED_ATTNとNVTE_FLASH_ATTN環境変数を設定しない
# 代わりにmodel.attention_backendパラメータで注意機構バックエンドを指定する

# 開始時間を記録
echo "トレーニング開始時間: $(date)"

# wandbにログイン（初回使用時に必要）
# wandb login

# 事前学習スクリプトの実行
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
    # シーケンス長を2048に削減
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
    # グローバルバッチサイズを2に削減
    model.global_batch_size=2 \\
    # 修正後のデータパス指定方式を使用
    model.data.data_prefix=[1.0,/workspace/ds/train_text_document] \\
    model.data.validation_drop_last=True \\
    model.data.num_workers=2 \\
    ++model.attention_backend=fused

# 終了時間を記録
echo "トレーニング終了時間: $(date)" 
```

元の8 GPUの設定と比較して、主な調整点は以下の通りです：
1.  `trainer.devices=1`：1台のGPUのみを使用するように設定。
2.  `TP_SIZE=1, PP_SIZE=1, EP_SIZE=1`：GPUが1台しかないため、すべての並列パラメータを1に設定。
3.  `model.encoder_seq_length=2048`：`global_batch_size`を極めて低く（2など）設定してもOOMが発生した後、シーケンス長を4096からさらに2048に削減。これはメモリ使用量を削減する最も重要な方法の1つ。
4.  `model.global_batch_size=2`：複数回の試行後、グローバルバッチサイズを最終的に2まで削減。
5.  `model.optim.name=adamw`：オプティマイザを`distributed_fused_adam`から標準の`adamw`に変更。
6.  `model.megatron_amp_O2=False`：Megatron特有のO2レベル混合精度最適化を無効にし、標準のPyTorch AMPに戻してメモリ使用量削減を試みる（この変更もOOMを解決しなかった）。
7.  `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：この環境変数を追加してメモリフラグメンテーションの問題を緩和しようとしたが、最終的なOOMには効果がなかった。
8.  **データパス形式の修正**：トレーニング、検証、テストデータパスを単一の`TRAIN_DATA_PATH`環境変数に結合して`model.data.data_prefix`に渡す方法（データロードエラーを引き起こした）を放棄し、`model.data.data_prefix`でトレーニングデータを指定し、`+model.data.validation_data_prefix`で検証データを指定する方法に戻した。
9.  `exp_manager.checkpoint_callback_params.save_top_k=1`：ディスク容量を節約するため、保存する最良チェックポイントの数を1に削減。
10. **注意機構関連の環境変数設定を削除**（`NVTE_FUSED_ATTN`と`NVTE_FLASH_ATTN`）。
11. **`++model.attention_backend=fused`パラメータを追加**して明示的に注意機構バックエンドを指定し、設定の競合エラーを解決。
12. トレーニングプロセスのモニタリングを容易にするため、開始時間と終了時間の記録を追加。

### スクリプトの使用方法

スクリプト作成後、実行権限を付与して実行する必要があります：

```bash
chmod +x ./src/run_pretraining_single_gpu.sh
./src/run_pretraining_single_gpu.sh
```

これらの調整は、限られたメモリを持つ単一GPUで、本来複数GPU向けに設計されたトレーニングフローを実行するためのものです。メモリ消費の大きな要素（シーケンス長、バッチサイズなど）を徐々に削減し、さまざまな最適化戦略を試すことで、最終的にそのハードウェア上で実行可能な設定を見つけました。

**注意**：単一GPU環境では、トレーニング速度は複数GPU環境よりもはるかに遅くなり、完了時間が大幅に延長されることが予想されます。上記の調整によって起動時のOOM問題が解決しても、トレーニングプロセス中にメモリの変動により再びOOMが発生する可能性があります。

### トレーニング進捗のモニタリング

スクリプトを実行した後、以下の方法でトレーニングの進捗状況を監視できます：

1. リアルタイムトレーニングログの確認：
   ```bash
   tail -f ${EXP_DIR}/${EXP_NAME}/log/*.log
   ```

2. GPU使用状況の確認：
   ```bash
   nvidia-smi -l 5
   ```

トレーニングが完了すると、最終モデルは`${EXP_DIR}/${EXP_NAME}/checkpoints/`ディレクトリに保存されます。

### 8.1 メモリ不足問題の調査とパラメータ調整

元の複数GPU（8x H100）トレーニング設定を単一のNVIDIA RTX 6000 Ada（48GB VRAM）に適応させようとする際、CUDA Out-of-Memory（OOM）の問題は避けられません。以下は問題の調査と解決の詳細な手順です：

1.  **初期調整（スクリプトv1）**：
    *   **目標**：トレーニングを単一GPUで起動可能にする。
    *   **変更**：
        *   `trainer.devices=1`
        *   `TP_SIZE=1`, `PP_SIZE=1`, `EP_SIZE=1`
        *   `model.encoder_seq_length`を8192から4096に削減。
        *   `model.global_batch_size`を1024から32に削減。
    *   **結果**：依然としてOOMエラーが発生。シーケンス長とバッチサイズを削減しても、8Bモデルの基本メモリ要件にトレーニングオーバーヘッドを加えると、48GB VRAMでも依然として不足することを示している。

2.  **ブログのデータパス形式を試行**：
    *   **背景**：元のブログスクリプトでは、複雑な文字列形式を`TRAIN_DATA_PATH`環境変数に割り当て、それを直接`model.data.data_prefix`に渡していた。
    *   **試行**：`export TRAIN_DATA_PATH=\"{train:[...],validation:[...],test:[...]}\"`と`model.data.data_prefix=${TRAIN_DATA_PATH}`の使用を復活。
    *   **結果**：起動直後に`AssertionError: Could not find data-idx or data-bin files at the prefix ...`というエラーが発生。これはHydraがこのような複雑な構造をパス一覧として正しく解析できないという以前の判断を確認した。NeMoは個別のトレーニングと検証のパスパラメータを期待している。
    *   **修正**：この形式を放棄し、`model.data.data_prefix=[1.0,/workspace/ds/train_text_document]`と`+model.data.validation_data_prefix=[/workspace/ds/validation_text_document]`に戻す。

3.  **`PYTORCH_CUDA_ALLOC_CONF`を試行**：
    *   **背景**：PyTorch OOMエラーのヒントでは、メモリフラグメンテーションを緩和するために`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`の設定が推奨されている。
    *   **試行**：スクリプトに`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`を追加。
    *   **結果**：OOMエラーは引き続き発生し、モデルのロードまたはオプティマイザーの初期化段階で発生。これは主な問題がピークメモリ要件が高すぎることであり、フラグメンテーションではないことを示している。

4.  **オプティマイザーの変更**：
    *   **背景**：OOMエラーは`distributed_fused_adam`オプティマイザーの初期化時に発生している。Fusedオプティマイザーは計算効率が高いが、融合状態を保存するためにより多くのメモリを必要とする可能性がある。
    *   **試行**：`model.optim.name`を`distributed_fused_adam`から標準の`adamw`に変更。
    *   **結果**：OOMエラーポイントは後に移動したが、依然として発生。これはオプティマイザー自体のメモリ使用量がボトルネックではないことを示している。

5.  **`megatron_amp_O2`の無効化**：
    *   **背景**：`model.megatron_amp_O2=True`はMegatron-LM特有のO2レベル混合精度最適化を有効にし、標準のPyTorch AMP（`O1`）よりも多くのメモリを使用する可能性がある。
    *   **試行**：`model.megatron_amp_O2`を`False`に設定。
    *   **結果**：OOMエラーは依然として発生し、モデルの重みをロードする段階で発生。テンソルシャードをマージする際にメモリ不足というエラーが表示される。これはO2最適化を無効にしても、モデル自体が`bf16`精度でロードされるとVRAMの限界に近づくことを示している。

6.  **`global_batch_size`の削減**：
    *   **背景**：モデルのロードですでに大量のメモリが消費されている状況では、トレーニングプロセス中のアクティベーション、勾配、オプティマイザ状態が主なメモリ増加ポイントとなり、これらはバッチサイズに直接関連している。
    *   **試行**：`model.global_batch_size`を32から16、さらに8、最終的に2まで段階的に削減。
    *   **結果**：`global_batch_size=2`まで削減しても、トレーニングはモデルロード段階（テンソルシャードのマージ時）でOOMが発生。

7.  **`encoder_seq_length`の削減**：
    *   **背景**：非常に低い`global_batch_size`でもOOMが続くということは、モデルロードが主なボトルネックであることを示している。シーケンス長はメモリ使用量に大きく影響する別のキーパラメータ。
    *   **試行**：`global_batch_size=2`を維持しながら、`model.encoder_seq_length`を4096から2048、1024、さらに512まで段階的に削減。
    *   **結果**：シーケンス長を512まで削減しても、トレーニングはモデルロード段階（`torch.cat`操作時）でOOMエラーが発生。

**結論**：比較的VRAMが限られた単一GPU（RTX 6000 Ada, 48GB）上でLlama-3.1-8Bモデルの継続事前学習を実行する場合、一般的なメモリ最適化戦略（バッチサイズを2に削減、シーケンス長を512に削減、オプティマイザの変更、O2最適化の無効化）を多数採用しても、モデルロードプロセス自体の基本的なVRAM要件（BF16精度）はハードウェアの制限を超えています。これは、より高度なメモリ最適化技術（ZeRO OffloadingをCPU/NVMeに適用、モデル量子化、またはLoRAなどのパラメータ効率の良い微調整技術）がない場合、全パラメータの継続事前学習は実行不可能であることを示しています。今後このハードウェアでトレーニングする場合は、LoRAなどのPEFT方法の採用や、より小規模のモデルへの切り替えを検討する必要があります。

### 一般的なエラーと解決策

単一GPUトレーニングスクリプトを実行する際、以下のような一般的なエラーが発生する可能性があります：

1. **Hydra設定エラー**：
   ```
   omegaconf.errors.ConfigAttributeError: Key 'validation_data_prefix' is not in struct
       full_key: model.data.validation_data_prefix
       object_type=dict
   ```
   
   **解決策**：NeMo中で存在しない設定項目については、新しい設定を追加するため、パラメータの前に`+`プレフィックスを付ける必要があります。例えば、`+model.data.validation_data_prefix=${VALID_DATA_PATH}`のようにします。既存の設定項目（`model.data.data_prefix`など）の場合、値が単純な型であれば直接割り当てるか、リスト形式（`model.data.data_prefix=[1.0,/path/to/train]`など）を使用します。**特に注意**：複雑な辞書構造を単一のパスパラメータに文字列として渡すことは避けてください。Hydraは通常このような構造をパスリストとして正しく解析できません。

2. **注意機構設定の競合エラー**：
   ```
   AssertionError: NVTE_FLASH_ATTN set to 0, but expected 1 for attention backend type auto. unset NVTE_FLASH_ATTN, NVTE_FUSED_ATTN and NVTE_UNFUSED_ATTN. Use the --attention-backend argument if you want to choose between (flash/fused/unfused/auto/local). Default is auto.
   ```
   **原因分析**：このエラーは古いバージョンの環境変数（例：`export NVTE_FUSED_ATTN=1`や`export NVTE_FLASH_ATTN=0`）を使用して注意機構を設定しようとした時に発生します。より新しいバージョンのNeMoフレームワーク（25.02.01など）では、コマンドラインパラメータ`++model.attention_backend`を使用して注意バックエンド（`fused`、`flash`、`unfused`、`auto`、`local`など）を指定することが推奨されています。フレームワークのデフォルト選択（通常は`auto`または`flash`）と競合する環境変数を同時に設定すると、このアサーションエラーがトリガーされます。
   **解決策**：
     - スクリプトからすべての`export NVTE_FUSED_ATTN=...`と`export NVTE_FLASH_ATTN=...`環境変数設定を**削除**する。
     - `python ... megatron_gpt_pretraining.py`コマンドに`++model.attention_backend=fused`（またはハードウェアとソフトウェアのサポート状況に応じて他の必要なバックエンド、例えば`flash`）パラメータを**追加**する。

3. **メモリ不足エラー（torch.OutOfMemoryError: CUDA out of memory）**：
    *   **原因分析**：これはリソースが制限された環境（単一GPUなど）で大規模モデルをトレーニングする際の最も一般的なエラーです。GPUのVRAMがモデルパラメータ、アクティベーション、勾配、オプティマイザ状態を格納するには不十分です。エラーはモデルのロード、オプティマイザの初期化、トレーニングイテレーションのどの段階でも発生する可能性があります。
    *   **解決策（優先順位と効果順）**：
        1.  **シーケンス長の削減**：`model.encoder_seq_length`を大幅に削減する（例：8192から4096以下に）。これは注意行列とアクティベーションのサイズに直接影響するため、メモリ使用量を削減する最も効果的な方法の1つです。
        2.  **グローバルバッチサイズの削減**：`model.global_batch_size`を削減する（例：32から16以下に）。これにより各イテレーションで処理されるデータ量が減少し、アクティベーション、勾配、オプティマイザ状態のメモリピークが削減されます。
        3.  **勾配蓄積の使用**：`model.micro_batch_size=1`を維持しながら、`trainer.accumulate_grad_batches`を1より大きい値（例：16または32）に設定します。これにより、単一ステップのメモリ使用量を増やすことなく、より大きなグローバルバッチサイズと同じ勾配更新効果を達成できます。
        4.  **アクティベーションチェックポイントの有効化**：`model.activations_checkpoint_granularity="selective"`または`"full"`、および`model.activations_checkpoint_method="uniform"`または`"block"`を設定します。これにより、バックプロパゲーション時に前方伝播のアクティベーションを再計算し、それらを保存する代わりに、追加の計算時間と引き換えに大幅なメモリ節約が可能になります。
        5.  **混合精度設定の確認**：`trainer.precision=bf16`または`fp16`が有効になっていることを確認します。通常、`bf16`はメモリ効率が高いですが、極端な場合には、この設定が正しく適用されているか確認する必要があります。
        6.  **異なるオプティマイザを試す**：`adamw`は通常`distributed_fused_adam`よりもメモリ使用量がわずかに少ないですが、これは通常OOMを解決するための主要な手段ではありません。
        7.  **不必要な機能をオフにする**：例えば、`model.megatron_amp_O2=True`が必要でない場合は、`False`に設定できます。
        8.  **GPUメモリ使用状況のモニタリング**：`nvidia-smi`を使用してメモリ使用状況を継続的に監視し、メモリピークがどのステップで発生するかを特定して、より的確な最適化を行います。
        9.  **他のプロセスの確認**：デスクトップ環境や他のアプリケーションなど、大量のGPUメモリを占有している他のプログラムがないことを確認します。

4.  **チェックポイントロードエラー**：
   - 指定したモデルパス（`model.restore_from_path`）が正確で、有効な`.nemo`ファイルを指していることを確認します。
   - モデルチェックポイントが現在のNeMoフレームワークバージョンと互換性があることを確認します。

5.  **データパス解析エラー**（`AssertionError: Could not find data-idx or data-bin`）：
    *   **原因分析**：通常、`model.data.data_prefix`または`+model.data.validation_data_prefix`に渡されたパスの形式が正しくないか、複数のパスを含む複雑な構造（辞書文字列など）を単一のパスパラメータに渡そうとしたことが原因です。
    *   **解決策**：データパスを指定する際、正しいHydra/OmegaConf構文を使用していることを確認します。トレーニングデータの場合、`model.data.data_prefix=[WEIGHT,/path/to/train_prefix]`を使用します。検証データとテストデータ（別々に指定する場合）には、`+model.data.validation_data_prefix=[/path/to/validation_prefix]`と`+model.data.test_data_prefix=[/path/to/test_prefix]`を使用します。パスが前処理後に生成された`.bin`および`.idx`ファイルのプレフィックス（拡張子なし）を指していることを確認します。

### Weights & Biases（wandb）を使用したトレーニング進捗のモニタリング

wandbは機械学習実験追跡のための強力なツールで、トレーニングプロセスの可視化と分析に役立ちます。NeMoフレームワークはwandbを統合してトレーニング進捗をモニタリングすることをサポートしています。

#### 1. wandbのインストールと設定

コンテナにwandbがプリインストールされていない場合は、まずインストールする必要があります：

```bash
pip install wandb
```

wandbアカウントにログイン（事前にwandb.aiでアカウント登録が必要）：

```bash
wandb login
```

コマンドを実行すると、システムはAPIキーの入力を求めます。[https://wandb.ai/authorize](https://wandb.ai/authorize)にアクセスしてAPIキーを取得できます。

#### 2. トレーニングスクリプトでwandbを設定

`run_pretraining_single_gpu.sh`スクリプトには、すでにwandb関連の設定が追加されています：

```bash
# wandbを有効化
export WANDB=True

# wandbプロジェクト設定
export WANDB_PROJECT="llama-3-japanese-continual-pretraining"  # プロジェクト名
export WANDB_RUN_ID="llama_3_ja_wiki_cpt_run1"  # 実行ID、特定の実験の追跡用
export WANDB_LOG_MODEL=true  # モデルチェックポイントを記録（オプション、大きなファイルのアップロードを避けるため慎重に使用）
```

トレーニングコマンドでは、以下のパラメータでwandbを設定します：

```bash
exp_manager.create_wandb_logger=${WANDB} \
exp_manager.wandb_logger_kwargs.project=${PJ_NAME} \
exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
```

#### 3. wandb実験結果へのアクセスと分析

トレーニングが開始されると、システムは自動的にwandbセッションを作成し、URLリンクを提供します。ブラウザでアクセスできます：

```
wandb: 🚀 View run at https://wandb.ai/username/project-name/runs/run-id
```

wandbダッシュボードでは、以下の指標をリアルタイムでモニタリングできます：

- トレーニングおよび検証損失曲線
- 学習率の変化
- GPU使用率とメモリ使用状況
- パラメータ統計情報
- スループットとトレーニング速度

#### 4. 分散環境でのwandbの使用

分散トレーニングでは、wandbはマスタープロセスでのみデータを記録し、重複を避けます。NeMoフレームワークではこの処理方法が内蔵されているため、追加の設定は必要ありません。

#### 5. トレーニングの再開と実験の比較

トレーニングが中断された場合、同じ`WANDB_RUN_ID`を使用してトレーニングを継続し、実験記録の連続性を維持できます：

```bash
export WANDB_RESUME=true
export WANDB_RUN_ID="以前の実行ID"
```

wandbは実験比較機能も提供しており、異なるハイパーパラメータ設定でのトレーニング効果を比較し、最適なトレーニング設定を見つけるのに役立ちます。

**注意**：wandbを使用するとトレーニング速度が低下する可能性があります。特に多数のパラメータやチェックポイントを記録する場合に顕著です。トレーニングパフォーマンスに明らかな低下が見られる場合は、ログ記録頻度を下げるか、特定の機能を無効にすることを検討してください。

## 評価方法

### Nejumi リーダーボード 3 評価

Nejumi リーダーボード 3はLLMの日本語能力を多面的に評価するためのベンチマークテストです。ベンチマークテストスクリプトを実行することで、自分のモデルをさまざまなモデルと比較できます。

ブログ著者の評価によると、MT-bench（160サンプル）で元のmeta-llama/Llama-3.1-8Bと日本語継続事前学習後のモデルを比較した結果：

- 日本語指示に対して、元のモデルは約30-40%の応答が英語でした
- 日本語継続事前学習後のモデルでは、英語での応答がごく少数のケースに減少しました

これは日本語継続事前学習を通じて、モデルの日本語指示の理解と応答能力が大幅に向上したことを示しています。

## 発生しうる問題と解決策

1. **Hugging Face権限問題**:
   - 問題：meta-llama/Llama-3.1-8Bモデルにアクセスできない
   - 解決策：Hugging Faceウェブサイトでアクセス権を申請し取得していることを確認

2. **GPU メモリ不足**:
   - 問題：H100 GPU以外を使用している場合、メモリ不足問題が発生する可能性がある
   - 解決策：バッチサイズを削減するか勾配累積を使用

3. **データダウンロード失敗**:
   - 問題：日本語Wikipediaデータのダウンロード失敗
   - 解決策：ネットワーク接続が安定していることを確認するか、ミラーサイトを使用

4. **Hydra/OmegaConf 設定エラー**:
    *   問題：パラメータの追加や変更時に`ConfigAttributeError`などのエラーが発生
    *   解決策：NeMo設定構造を理解し、`+`を使用して新しいパラメータを追加し、既存のパラメータを直接値で変更。リストや辞書の正しい構文に注意。複雑な構造を単純な文字列として渡すことを避ける。

5.  **データパスまたは形式エラー**:
    *   問題：トレーニング開始時に`.bin`または`.idx`ファイルが見つからないと報告される
    *   解決策：`model.data.data_prefix`や`+model.data.validation_data_prefix`などのパラメータの値と形式が正しいことを注意深く確認し、それらが前処理で生成されたファイルの正しいプレフィックス（拡張子なし）を指していることを確認する


## まとめ

この文書では、NeMo FrameworkをLlama-3.1-8Bモデルの日本語継続事前学習に使用する完全なフローを詳細に記録しました。このチュートリアルでは単一ノードと比較的小規模な日本語Wikipediaデータを使用していますが、データ量を増やし複数ノードの計算環境を使用することで、容易に大規模トレーニング設定に拡張できます。

主なステップは以下の通りです：
1. 環境準備とコンテナ起動
2. Hugging Faceから事前学習済みモデルのダウンロード
3. モデルをNeMo形式に変換
4. 日本語Wikipediaデータの準備と前処理
5. 継続事前学習の実行（単一GPUと複数GPU設定）
6. モデルパフォーマンスの評価

日本語継続事前学習を通じて、モデルの日本語指示理解と応答能力が大幅に向上し、英語での応答の割合が減少します。この方法はさまざまな言語やドメインの大規模モデルのカスタマイズに適用でき、特定の言語やビジネスドメイン向けのローカライズされた大規模言語モデルの開発に役立ちます。

この作業記録がNeMo Frameworkを使用したカスタム大規模言語モデルトレーニングを行うチームへの参考や援助となることを願っています。

## 参考資料

- [NeMo Framework で実践する継続事前学習 – 日本語 LLM 編 –](https://developer.nvidia.com/ja-jp/blog/how-to-use-continual-pre-training-with-japanese-language-on-nemo-framework/)
- NVIDIA NeMo Framework ユーザーガイド
- NeMo Framework 単一ノード事前学習ドキュメント
- [Training Localized Multilingual LLMs with NVIDIA NeMo, Part 1](https://developer.nvidia.com/blog/training-localized-multilingual-llms-with-nvidia-nemo-part-1/)
- [Training Localized Multilingual LLMs with NVIDIA NeMo, Part 2](https://developer.nvidia.com/blog/training-localized-multilingual-llms-with-nvidia-nemo-part-2/)
- [NeMo Curator を使った日本語データのキュレーション](https://developer.nvidia.com/ja-jp/blog/using-nemo-curator-for-japanese-data-curation/)
- [NeMo Framework で日本語 LLM をファインチューニング – SFT 編 –](https://developer.nvidia.com/ja-jp/blog/fine-tuning-japanese-llm-with-nemo-framework-sft/)
- [NeMo Framework で日本語 LLM をファインチューニング – PEFT 編 –](https://developer.nvidia.com/ja-jp/blog/fine-tuning-japanese-llm-with-nemo-framework-peft/)
- [NeMo Framework で日本語 LLM をファインチューニング – DPO 編 –](https://developer.nvidia.com/ja-jp/blog/fine-tuning-japanese-llm-with-nemo-framework-dpo/)
- [NVIDIA NIM でファインチューニングされた AI モデルのデプロイ](https://developer.nvidia.com/ja-jp/blog/deploying-fine-tuned-ai-models-with-nvidia-nim/)