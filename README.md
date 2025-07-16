# NeMo 2.0 Qwen2.5 日本語ファインチューニング

NeMo 2.0を使用してQwen2.5モデルの日本語ファインチューニングを行うプロジェクトです。[llm-jp-corpus-v3](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3)の日本語Wikipediaデータセットを使用して、効率的なPEFT（Parameter-Efficient Fine-Tuning）による学習を実現します。

## 目次

- [概要](#概要)
- [環境要件](#環境要件)
- [Docker環境のセットアップ](#docker環境のセットアップ)
- [追加パッケージのインストール](#追加パッケージのインストール)
- [データ準備](#データ準備)
- [使用方法](#使用方法)
- [プロジェクト構成](#プロジェクト構成)
- [サンプルコード](#サンプルコード)
- [トラブルシューティング](#トラブルシューティング)
- [参考資料](#参考資料)

## 概要

このプロジェクトでは以下の機能を提供します：

- **Qwen2.5モデルの日本語対応**: HuggingFaceモデルからNeMo形式への変換
- **効率的なファインチューニング**: LoRA PEFTを使用したメモリ効率的な学習
- **日本語データセット**: llm-jp-corpus-v3のja_wikiデータセットを使用
- **実践的なサンプル**: 学習から推論までの完全なワークフロー

## 環境要件

- NVIDIA GPUドライバー（推奨: 最新版）
- Docker & NVIDIA Container Toolkit
- CUDA対応GPU
- 十分なディスク容量

## Docker環境のセットアップ

### 1. NVIDIA Container Toolkitのインストール

```bash
# Ubuntu/Debian の場合
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. NeMo 2.0 Dockerイメージのダウンロード

[NGC Catalog](https://catalog.ngc.nvidia.com/?filters=&orderBy=weightPopularDESC&query=&page=&pageSize=)からNeMo 2.0の公式Dockerイメージを使用します。

```bash
# NeMo 2.0 Docker imageのプル
docker pull nvcr.io/nvidia/nemo:25.04
```

### 3. Dockerコンテナの起動

```bash
# ワークスペースディレクトリでDockerコンテナを起動
docker run -it --gpus all \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --network=host \
    -v '/path/to/your/workspace:/workspace' \
    -w /workspace \
    nvcr.io/nvidia/nemo:25.04 bash
```
**パラメータ説明:**
- `--gpus all`: 全てのGPUにアクセス
- `--shm-size=16g`: 共有メモリサイズを16GBに設定
- `--ulimit memlock=-1`: メモリロック制限を無制限に
- `--network=host`: ホストネットワークを使用
- `-v '/path/to/your/workspace:/workspace'`: ローカルディレクトリをマウント

### 4. コンテナ内での確認

```bash
# GPUの確認
nvidia-smi

# NeMoのバージョン確認
python -c "import nemo; print(nemo.__version__)"
```

## 追加パッケージのインストール

Docker環境内で必要に応じて追加パッケージをインストールします：

```bash
# 追加の依存関係（コンテナ内で実行）
pip install datasets
pip install jupyter
pip install matplotlib seaborn
```

## データ準備

### 日本語Wikipediaデータセットのダウンロード

[llm-jp-corpus-v3](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3)の[ja_wiki](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/tree/main/ja/ja_wiki)データセットを使用します。

```bash
cd /workspace/
mkdir -p data/ja_wiki

# 訓練データのダウンロード
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

# バリデーションデータのダウンロード
wget -O data/ja_wiki/validation_0.jsonl.gz --no-check-certificate https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/validation_0.jsonl.gz?ref_type=heads

# 圧縮ファイルの展開
gunzip data/ja_wiki/*
```

## 使用方法

### 1. HuggingFaceモデルのNeMo形式への変換

```bash
python src/01_convert_hf_to_nemo.py
```

### 2. データ前処理

```bash
# 日本語Wikipediaデータをインストラクション形式に変換
python src/00convert_ja_wiki_to_sft.py

# 一般的な日本語データ変換（必要に応じて）
python src/00convert_ja.py
```

### 3. ファインチューニング実行

```bash
python src/02_qwen25_sft_training.py
```

### 4. Jupyter Notebookでの実行

対話的な実行には、提供されているNotebookを使用してください：

- `sft.ipynb`: 基本的なSFTファインチューニング
- `sft_ja_wiki.ipynb`: 日本語Wikipedia特化のファインチューニング

## プロジェクト構成

```
.
├── README.md                          # このファイル
├── .gitignore                         # Git除外設定
├── src/                               # メインスクリプト
│   ├── 01_convert_hf_to_nemo.py      # HF→NeMo変換
│   ├── 00convert_ja_wiki_to_sft.py   # WikiデータSFT変換
│   ├── 00convert_ja.py               # 一般日本語データ変換
│   └── 02_qwen25_sft_training.py     # SFTトレーニング
├── sft.ipynb                         # SFTトレーニング notebook
├── sft_ja_wiki.ipynb                 # Wiki特化 notebook
├── NeMo2.0 PEFT Samle/               # 追加サンプルコード
│   ├── abeja-qwen-peft-tuning-example.py
│   ├── abeja-qwen-peft-inference.py
│   ├── elyza-peft-tuning-example.py
│   └── elyza-peft-inference.py
└── data/                             # データディレクトリ
    └── ja_wiki/                      # 日本語Wikipediaデータ
```

## サンプルコード

### 基本的なファインチューニング

```python
import nemo_run as run
from nemo.collections import llm

# ファインチューニングの実行
result = llm.finetune(
    model="/path/to/qwen2.5-0.5b.nemo",
    data="/path/to/processed_data/",
    name="qwen25_ja_wiki_sft",
    num_nodes=1,
    num_gpus_per_node=1,
    max_steps=1000,
    peft_scheme="lora",
    seq_length=2048,
    micro_batch_size=2,
    global_batch_size=16,
)
```

### LoRA PEFT設定

```python
recipe = llm.qwen25_500m.finetune_recipe(
    name="qwen25_500m_ja_wiki_sft",
    dir="/workspace/checkpoints",
    num_nodes=1,
    num_gpus_per_node=1,
    peft_scheme='lora',
    packed_sequence=False,
)

# 実行
run.run(recipe, executor=run.LocalExecutor())
```

## トラブルシューティング

### よくある問題

1. **メモリ不足エラー**
   ```bash
   # micro_batch_sizeを小さくする
   micro_batch_size=1
   ```

2. **CUDA Out of Memory**
   ```bash
   # LoRA PEFTを使用
   peft_scheme="lora"
   ```

3. **データ形式エラー**
   ```bash
   # データ前処理スクリプトを再実行
   python src/00convert_ja_wiki_to_sft.py
   ```

### パフォーマンス最適化

- **高速化**: `packed_sequence=True`を使用
- **メモリ削減**: LoRA rank値を調整（デフォルト: 32）
- **バッチサイズ**: GPUメモリに応じて調整

## 参考資料

- [NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [Qwen2.5 Model](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [LLM-jp Corpus v3](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

---

**Author**: Kosuke
**Last Updated**: 2025-01
