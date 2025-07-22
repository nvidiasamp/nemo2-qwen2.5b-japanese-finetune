# NeMo 2.0 Qwen2.5 日本語ファインチューニング

NeMo 2.0を使用してQwen2.5モデルの日本語ファインチューニングを行うプロジェクトです。[llm-jp-corpus-v3](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3)の日本語Wikipediaデータセットを使用して、効率的なLoRA PEFTファインチューニングとフルモデルSFTファインチューニングの両方に対応します。

## 📋 目次

- [概要](#概要)
- [環境要件](#環境要件)
- [Docker環境のセットアップ](#docker環境のセットアップ)
- [データ準備](#データ準備)
- [使用方法](#使用方法)
- [プロジェクト構成](#プロジェクト構成)
- [詳細仕様](#詳細仕様)
- [トラブルシューティング](#トラブルシューティング)
- [参考資料](#参考資料)

## 🎯 概要

このプロジェクトでは以下の機能を提供します：

- **Qwen2.5-0.5Bモデルの日本語対応**: HuggingFaceモデルからNeMo形式への変換
- **効率的なファインチューニング**: LoRA PEFTとフルモデルSFTの両方をサポート
- **日本語データセット**: llm-jp-corpus-v3のja_wikiデータセットを質問回答形式に自動変換
- **完全なワークフロー**: データ変換から学習まで4つのスクリプトで完結

## 🔧 環境要件

- NVIDIA GPUドライバー（推奨: 最新版）
- Docker & NVIDIA Container Toolkit
- CUDA対応GPU（推奨: 16GB+ VRAM）
- 十分なディスク容量（50GB+推奨）

## 🐳 Docker環境のセットアップ

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

### 2. NeMo 2.0 Dockerコンテナの起動

```bash
# NeMo 2.0 Docker imageを使用してコンテナを起動
docker run -it --gpus all \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --network=host \
    -v '/path/to/your/workspace:/workspace' \
    -w /workspace \
    nvcr.io/nvidia/nemo:25.04 bash
```

### 3. 動作確認

```bash
# GPUの確認
nvidia-smi

# NeMoのバージョン確認
python -c "import nemo; print(nemo.__version__)"
```

## 📊 データ準備

### 日本語Wikipediaデータセットのダウンロード

```bash
cd /workspace/
mkdir -p data/ja_wiki

# 訓練データのダウンロード（全14ファイル）
for i in {0..13}; do
    wget -O data/ja_wiki/train_${i}.jsonl.gz --no-check-certificate \
    https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_${i}.jsonl.gz?ref_type=heads
done

# バリデーションデータのダウンロード
wget -O data/ja_wiki/validation_0.jsonl.gz --no-check-certificate \
https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/validation_0.jsonl.gz?ref_type=heads

# 圧縮ファイルの展開
gunzip data/ja_wiki/*.gz
```

## 🚀 使用方法

以下の順序でスクリプトを実行してください：

### ステップ1: HuggingFaceモデルのNeMo形式への変換

```bash
python src/01_convert_hf_to_nemo.py
```

**処理内容:**
- `Qwen/Qwen2.5-0.5B`モデルをHuggingFaceからダウンロード
- NeMo形式の`qwen2.5-0.5b.nemo`ファイルに変換
- ワークスペースルートに保存

### ステップ2: 日本語Wikipediaデータの前処理

```bash
python src/00convert_ja.py
```

**処理内容:**
- 入力: `{"text": "記事内容", "meta": {"id": "ID", "title": "タイトル", "url": "URL"}}`
- 出力: `{"input": "質問", "output": "回答"}`形式に変換
- 各記事から複数の質問パターンを自動生成
- 適切な長さのチャンクに分割（50-1500文字）
- 結果は`/workspace/data/training_data/`に保存

### ステップ3: ファインチューニング実行

#### Option A: LoRA PEFTファインチューニング（推奨、メモリ効率的）

```bash
python src/02_qwen25_peft.py
```

#### Option B: フルモデルSFTファインチューニング

```bash
python src/03_qwen25_sft.py
```

## 📁 プロジェクト構成

```
.
├── README.md                          # このファイル
├── src/                               # メインスクリプト
│   ├── 01_convert_hf_to_nemo.py      # HF→NeMo変換（13行）
│   ├── 00convert_ja.py               # ja_wiki→SFT形式変換（276行）
│   ├── 02_qwen25_peft.py             # LoRA PEFTファインチューニング（107行）
│   └── 03_qwen25_sft.py              # フルモデルSFTファインチューニング（102行）
├── data/                             # データディレクトリ
│   ├── ja_wiki/                      # 日本語Wikipediaデータ（生データ）
│   │   ├── train_0.jsonl ~ train_13.jsonl
│   │   └── validation_0.jsonl
│   └── training_data/                # SFT形式変換済みデータ
│       ├── training.jsonl            # 訓練データ
│       └── validation.jsonl          # バリデーションデータ
├── models/                           # モデル保存ディレクトリ
│   └── checkpoints/                  # ファインチューニングチェックポイント
│       ├── qwen25_500m_peft/         # LoRA PEFTモデル
│       └── qwen25_500m_sft/          # フルSFTモデル
└── qwen2.5-0.5b.nemo                # 変換済みNeMoモデル
```

## 🔬 詳細仕様

### データ変換仕様（00convert_ja.py）

- **チャンクサイズ**: 800文字
- **出力長制限**: 50-1500文字
- **質問長制限**: 最大200文字
- **生成パターン**: 要約版 + チャンク版 + キーワード版

### ファインチューニング設定

| パラメータ | LoRA PEFT | フルSFT |
|------------|-----------|---------|
| `peft_scheme` | `"lora"` | `"none"` |
| `micro_batch_size` | 2 | 2 |
| `global_batch_size` | 16 | 16 |
| `seq_length` | 2048 | 2048 |
| `max_steps` | 1000 | 1000 |
| `val_check_interval` | 100 | 100 |

### カスタムデータモジュール

両スクリプトで使用される`CustomFineTuningDataModule`は以下の機能を提供：

- JSONLファイルの自動検出
- ファイル存在確認
- 必要に応じたシンボリックリンク作成

## 🛠️ トラブルシューティング

### よくある問題

1. **メモリ不足エラー**
   ```bash
   # LoRA PEFTを使用（推奨）
   python src/02_qwen25_peft.py
   ```

2. **データファイルが見つからない**
   ```bash
   # データ前処理を再実行
   python src/00convert_ja.py
   ```

3. **モデル変換エラー**
   ```bash
   # HuggingFaceモデル変換を再実行
   python src/01_convert_hf_to_nemo.py
   ```

4. **マルチプロセッシングエラー**
   - すべてのスクリプトに`if __name__ == "__main__":`が含まれているため問題なし

### パフォーマンス比較

| 手法 | メモリ使用量 | 学習速度 | 品質 | 推奨用途 |
|------|-------------|----------|------|----------|
| LoRA PEFT | 低 | 高速 | 良好 | 実験・プロトタイプ |
| フルSFT | 高 | 低速 | 最高 | 本格運用 |

### チェックポイント場所

- **LoRA PEFT**: `/workspace/models/checkpoints/qwen25_500m_peft/`
- **フルSFT**: `/workspace/models/checkpoints/qwen25_500m_sft/`

## 📚 参考資料

- [NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [Qwen2.5 Model](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [LLM-jp Corpus v3](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

## 📄 ライセンス

このプロジェクトは各々のコンポーネントのライセンスに従います：
- NeMo: Apache 2.0
- Qwen2.5: 各モデルのライセンス
- LLM-jp Corpus: 各データセットのライセンス

---

**Author**: Kosuke
**Last Updated**: 2025-01
**Version**: 2.0
