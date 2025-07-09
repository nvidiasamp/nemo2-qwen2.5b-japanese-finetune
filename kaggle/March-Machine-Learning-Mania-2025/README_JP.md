# NCAA バスケットボールトーナメント予測システム

[English](README_EN.md) | [中文](README_CN.md) | [日本語](README_JP.md)

## はじめに

NCAA バスケットボールトーナメント予測システムは、NCAA バスケットボールトーナメントの試合結果を高精度で予測するために設計された最先端の機械学習ソリューションです。このシステムは、歴史的バスケットボールデータを処理し、関連特徴量を生成し、最適化された XGBoost モデルを訓練し、トーナメントの対戦における勝率予測を生成する高度な予測パイプラインを実装しています。

このシステムは、NCAA バスケットボールトーナメントの結果を予測することを参加者に挑戦させる [March Machine Learning Mania 2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025) Kaggle コンペティション向けに特別に設計されています。

### バージョン 3.0 の主な改良点

- **GPU アクセラレーション**：cudf と cupy を通じた CUDA サポートの追加により、互換性のあるハードウェアでのパフォーマンスが劇的に向上
- **メモリ最適化**：適応的バッチ処理と精度削減によるメモリ管理の強化
- **エラー回復能力**：パイプライン全体での検証、優雅なフォールバック、エラー回復の改善
- **拡張可視化**：キャリブレーション曲線や性別比較分析を含む包括的な視覚的分析
- **多言語ドキュメント**：英語、中国語、日本語での完全なドキュメント提供

### 以前のバージョンの改良点

- **男女両方の予測サポート**：男子と女子の両方の NCAA バスケットボールトーナメントをサポート
- **パフォーマンス最適化**：並列処理とベクトル化操作の改善によるデータ処理の高速化
- **メモリ効率**：大規模データセットを処理するためのメモリ使用量とキャッシング戦略の改善
- **堅牢なエラー処理**：パイプライン全体での検証とエラー回復の改善

## システム要件

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- XGBoost
- joblib
- tqdm
- psutil（メモリ監視用）
- concurrent.futures（並列処理用）
- cupy および cudf（オプション、GPU アクセラレーション用）

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/ncaa-prediction-system.git
cd ncaa-prediction-system

# 仮想環境を作成（オプションですが推奨）
python -m venv myenv
source myenv/bin/activate  # Windowsの場合: myenv\Scripts\activate

# 依存関係をインストール
pip install -r requirements.txt

# GPU依存関係をインストール（オプション）
pip install cupy-cuda11x cudf-cuda11x
```

## システムアーキテクチャ

システムは柔軟性、再現性、およびパフォーマンスを念頭に置いたモジュラーアーキテクチャに従っています：

```
NCAA予測システム
├── データ取得レイヤー
│   ├── 過去の試合データ読み込み
│   ├── チーム情報処理
│   └── トーナメント構造分析
├── 特徴量エンジニアリングレイヤー
│   ├── チームパフォーマンス統計
│   ├── トーナメント進行モデリング
│   ├── 対戦履歴分析
│   └── シードベースの特徴量生成
├── モデル訓練レイヤー
│   ├── 性別特定モデル訓練
│   ├── ハイパーパラメータ最適化
│   ├── 交差検証フレームワーク
│   └── GPU加速学習
├── 予測と評価レイヤー
│   ├── キャリブレーション曲線分析
│   ├── Brierスコア最適化
│   ├── 予測分布分析
│   └── リスク調整戦略
└── 可視化とレポートレイヤー
    ├── インタラクティブなパフォーマンスチャート
    ├── 性別比較分析
    ├── 特徴量重要度の可視化
    └── 予測信頼度分析
```

## コード構造

プロジェクトは予測パイプラインの特定の側面を扱ういくつかのモジュールに編成されています：

- **main.py**：全体のワークフローを調整し、コマンドラインインターフェースを提供
- **data_preprocessing.py**：データ読み込み、探索、訓練-検証分割を処理
- **feature_engineering.py**：生データから特徴量を作成（チーム統計、シード、対戦カード）
- **train_model.py**：性別特定のモデルを持つXGBoostモデル訓練を実装
- **submission.py**：提出用のトーナメント予測を生成
- **evaluate.py**：評価指標と可視化ツールを含む
- **utils.py**：GPUアクセラレーションサポートを含むユーティリティ関数を提供

## 使用方法

### 基本的な使用法

```bash
python main.py --data_path ./data --output_path ./output --target_year 2025
```

### 高度なオプション

```bash
python main.py --data_path ./data \
               --output_path ./output \
               --train_start_year 2016 \
               --train_end_year 2024 \
               --target_year 2025 \
               --explore \
               --random_seed 42 \
               --n_cores 8 \
               --use_gpu \
               --generate_predictions
```

### コマンドライン引数

- `--data_path`：データディレクトリへのパス（デフォルト：'../input'）
- `--output_path`：出力ファイルのパス（デフォルト：'../output'）
- `--explore`：データ探索と可視化を有効にする（デフォルト：False）
- `--train_start_year`：訓練データの開始年（デフォルト：2016）
- `--train_end_year`：訓練データの終了年（デフォルト：2024）
- `--target_year`：予測対象年（デフォルト：2025）
- `--random_seed`：再現性のための乱数シード（デフォルト：42）
- `--n_cores`：並列処理に使用するCPUコア数（デフォルト：自動検出）
- `--use_cache`：処理を高速化するためにキャッシュデータを使用（デフォルト：False）
- `--use_gpu`：互換性のある操作にGPUアクセラレーションを有効にする（デフォルト：False）
- `--xgb_trees`：XGBoostモデルのツリー数（デフォルト：500）
- `--xgb_depth`：XGBoostモデルの最大ツリー深度（デフォルト：6）
- `--xgb_lr`：XGBoostモデルの学習率（デフォルト：0.05）
- `--generate_predictions`：すべての可能な対戦の予測を生成（デフォルト：False）
- `--output_file`：予測の出力ファイル名（デフォルト：タイムスタンプベース）
- `--load_models`：新しいモデルを訓練する代わりに事前訓練されたモデルを読み込む（デフォルト：False）
- `--men_model`：男子モデルファイルへのパス（デフォルト：None）
- `--women_model`：女子モデルファイルへのパス（デフォルト：None）
- `--men_features`：男子特徴量ファイルへのパス（デフォルト：None）
- `--women_features`：女子特徴量ファイルへのパス（デフォルト：None）

## データ要件

システムはデータディレクトリに以下のCSVファイルを期待します：

- **MTeams.csv**：男子チーム情報
- **WTeams.csv**：女子チーム情報
- **MRegularSeasonCompactResults.csv**：男子レギュラーシーズン結果
- **WRegularSeasonCompactResults.csv**：女子レギュラーシーズン結果
- **MNCAATourneyCompactResults.csv**：男子トーナメント結果
- **WNCAATourneyCompactResults.csv**：女子トーナメント結果
- **MRegularSeasonDetailedResults.csv**：男子レギュラーシーズン詳細統計
- **WRegularSeasonDetailedResults.csv**：女子レギュラーシーズン詳細統計
- **MNCAATourneySeeds.csv**：男子トーナメントシード情報
- **WNCAATourneySeeds.csv**：女子トーナメントシード情報
- **SampleSubmissionStage1.csv**：サンプル提出フォーマット

## 主な機能

### GPU アクセラレーション

- cupy と cudf ライブラリによる CUDA ベースのアクセラレーション
- フォールバックメカニズムを備えた適応的 GPU メモリ管理
- 特徴量エンジニアリングとモデル訓練のための最適化されたテンソル操作
- CPU への優雅な低下を伴う自動ハードウェア検出

### 男女両方の予測

- 男子と女子のトーナメントそれぞれに訓練された個別のモデル
- 各トーナメントの特性に合わせた性別特有の特徴量エンジニアリング
- 包括的なトーナメントカバレッジのための結合予測出力
- 性別間の予測パターンの比較分析

### 高度な特徴量エンジニアリング

- チームパフォーマンス統計の計算
- シード情報処理
- 過去の対戦分析
- トーナメント進行確率推定
- お気に入り-ロングショットバイアス補正
- 性別特有の特徴量調整

### パフォーマンス最適化

- 計算集約型操作のマルチコア並列処理
- 互換性のある操作のGPUアクセラレーション
- 冗長な計算を避けるメモリキャッシング
- 効率向上のためのベクトル化操作
- メモリ使用量のモニタリングと最適化
- パフォーマンス追跡のための時間認識関数デコレータ

### 堅牢な評価

- 複数の指標（Brierスコア、対数損失、精度、ROC AUC）
- キャリブレーション曲線分析
- 性別による予測分布の可視化
- Brierスコア特性に基づくリスク最適化提出戦略
- 男子と女子の予測モデル間の比較分析

## 予測パイプライン

1. **データ読み込み**：男女両方の過去のバスケットボールデータを読み込み、前処理
2. **特徴量エンジニアリング**：性別特有の考慮事項を持つ生データから予測特徴量を作成
3. **モデル訓練**：男子と女子のトーナメントに別々のXGBoostモデルを訓練
4. **評価**：複数の指標を使用してモデルのパフォーマンスを評価
5. **予測生成**：すべての可能なトーナメント対戦の予測を作成
6. **リスク戦略適用**：Brierスコアの最適リスク戦略を適用
7. **提出作成**：コンペティション提出用に予測をフォーマット

## 理論的洞察

システムは予測精度を向上させるためにいくつかの理論的洞察を実装しています：

- **Brierスコア最適化**：約33.3%の勝率を持つ予測には、期待Brierスコアを最適化するための戦略的リスク調整が適用されます。
- **お気に入り-ロングショットバイアス補正**：システムは、強いチーム（低シード）の系統的過小評価と弱いチーム（高シード）の過大評価を修正します。
- **時間認識検証**：バスケットボール予測の時間的性質をより良く反映するために、より最近のシーズンを使用して検証が行われます。
- **性別特有のモデリング**：別々のモデルが男子と女子のバスケットボールトーナメントのユニークな特性を捉えます。
- **キャリブレーション理論**：予測された確率が真の勝利確率を正確に反映することを確保する確率キャリブレーション技術を実装します。

## 結果例

システムはいくつかの出力ファイルを生成します：

- 男子と女子両方のトーナメントの訓練済みモデルファイル（men_model.pkl, women_model.pkl）
- 特徴量キャッシュファイル（men_features.pkl, women_features.pkl）
- 予測提出ファイル（submission_YYYYMMDD_HHMMSS.csv）
- モデル評価指標と可視化
- 男子と女子の予測間の比較分析

## 高度な使用法

### GPU アクセラレーション

```python
from utils import gpu_context, to_gpu, to_cpu

# GPUが利用可能かどうかを確認
with gpu_context(use_gpu=True) as gpu_available:
    if gpu_available:
        print("GPUアクセラレーションが有効")
        # データをGPUに移動
        X_gpu = to_gpu(X_train)
        y_gpu = to_gpu(y_train)
        
        # GPUで処理
        # ... 処理ステップ ...
        
        # 結果をCPUに戻す
        X_processed = to_cpu(X_gpu)
        y_processed = to_cpu(y_gpu)
    else:
        print("GPUが利用できません、CPUを使用")
        X_processed = X_train
        y_processed = y_train
```

### 性別特有のモデルの訓練

```python
from train_model import train_gender_specific_models
from utils import save_features

# 両性別の特徴量を準備
m_features, m_targets = merge_features(m_train_data, m_team_stats, m_seed_features, m_matchup_history)
w_features, w_targets = merge_features(w_train_data, w_team_stats, w_seed_features, w_matchup_history)

# 性別特有のモデルを訓練
models = train_gender_specific_models(
    m_features, m_targets, w_features, w_targets,
    m_tourney_train, w_tourney_train,
    random_seed=42, save_models_dir='./models'
)

# 個別のモデルにアクセス
men_model = models['men']['model']
women_model = models['women']['model']
```

### 結合予測の生成

```python
from submission import prepare_all_predictions, create_submission

# 両性別の予測を生成
all_predictions = prepare_all_predictions(
    model, features_dict, data_dict, 
    model_columns=model_columns,
    year=2025, 
    gender='both'  # 男子と女子の両方の対戦を処理
)

# 提出ファイルを作成
submission = create_submission(all_predictions, sample_submission, 'submission_2025.csv')
```

## パフォーマンスに関する注意

- 特徴量エンジニアリングはパイプラインで最も時間のかかる部分です。以前に計算された特徴量を再利用するには `--use_cache` フラグを使用してください。
- GPUアクセラレーションはパフォーマンスを大幅に向上させますが、互換性のあるハードウェアとドライバが必要です。
- 非常に大きなデータセットの場合は、速度とメモリ使用量のバランスを取るために `n_cores` パラメータを調整してください。
- システムにはメモリ使用量を効果的に管理するための自動バッチサイズ最適化が含まれています。

## 可視化

システムはモデルのパフォーマンスを理解するのに役立ついくつかの可視化を生成します：

- 男子と女子のトーナメントの両方の予測分布チャート
- 予測された確率と実際の勝率を示すキャリブレーション曲線
- 最も予測力のある要因を強調する特徴量重要度プロット
- 男子と女子の予測の違いを示す比較プロット
- メモリとパフォーマンスのプロファイリングチャート

## 参考文献

- March Machine Learning Mania 2025: [https://www.kaggle.com/competitions/march-machine-learning-mania-2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025)
- XGBoost: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
- Brierスコア: [https://en.wikipedia.org/wiki/Brier_score](https://en.wikipedia.org/wiki/Brier_score)
- NCAAトーナメント: [https://www.ncaa.com/march-madness](https://www.ncaa.com/march-madness)
- RAPIDS cuDF: [https://docs.rapids.ai/api/cudf/stable/](https://docs.rapids.ai/api/cudf/stable/)
- CuPy: [https://cupy.dev/](https://cupy.dev/)

## 著者

趙俊茗 (Junming Zhao)

## ライセンス

MITライセンス

---

このREADMEは、NCAAバスケットボールトーナメント予測システムの包括的な概要を提供し、セットアップ手順、使用例、および主要な技術的詳細を含みます。質問や貢献については、リポジトリでissueを開いてください。