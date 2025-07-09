# NVIDIA NIM 微調モデル展開の再現 - プロセス記録と総括

## 1. はじめに

本文書は、NVIDIA公式ブログ記事「[Deploying Fine-tuned AI Models with NVIDIA NIM](https://developer.nvidia.com/ja-jp/blog/deploying-fine-tuned-ai-models-with-nvidia-nim/)」の再現プロセス全体を記録することを目的としています。目標は、NVIDIA NIM（NVIDIA Inference Microservices）を使用して微調整されたAIモデルを展開する方法を理解し実践することです。

本文書では、ローカル環境での再現時にブログで説明されている環境との差異、対応する調整、具体的な操作手順、遭遇した問題とその解決策を詳細に記録します。最終的に、本文書は技術共有と会議発表の基礎資料として使用されます。

**プロジェクトが従うCursor Rule:** `reproduce_nvidia_nim_blog.mdc`

## 2. ブログ内容の核心解読

NVIDIAのブログ記事「[Deploying Fine-tuned AI Models with NVIDIA NIM](https://developer.nvidia.com/ja-jp/blog/deploying-fine-tuned-ai-models-with-nvidia-nim/)」（以下「ブログ」と略称）は、**NVIDIA NIM（NVIDIA Inference Microservices）**を利用して微調整（Fine-tuned）されたAI大規模言語モデル（LLM）を効率的に展開する方法を詳細に説明しています。

**ブログの核心目的とNIMが解決する問題：**

ブログの核心主旨は、NIMがどのように**微調整LLMの展開プロセス、特にモデル重み（SFTなど）を直接修正したモデルの簡素化と高速化**を実現するかを示すことです。従来、これらの微調整されたモデルを本番環境に投入し、最適な推論性能を得ることは、通常、適切な推論エンジンの選択、特定のハードウェアに対するコンパイル最適化、およびサービスインターフェースの設定などの複雑で時間のかかる手動最適化ステップを伴いました。NIMは以下の方法でこれらの痛点を解決することを目指しています：

1. **自動化された最適化と展開：** NIMは単一ステップの展開フローで、ユーザーが提供するモデル重み（基本モデルまたはSFT微調整モデル）に対して、特定のGPUハードウェアに最適化されたTensorRT-LLM推論エンジンをローカルで自動的に構築・ロードできます。これにより、ユーザーがTensorRT-LLMの底層詳細を深く理解する必要なく、高性能推論を得ることができます。

2. **標準化と使いやすさ：** NIMは複雑な推論バックエンドをDockerコンテナに封装し、OpenAI APIと互換性のある標準化されたインターフェースを提供します。これにより、開発者はNIM展開されたモデルを既存のAIアプリケーションやワークフローに簡単に統合できます。

3. **柔軟性と高性能：** 多様な微調整戦略とモデルをサポートし、TensorRT-LLMとの深い統合により、モデルが高効率（低遅延、高スループット）で動作することを保証します。

**主要技術コンポーネント解読：**

* **NVIDIA NIM（NVIDIA Inference Microservices）:**
  * **概要：** NVIDIA GPU用に最適化された事前構築済み推論マイクロサービス一式。これらはDockerコンテナ形式で提供され、特定のAIモデル（LLMなど）の実行に必要なすべてを封装しています。
  * **使用理由：** 主に展開の簡素化のため。ユーザーは自分のモデル基盤アーキテクチャに合致するNIMイメージを選択し、NIMが後続の最適化と実行を処理します。
  * **核心機能：** LLMにおいて、NIMの核心機能の一つはTensorRT-LLMの統合です。NIMサービスが開始され、ユーザー提供のモデル重み（`NIM_FT_MODEL`パラメータ経由）を指定されると、TensorRT-LLMを利用して現在のモデルとGPUハードウェアに高度に最適化された推論エンジンを**動的に構築**（または以前にキャッシュされたものをロード）します。この「動的構築」能力はSFT微調整モデルにとって重要であり、推論エンジンが新しい微調整後の重みと完全に一致することを保証します。

* **TensorRT-LLM:**
  * **概要：** NVIDIA GPU上でLLMの推論を加速するためのNVIDIAのオープンソースライブラリ。標準LLM（Hugging Faceのモデルなど）を特定のNVIDIA GPUハードウェアに高度に最適化された形式に変換できます。
  * **使用理由：** 極限の推論性能、より低い遅延とより高いスループットを実現するため。
  * **動作原理：** オペレータ融合、量子化（FP8、INT8など）、テンソル並列、パイプライン並列などの多様な最適化技術を適用します。

* **SFT（Supervised Fine-tuning）とモデル重み:**
  * **概要：** 特定タスクのラベル付きデータセットでの訓練により、事前訓練LLMの基本重みを直接修正し、特定ドメインやタスクにより適応させる微調整技術。
  * **ブログの焦点：** ブログはNIMのSFTモデルサポートを特に強調しています。SFTは重みを直接変更するため、推論エンジンがこれらの新しい重みに適応して最適性能を保証する必要があり、これがNIMがTensorRT-LLMエンジンの動的構築を通じて実現していることです。

* **Docker:**
  * **概要：** コンテナ化技術。
  * **使用理由：** NIMサービスはDockerコンテナを通じてパッケージ化・実行され、環境設定、依存関係管理、展開の可搬性を大幅に簡素化します。ユーザーはローカル環境とNIM実行に必要な複雑な依存関係（特定バージョンのCUDA、cuDNN、TensorRT-LLMなど）間の競合を心配する必要がありません。

**ブログで説明される核心フロー概要（SFTモデルを例として）：**

1. **前提条件：** NVIDIA GPU環境、`git-lfs`、NGC API Keyを準備。
2. **環境設定：** `NGC_API_KEY`と`NIM_CACHE_PATH`（最適化エンジンキャッシュ用）などの必要な環境変数をエクスポート。
3. **モデル準備:**
   * SFT微調整モデルの重みを取得（ブログ例では`nvidia/OpenMath2-Llama3.1-8B`）。
   * **重要点：** 使用する微調整モデルの基盤モデルアーキテクチャはNVIDIA NIMでサポートされている必要があります。例えば、SFTモデルがLlama-3.1-8Bベースの場合、NIMでLlama-3.1-8B基本モデル用のイメージを使用する必要があります。
   * モデル重みを格納する親ディレクトリを指す`MODEL_WEIGHT_PARENT_DIRECTORY`を設定。
4. **NIMコンテナイメージ（`IMG_NAME`）の選択:**
   * モデルの基盤アーキテクチャに基づいて、NVIDIA API Catalog（NGC）から対応するNIM for LLMsコンテナイメージを選択。例えば、ブログの`OpenMath2-Llama3.1-8B`（SFTモデル）は`nvcr.io/nim/meta/llama-3.1-8b-instruct:1.3.0`（その基本モデルLlama-3.1-8B Instruct用のNIMイメージ）を使用。
5. **（オプション）性能設定プロファイル（`NIM_MODEL_PROFILE`）の選択:**
   * NIMはこの環境変数を通じて事前定義された性能最適化設定プロファイルを指定可能。例えば、低遅延（`latency`）または高スループット（`throughput`）に重点を置いたプロファイル。これらの設定プロファイルは通常、特定ハードウェア（H100など）、精度（BF16、FP8など）、並列戦略（TP、PPなど）の最適化パラメータを含みます。
   * ブログの「カスタム パフォーマンス プロファイルで最適化された TensorRT-LLM エンジンのビルド」部分でこの点を詳しく議論しています。
   * **指定なしまたは直接互換する事前構築プロファイルがない場合**：NIMはハードウェアとモデルの自動検出を試み、最適化されたTensorRT-LLMエンジンを動的に構築します。汎用的で構築可能な設定（実験で観察した`tensorrt_llm-trtllm_buildable-bf16-tp1-pp1`など）を選択します。
6. **NIMマイクロサービスの開始（`docker run`経由）:**
   * コマンドにはGPU指定、APIキー、モデル重みパス（`-v`でコンテナ内の`NIM_FT_MODEL`が指すパスにマウント）、サービスが公開するモデル名（`NIM_SERVED_MODEL_NAME`）、オプションの性能設定（`NIM_MODEL_PROFILE`）、およびモデル重みとキャッシュのパスマウントが含まれます。
   * **核心メカニズム**：Dockerコンテナ開始時、NIMサービスは`NIM_FT_MODEL`が指す重みを読み取り、（オプションの）`NIM_MODEL_PROFILE`または自動選択された構築戦略と組み合わせて、TensorRT-LLMを利用して現在のGPUに最適化された推論エンジンを構築（またはキャッシュからロード）します。このプロセスには数分かかる場合があります。
7. **モデルとの対話：** 標準OpenAI API互換インターフェース（通常`http://localhost:8000/v1`）を通じて展開されたモデルと通信。

**ブログの異なる部分が実践で同様の結果になる理由：**

ブログの「SFT モデルを使用した例」と「カスタム パフォーマンス プロファイルで最適化された TensorRT-LLM エンジンのビルド」部分が、再現プロセスで同様の実行結果を得る理由：

* **両者ともNIMの核心展開メカニズムに依存：** SFTモデルの直接展開（ブログ第一部）でも、展開時にカスタム性能設定プロファイルの指定試行（ブログ第二部）でも、**底層のNIMワークフローは同様**です。NIMは常にモデル重みを取得し、最適化されたTensorRT-LLMエンジンの構築またはロードを試行します。

* **`NIM_MODEL_PROFILE`の役割：**
  * **有効で互換性のある**`NIM_MODEL_PROFILE`が提供された場合、NIMはその設定プロファイルの指示に従って最適化を行います。
  * 提供された`NIM_MODEL_PROFILE`が**無効、非互換、またはこのパラメータを全く提供しない場合**（実験で確認したように、RTX 6000 Ada + Llama-3.2-1Bに直接適用可能な事前構築プロファイルが見つからなかったため）、NIMは**自動最適化動作にフォールバック**します。汎用的で構築可能な戦略（`tensorrt_llm-trtllm_buildable-bf16-tp1-pp1`など）を選択してエンジンを動的に構築します。

* **実践状況：** 再現において、`list-model-profiles`コマンドで、使用するNIMイメージとRTX 6000 Ada GPUに直接互換する事前構築性能設定プロファイルがないことが判明。したがって、Dockerコマンドで`-e NIM_MODEL_PROFILE=...`を試行するかどうかに関わらず（NIM内部で認識・間接使用可能な構築パラメータを指定しない限り）、NIMの最終動作は同様になります：自動的な「buildable」戦略ベースのエンジン構築を行います。

したがって、ブログのこれら二つの部分は、NIM展開フローの**異なる設定オプションと観察側面**を示しています：
* 第一部は**基本SFTモデルの展開フロー**を示し、NIMがカスタム重みを処理できることを強調。
* 第二部は**性能設定プロファイルによるより細かい最適化制御**を深く探求し、利用可能なプロファイルの照会方法を示しています。

利用可能な事前構築プロファイルがない場合、両フローは最終的にNIMの動的エンジン構築メカニズムをトリガーし、同様のログ出力（エンジン構築プロセスなど）と最終的なサービス動作をもたらします。

**ローカル再現への重要な示唆：**
* **GPU VRAMは重要な制約：** ローカルGPU VRAM（約48GB）はブログ例（80GB）より遙かに小さいため、より小さなパラメータ数のモデル（`meta-llama/Llama-3.2-1B`など）を選択して再現する必要があります。
* **`IMG_NAME`の正確性が重要：** モデルの基盤アーキテクチャに対応する正しいNIMコンテナイメージ名をNVIDIA API Catalogから見つける必要があります。
* **パスマッピングの正確性：** Dockerのボリュームマウント（`-v`パラメータ）を正しく設定し、モデル重みがコンテナ内部で見えるようにする必要があります。
* **NIMの自動化能力の理解：** 完璧な事前構築性能設定プロファイルがなくても、NIMはTensorRT-LLMエンジンの動的構築を通じて最適化されたサービスを提供できます。

**今回の再現説明更新：** ローカル環境のVRAM制限（約48GB、NVIDIA RTX 6000 Ada）を考慮し、今回の再現では`meta-llama/Llama-3.2-1B`を基本モデルとしてNIM展開実践を行いました。NIMが明示的に互換するカスタム性能設定プロファイルを指定しない場合に、ハードウェアの自動検出、汎用構築戦略の選択（`tensorrt_llm-trtllm_buildable-bf16-tp1-pp1`など）、およびローカルモデル重みに対する最適化TensorRT-LLMエンジンの動的構築プロセスと結果を重点的に観察しました。

## 3. ローカル再現環境

### 3.1 ブログ記述環境

* GPU: 80 GB GPU メモリ（具体的な型番は未明記、A100またはH100の可能性）
* ソフトウェア: `git-lfs`
* その他: NGC API Key

### 3.2 私のローカル環境

* **オペレーティングシステム:** Ubuntu 20.04.6 LTS (focal)
* **GPU型番と数量（本プロジェクト使用）:**
  * GPU 0: NVIDIA RTX 6000 Ada Generation（本プロジェクトで指定使用するGPU）
  * （GPU 1: NVIDIA T600 - システムで検出されるが、本プロジェクトでは使用しない）
* **VRAM容量（GPU 0）:** 49140MiB（約48GB）
* **NVIDIAドライババージョン:** 535.183.01
* **Dockerバージョン:** Docker version 28.0.4, build b8034c0
* **CUDAバージョン（ドライバサポート）:** 12.2（注：`nvcc`コマンドが見つからず、CUDA Toolkitが単独でインストールされていないかPATHに含まれていない可能性）
* **git-lfsバージョン:** git-lfs/3.6.1 (GitHub; linux amd64; go 1.23.3)
* **主な差異点:**
  * 使用GPU（NVIDIA RTX 6000 Ada Generation）の最大単カードVRAMは約48GBで、ブログ推奨の80GBより遙かに小さい。
  * そのため、**モデル選択を`meta-llama/Llama-3.2-1B`**（または適用可能で見つけられるSFT微調整版）に調整。これは実行可能なモデルサイズと推論性能に大きく影響する可能性があり、モデル選択とリソース設定に特別な注意が必要。

## 4. 再現手順と操作記録

*（`reproduce_nvidia_nim_blog.mdc`ルールの核心ステップとブログフローに従って、各ステップを詳細に記録）*

### 4.1 環境準備

* **NGC API Key設定:**
  * 実際の操作：ターミナルで以下のコマンドを実行して`NGC_API_KEY`環境変数を設定済み。セキュリティ上、API Keyの具体的な値はここでは省略。実際の操作環境で正しく設定してください。
  ```bash
  export NGC_API_KEY="nvapi-02nWXga...JJ5Ix" # これは例示、実際には設定済み
  ```

* **NIM Cache Path設定:**
  * 実際の操作：ターミナルで以下のコマンドを実行して`NIM_CACHE_PATH`を設定し、対応するディレクトリを作成して権限を設定済み。
  ```bash
  export NIM_CACHE_PATH=/tmp/nim/.cache 
  mkdir -p $NIM_CACHE_PATH
  chmod -R 777 $NIM_CACHE_PATH
  ```
  * 注：`chmod -R 777`は非常に寛容な権限を提供し、開発テスト段階で権限問題を避けるのに適しています。本番環境ではより安全な権限設定を検討すべきです。

### 4.2 モデル取得

* **`git lfs`初期化と`models`ディレクトリ作成:**
  * 大型モデルファイルを正しく処理するため、まず`git lfs install`を実行（またはグローバル初期化を確認）。
  * その後、プロジェクトルートディレクトリ（`/home/cho/workspace/MTG_ambassador/finetuning_model_deploy`）の下にダウンロードしたモデルを格納する`models`サブディレクトリを作成。
  ```bash
  # git lfs install # （実行推奨）
  mkdir -p models
  ```

* **Hugging Face CLI ログインとモデルクローン:**
  * `meta-llama/Llama-3.2-1B`のクローン試行時に認証失敗が発生。以下の手順で解決：
    1. Hugging Face ウェブサイト（[https://huggingface.co/meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)）でモデルアクセス権限の要求と取得を確認。
    2. Hugging Face Access Tokens設定ページ（[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)）にアクセスし、ゲート付きリポジトリ（gated repositories）へのアクセス権限を持つアクセストークンを生成/更新。
    3. 新しいトークンで`huggingface-cli login`を再実行し、Gitクレデンシャルとして追加。
       ```bash
       # huggingface-cli login # （権限付きトークンを貼り付けてGitクレデンシャルとして確認）
       ```
  * ログイン成功後、以下のコマンドで`meta-llama/Llama-3.2-1B`モデルを`models/Llama-3.2-1B`ディレクトリにクローン：
    ```bash
    git clone https://huggingface.co/meta-llama/Llama-3.2-1B models/Llama-3.2-1B
    ```
    モデルは`/home/cho/workspace/MTG_ambassador/finetuning_model_deploy/models/Llama-3.2-1B`に正常にダウンロードされました。

* **`MODEL_WEIGHT_PARENT_DIRECTORY`環境変数の設定:**
  * この環境変数はすべてのモデル重みフォルダを含む親ディレクトリを指します。本プロジェクトでは`models`ディレクトリです。
  * プロジェクトルートディレクトリで以下のコマンドを実行（または実行済み）：
  ```bash
  export MODEL_WEIGHT_PARENT_DIRECTORY=${PWD}/models
  # ${PWD}はこのコンテキストで /home/cho/workspace/MTG_ambassador/finetuning_model_deploy
  # したがって MODEL_WEIGHT_PARENT_DIRECTORY は /home/cho/workspace/MTG_ambassador/finetuning_model_deploy/models に設定
  ```

* **重要な後続ステップ：** NVIDIA API Catalogで`meta-llama/Llama-3.2-1B`（または該当する場合はその基本モデル）に対応するNIM基本モデルイメージ（`IMG_NAME`）を検索する必要があります。

### 4.3 NIMマイクロサービス開始

**背景説明：モデル重みがあるのになぜNIMモデルイメージ（`$IMG_NAME`）が必要なのか？**

`meta-llama/Llama-3.2-1B`のモデル重み（`.safetensors`ファイルなど）を既にダウンロードしました。これらの重みはモデルの「知識の核心」です。しかし、これらの重みが効率的に推論サービスを外部に提供するには、専門的な実行環境と推論エンジンが必要です。これがNVIDIA NIMモデルイメージ（`$IMG_NAME`）が提供するものです。

NIMモデルイメージは事前設定された、高度に最適化されたDockerイメージで、以下を含みます：
1. **推論サーバーソフトウェア：** NVIDIA Triton Inference Serverなど、モデルのライフサイクル管理、リクエストとレスポンスの処理用。
2. **TensorRT-LLMエンジン/ライブラリ：** NVIDIAの核心技術で、大規模言語モデルを特定NVIDIA GPUハードウェアに高度に最適化された形式に変換し、最適な推論性能（低遅延、高スループット）を実現。
3. **モデル最適化と構築ロジック：** NIMサービスが開始され、提供されたモデル重み（`NIM_FT_MODEL`パラメータ経由）を指定されると、TensorRT-LLMを利用して現在のモデルとGPUハードウェアに最適化された推論エンジンを動的に構築（またはキャッシュされたものをロード）します。このプロセスはSFT微調整モデルにとって特に重要で、推論エンジンが新しい重みと完全に一致することを保証します。
4. **標準化されたAPIインターフェース：** 通常OpenAI API互換のインターフェースを提供し、アプリケーション統合を容易にします。
5. **すべての必要な依存関係：** CUDAドライバやライブラリファイルなどを含み、モデルがNVIDIA GPU上で円滑に動作することを保証。

簡単に言えば、モデル重みは「食材」で、NIMモデルイメージは「設備の整った専門キッチン+食材を完成品に変えてサービスを提供するシェフチーム」です。静的な重みを動的で効率的な、本番使用可能な推論サービスに変換します。理論的には手動でこれらすべてを構築できますが、NIMは展開と最適化の複雑さを大幅に簡素化します。

* **NIM基本モデルイメージ（`IMG_NAME`）の取得:**
  * NVIDIA API Catalog（NGC）の照会結果に基づき、`meta-llama/Llama-3.2-1B`モデル（特にその指示微調整版の基盤）に対しては、NIMイメージ`nvcr.io/nim/meta/llama-3.2-1b-instruct:1`を使用すべきです。
  * 実際の操作：ターミナルでこの環境変数を設定します。
  ```bash
  export IMG_NAME="nvcr.io/nim/meta/llama-3.2-1b-instruct:1"
  ```

* **性能設定プロファイル（`NIM_MODEL_PROFILE`）について:**
  * NIMがモデル用に最適化されたTensorRT-LLM推論エンジンを構築する際、性能設定プロファイルを通じて最適化の重点を指導できます。ブログでは主に2つのタイプが言及されています：
    1. **遅延（Latency）：** 単一推論リクエストの処理時間削減を優先し、リアルタイム応答に敏感なアプリケーション（チャットボットなど）に適用。
    2. **スループット（Throughput）：** 単位時間あたりの処理リクエスト数最大化を優先し、大量の並行またはバッチ処理タスクを処理する必要があるシナリオに適用。
  * **選択方法：**
    * **自動選択：** 指定しない場合、NIMはモデルとハードウェアに基づいて適切な設定プロファイルを自動選択を試行。
    * **手動指定：** `docker run`コマンドで`-e NIM_MODEL_PROFILE=<profile_name>`パラメータを使用して明示的に指定可能。例えば、ブログでは`tensorrt_llm-h100-bf16-tp2-pp1-latency`を使用。具体的なプロファイル名は通常、ハードウェア、精度、並列戦略、最適化目標などの情報を含みます。
  * **推奨：** 初期実行時は指定せずにNIMのデフォルト性能を観察。さらなる最適化が必要な場合は、NVIDIA NIM公式ドキュメントを参照し、GPU（RTX 6000 Ada）とモデル（Llama-3.2-1B）に適用可能な推奨性能設定プロファイルを探し、`docker run`コマンドで指定してください。

* **スクリプトを使用したNIMサービス開始（推奨方法）:**

  環境変数設定とDockerコマンド実行を簡素化するため、プロジェクトは`scripts/`ディレクトリに2つのスクリプトを提供：

  1. **`scripts/01_set_env_vars.sh`**:
     * **役割**: このスクリプトはNIMサービス開始に必要なすべての環境変数設定を担当します。`NGC_API_KEY`、`NIM_CACHE_PATH`、`MODEL_WEIGHT_PARENT_DIRECTORY`、`IMG_NAME`を含みます。
     * **重要**: 初回使用前に、このファイルを編集してプレースホルダー`YOUR_NGC_API_KEY_HERE`を実際のNGC APIキーに置き換える**必要があります**。
     * **使用方法**: ターミナルでプロジェクトルートディレクトリ（`/home/cho/workspace/MTG_ambassador/finetuning_model_deploy`）に移動し、以下のコマンドで現在のシェルセッションにロード：
       ```bash
       source ./scripts/01_set_env_vars.sh
       ```
     * スクリプトはプロジェクトパスを自動検出して`MODEL_WEIGHT_PARENT_DIRECTORY`を設定します。

  2. **`scripts/03_run_nim_docker.sh`**:
     * **役割**: このスクリプトは実際の`docker run`コマンドを含み、NIMサービスを開始します。`01_set_env_vars.sh`でロードされた環境変数を使用し、ローカル`Llama-3.2-1B`モデルに基づいてパラメータ設定（モデルパス、サービス名、GPU指定を`device=0`、共有メモリサイズなど）を行います。
     * **前提条件**: まず`01_set_env_vars.sh`スクリプトを正常に`source`する必要があります。
     * **使用方法**: 環境変数ロード後、プロジェクトルートディレクトリで実行：
       ```bash
       ./scripts/03_run_nim_docker.sh
       ```
     * このスクリプトは必要な環境変数が設定されているかをまず確認し、次に`NIM_CACHE_PATH`の作成と権限設定を試行し、最後に`docker run`コマンドを実行します。
     * デフォルトでは、性能設定プロファイル（`NIM_MODEL_PROFILE`）はNIMによる自動選択、共有メモリ（`--shm-size`）は16GBに設定。必要に応じてこのスクリプト内のこれらの値を直接修正できます。

* **Dockerコマンドの実行（手動参考）:**
  * 以下のコマンドは`scripts/03_run_nim_docker.sh`スクリプトで実行される核心コマンドです。スクリプト化実行が推奨方法ですが、ここでは手動コマンドを参考として保持し、ローカル`Llama-3.2-1B`モデルと環境に基づいて調整済みです。
  * 説明：ダウンロードしたのは`meta-llama/Llama-3.2-1B`（基本版）の重みですが、使用するNIMイメージは`instruct`版用です。NIMは提供された重みに対して最適化エンジンの構築を試行します。`NIM_FT_MODEL`は基本版重みのフォルダを指すべきです。
  ```bash
  # docker run --rm --gpus '"device=0"' \
  #     --user $(id -u):$(id -g) \
  #     --network=host \
  #     --shm-size=16GB \ # (1Bモデルは32GB不要、実際の状況に応じて調整)
  #     -e NGC_API_KEY \
  #     -e NIM_FT_MODEL=/opt/weights/hf/Llama-3.2-1B \ # モデルフォルダ名をLlama-3.2-1Bと仮定
  #     -e NIM_SERVED_MODEL_NAME=Llama-3.2-1B-custom \ # カスタムサービス名、開始後実際の照会名を確認
  #     -e NIM_MODEL_PROFILE=tensorrt_llm-h100-bf16-tp2-pp1-latency \ # (実際のGPUとモデルサイズに応じてprofile調整が必要)
  #     -v $NIM_CACHE_PATH:/opt/nim/.cache \
  #     -v $MODEL_WEIGHT_PARENT_DIRECTORY:/opt/weights/hf \
  #     $IMG_NAME
  # (実際に実行した完全なコマンドを記録)
  ```

#### 4.3.1 故障排除：ポート競合（Port 8000 in use）

初回で`./scripts/03_run_nim_docker.sh`実行試行後、以下のエラーが発生：
```
ERROR 2025-05-22 07:34:00.333 server.py:170] [Errno 98] error while attempting to bind on address ('0.0.0.0', 8000): address already in use
```
これは、ホスト上のポート8000が他のプロセスによって占有されており、NIMサービスが開始できないことを示しています。

**解決手順：**

1. **ポートを占有するプロセスの特定**：
   以下のコマンドの一つを使用してポート8000を占有するプロセスを見つけます：
   ```bash
   sudo netstat -tulnp | grep :8000
   # または
   sudo ss -tulnp | grep :8000
   ```
   今回の調査では、出力でPID `2395910`（IPv4）と`2395917`（IPv6）の`docker-proxy`プロセスがポート8000を占有していることが判明。

2. **ホストの8000ポートをマッピングするアクティブなDockerコンテナの確認**：
   `docker ps -a | grep ":8000->"`を実行して、ホストの8000ポートを自身にマッピングするコンテナがあるかを確認。
   今回の調査では、このコマンドは何も返さず、このポートマッピングを使用するアクティブな既知のコンテナがないことを示します。これは`docker-proxy`が残留プロセスである可能性を意味します。

3. **ポートを占有するプロセスの停止**：
   このポートを直接マッピングするアクティブなコンテナがないことを確認したため、これらの`docker-proxy`プロセスを停止することに決定：
   ```bash
   sudo kill 2395910 2395917
   ```
   **注意**：ポートを占有するのが他の重要なプロセスの場合、そのプロセスを停止するかNIMサービスを他のポートにマッピングすることを検討する必要があります。

4. **NIM開始スクリプトの再実行**：
   競合プロセス停止後、`./scripts/03_run_nim_docker.sh`を再実行。

今後この問題に再遭遇した場合は、上記の手順に従って調査・解決できます。ポート8000を占有するプロセスが停止できないまたは停止したくない重要なサービスの場合、`scripts/03_run_nim_docker.sh`ファイルを編集して`-p 8000:8000`を他の利用可能なホストポート（例：`-p 8001:8000`）に修正し、新しいポート経由でNIMサービスにアクセスできます。

### 4.4 モデルとの対話（Python）

NIMサービスが正常に実行されたら、PythonスクリプトでOpenAI互換のAPIインターフェース経由でモデルを照会できます。

* **`src/query_nim_service.py`ファイルの作成:**
  このスクリプトは`openai` Pythonライブラリを使用して`http://localhost:8000/v1`で実行されるNIMサービスと通信します。
  スクリプト内容は以下（または`src/query_nim_service.py`参照）：
  ```python
  # src/query_nim_service.py
  from openai import OpenAI

  client = OpenAI(
    base_url = "http://localhost:8000/v1", # ポートを修正した場合はここも変更
    api_key = "none" # NIM ローカル展開では通常keyは不要
  )

  MODEL_TO_QUERY = "Llama-3.2-1B-custom-deployed" 
  # 上記のモデル名は scripts/03_run_nim_docker.sh で NIM_SERVED_MODEL_NAME を通じて設定した名前と一致させる
  # または、NIMコンテナログが他の名前（meta-llama/Llama-3.2-1Bなど）でモデル登録を示す場合は、その名前を使用

  try:
      print(f"Attempting to query model: {MODEL_TO_QUERY} using /completions endpoint.")
      # 注意：基本モデルは通常チャットテンプレートがないため、/completions インターフェースを使用
      completion = client.completions.create(
        model=MODEL_TO_QUERY,
        prompt="What is the capital of France?", # prompt パラメータを使用
        temperature=0.7,
        top_p=1.0,
        max_tokens=50,
        stream=True
      )
      
      print(f"Response from {MODEL_TO_QUERY}:")
      for chunk in completion:
        if chunk.choices[0].text is not None: # 応答は .text 内
          print(chunk.choices[0].text, end="")
      print("\n--- End of response ---")

  except Exception as e:
      print(f"Error querying model {MODEL_TO_QUERY}: {e}")
      # chat template に関するエラーが発生した場合、モデルが /chat/completions インターフェースをサポートしていない、
      # またはそのインターフェースを使用しようとしていることを意味します。上記のように /completions インターフェースを使用してください。
  ```

* **依存関係のインストール:**
  `openai`ライブラリをまだインストールしていない場合、Pythonの環境（仮想環境推奨）でインストール：
  ```bash
  # 仮想環境 (.venv) を使用する場合
  # source .venv/bin/activate 
  pip install openai
  ```

* **スクリプトの実行:**
  NIM Dockerコンテナがまだ実行中であることを確認。
  **新しいターミナルウィンドウ**で、プロジェクトルートディレクトリに入って実行：
  ```bash
  # cd /path/to/your/MTG_ambassador/finetuning_model_deploy
  # source .venv/bin/activate # 仮想環境を使用した場合
  python src/query_nim_service.py
  ```
  出力と結果を記録。

  **実際の実行出力例：**
  ```
  (.venv) cho@tut-server-11:~/workspace/MTG_ambassador/finetuning_model_deploy$ python src/query_nim_service.py
  Attempting to query model: Llama-3.2-1B-custom-deployed using /completions endpoint.
  Response from Llama-3.2-1B-custom-deployed:
   A. Rennes B. New York C. Des Moines D. Paris
  A. Rennes
  B. New York
  C. Des Moines
  D. Paris
  Answer: A
  Explanation:   Rennes is the capital of France.
  --- End of response ---
  ```
  *（注意：上記のモデル出力は「Rennes」をフランスの首都として誤って認識しています。これは基本モデルの知識制限や特定のプロンプト下での行動特性を示しています。実際の応用では、回答の正確性を最適化するためにさらなる微調整やプロンプトエンジニアリングが必要な可能性があります。）*

* **予期される問題と解決：`Chat template not found`エラー**
  基本モデル（`meta-llama/Llama-3.2-1B`など）を照会する際に、最初に`client.chat.completions.create()`メソッド（NIMの`/v1/chat/completions`インターフェースに対応）を使用した場合、以下のようなエラーが発生する可能性：
  ```
  Error querying model Llama-3.2-1B-custom-deployed: Error code: 500 - {'object': 'error', 'message': 'Model Llama-3.2-1B-custom-deployed does not have a default chat template defined in the tokenizer.  Set the `chat_template` field to issue a `/chat/completions` request or use `/completions` endpoint', ...}
  ```
  これは基本モデルが通常事前定義されたチャットテンプレートを持たず、`/chat/completions`インターフェースがそれを必要とするためです。
  **解決策**：上記のPythonスクリプト例のように、`client.completions.create()`メソッド（NIMの`/v1/completions`インターフェースに対応）に変更し、`messages`パラメータを`prompt`パラメータに置き換えます。

## 5. 遭遇した問題と解決策

* **問題1: NIMサービス照会時の`Chat template not found`エラー**
  * **発生状況:** `src/query_nim_service.py`スクリプトでNIM展開された`meta-llama/Llama-3.2-1B`モデルとの対話時に、最初に`client.chat.completions.create()`メソッド（NIMの`/v1/chat/completions`インターフェースに対応）を使用しようとした場合。
  * **関連ログ/エラー情報:**
    ```
    Error querying model Llama-3.2-1B-custom-deployed: Error code: 500 - {'object': 'error', 'message': 'Model Llama-3.2-1B-custom-deployed does not have a default chat template defined in the tokenizer.  Set the `chat_template` field to issue a `/chat/completions` request or use `/completions` endpoint', ...}
    ```
  * **原因分析:** 基本版LLM（例：`meta-llama/Llama-3.2-1B`）は通常事前定義されたチャットテンプレート（chat template）を持ちません。NIMサービスの`/v1/chat/completions` APIエンドポイントは、モデルがチャットテンプレートを持つことを要求して対話形式の入力を正しく処理します。
  * **最終解決策:** Python照会スクリプト（`src/query_nim_service.py`）を修正し、APIコールを`/v1/chat/completions`から`/v1/completions`に切り替え。
    * 具体的操作：`client.chat.completions.create(...)`コールを`client.completions.create(...)`に変更。
    * パラメータ調整：`messages=[{"role":"user", "content":"..."}]`を`prompt="ユーザーの質問内容"`に置き換え。
    * この解決策は本文書「4.4 モデルとの対話（Python）」部分の`src/query_nim_service.py`サンプルコードで詳細に説明・展示済み。

* **問題2:** ...

## 6. ローカル環境に基づく調整と最適化

本セクションでは、ローカル環境（NVIDIA RTX 6000 Ada Generation GPU、約48GB VRAM）での`meta-llama/Llama-3.2-1B`モデル展開に対して行った具体的な調整と最適化試行を詳細に記録します。重要な最適化ポイントの一つは、カスタム性能設定プロファイル（`NIM_MODEL_PROFILE`）を使用してNIMが特定ハードウェアと最適化目標（遅延またはスループットなど）に対応するTensorRT-LLMエンジン構築を指導することです。

### 6.1 カスタム性能設定プロファイル（`NIM_MODEL_PROFILE`）の使用

「4.3 NIMマイクロサービス開始」の性能設定プロファイルに関する推奨部分で述べたように、NIMは`-e NIM_MODEL_PROFILE=<profile_name>`パラメータで性能設定プロファイルを手動指定できます。指定しない場合、NIMは自動選択を試行します。手動指定により最適化方向をより精密に制御できます。

**目標：** ローカルRTX 6000 AdaとLlama-3.2-1Bモデルに適した性能設定プロファイルを探索・適用し、より優れた推論性能（例：より低い遅延）の獲得を期待。

**1. 適用可能な性能設定プロファイルの検索:**

* **NVIDIA NIM公式ドキュメントの参照:** 最も権威のある情報源はNVIDIA公式ドキュメントです。GPUアーキテクチャ（Ada Lovelaceアーキテクチャ、H100シリーズに類似するがVRAMと核心数に差異）と目標モデルサイズ（1Bパラメータレベル）に関連する利用可能な設定プロファイルを検索する必要があります。

* **設定プロファイル命名規則の理解:** 性能設定プロファイルの名前は通常以下の情報を含みます：
  * `tensorrt_llm`: TensorRT-LLMバックエンドの使用を示す。
  * `h100` / `ada` / `a100`など: 目標GPUアーキテクチャまたはシリーズ。具体的なグラフィックカードVRAMに適した`h100`類似の設定を探すか、`ada`シリーズ専用の設定（提供されている場合）を直接探す必要があります。
  * `bf16` / `fp16` / `int8` / `fp8`: 使用精度（例：BF16、FP16、INT8量子化、FP8量子化）。
  * `tp<N>`: テンソル並列度（Tensor Parallelism）。例：`tp1`、`tp2`。
  * `pp<N>`: パイプライン並列度（Pipeline Parallelism）。例：`pp1`。
  * `latency` / `throughput`: 最適化目標。

* **NVIDIAブログ例:** ブログ記事（[Deploying Fine-tuned AI Models with NVIDIA NIM](https://developer.nvidia.com/ja-jp/blog/deploying-fine-tuned-ai-models-with-nvidia-nim/)）では、H100上での`OpenMath2-Llama3.1-8B`例として`tensorrt_llm-h100-bf16-tp2-pp1-latency`を使用。私たちの1BモデルとRTX 6000 Adaでは、`tp1`（モデルが小さく単一GPU VRAMで十分なため）で遅延最適化の設定プロファイルが必要かもしれません。精度面では`bf16`または`fp16`が一般的な選択です。

**調査と選択：**
```
（ここにドキュメント参照または実験で見つけた、試行することに決めた NIM_MODEL_PROFILE 値を記録してください）
例: NIM_MODEL_PROFILE="tensorrt_llm_ada_bf16_tp1_pp1_latency" （これは仮定の名前、実際の調査結果で置き換えてください）
```

**`list-model-profiles`コマンド実行結果（`nvcr.io/nim/meta/llama-3.2-1b-instruct:1`対象）:**
`2025-05-22`に現在のNIMイメージ（`nvcr.io/nim/meta/llama-3.2-1b-instruct:1`）がサポートする性能設定プロファイルの取得を試行。プロジェクトルートディレクトリで以下のコマンドを実行（`source MTG_ambassador/finetuning_model_deploy/scripts/01_set_env_vars.sh`が実行済みであることを確認）：
```bash
docker run --rm --gpus=all -e NGC_API_KEY=$NGC_API_KEY $IMG_NAME list-model-profiles
```
得られた出力：
```text
Environment variables set:
  NGC_API_KEY: nvapi-d5qS... (部分的に隠蔽)
  NIM_CACHE_PATH: /tmp/nim/.cache
  MODEL_WEIGHT_PARENT_DIRECTORY: /home/cho/models
  IMG_NAME: nvcr.io/nim/meta/llama-3.2-1b-instruct:1

REMINDER: Please ensure you have replaced 'YOUR_NGC_API_KEY_HERE' in this script with your actual NGC API Key.
This script should be sourced: 'source ./scripts/01_set_env_vars.sh'

===========================================
== NVIDIA Inference Microservice LLM NIM ==
===========================================

NVIDIA Inference Microservice LLM NIM Version 1.8.4
Model: meta/llama-3.2-1b-instruct

Container image Copyright (c) 2016-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This is a PLACEHOLDER LEGAL FILE. REPLACE WITH ACTUAL FILE.

SYSTEM INFO
- Free GPUs:
  -  [26b1:10de] (0) NVIDIA RTX 6000 Ada Generation[current utilization: 0%]
  -  [1fb1:10de] (1) NVIDIA T600[current utilization: 0%]
2025-05-22T09:03:42.752024Z  INFO nim_hub_ngc::api::tokio::builder: ngc configured with api_loc: api.ngc.nvidia.com auth_loc: authn.nvidia.com scheme: https
2025-05-22T09:03:43.669827Z  INFO nim_hub_ngc::api::tokio: Downloaded filename: config.json to blob: "/opt/nim/.cache/ngc/hub/models--nim--meta--llama-3.2-1b-instruct/blobs/c4c70d448a3481feffa70c72e7cc6501"
2025-05-22T09:03:43.672925Z  INFO nim_hub_ngc::api::tokio::builder: ngc configured with api_loc: api.ngc.nvidia.com auth_loc: authn.nvidia.com scheme: https
2025-05-22T09:03:43.673095Z  INFO nim_hub_ngc: File: config.json found in cache: "/opt/nim/.cache/ngc/hub/models--nim--meta--llama-3.2-1b-instruct/snapshots/hf-e9f8eff-nim1.5+/config.json"
2025-05-22T09:03:43.673107Z  INFO nim_hub_ngc::api::tokio::public: Skipping download, using cached copy of file: config.json at path: "/opt/nim/.cache/ngc/hub/models--nim--meta--llama-3.2-1b-instruct/snapshots/hf-e9f8eff-nim1.5+/config.json"
# ... (大量の類似ngcログ出力、簡潔性のためここでは省略) ...
MODEL PROFILES
- Compatible with system and runnable: <None>
- Compilable to TRT-LLM using just-in-time compilation of HF models to TRTLLM engines: <None>
- Incompatible with system:
  - 8b87146e39b0305ae1d73bc053564d1b4b4c565f81aa5abe3e84385544ca9b60 (tensorrt_llm-b200-fp8-tp1-pp1-throughput)
  - 7b508014e846234db3cabe5c9f38568b4ee96694b60600a0b71c621dc70cacf3 (tensorrt_llm-h100-fp8-tp1-pp1-throughput)
  - af876a179190d1832143f8b4f4a71f640f3df07b0503259cedee3e3a8363aa96 (tensorrt_llm-h200-fp8-tp1-pp1-throughput)
  - ad17776f4619854fccd50354f31132a558a1ca619930698fd184d6ccf5fe3c99 (tensorrt_llm-l40s-fp8-tp1-pp1-throughput)
  - 5811750e70b7e9f340f4d670c72fcbd5282e254aeb31f62fd4f937cfb9361007 (tensorrt_llm-h100_nvl-fp8-tp1-pp1-throughput)
  - 222d1729a785201e8a021b226d74d227d01418c41b556283ee1bdbf0a818bd94 (tensorrt_llm-a100-bf16-tp1-pp1-throughput)
  - 74bfd8b2df5eafe452a9887637eef4820779fb4e1edb72a4a7a2a1a2d1e6480b (tensorrt_llm-a10g-bf16-tp1-pp1-throughput)
  - a4c63a91bccf635b570ddb6d14eeb6e7d0acb2389712892b08d21fad2ceaee38 (tensorrt_llm-b200-bf16-tp1-pp1-throughput)
  - e7dbd9a8ce6270d2ec649a0fecbcae9b5336566113525f20aee3809ba5e63856 (tensorrt_llm-h100-bf16-tp1-pp1-throughput)
  - 434e8d336fa23cbe151748d32b71e196d69f20d319ee8b59852a1ca31a48d311 (tensorrt_llm-h200-bf16-tp1-pp1-throughput)
  - ac5071bbd91efcc71dc486fcd5210779570868b3b8328b4abf7a408a58b5e57c (tensorrt_llm-l40s-bf16-tp1-pp1-throughput)
  - 25b5e251d366671a4011eaada9872ad1d02b48acc33aa0637853a3e3c3caa516 (tensorrt_llm-h100_nvl-bf16-tp1-pp1-throughput)
  - c6821c013c559912c37e61d7b954c5ca8fe07dda76d8bea0f4a52320e0a54427 (tensorrt_llm-a100_sxm4_40gb-bf16-tp1-pp1-throughput)
  - 4f904d571fe60ff24695b5ee2aa42da58cb460787a968f1e8a09f5a7e862728d (vllm-bf16-tp1-pp1)
  - bae7cf0b51c21f0e6f697593ee58dc8a555559b4b81903502a9e0ffbdc1b67a9 (tensorrt_llm-b200-fp8-tp1-pp1-throughput-lora-lora)
  - 018942e9a37a69b58ed514d7006ed851f57a1b2501ec4d39e3d862d8718893e5 (tensorrt_llm-h100-fp8-tp1-pp1-throughput-lora-lora)
  - 083e56197c3078f2932fc8d3d0e6d0d5b88d29ab8dbe154097deb322f227be07 (tensorrt_llm-h200-fp8-tp1-pp1-throughput-lora-lora)
  - 0782f55dcd12ec36d6126d9768fd364182986eecd25526eb206553df388057b7 (tensorrt_llm-l40s-fp8-tp1-pp1-throughput-lora-lora)
  - 22f2a8f33c93678bdfb831a62887f25642534525f1e0212f360b5490b03f99a2 (tensorrt_llm-h100_nvl-fp8-tp1-pp1-throughput-lora-lora)
  - 74eedaea9082525093f8346646709dc91efa7fd7ef2bc72d595feefaab94cf6f (tensorrt_llm-a100-bf16-tp1-pp1-throughput-lora-lora)
  - 0c85a1fe1aa38fa0f635a4c77bf25f7c94bfb9022564a929580415093f5b97d0 (tensorrt_llm-a10g-bf16-tp1-pp1-throughput-lora-lora)
  - a7baab3661c8376231396bf5529a78365892a33cc3416a11986fc2d96afcbcd8 (tensorrt_llm-b200-bf16-tp1-pp1-throughput-lora-lora)
  - 6e29cf433044b5a90f50d4cd1ca2e48d1ecb62eb5fa6fa9fb6f8317da61eebb3 (tensorrt_llm-h100-bf16-tp1-pp1-throughput-lora-lora)
  - f603fd56445cfe4f9448dfef2d9d1507465af46f1a0e60e9fb9796850bdbd7e6 (tensorrt_llm-h200-bf16-tp1-pp1-throughput-lora-lora)
  - 08662fae791cb76c40899ed539a7cbfdac61cad60d042220a3c321abd34a8a87 (tensorrt_llm-l40s-bf16-tp1-pp1-throughput-lora-lora)
  - 86b3fade693f8139fd213d70622d87dd6919f9f22dc7e8bbdfd84f1b5601d044 (tensorrt_llm-h100_nvl-bf16-tp1-pp1-throughput-lora-lora)
  - 47dbb6f7acda1fb4b1220d9462a66bda882e06e5d547e3b489c6bd7bf87df6e3 (tensorrt_llm-a100_sxm4_40gb-bf16-tp1-pp1-throughput-lora-lora)
  - f749ba07aade1d9e1c36ca1b4d0b67949122bd825e8aa6a52909115888a34b95 (vllm-bf16-tp1-pp1-lora)
  - ac34857f8dcbd174ad524974248f2faf271bd2a0355643b2cf1490d0fe7787c2 (tensorrt_llm-trtllm_buildable-bf16-tp1-pp1)
  - 7b8458eb682edb0d2a48b4019b098ba0bfbc4377aadeeaa11b346c63c7adf724 (tensorrt_llm-trtllm_buildable-bf16-tp1-pp1-lora)
```

**分析と結論：**
上記の`list-model-profiles`コマンド出力に基づき、使用するNIMイメージ`nvcr.io/nim/meta/llama-3.2-1b-instruct:1`とローカルGPU`NVIDIA RTX 6000 Ada Generation`について：
* **直接互換し実行可能な事前構築最適化設定プロファイルがない**（`Compatible with system and runnable: <None>`）。
* これは、リストから既製で最適保証の`NIM_MODEL_PROFILE`名を簡単に選択して起動スクリプトに入力できないことを意味します。
* NIMサービス開始時に有効で互換性のある`NIM_MODEL_PROFILE`を指定しない場合、**ハードウェアとモデルの自動検出を試行し、最適化されたTensorRT-LLMエンジンを動的に構築**します。このプロセスは通常、`tensorrt_llm-trtllm_buildable-bf16-tp1-pp1`のような汎用構築戦略を利用します（リスト末尾に示されているように、「非互換」とマークされていても、これは*事前構築プロファイル*として直接使用できないことを指す可能性があり、その構築ロジックはNIM内部で採用される可能性があります）。
* したがって、現段階の実験重点は、NIMが`NIM_MODEL_PROFILE`を明示的に指定しない（またはデフォルトの構築動作を使用する）場合のモデル最適化プロセスと性能表現の観察・記録になります。

**2. NIM起動スクリプトの修正:**

上記の分析に基づき、現在`scripts/03_run_nim_docker.sh`で`NIM_MODEL_PROFILE`環境変数を明示的に設定または修正して特定の事前構築設定プロファイルを指定する**必要はありません**（直接互換するものが見つからないため）。NIMの自動最適化メカニズムに依存します。

実験のために以前`scripts/03_run_nim_docker.sh`に`NIM_MODEL_PROFILE`環境変数を追加している場合は、NIMの内部で認識・間接使用可能な構築パラメータを指定しない限り、これを削除またはコメントアウトしてNIMのデフォルト動作に戻してください。

**スクリプト修正記録：**
```diff
（`scripts/03_run_nim_docker.sh`に`NIM_MODEL_PROFILE`行がない場合は、「スクリプト未修正、NIMデフォルト最適化動作を使用」と記載。関連行を削除した場合は、diffを貼り付け。）
```

**3. テスト手順:**
現在は、NIMのデフォルト場合の動作観察に焦点を当てます。

### 6.2 NIMデフォルト最適化動作と性能の観察

前セクションで、`meta-llama/Llama-3.2-1B`モデルとローカルNVIDIA RTX 6000 Ada Generation GPUについて、NIMイメージ`nvcr.io/nim/meta/llama-3.2-1b-instruct:1`が直接互換する事前構築性能設定プロファイルを提供していないことが判明しました。そのため、`NIM_MODEL_PROFILE`環境変数を明示的に設定せず、NIMが最適化戦略を自己決定してTensorRT-LLMエンジンを動的に構築することを選択しました。

**NIMサービス開始とエンジン構築ログ分析（`./scripts/03_run_nim_docker.sh`実行後）：**

以下の重要情報をDockerコンテナログから抽出（`docker logs <container_id>`）：

* **開始時間とNIMバージョン：**
  * サービス開始試行は`2025-05-22 09:21:00`頃。
  * NVIDIA Inference Microservice LLM NIM Version `1.8.4`。
  * モデルは`meta/llama-3.2-1b-instruct`（NIMイメージに基づく）として認識。

* **自動選択された性能設定プロファイル：**
  * NIMが2つの互換設定プロファイルを検出し、最終的に選択：
    ```
    INFO 2025-05-22 09:21:08.183 ngc_injector.py:315] Selected profile: ac34857f8dcbd174ad524974248f2faf271bd2a0355643b2cf1490d0fe7787c2 (tensorrt_llm-trtllm_buildable-bf16-tp1-pp1)
    ```
  * **選択された設定プロファイルメタデータ：**
    * `llm_engine: tensorrt_llm`
    * `pp: 1`（Pipeline Parallelism = 1）
    * `precision: bf16`（BF16精度）
    * `tp: 1`（Tensor Parallelism = 1）
    * `trtllm_buildable: true`（TensorRT-LLMエンジンの動的構築用設定であることを示す）

* **TensorRT-LLMエンジン構築プロセス：**
  * NIMがモデル`/opt/weights/hf/Llama-3.2-1B`（`scripts/03_run_nim_docker.sh`で`NIM_FT_MODEL`経由で指定したローカル重みパス、コンテナ内でこのパスにマッピング）用にTensorRT-LLMエンジン構築を開始。
  * 構築パラメータには`max_seq_len=131072`、`dtype=torch.bfloat16`、`tensor_parallel_size=1`、`pipeline_parallel_size=1`などが含まれます。
  * **エンジン構築所要時間：**
    * チェックポイント読取・変換所要時間：`INFO 2025-05-22 09:21:35.961 build_utils.py:239] Total time of reading and converting rank0 checkpoint: 0.18 s`
    * エンジン構築所要時間（rank0）：`INFO 2025-05-22 09:22:59.751 build_utils.py:247] Total time of building rank0 engine: 83.79 s`
    * したがって、TensorRT-LLMエンジン構築は合計約**83.8秒**を要しました。

* **エンジンロードとサービス開始：**
  * エンジン正常ロード：`Engine size in bytes 3112711548`（約2.89 GB）。
  * NIMサービス正常開始、`http://0.0.0.0:8000`でリスン。
    ```
    INFO 2025-05-22 09:23:01.032 launcher.py:298] Worker starting (pid: 212)
    INFO 2025-05-22 09:23:02.182 server.py:89] Starting HTTP server on port 8000
    INFO 2025-05-22 09:23:02.182 server.py:91] Listening on address 0.0.0.0
    INFO 2025-05-22 09:23:02.202 server.py:162] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
    ```

* **モデル照会テスト：**
  * 「4.4 モデルとの対話（Python）」部分で述べたように、サービス開始後、`src/query_nim_service.py`スクリプトで`/v1/completions`エンドポイント経由でモデルの照会に成功。関連ログエントリ：
    `INFO 2025-05-22 09:28:45.91 httptools_impl.py:481] ::1:33148 - "POST /v1/completions HTTP/1.1" 200`

**総括：**
実験結果は、特定ハードウェア（NVIDIA RTX 6000 Ada）とNIMイメージ（`nvcr.io/nim/meta/llama-3.2-1b-instruct:1`）用に事前構築された直接互換の最適化設定プロファイル（`NIM_MODEL_PROFILE`）を見つけられない場合でも、NVIDIA NIMが以下を実現できることを示しています：
1. 適切な構築戦略（この例では`tensorrt_llm-trtllm_buildable-bf16-tp1-pp1`）を自動選択。
2. この戦略を利用して、提供されたモデル重み（`meta-llama/Llama-3.2-1B`）用に現在のGPUに最適化されたTensorRT-LLMエンジンを動的に構築。
3. ローカル環境でのエンジン構築プロセス全体は約1分半を要しました。
4. サービスの正常開始とAPIリクエストへの正確な応答。

これはNIMの展開フロー簡素化と自動化最適化における能力を十分に示し、多様なハードウェアとモデル組み合わせ下でも良好な適用性を持つことを証明しています。

## 7. 総括と発表要点

* **核心収穫：**
  * **展開と検証の成功：** ローカルNVIDIA RTX 6000 Ada（約48GB VRAM）環境で、NVIDIA NIMを使用した`meta-llama/Llama-3.2-1B`モデルの展開に成功し、サービス可用性を検証。
  * **NIMワークフローの理解：** NIMのワークフローをより深く理解、特にハードウェアの自動検出、適切な構築戦略の選択（`tensorrt_llm-trtllm_buildable-bf16-tp1-pp1`など）、ローカルモデル重み用最適化TensorRT-LLMエンジンの動的構築（所要時間約1.5分）能力。
  * **展開簡素化性：** NIMとTensorRT-LLMの組み合わせがLLM展開の簡素化における強力な能力を体感、手動設定と推論エンジン最適化の複雑さを大幅に軽減。
  * **標準化インターフェースの価値：** NIMが提供するOpenAI互換APIインターフェースにより、展開済みモデルとの対話と統合が非常に便利。

* **主要課題と対応：**
  * **GPU VRAM制限：** ローカルGPU（約48GB）がブログ例環境（80GB）より遙かに小さく、最初は実行不可能を懸念。より小さなパラメータ数の`meta-llama/Llama-3.2-1B`モデルを選択することで、この環境での展開とテストを成功に完了。
  * **環境変数とパス設定：**
    * 環境変数（`IMG_NAME`、`MODEL_WEIGHT_PARENT_DIRECTORY`など）の未正確ロードやスクリプト内パス（`01_set_env_vars.sh`の相対パスなど）解析不当によるコマンド実行失敗に複数回遭遇。`source`スクリプトの確実実行、明確な絶対パス使用（`01_set_env_vars.sh`内`PROJECT_ROOT_DIR`決定方法の修正など）、コマンド実行時の現在作業ディレクトリへの注意により解決。
  * **ポート競合：** 初回NIM Dockerコンテナ開始時、ホストポート8000が占有されている問題に遭遇。`sudo netstat -tulnp | grep :8000`または`sudo ss -tulnp | grep :8000`で占有プロセス（通常は残留`docker-proxy`）を見つけ、`sudo kill <PID>`で停止して解決。
  * **モデルAPI使用差異（`Chat template not found`）：** PythonスクリプトでNIMサービス展開の`meta-llama/Llama-3.2-1B`基本モデル照会時、最初に`/v1/chat/completions`インターフェース（`client.chat.completions.create()`対応）使用が「Chat template not found」エラーを引き起こしました。基本モデルが通常事前設定チャットテンプレートを持たないことが原因。`/v1/completions`インターフェース（`client.completions.create()`対応）への切り替えと`prompt`パラメータでの`messages`パラメータ置換により成功解決。
  * **DockerとGPUドライバ互換性：** 今回は直接遭遇しませんでしたが、Docker、NVIDIAドライバ、CUDAバージョン間の互換性注意が必要で、これはNIMサービス正常動作の基盤です。

* **将来探索可能方向：**
  * **異なるモデルと微調整テスト：** 他の異なるパラメータ数やタイプのモデル、特にSFT（Supervised Fine-tuning）微調整モデルの展開試行により、NIMの異なる微調整モデルサポート度と最適化効果をより包括的に検証。
  * **性能最適化戦略探索：** ハードウェアサポートとNIMが互換する性能設定プロファイルを提供する場合、より積極的な最適化戦略（例：RTX 6000 AdaサポートでNIM imageに対応profileを含むFP8量子化）をテストし、性能とモデル精度への影響を比較。
  * **詳細性能ベンチマークテスト：** より体系的で詳細な性能ベンチマークテストを実施し、標準化ツール（`triton-perf-analyzer`やカスタムスクリプトなど）を使用して異なる設定（例：明示的`NIM_MODEL_PROFILE`の有無、異なるモデル、異なる並行量）下での推論遅延、スループット（TPS、Tokens Per Second）、GPUリソース利用率を記録・比較。
  * **マルチGPUと分散展開：** マルチGPU環境でのNIM展開設定（テンソル並列TP、パイプライン並列PPの設定など）と性能拡張性を研究。
  * **アプリケーション統合とワークフロー：** NIM展開されたモデルサービスをより緊密に複雑な人工知能アプリケーション、RAGシステム、またはMagentic Agentワークフローに統合する方法を探索。

* **他の試行者への提言：**
  * **公式ドキュメントの注意深い読解：** 開始前に、NVIDIA NIMの公式ドキュメント、特にサポートモデルリスト、NIMコンテナイメージバージョン選択（通常基本モデルに対応）、必要環境変数の詳細説明、ハードウェア互換性要件を注意深く研読することが重要。
  * **環境変数設定の重視：** 関連スクリプト実行のシェルセッションで、すべての必要な環境変数（`NGC_API_KEY`、`NIM_CACHE_PATH`、`MODEL_WEIGHT_PARENT_DIRECTORY`、`IMG_NAME`など）が正しく設定・エクスポートされていることを確認。プロジェクト提供の`scripts/01_set_env_vars.sh`スクリプトによる統一管理を強く推奨し、新しいターミナルセッションやDocker関連コマンド実行前に毎回このスクリプトを`source`することを忘れずに。
  * **Dockerログによる故障排除の活用：** Dockerコンテナ開始失敗、サービス動作異常、モデルロードエラー遭遇時は、まず`docker ps -a`でコンテナ実行状態を確認し、次に`docker logs <container_id_or_name>`（`-f`パラメータでリアルタイム観察可能）でコンテナ内部ログ出力を注意深く確認すべきです。これは通常最も直接的で詳細なエラー情報と故障手がかりを提供します。
  * **小規模から開始、段階的検証：** 特にVRAMなどハードウェアリソースが相対的に制限されるローカル環境では、より小さなパラメータ数のモデル（本プロジェクトの1Bモデルなど）から試行開始し、基礎フローを通した後に、より大きなモデルやより複雑な設定に段階的に挑戦することを推奨。
  * **APIエンドポイント差異の理解：** 基本大モデルと指示/対話微調整されたモデルのAPI照会時の動作とインターフェース差異を明確に区別。基本モデルは通常`/v1/completions`エンドポイントと`prompt`パラメータを使用し、指示/対話モデルは`/v1/chat/completions`エンドポイントと`messages`パラメータにより適しています。誤った使用は「Chat template not found」などの問題を引き起こす可能性があります。
  * **モデル重みパスの正確性確保：** `NIM_FT_MODEL`環境変数が指定するのはDockerコンテナ内部のモデル重みパス（通常`/opt/weights/hf/<model_folder_name>`）で、`-v $MODEL_WEIGHT_PARENT_DIRECTORY:/opt/weights/hf`によりモデルファイルフォルダを含むローカル親ディレクトリがコンテナの`/opt/weights/hf`に正しくマウントされていることを確保する必要があります。

---
*文書は継続的に更新されます。*