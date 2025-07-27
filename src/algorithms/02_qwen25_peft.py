import json
import os
import nemo_run as run
from nemo.collections import llm
from nemo import lightning as nl
from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule

class CustomFineTuningDataModule(FineTuningDataModule):
    """
    準備済みのJSONLファイルを使用するカスタムデータモジュール
    データ形式: {"input": "質問", "output": "回答"}
    """

    def __init__(self, dataset_root: str, **kwargs):
        self.dataset_root = dataset_root
        super().__init__(dataset_root=dataset_root, **kwargs)

    def prepare_data(self) -> None:
        """
        データの準備処理 - 既にJSONL形式で準備済みなので、
        ファイルの存在確認のみ実行
        """
        train_path = os.path.join(self.dataset_root, "training.jsonl")
        val_path = os.path.join(self.dataset_root, "validation.jsonl")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training file not found: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation file not found: {val_path}")

        print(f"Found training data: {train_path}")
        print(f"Found validation data: {val_path}")

        # ファイル名をNeMoが期待する形式にリネーム（必要に応じて）
        training_jsonl = os.path.join(self.dataset_root, "training.jsonl")
        validation_jsonl = os.path.join(self.dataset_root, "validation.jsonl")

        if not os.path.exists(training_jsonl):
            print(f"Creating symlink: {training_jsonl} -> {train_path}")
            os.symlink(os.path.basename(train_path), training_jsonl)

        if not os.path.exists(validation_jsonl):
            print(f"Creating symlink: {validation_jsonl} -> {val_path}")
            os.symlink(os.path.basename(val_path), validation_jsonl)

        super().prepare_data()

def main():
    print("=== Qwen2.5 PEFT Training Script ===")

    # データセットの設定
    data_config = run.Config(
        CustomFineTuningDataModule,
        dataset_root='/workspace/data/training_data/',
        seq_length=2048,  # シーケンス長
        micro_batch_size=2,  # 各GPUでのバッチサイズ
        global_batch_size=16,  # 全体のバッチサイズ
        dataset_kwargs={}
    )

    # ファインチューニングレシピの設定
    recipe = llm.qwen25_500m.finetune_recipe(
        dir="/workspace/models/checkpoints/qwen25_500m_peft",  # チェックポイント保存先
        name="qwen25_500m_peft",
        num_nodes=1,
        num_gpus_per_node=1,
        peft_scheme="lora",  # LoRA使用（メモリ効率のため）
    )

    # 事前訓練済みモデルの復元設定
    recipe.resume.restore_config = run.Config(
        nl.RestoreConfig,
        path='/workspace/qwen2.5-0.5b.nemo'  # 変換済みのNeMoモデル
    )

    # トレーニング設定
    recipe.trainer.max_steps = 1000  # 最大ステップ数
    recipe.trainer.num_sanity_val_steps = 2  # バリデーション確認ステップ
    recipe.trainer.val_check_interval = 100  # バリデーション実行間隔
    recipe.trainer.log_every_n_steps = 10  # ログ出力間隔
    recipe.trainer.enable_checkpointing = True  # チェックポイント有効化

    # PEFTの場合の設定
    recipe.trainer.strategy.ckpt_async_save = False  # 非同期チェックポイント無効
    recipe.trainer.strategy.context_parallel_size = 1  # コンテキスト並列サイズ
    recipe.trainer.strategy.ddp = "megatron"  # LoRA/PEFTに必要

    # データモジュールの設定
    recipe.data = data_config

    print("=== Training Configuration ===")
    print(f"Model: Qwen2.5-0.5B")
    print(f"PEFT Scheme: LoRA")
    print(f"Max Steps: {recipe.trainer.max_steps}")
    print(f"Micro Batch Size: {data_config.micro_batch_size}")
    print(f"Global Batch Size: {data_config.global_batch_size}")
    print(f"Sequence Length: {data_config.seq_length}")
    print(f"Checkpoint Dir: {recipe.log.log_dir}")
    print(f"Data Root: {data_config.dataset_root}")

    # トレーニング実行
    print("\n=== Starting Training ===")
    run.run(recipe, executor=run.LocalExecutor())

if __name__ == "__main__":
    main()
