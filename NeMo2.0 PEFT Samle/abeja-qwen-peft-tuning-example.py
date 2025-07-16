import json
import os
import random
import nemo_run as run
from nemo.collections import llm
from nemo import lightning as nl
from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule

from nemo.collections import llm

class MyFineTuningDataModule(FineTuningDataModule):
    INPUT_TRAIN = "./JGLUE/datasets/jcommonsenseqa-v1.3/train-v1.3.json"
    INPUT_VALID = "./JGLUE/datasets/jcommonsenseqa-v1.3/valid-v1.3.json"
    OUTPUT_DIR = "./data/jcommonsenseqa-v1.3"

    def prepare_data(self) -> None:
        self._download_data()
        self._preprocess_and_split_data()
        super().prepare_data()

    def _download_data(self) -> None:
        os.system("git clone https://github.com/yahoojapan/JGLUE.git")

    def _preprocess_and_split_data(self) -> None:
        random.seed(42)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.prosess(self.INPUT_TRAIN, train=True)
        self.prosess(self.INPUT_VALID)

    def write_jsonl(self, fname, json_objs):
        with open(fname, 'wt') as f:
            for o in json_objs:
                f.write(json.dumps(o, ensure_ascii=False)+"\n")

    def form_question(self, obj):
        st = ""
        st += "### 指示:\n"
        st += "与えられた選択肢の中から、最適な答えを選んでください。出力は以下から選択してください：\n"
        st += f"- {obj['choice0']}\n"
        st += f"- {obj['choice1']}\n"
        st += f"- {obj['choice2']}\n"
        st += f"- {obj['choice3']}\n"
        st += f"- {obj['choice4']}\n"
        st += "### 入力:\n"
        st += f"{obj['question']}\n"
        st += "### 応答:"
        return st


    def prosess(self, input_path, train=False):
        with open(input_path) as f:
            dataset = [json.loads(line) for line in f.readlines()]

        processed = []
        for data in dataset:
            prompt = self.form_question(data)
            answer = data[f"choice{data['label']}"]
            processed.append({"input": prompt, "output": f"{answer}"})

        if train:
            random.shuffle(processed)
            train_ds = processed[:-1000]
            valid_ds = processed[-1000:]
            self.write_jsonl(f"{self.OUTPUT_DIR}/training.jsonl", train_ds)
            self.write_jsonl(f"{self.OUTPUT_DIR}/validation.jsonl", valid_ds)
        else:
            self.write_jsonl(f"{self.OUTPUT_DIR}/test.jsonl", processed)

if __name__ == "__main__":


    data = run.Config(MyFineTuningDataModule, dataset_root='./data/jcommonsenseqa-v1.3/', seq_length=4096,
                      micro_batch_size=1, global_batch_size=32, dataset_kwargs={})
    recipe = llm.qwen25_7b.finetune_recipe(
        dir="./models/checkpoints/abeja_qwen25_7b_finetuning",  # Path to store checkpoints
        name="qwen25_lora",
        num_nodes=1,
        num_gpus_per_node=1,
        peft_scheme="lora",
    )

    recipe.resume.restore_config = run.Config(
        nl.RestoreConfig,
        path='./models/qwen2.5'
    )

    recipe.trainer.max_steps = 10
    recipe.trainer.num_sanity_val_steps = 0

    # Async checkpointing doesn't work with PEFT
    recipe.trainer.strategy.ckpt_async_save = False

    # Need to set this to 1 since the default is 2
    recipe.trainer.strategy.context_parallel_size = 1
    recipe.trainer.val_check_interval = 10
    recipe.data = data

    # This is currently required for LoRA/PEFT
    recipe.trainer.strategy.ddp = "megatron"

    run.run(recipe, executor=run.LocalExecutor())
