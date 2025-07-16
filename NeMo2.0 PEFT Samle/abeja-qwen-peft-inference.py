import torch
from megatron.core.inference.common_inference_params import CommonInferenceParams
import nemo.lightning as nl
from nemo.collections.llm import api

strategy = nl.MegatronStrategy(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    sequence_parallel=False,
    setup_optimizers=False,
)

trainer = nl.Trainer(
    accelerator="gpu",
    devices=1,
    num_nodes=1,
    strategy=strategy,
    plugins=nl.MegatronMixedPrecision(
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    ),
)
prompts = [
    "きりたんぽが名物である県はどこか？",
    #"Please tell me any Japanese word you know."
]

adapter_checkpoint = './models/checkpoints/abeja_qwen25_7b_finetuning/qwen25_lora/2025-06-17_10-53-58/checkpoints/model_name=0--val_loss=0.22-step=9-consumed_samples=320.0-last'
# adapter_checkpoint = '/workspace/results/qwen2_500m_finetune_llm_jp_wiki/checkpoints/model_name=0--val_loss=2.40-step=1999-consumed_samples=8000.0-last'
results = api.generate(
    path=adapter_checkpoint,
    prompts=prompts,
    trainer=trainer,
    #inference_params=CommonInferenceParams(temperature=0.1, top_k=10, num_tokens_to_generate=4096),
    inference_params=CommonInferenceParams(temperature=0.1, top_k=1, num_tokens_to_generate=512),
    text_only=True,
)
print(results)
