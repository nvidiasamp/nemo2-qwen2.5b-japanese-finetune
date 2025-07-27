from nemo.collections import llm

if __name__ == "__main__":
    print("Starting Hugging Face to NeMo model conversion...")
    model_config = llm.Qwen25Config500M()
    model = llm.Qwen2Model(model_config)

    # .nemo ファイルの出力先を指定
    output_path = "qwen2.5-0.5b.nemo"

    llm.import_ckpt(model=model, source='hf://Qwen/Qwen2.5-0.5B', output_path=output_path, overwrite=True)
    print(f"Model conversion successful. Saved to {output_path}")
