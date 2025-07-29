# Basic Usage Examples

This directory contains basic usage examples for the NeMo Japanese Fine-tuning package.

## Quick Start Example

The `quick_start.py` script demonstrates the basic workflow:

1. **Data Conversion**: Convert Japanese Wikipedia to Q&A format
2. **Model Configuration**: Setup Qwen model parameters
3. **Trainer Setup**: Configure PEFT/LoRA training
4. **Training**: Execute the fine-tuning process

### Running the Quick Start

```bash
# Navigate to the example directory
cd examples/basic_usage/

# Run the quick start example
python quick_start.py
```

**Note**: The quick start example uses placeholder paths. You'll need to:

1. Update file paths to point to your actual data
2. Ensure you have converted your data using the data processing scripts
3. Have a converted `.nemo` model file ready
4. Uncomment the training lines when ready to run

### Required Setup

Before running the examples:

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   ```bash
   # Convert Japanese Wikipedia data
   python scripts/data_processing/convert_japanese_data.py \
       --input-dir /path/to/japanese/wiki \
       --output-dir /path/to/converted/data
   ```

3. **Convert Model**:
   ```bash
   # Convert HuggingFace model to NeMo format
   python scripts/model_conversion/hf_to_nemo.py \
       --model-name Qwen/Qwen2.5-0.5B \
       --output-path /path/to/qwen2.5-0.5b.nemo
   ```

### Example Directory Structure

After setup, your directory structure should look like:

```
your_project/
├── data/
│   ├── raw/ja_wiki/          # Raw Japanese Wikipedia JSONL
│   └── converted/            # Converted Q&A format
│       ├── training.jsonl
│       └── validation.jsonl
├── models/
│   └── qwen2.5-0.5b.nemo    # Converted NeMo model
└── checkpoints/             # Training checkpoints
    └── japanese_fine_tune/
```

### Customization

The quick start example can be customized by:

- Changing model size in `QwenModelConfig`
- Adjusting LoRA parameters for PEFT
- Modifying training hyperparameters
- Switching between PEFT and SFT training

For more advanced examples, see the `advanced_training/` directory. 