#!/bin/bash
# scripts/data_processing/process_llm_jp_data.sh
# Process LLM-JP Wikipedia corpus for continual pre-training

set -e

echo "=== LLM-JP Wikipedia Data Processing ==="
echo "This script processes Japanese Wikipedia data from LLM-JP corpus"
echo "Based on M1nG branch implementation"

# Configuration
DATA_DIR="${DATA_DIR:-data/llm_jp_wiki}"
RAW_DIR="${DATA_DIR}/raw/ja_wiki"
BINARY_DIR="${DATA_DIR}/nemo_binary"
MERGED_FILE="${RAW_DIR}/train_merged.jsonl"

# Create directories
mkdir -p "${RAW_DIR}"
mkdir -p "${BINARY_DIR}"

# Function to download LLM-JP data
download_llm_jp_data() {
    echo "📥 Downloading LLM-JP Wikipedia corpus..."
    
    # Download training data (14 files)
    for i in {0..13}; do
        FILE="train_${i}.jsonl"
        URL="https://huggingface.co/datasets/llm-jp/llm-jp-corpus/resolve/main/data/ja_wiki/v3.0.0/${FILE}"
        
        if [ ! -f "${RAW_DIR}/${FILE}" ]; then
            echo "Downloading ${FILE}..."
            wget -q --show-progress "${URL}" -O "${RAW_DIR}/${FILE}"
        else
            echo "✓ ${FILE} already exists"
        fi
    done
    
    # Download validation data
    if [ ! -f "${RAW_DIR}/validation_0.jsonl" ]; then
        echo "Downloading validation_0.jsonl..."
        wget -q --show-progress \
            "https://huggingface.co/datasets/llm-jp/llm-jp-corpus/resolve/main/data/ja_wiki/v3.0.0/validation_0.jsonl" \
            -O "${RAW_DIR}/validation_0.jsonl"
    else
        echo "✓ validation_0.jsonl already exists"
    fi
    
    echo "✅ Download complete"
}

# Function to merge training files
merge_training_files() {
    echo "🔄 Merging training files..."
    
    if [ -f "${MERGED_FILE}" ]; then
        echo "✓ Merged file already exists"
        return
    fi
    
    # Merge all training files
    > "${MERGED_FILE}"  # Create empty file
    
    for i in {0..13}; do
        FILE="${RAW_DIR}/train_${i}.jsonl"
        if [ -f "${FILE}" ]; then
            echo "Adding ${FILE} to merged file..."
            cat "${FILE}" >> "${MERGED_FILE}"
        fi
    done
    
    echo "✅ Merge complete"
    
    # Check file size
    SIZE=$(du -h "${MERGED_FILE}" | cut -f1)
    echo "📊 Merged file size: ${SIZE}"
}

# Function to convert to NeMo binary format
convert_to_nemo_format() {
    echo "🔄 Converting to NeMo binary format..."
    
    # Check if binary files already exist
    if [ -f "${BINARY_DIR}/ja_wiki_train_text_document.bin" ] && \
       [ -f "${BINARY_DIR}/ja_wiki_train_text_document.idx" ]; then
        echo "✓ Binary files already exist"
        return
    fi
    
    # Run conversion in Docker container
    docker run --rm --gpus all --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v "$(pwd):/workspace" -w "/workspace" \
        nvcr.io/nvidia/nemo:25.04 \
        python -c "
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import create_shard_args
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import build_train_valid_test_datasets

# Configuration
input_files = {
    'train': '/workspace/${MERGED_FILE}',
    'validation': '/workspace/${RAW_DIR}/validation_0.jsonl',
    'test': '/workspace/${RAW_DIR}/validation_0.jsonl'
}

output_prefix = '/workspace/${BINARY_DIR}/ja_wiki'

# Create binary files
for split, input_file in input_files.items():
    print(f'Processing {split} split...')
    args = create_shard_args(
        input_file,
        f'{output_prefix}_{split}',
        tokenizer_model_name='Qwen/Qwen2.5-0.5B',
        append_eod=True
    )
    
    # Build dataset
    build_train_valid_test_datasets(**args)
    print(f'✅ {split} split complete')
"
    
    echo "✅ Conversion complete"
}

# Function to validate output
validate_output() {
    echo "🔍 Validating output files..."
    
    local all_exist=true
    local required_files=(
        "${BINARY_DIR}/ja_wiki_train_text_document.bin"
        "${BINARY_DIR}/ja_wiki_train_text_document.idx"
        "${BINARY_DIR}/ja_wiki_validation_text_document.bin"
        "${BINARY_DIR}/ja_wiki_validation_text_document.idx"
    )
    
    for file in "${required_files[@]}"; do
        if [ -f "${file}" ]; then
            SIZE=$(du -h "${file}" | cut -f1)
            echo "✓ ${file} (${SIZE})"
        else
            echo "✗ ${file} missing"
            all_exist=false
        fi
    done
    
    if [ "${all_exist}" = true ]; then
        echo "✅ All output files validated"
    else
        echo "❌ Some output files are missing"
        exit 1
    fi
}

# Main execution
main() {
    echo "Starting at: $(date)"
    
    # Step 1: Download data
    download_llm_jp_data
    
    # Step 2: Merge training files
    merge_training_files
    
    # Step 3: Convert to NeMo format
    convert_to_nemo_format
    
    # Step 4: Validate output
    validate_output
    
    echo "Completed at: $(date)"
    echo "✅ LLM-JP data processing complete!"
}

# Run main function
main "$@" 