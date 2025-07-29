#!/bin/bash

# Container data processing script - Fixed version
set -e

# Configuration variables
WORKSPACE="/workspace"
DATA_ROOT="${WORKSPACE}/data/llm_jp_wiki"
RAW_DATA_DIR="${DATA_ROOT}/raw/ja_wiki"
OUTPUT_DIR="${DATA_ROOT}/nemo_binary"
TOKENIZER="Qwen/Qwen2.5-0.5B"

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Create directory structure
log_step "Creating directory structure..."
mkdir -p "$RAW_DATA_DIR"
mkdir -p "$OUTPUT_DIR"

# Download LLM-JP data (if not already exists)
log_step "Checking and downloading LLM-JP Japanese Wikipedia data..."
cd "$RAW_DATA_DIR"

# Download training data files (0-13)
for i in {0..13}; do
    if [[ ! -f "train_${i}.jsonl" ]]; then
        log_info "Downloading train_${i}.jsonl.gz..."
        wget -O "train_${i}.jsonl.gz" --no-check-certificate \
            "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_${i}.jsonl.gz?ref_type=heads"
        gunzip "train_${i}.jsonl.gz"
    else
        log_info "train_${i}.jsonl already exists, skipping download"
    fi
done

# Download validation data
if [[ ! -f "validation_0.jsonl" ]]; then
    log_info "Downloading validation_0.jsonl.gz..."
    wget -O "validation_0.jsonl.gz" --no-check-certificate \
        "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/validation_0.jsonl.gz?ref_type=heads"
    gunzip "validation_0.jsonl.gz"
else
    log_info "validation_0.jsonl already exists, skipping download"
fi

# Merge all training files
log_step "Merging all training data files..."
if [[ ! -f "train_merged.jsonl" ]]; then
    log_info "Merging train_0.jsonl to train_13.jsonl..."
    cat train_{0..13}.jsonl > train_merged.jsonl
    log_info "✅ Training data merge completed: $(wc -l train_merged.jsonl | cut -d' ' -f1) lines"
else
    log_info "train_merged.jsonl already exists, skipping merge"
fi

# Use NeMo preprocessing script to process data
log_step "Processing data using NeMo preprocessing script..."

# Process training data
log_info "Processing merged training data..."
python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input="${RAW_DATA_DIR}/train_merged.jsonl" \
    --json-keys=text \
    --tokenizer-library=huggingface \
    --tokenizer-type="$TOKENIZER" \
    --dataset-impl mmap \
    --append-eod \
    --output-prefix="${OUTPUT_DIR}/ja_wiki_train" \
    --workers=4

# Process validation data
log_info "Processing validation data..."
python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input="${RAW_DATA_DIR}/validation_0.jsonl" \
    --json-keys=text \
    --tokenizer-library=huggingface \
    --tokenizer-type="$TOKENIZER" \
    --dataset-impl mmap \
    --append-eod \
    --output-prefix="${OUTPUT_DIR}/ja_wiki_val" \
    --workers=2

# Verify output files
log_step "Verifying output files..."
if [[ -f "${OUTPUT_DIR}/ja_wiki_train_text_document.bin" && \
      -f "${OUTPUT_DIR}/ja_wiki_train_text_document.idx" && \
      -f "${OUTPUT_DIR}/ja_wiki_val_text_document.bin" && \
      -f "${OUTPUT_DIR}/ja_wiki_val_text_document.idx" ]]; then
    log_info "✅ Data processing completed!"
    
    # Display file information
    echo ""
    echo "📁 Generated files:"
    ls -lh "${OUTPUT_DIR}"/*.{bin,idx}
    
    # Display file size statistics
    echo ""
    echo "📊 File size statistics:"
    echo "Training data: $(du -h ${OUTPUT_DIR}/ja_wiki_train_text_document.bin | cut -f1)"
    echo "Validation data: $(du -h ${OUTPUT_DIR}/ja_wiki_val_text_document.bin | cut -f1)"
    echo "Total: $(du -sh ${OUTPUT_DIR} | cut -f1)"
    
    # Display dataset statistics
    echo ""
    echo "📈 Dataset statistics:"
    echo "Original training lines: $(wc -l ${RAW_DATA_DIR}/train_merged.jsonl | cut -d' ' -f1)"
    echo "Original validation lines: $(wc -l ${RAW_DATA_DIR}/validation_0.jsonl | cut -d' ' -f1)"
else
    echo "❌ Data processing failed, output files are incomplete"
    echo "Check files:"
    ls -la "${OUTPUT_DIR}/"
    exit 1
fi

log_info "🎉 LLM-JP data processing successfully completed!"
log_info "Data location: ${OUTPUT_DIR}"
echo ""
echo "🚀 Next step: You can start Task 7 - Implement PEFT-LoRA fine-tuning script"