#!/bin/bash

# 容器内数据处理脚本 - 修正版
set -e

# 配置变量
WORKSPACE="/workspace"
DATA_ROOT="${WORKSPACE}/data/llm_jp_wiki"
RAW_DATA_DIR="${DATA_ROOT}/raw/ja_wiki"
OUTPUT_DIR="${DATA_ROOT}/nemo_binary"
TOKENIZER="Qwen/Qwen2.5-0.5B"

# 颜色输出
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

# 创建目录结构
log_step "创建目录结构..."
mkdir -p "$RAW_DATA_DIR"
mkdir -p "$OUTPUT_DIR"

# 下载LLM-JP数据（如果尚未存在）
log_step "检查和下载 LLM-JP 日语 Wikipedia 数据..."
cd "$RAW_DATA_DIR"

# 下载训练数据文件（0-13）
for i in {0..13}; do
    if [[ ! -f "train_${i}.jsonl" ]]; then
        log_info "下载 train_${i}.jsonl.gz..."
        wget -O "train_${i}.jsonl.gz" --no-check-certificate \
            "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/train_${i}.jsonl.gz?ref_type=heads"
        gunzip "train_${i}.jsonl.gz"
    else
        log_info "train_${i}.jsonl 已存在，跳过下载"
    fi
done

# 下载验证数据
if [[ ! -f "validation_0.jsonl" ]]; then
    log_info "下载 validation_0.jsonl.gz..."
    wget -O "validation_0.jsonl.gz" --no-check-certificate \
        "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_wiki/validation_0.jsonl.gz?ref_type=heads"
    gunzip "validation_0.jsonl.gz"
else
    log_info "validation_0.jsonl 已存在，跳过下载"
fi

# 合并所有训练文件
log_step "合并所有训练数据文件..."
if [[ ! -f "train_merged.jsonl" ]]; then
    log_info "合并 train_0.jsonl 到 train_13.jsonl..."
    cat train_{0..13}.jsonl > train_merged.jsonl
    log_info "✅ 训练数据合并完成: $(wc -l train_merged.jsonl | cut -d' ' -f1) 行"
else
    log_info "train_merged.jsonl 已存在，跳过合并"
fi

# 使用NeMo预处理脚本处理数据
log_step "使用 NeMo 预处理脚本处理数据..."

# 处理训练数据
log_info "处理合并后的训练数据..."
python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input="${RAW_DATA_DIR}/train_merged.jsonl" \
    --json-keys=text \
    --tokenizer-library=huggingface \
    --tokenizer-type="$TOKENIZER" \
    --dataset-impl mmap \
    --append-eod \
    --output-prefix="${OUTPUT_DIR}/ja_wiki_train" \
    --workers=4

# 处理验证数据
log_info "处理验证数据..."
python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input="${RAW_DATA_DIR}/validation_0.jsonl" \
    --json-keys=text \
    --tokenizer-library=huggingface \
    --tokenizer-type="$TOKENIZER" \
    --dataset-impl mmap \
    --append-eod \
    --output-prefix="${OUTPUT_DIR}/ja_wiki_val" \
    --workers=2

# 验证输出文件
log_step "验证输出文件..."
if [[ -f "${OUTPUT_DIR}/ja_wiki_train_text_document.bin" && \
      -f "${OUTPUT_DIR}/ja_wiki_train_text_document.idx" && \
      -f "${OUTPUT_DIR}/ja_wiki_val_text_document.bin" && \
      -f "${OUTPUT_DIR}/ja_wiki_val_text_document.idx" ]]; then
    log_info "✅ 数据处理完成！"
    
    # 显示文件信息
    echo ""
    echo "📁 生成的文件："
    ls -lh "${OUTPUT_DIR}"/*.{bin,idx}
    
    # 显示文件大小统计
    echo ""
    echo "📊 文件大小统计："
    echo "训练数据: $(du -h ${OUTPUT_DIR}/ja_wiki_train_text_document.bin | cut -f1)"
    echo "验证数据: $(du -h ${OUTPUT_DIR}/ja_wiki_val_text_document.bin | cut -f1)"
    echo "总计: $(du -sh ${OUTPUT_DIR} | cut -f1)"
    
    # 显示数据集统计
    echo ""
    echo "📈 数据集统计："
    echo "原始训练行数: $(wc -l ${RAW_DATA_DIR}/train_merged.jsonl | cut -d' ' -f1)"
    echo "原始验证行数: $(wc -l ${RAW_DATA_DIR}/validation_0.jsonl | cut -d' ' -f1)"
else
    echo "❌ 数据处理失败，输出文件不完整"
    echo "检查文件："
    ls -la "${OUTPUT_DIR}/"
    exit 1
fi

log_info "🎉 LLM-JP 数据处理成功完成！"
log_info "数据位置: ${OUTPUT_DIR}"
echo ""
echo "🚀 下一步：您可以开始任务7 - 实现PEFT-LoRA微调脚本"