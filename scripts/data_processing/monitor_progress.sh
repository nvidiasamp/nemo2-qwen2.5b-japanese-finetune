#!/bin/bash

# 数据处理进度监控脚本

PROJECT_ROOT="/home/cho/workspace/workshop-25-08-02"
OUTPUT_DIR="${PROJECT_ROOT}/data/llm_jp_wiki/nemo_binary"

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🔍 LLM-JP 数据处理进度监控${NC}"
echo "================================"
echo ""

# 检查Docker容器状态
echo -e "${BLUE}📦 Docker容器状态:${NC}"
docker_count=$(ps aux | grep -c "nvcr.io/nvidia/nemo:25.04" | head -1)
if [[ $docker_count -gt 1 ]]; then
    echo -e "${GREEN}✅ NeMo容器正在运行${NC}"
else
    echo -e "${YELLOW}⚠️  NeMo容器可能已完成或停止${NC}"
fi
echo ""

# 检查输出文件
echo -e "${BLUE}📁 输出文件状态:${NC}"
if [[ -d "$OUTPUT_DIR" ]]; then
    echo "位置: $OUTPUT_DIR"
    echo ""
    
    # 检查每个预期文件
    files=("ja_wiki_train_text_document.bin" "ja_wiki_train_text_document.idx" 
           "ja_wiki_val_text_document.bin" "ja_wiki_val_text_document.idx")
    
    for file in "${files[@]}"; do
        if [[ -f "${OUTPUT_DIR}/${file}" ]]; then
            size=$(du -h "${OUTPUT_DIR}/${file}" | cut -f1)
            echo -e "${GREEN}✅ ${file}${NC} - ${size}"
        else
            echo -e "${YELLOW}⏳ ${file}${NC} - 待生成"
        fi
    done
    
    echo ""
    echo -e "${BLUE}📊 目录总大小:${NC} $(du -sh $OUTPUT_DIR | cut -f1)"
else
    echo -e "${YELLOW}⚠️  输出目录尚未创建${NC}"
fi

echo ""

# 检查完成状态
train_bin="${OUTPUT_DIR}/ja_wiki_train_text_document.bin"
train_idx="${OUTPUT_DIR}/ja_wiki_train_text_document.idx"
val_bin="${OUTPUT_DIR}/ja_wiki_val_text_document.bin"
val_idx="${OUTPUT_DIR}/ja_wiki_val_text_document.idx"

if [[ -f "$train_bin" && -f "$train_idx" && -f "$val_bin" && -f "$val_idx" ]]; then
    echo -e "${GREEN}🎉 数据处理已完成！${NC}"
    echo ""
    echo -e "${BLUE}📈 最终统计:${NC}"
    echo "训练数据: $(du -h $train_bin | cut -f1)"
    echo "验证数据: $(du -h $val_bin | cut -f1)"
    echo "总计大小: $(du -sh $OUTPUT_DIR | cut -f1)"
    echo ""
    echo -e "${GREEN}🚀 可以开始下一个任务：任务7 - 实现PEFT-LoRA微调脚本${NC}"
elif [[ -f "$train_bin" ]]; then
    echo -e "${YELLOW}⏳ 数据处理进行中... 训练数据转换完成，正在处理验证数据${NC}"
else
    echo -e "${YELLOW}⏳ 数据处理进行中... 正在转换训练数据${NC}"
fi

echo ""
echo "提示：运行 'watch -n 30 ./scripts/data_processing/monitor_progress.sh' 进行实时监控"