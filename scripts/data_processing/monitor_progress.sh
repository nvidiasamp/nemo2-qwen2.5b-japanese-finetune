#!/bin/bash

# Data processing progress monitoring script

PROJECT_ROOT="/home/cho/workspace/workshop-25-08-02"
OUTPUT_DIR="${PROJECT_ROOT}/data/llm_jp_wiki/nemo_binary"

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🔍 LLM-JP Data Processing Progress Monitor${NC}"
echo "================================"
echo ""

# Check Docker container status
echo -e "${BLUE}📦 Docker Container Status:${NC}"
docker_count=$(ps aux | grep -c "nvcr.io/nvidia/nemo:25.04" | head -1)
if [[ $docker_count -gt 1 ]]; then
    echo -e "${GREEN}✅ NeMo container is running${NC}"
else
    echo -e "${YELLOW}⚠️  NeMo container may have completed or stopped${NC}"
fi
echo ""

# Check output files
echo -e "${BLUE}📁 Output File Status:${NC}"
if [[ -d "$OUTPUT_DIR" ]]; then
    echo "Location: $OUTPUT_DIR"
    echo ""
    
    # Check each expected file
    files=("ja_wiki_train_text_document.bin" "ja_wiki_train_text_document.idx" 
           "ja_wiki_val_text_document.bin" "ja_wiki_val_text_document.idx")
    
    for file in "${files[@]}"; do
        if [[ -f "${OUTPUT_DIR}/${file}" ]]; then
            size=$(du -h "${OUTPUT_DIR}/${file}" | cut -f1)
            echo -e "${GREEN}✅ ${file}${NC} - ${size}"
        else
            echo -e "${YELLOW}⏳ ${file}${NC} - pending generation"
        fi
    done
    
    echo ""
    echo -e "${BLUE}📊 Directory Total Size:${NC} $(du -sh $OUTPUT_DIR | cut -f1)"
else
    echo -e "${YELLOW}⚠️  Output directory not created yet${NC}"
fi

echo ""

# Check completion status
train_bin="${OUTPUT_DIR}/ja_wiki_train_text_document.bin"
train_idx="${OUTPUT_DIR}/ja_wiki_train_text_document.idx"
val_bin="${OUTPUT_DIR}/ja_wiki_val_text_document.bin"
val_idx="${OUTPUT_DIR}/ja_wiki_val_text_document.idx"

if [[ -f "$train_bin" && -f "$train_idx" && -f "$val_bin" && -f "$val_idx" ]]; then
    echo -e "${GREEN}🎉 Data processing completed!${NC}"
    echo ""
    echo -e "${BLUE}📈 Final Statistics:${NC}"
    echo "Training data: $(du -h $train_bin | cut -f1)"
    echo "Validation data: $(du -h $val_bin | cut -f1)"
    echo "Total size: $(du -sh $OUTPUT_DIR | cut -f1)"
    echo ""
    echo -e "${GREEN}🚀 Ready for next task: Task 7 - Implement PEFT-LoRA fine-tuning script${NC}"
elif [[ -f "$train_bin" ]]; then
    echo -e "${YELLOW}⏳ Data processing in progress... Training data conversion completed, processing validation data${NC}"
else
    echo -e "${YELLOW}⏳ Data processing in progress... Converting training data${NC}"
fi

echo ""
echo "Tip: Run 'watch -n 30 ./scripts/data_processing/monitor_progress.sh' for real-time monitoring"