# 数据目录

## 目录结构
- `llm_jp_wiki/` - LLM-JP日语Wikipedia语料库
  - `processed/` - Uzushio预处理后的数据
  - `nemo_format/` - NeMo Curator处理后的Parquet格式数据
  - `nemo_binary/` - NeMo训练用二进制格式数据(.bin/.idx)

## 数据流程
1. 下载LLM-JP数据 → `llm_jp_wiki/`
2. Uzushio预处理 → `processed/`
3. NeMo Curator去重 → `nemo_format/`
4. 二值化处理 → `nemo_binary/`
