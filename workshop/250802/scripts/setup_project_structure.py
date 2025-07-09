#!/usr/bin/env python3
"""
项目目录结构设置脚本
基于NeMo 2.0 + Qwen2.5-0.5B日语持续学习Workshop项目需求
"""

import os
from pathlib import Path

def create_project_structure():
    """创建完整的项目目录结构"""
    
    # 项目根目录
    project_root = Path(".")
    
    # 定义目录结构
    directories = [
        # 数据目录
        "data",
        "data/llm_jp_wiki",
        "data/llm_jp_wiki/processed",
        "data/llm_jp_wiki/nemo_format", 
        "data/llm_jp_wiki/nemo_binary",
        
        # 模型目录
        "models",
        "models/checkpoints",
        "models/checkpoints/qwen25_continual",
        "models/checkpoints/qwen25_peft",
        
        # 脚本目录
        "scripts",
        "scripts/data_processing",
        "scripts/training",
        "scripts/inference",
        "scripts/evaluation",
        
        # 日志目录
        "logs",
        "logs/training",
        "logs/inference",
        "logs/data_processing",
        
        # 文档目录
        "docs",
        "docs/model_cards",
        "docs/api_docs", 
        "docs/user_guides",
        
        # 配置目录
        "configs",
        "configs/training",
        "configs/data",
        "configs/wandb",
        
        # 输出目录
        "outputs",
        "outputs/inference_results",
        "outputs/evaluation_reports",
        "outputs/demo_materials",
        
        # 工具目录
        "tools",
        "tools/data_validation",
        "tools/model_analysis",
        "tools/demo_tools",
        
        # 测试目录
        "tests",
        "tests/unit_tests",
        "tests/integration_tests",
        "tests/performance_tests"
    ]
    
    # 创建目录
    print("🚀 创建项目目录结构...")
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    # 创建README文件
    create_readme_files(project_root)
    
    # 创建配置文件模板
    create_config_templates(project_root)
    
    # 创建gitignore
    create_gitignore(project_root)
    
    print("\n🎉 项目目录结构创建完成！")
    print("📁 主要目录：")
    print("   📂 data/          - 数据存储")
    print("   📂 models/        - 模型和checkpoint")
    print("   📂 scripts/       - 所有脚本文件")
    print("   📂 logs/          - 日志文件")
    print("   📂 docs/          - 文档")
    print("   📂 configs/       - 配置文件")
    print("   📂 outputs/       - 输出结果")
    print("   📂 tools/         - 工具脚本")
    print("   📂 tests/         - 测试文件")

def create_readme_files(project_root):
    """创建各目录的README文件"""
    
    readme_contents = {
        "data/README.md": """# 数据目录

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
""",
        
        "models/README.md": """# 模型目录

## 目录结构
- `qwen25_0.5b.nemo` - 导入的基础模型
- `checkpoints/` - 训练checkpoint
  - `qwen25_continual/` - 继续学习checkpoint
  - `qwen25_peft/` - PEFT-LoRA微调checkpoint

## 模型流程
1. HuggingFace → NeMo格式 (`qwen25_0.5b.nemo`)
2. 继续学习训练 → `qwen25_continual/`
3. PEFT微调 → `qwen25_peft/`
""",
        
        "scripts/README.md": """# 脚本目录

## 核心脚本
1. `01_import_model.py` - 模型导入
2. `02a_process_data_uzushio.py` - Uzushio数据预处理
3. `02b_process_data_curator.py` - NeMo Curator数据处理
4. `02c_binarize_data.py` - 数据二值化
5. `03_run_continual_learning.py` - 继续学习训练
6. `04_run_peft_tuning.py` - PEFT微调
7. `05_run_inference.py` - 模型推理

## 目录说明
- `data_processing/` - 数据处理相关脚本
- `training/` - 训练相关脚本
- `inference/` - 推理相关脚本
- `evaluation/` - 评估相关脚本
""",
        
        "logs/README.md": """# 日志目录

## 目录结构
- `training/` - 训练日志
- `inference/` - 推理日志
- `data_processing/` - 数据处理日志

## 日志管理
- 自动按日期创建子目录
- 保留最近30天的日志
- WandB链接记录在对应日志中
""",
        
        "docs/README.md": """# 文档目录

## 目录结构
- `model_cards/` - 模型卡片和交付文档
- `api_docs/` - API文档
- `user_guides/` - 用户指南

## 交付文档
- `MODEL_CARD.md` - 模型交付卡片
- `USAGE_GUIDE.md` - 使用指南
- `WANDB_LINKS.md` - WandB实验链接
"""
    }
    
    for file_path, content in readme_contents.items():
        readme_file = project_root / file_path
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📄 Created: {file_path}")

def create_config_templates(project_root):
    """创建配置文件模板"""
    
    # WandB配置模板
    wandb_config = """# WandB配置模板
# 团队信息
entity: "your-team"

# 项目名称
projects:
  continual: "qwen25-japanese-continual"
  peft: "qwen25-japanese-peft"
  demo: "qwen25-demo-live"

# 通用标签
common_tags:
  - "nemo2.0"
  - "qwen2.5-0.5b"
  - "japanese"
  - "workshop"

# 环境信息
environment:
  gpu: "RTX_6000_Ada"
  container: "nemo:25.04"
  python: "3.10"
"""
    
    with open(project_root / "configs/wandb/wandb_config.yaml", 'w', encoding='utf-8') as f:
        f.write(wandb_config)
    
    # 训练配置模板
    training_config = """# 训练配置模板

# 继续学习配置
continual_learning:
  model: "qwen2.5-0.5b"
  max_steps: 1000
  micro_batch_size: 4
  global_batch_size: 64
  seq_length: 2048
  log_every_n_steps: 10

# PEFT配置
peft:
  scheme: "lora"
  max_steps: 500
  micro_batch_size: 4
  global_batch_size: 64
  seq_length: 2048
  log_every_n_steps: 5
  # PEFT特殊配置
  ckpt_async_save: false
  context_parallel_size: 1
  ddp: "megatron"
"""
    
    with open(project_root / "configs/training/training_config.yaml", 'w', encoding='utf-8') as f:
        f.write(training_config)
    
    print(f"📄 Created: configs/wandb/wandb_config.yaml")
    print(f"📄 Created: configs/training/training_config.yaml")

def create_gitignore(project_root):
    """创建.gitignore文件"""
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# PyCharm
.idea/

# VSCode
.vscode/

# Data files (too large for git)
data/llm_jp_wiki/
*.bin
*.idx
*.parquet

# Model files (too large for git)
models/*.nemo
models/checkpoints/

# Logs
logs/*.log
logs/**/*.log

# Temporary files
*.tmp
*.temp
.DS_Store
Thumbs.db

# WandB
wandb/

# Docker
.dockerignore

# Environment variables
.env
.env.local

# Outputs
outputs/inference_results/
outputs/evaluation_reports/

# Cache
.cache/
"""
    
    gitignore_file = project_root / ".gitignore"
    # 读取现有内容避免重复
    existing_content = ""
    if gitignore_file.exists():
        with open(gitignore_file, 'r', encoding='utf-8') as f:
            existing_content = f.read()
    
    # 只添加新内容
    if "# Python" not in existing_content:
        with open(gitignore_file, 'a', encoding='utf-8') as f:
            f.write(gitignore_content)
        print(f"📄 Updated: .gitignore")

if __name__ == "__main__":
    create_project_structure() 