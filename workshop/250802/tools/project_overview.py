#!/usr/bin/env python3
"""
NeMo 2.0 + Qwen2.5-0.5B 日语Workshop项目概览工具
快速查看项目状态、任务进度和文件结构
"""

import os
import json
from pathlib import Path
from datetime import datetime

def main():
    """主函数：显示项目概览"""
    print("🎯 NeMo 2.0 + Qwen2.5-0.5B 日语Workshop项目")
    print("=" * 60)
    print(f"📅 查看时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 项目根目录: {Path.cwd()}")
    print()
    
    # 显示任务状态
    show_task_status()
    
    # 显示目录结构
    show_directory_structure()
    
    # 显示关键文件状态
    show_key_files()
    
    # 显示下一步操作
    show_next_steps()

def show_task_status():
    """显示任务状态概览"""
    print("📋 任务状态概览")
    print("-" * 30)
    
    tasks_file = Path(".taskmaster/tasks/tasks.json")
    if not tasks_file.exists():
        print("❌ 任务文件不存在")
        return
    
    with open(tasks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tasks = data.get("master", {}).get("tasks", [])
    
    # 统计状态
    status_count = {}
    for task in tasks:
        status = task.get("status", "unknown")
        status_count[status] = status_count.get(status, 0) + 1
    
    print(f"📊 总任务数: {len(tasks)}")
    for status, count in status_count.items():
        emoji = {
            "pending": "⏳",
            "in-progress": "🔄", 
            "done": "✅",
            "blocked": "🚫",
            "deferred": "⏸️"
        }.get(status, "❓")
        print(f"   {emoji} {status}: {count}")
    
    # 显示下一个任务
    next_task = None
    for task in tasks:
        if task.get("status") == "pending":
            # 检查依赖是否完成
            deps = task.get("dependencies", [])
            deps_completed = all(
                any(t.get("id") == dep and t.get("status") == "done" for t in tasks)
                for dep in deps
            ) if deps else True
            
            if deps_completed:
                next_task = task
                break
    
    if next_task:
        print(f"\n🎯 下一个任务: {next_task.get('id')} - {next_task.get('title')}")
    
    print()

def show_directory_structure():
    """显示关键目录结构"""
    print("📂 项目目录结构")
    print("-" * 30)
    
    key_dirs = [
        ("data", "数据存储"),
        ("models", "模型和checkpoint"),
        ("scripts", "脚本文件"),
        ("logs", "日志文件"),
        ("configs", "配置文件"),
        ("docs", "文档"),
        ("outputs", "输出结果"),
        (".taskmaster/tasks", "任务文件")
    ]
    
    for dir_path, description in key_dirs:
        path = Path(dir_path)
        if path.exists():
            file_count = len(list(path.rglob("*"))) if path.is_dir() else 1
            print(f"✅ {dir_path:<20} - {description} ({file_count} 项)")
        else:
            print(f"❌ {dir_path:<20} - {description} (不存在)")
    
    print()

def show_key_files():
    """显示关键文件状态"""
    print("📄 关键文件状态")
    print("-" * 30)
    
    key_files = [
        (".taskmaster/docs/prd.txt", "产品需求文档"),
        (".taskmaster/config.json", "TaskMaster配置"),
        ("configs/wandb/wandb_config.yaml", "WandB配置模板"),
        ("configs/training/training_config.yaml", "训练配置模板"),
        (".gitignore", "Git忽略文件"),
        ("scripts/setup_project_structure.py", "项目结构脚本"),
    ]
    
    for file_path, description in key_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            size_str = f"{size/1024:.1f}KB" if size > 1024 else f"{size}B"
            print(f"✅ {file_path:<35} - {description} ({size_str})")
        else:
            print(f"❌ {file_path:<35} - {description} (不存在)")
    
    print()

def show_next_steps():
    """显示下一步操作建议"""
    print("🚀 下一步操作建议")
    print("-" * 30)
    
    steps = [
        "1. 查看任务详情: `cat .taskmaster/tasks/task_001.txt`",
        "2. 开始第一个任务: TaskMaster set-status --id=1 --status=in-progress",
        "3. 启动NeMo容器: `docker run --gpus all -it --rm -v .:/workspace nvcr.io/nvidia/nemo:25.04`",
        "4. 监控进度: 定期查看WandB面板",
        "5. 更新任务状态: TaskMaster update-subtask --id=1.x --prompt='进展情况'"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print()
    print("🎯 项目目标: 14天完成NeMo 2.0 + Qwen2.5-0.5B日语持续学习与PEFT微调")
    print("📊 当前状态: 项目结构已完成，准备开始环境设置")
    print("🤝 团队协作: WandB统一监控，模型标准化交付")

if __name__ == "__main__":
    main() 