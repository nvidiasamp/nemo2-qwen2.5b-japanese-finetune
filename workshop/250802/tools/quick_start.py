#!/usr/bin/env python3
"""
NeMo 2.0 + Qwen2.5-0.5B Workshop 快速启动脚本
"""

import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1]
    
    if command == "overview":
        show_overview()
    elif command == "task":
        if len(sys.argv) >= 3:
            show_task(sys.argv[2])
        else:
            show_next_task()
    elif command == "start":
        if len(sys.argv) >= 3:
            start_task(sys.argv[2])
        else:
            start_next_task()
    elif command == "status":
        show_status()
    elif command == "docker":
        start_docker()
    else:
        show_help()

def show_help():
    """显示帮助信息"""
    print("🚀 NeMo 2.0 + Qwen2.5-0.5B Workshop 快速启动工具")
    print("=" * 50)
    print("用法: python3 tools/quick_start.py <command> [args]")
    print()
    print("命令:")
    print("  overview      - 显示项目概览")
    print("  task [id]     - 显示任务详情 (不指定id则显示下一个任务)")
    print("  start [id]    - 开始任务 (不指定id则开始下一个任务)")
    print("  status        - 显示所有任务状态")
    print("  docker        - 启动NeMo Docker容器")
    print()
    print("示例:")
    print("  python3 tools/quick_start.py overview")
    print("  python3 tools/quick_start.py task 1")
    print("  python3 tools/quick_start.py start 1")
    print("  python3 tools/quick_start.py docker")

def show_overview():
    """显示项目概览"""
    subprocess.run(["python3", "tools/project_overview.py"])

def show_task(task_id):
    """显示指定任务详情"""
    task_file = Path(f".taskmaster/tasks/task_{task_id:0>3}.txt")
    if task_file.exists():
        print(f"📋 任务 {task_id} 详情:")
        print("=" * 40)
        with open(task_file, 'r', encoding='utf-8') as f:
            print(f.read())
    else:
        print(f"❌ 任务文件不存在: {task_file}")

def show_next_task():
    """显示下一个任务"""
    print("🎯 下一个任务详情:")
    show_task("001")

def start_task(task_id):
    """开始指定任务"""
    print(f"🚀 开始任务 {task_id}...")
    # 这里可以集成TaskMaster命令
    print(f"💡 手动执行: TaskMaster set-status --id={task_id} --status=in-progress")
    show_task(task_id)

def start_next_task():
    """开始下一个任务"""
    print("🚀 开始下一个任务...")
    start_task("1")

def show_status():
    """显示所有任务状态"""
    print("📊 所有任务状态:")
    print("=" * 40)
    for i in range(1, 16):  # 15个任务
        task_file = Path(f".taskmaster/tasks/task_{i:0>3}.txt")
        if task_file.exists():
            with open(task_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                title = ""
                status = ""
                for line in lines:
                    if line.startswith("# Title:"):
                        title = line.replace("# Title:", "").strip()
                    elif line.startswith("# Status:"):
                        status = line.replace("# Status:", "").strip()
                
                status_emoji = {
                    "pending": "⏳",
                    "in-progress": "🔄",
                    "done": "✅",
                    "blocked": "🚫"
                }.get(status, "❓")
                
                print(f"{status_emoji} {i:2d}. {title} ({status})")

def start_docker():
    """启动NeMo Docker容器"""
    print("🐳 启动NeMo Docker容器...")
    print("执行命令:")
    cmd = "docker run --gpus all -it --rm -v .:/workspace nvcr.io/nvidia/nemo:25.04"
    print(f"  {cmd}")
    print()
    print("容器启动后请执行:")
    print("  1. pip install nemo-run wandb")
    print("  2. wandb login [YOUR_API_KEY]")
    print("  3. cd /workspace")
    print("  4. python3 tools/project_overview.py")

if __name__ == "__main__":
    main() 