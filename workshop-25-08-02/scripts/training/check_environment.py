#!/usr/bin/env python3
# scripts/training/check_environment.py

"""
Environment validation script - Ensures proper training environment
Prevents running scripts in incorrect environments that require Docker
"""

import os
import sys
import subprocess
from pathlib import Path

def check_docker_available():
    """Check if Docker is available"""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Docker available: {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker not available")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Docker not installed or unavailable")
        return False

def check_gpu_available():
    """Check if GPU is available"""
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA GPU available")
            return True
        else:
            print("❌ NVIDIA GPU not available")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi not found, GPU may not be available")
        return False

def check_data_files():
    """Check training data files"""
    required_files = [
        "data/llm_jp_wiki/nemo_binary/ja_wiki_train_text_document.bin",
        "data/llm_jp_wiki/nemo_binary/ja_wiki_val_text_document.bin",
        "data/llm_jp_wiki/nemo_binary/ja_wiki_train_text_document.idx",
        "data/llm_jp_wiki/nemo_binary/ja_wiki_val_text_document.idx",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing training data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print("✅ All training data files exist")
        return True

def check_nemo_environment():
    """Check if running in NeMo environment"""
    try:
        import nemo_run
        print("✅ Running in NeMo environment - can execute Python scripts directly")
        return True
    except ImportError:
        print("ℹ️  Not in NeMo environment - Docker execution required")
        return False

def main():
    """Main validation function"""
    print("🔍 Environment validation in progress...")
    print("=" * 50)
    
    # Check environment components
    docker_ok = check_docker_available()
    gpu_ok = check_gpu_available()
    data_ok = check_data_files()
    nemo_env = check_nemo_environment()
    
    print("=" * 50)
    
    # Provide recommendations
    if nemo_env:
        print("\n🎯 Recommendation: You are in NeMo environment, can run directly:")
        print("   python scripts/training/nemo_official_fixed_final.py --clean-checkpoints --fresh-start")
    elif docker_ok and gpu_ok and data_ok:
        print("\n🎯 Recommendation: Use Docker to run training:")
        print("   bash scripts/training/run_fixed_training.sh")
        print("   or:")
        print("   ./scripts/training/run_fixed_training.sh")
    else:
        print("\n❌ Environment validation failed, please resolve the following issues:")
        if not docker_ok:
            print("   - Install and start Docker")
        if not gpu_ok:
            print("   - Ensure NVIDIA GPU and drivers are working")
        if not data_ok:
            print("   - Prepare training data files")
    
    print("\n📋 Quick Start Guide:")
    print("   1. Ensure all validation checks pass")
    print("   2. Run: bash scripts/training/run_fixed_training.sh")
    print("   3. Select option 1 (clean checkpoints and restart)")

if __name__ == "__main__":
    main() 