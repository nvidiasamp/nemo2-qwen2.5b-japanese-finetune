#!/usr/bin/env python3
# scripts/training/monitor_training.py

"""
Real-time training monitoring tool
Provides continuous monitoring of training progress and system status
"""

import time
import os
import subprocess
import glob
from pathlib import Path

def check_docker_processes():
    """Check for running Docker training processes"""
    try:
        result = subprocess.run(['docker', 'ps', '--filter', 'ancestor=nvcr.io/nvidia/nemo:25.04'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and len(result.stdout.split('\n')) > 2:
            return result.stdout
        return None
    except Exception:
        return None

def check_training_logs():
    """Check latest training logs"""
    log_patterns = [
        "experiments/*/nemo_log_globalrank-0_localrank-0.txt",
        "experiments/*/lightning_logs.txt"
    ]
    
    latest_logs = []
    for pattern in log_patterns:
        files = glob.glob(pattern)
        if files:
            latest_file = max(files, key=os.path.getmtime)
            try:
                with open(latest_file, 'r') as f:
                    lines = f.readlines()
                    latest_logs.append((latest_file, lines[-5:] if len(lines) >= 5 else lines))
            except Exception:
                pass
    
    return latest_logs

def check_checkpoints():
    """Check checkpoint status"""
    checkpoint_dirs = glob.glob("experiments/*/checkpoints")
    checkpoint_info = []
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_files = glob.glob(f"{checkpoint_dir}/*")
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
            size = os.path.getsize(latest_checkpoint) / (1024 * 1024)  # MB
            checkpoint_info.append((latest_checkpoint, f"{size:.1f}MB"))
    
    return checkpoint_info

def monitor_training():
    """Main monitoring function"""
    print("🔍 === Training Status Monitor ===")
    print("Press Ctrl+C to exit monitoring")
    
    try:
        while True:
            print("\n" + "="*60)
            print(f"⏰ Monitor time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Check Docker processes
            print("\n🐳 Docker process status:")
            docker_status = check_docker_processes()
            if docker_status:
                print(docker_status)
            else:
                print("   No active NeMo training containers found")
            
            # Check checkpoint status
            print("\n💾 Checkpoint status:")
            checkpoints = check_checkpoints()
            if checkpoints:
                for checkpoint_path, size in checkpoints:
                    print(f"   Latest: {os.path.basename(checkpoint_path)} ({size})")
            else:
                print("   No checkpoints found")
            
            # Check training logs
            print("\n📋 Training logs:")
            logs = check_training_logs()
            if logs:
                for log_path, recent_lines in logs:
                    print(f"   {os.path.basename(log_path)}:")
                    for line in recent_lines:
                        print(f"     {line.strip()}")
            else:
                print("   No recent training logs found")
            
            print(f"\n⏳ Waiting 30 seconds before refresh... (Press Ctrl+C to exit)")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

if __name__ == "__main__":
    monitor_training() 