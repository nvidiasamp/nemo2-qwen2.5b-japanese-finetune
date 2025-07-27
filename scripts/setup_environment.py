#!/usr/bin/env python3
"""
Environment setup script for Japanese Continual Learning with NeMo 2.0
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel

console = Console()

def run_command(cmd, description="Running command"):
    """Run a shell command with progress indication"""
    with Progress() as progress:
        task = progress.add_task(f"[green]{description}...", total=None)
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            progress.update(task, total=1, completed=1)
            return result.stdout
        except subprocess.CalledProcessError as e:
            progress.update(task, total=1, completed=1)
            console.print(f"[red]Error: {e}")
            console.print(f"[red]stderr: {e.stderr}")
            return None

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available"""
    console.print("[bold blue]Checking GPU availability...[/bold blue]")
    
    result = run_command("nvidia-smi", "Checking NVIDIA GPU")
    if result:
        console.print("[green]✅ NVIDIA GPU detected[/green]")
        return True
    else:
        console.print("[red]❌ NVIDIA GPU not found[/red]")
        return False

def check_docker():
    """Check if Docker is available"""
    console.print("[bold blue]Checking Docker...[/bold blue]")
    
    result = run_command("docker --version", "Checking Docker")
    if result:
        console.print(f"[green]✅ Docker found: {result.strip()}[/green]")
        
        # Check for NVIDIA Docker runtime
        result = run_command("docker info | grep -i nvidia", "Checking NVIDIA Docker runtime")
        if result:
            console.print("[green]✅ NVIDIA Docker runtime detected[/green]")
            return True
        else:
            console.print("[yellow]⚠️  NVIDIA Docker runtime not found[/yellow]")
            return False
    else:
        console.print("[red]❌ Docker not found[/red]")
        return False

def pull_nemo_image():
    """Pull the NeMo Docker image"""
    console.print("[bold blue]Pulling NeMo 2.0 Docker image...[/bold blue]")
    
    result = run_command(
        "docker pull nvcr.io/nvidia/nemo:25.04", 
        "Pulling NeMo 25.04 image"
    )
    
    if result is not None:
        console.print("[green]✅ NeMo Docker image pulled successfully[/green]")
        return True
    else:
        console.print("[red]❌ Failed to pull NeMo Docker image[/red]")
        return False

def setup_directories():
    """Setup project directories"""
    console.print("[bold blue]Setting up project directories...[/bold blue]")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/models",
        "data/outputs",
        "experiments/logs",
        "experiments/checkpoints",
        "experiments/results",
        "experiments/wandb",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✅ Created directory: {directory}[/green]")

def create_env_file():
    """Create .env file template"""
    console.print("[bold blue]Creating .env file template...[/bold blue]")
    
    env_content = """# Environment variables for Japanese Continual Learning with NeMo 2.0

# Weights & Biases
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=japanese-continual-learning-nemo
WANDB_ENTITY=your_team_name

# Model and data paths
MODEL_DIR=data/models
DATA_DIR=data/processed
CHECKPOINT_DIR=experiments/checkpoints
RESULTS_DIR=experiments/results

# Training configuration
BATCH_SIZE=8
LEARNING_RATE=5e-5
MAX_EPOCHS=10
PRECISION=16

# Hardware configuration
GPUS=1
NUM_WORKERS=4
"""
    
    with open(".env.template", "w") as f:
        f.write(env_content)
    
    console.print("[green]✅ Created .env.template file[/green]")
    console.print("[yellow]⚠️  Please copy .env.template to .env and fill in your values[/yellow]")

def verify_installation():
    """Verify the installation by running a simple test"""
    console.print("[bold blue]Verifying installation...[/bold blue]")
    
    test_cmd = """
    docker run --rm --gpus all \
        -v $(pwd):/workspace -w /workspace \
        nvcr.io/nvidia/nemo:25.04 \
        python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
try:
    import nemo_run
    print(f'NeMo-Run version: {nemo_run.__version__}')
except ImportError as e:
    print(f'NeMo-Run import error: {e}')
"
    """
    
    result = run_command(test_cmd, "Verifying NeMo installation")
    if result:
        console.print("[green]✅ Installation verified successfully[/green]")
        console.print(f"[blue]Output:\n{result}[/blue]")
        return True
    else:
        console.print("[red]❌ Installation verification failed[/red]")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup environment for Japanese Continual Learning with NeMo 2.0")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker setup")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification step")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold green]Japanese Continual Learning with NeMo 2.0[/bold green]\n"
        "[blue]Environment Setup Script[/blue]",
        title="Setup"
    ))
    
    # Check prerequisites
    if not args.skip_docker:
        if not check_docker():
            console.print("[red]Docker with NVIDIA runtime is required. Please install Docker and nvidia-docker2.[/red]")
            sys.exit(1)
        
        if not check_nvidia_gpu():
            console.print("[red]NVIDIA GPU is required for this project.[/red]")
            sys.exit(1)
    
    # Setup steps
    setup_directories()
    create_env_file()
    
    if not args.skip_docker:
        if not pull_nemo_image():
            console.print("[red]Failed to pull NeMo Docker image.[/red]")
            sys.exit(1)
    
    if not args.skip_verify and not args.skip_docker:
        if not verify_installation():
            console.print("[red]Installation verification failed.[/red]")
            sys.exit(1)
    
    console.print(Panel.fit(
        "[bold green]✅ Environment setup completed successfully![/bold green]\n"
        "[blue]Next steps:[/blue]\n"
        "1. Copy .env.template to .env and fill in your values\n"
        "2. Run: ./scripts/start_container.sh\n"
        "3. Start with: python src/models/import_qwen25.py",
        title="Setup Complete"
    ))

if __name__ == "__main__":
    main() 