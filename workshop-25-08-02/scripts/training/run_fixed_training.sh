#!/bin/bash
# scripts/training/run_fixed_training.sh

echo "🚀 Starting NeMo fixed version training..."

# Option 1: Clean checkpoint and restart training (recommended)
echo "📋 Select training option:"
echo "1. Clean checkpoints and restart (recommended)"
echo "2. Fresh start only (preserve checkpoint directory)"
echo "3. Attempt recovery training (not recommended)"
read -p "Please select (1/2/3): " choice

case $choice in
    1)
        echo "🧹 Cleaning corrupted checkpoints and restarting training..."
        docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
            -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/nemo:25.04 \
            python scripts/training/nemo_official_fixed_final.py --clean-checkpoints --fresh-start
        ;;
    2)
        echo "🔄 Starting fresh training (preserving checkpoint directory)..."
        docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
            -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/nemo:25.04 \
            python scripts/training/nemo_official_fixed_final.py --fresh-start
        ;;
    3)
        echo "⚠️ Attempting recovery training (may fail)..."
        docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
            -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/nemo:25.04 \
            python scripts/training/nemo_official_fixed_final.py
        ;;
    *)
        echo "❌ Invalid selection, exiting."
        exit 1
        ;;
esac

echo "✅ Training command has been launched!" 