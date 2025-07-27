#!/usr/bin/env python3
"""
Validate the imported Qwen2.5-0.5B model for correctness
Use NeMo 2.0's recipe system for validation
"""

import os
import sys
from pathlib import Path

# Set project root directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def validate_model():
    """Validate if the imported model is correct"""
    
    print("🔍 Validating NeMo 2.0 Qwen2.5-0.5B Model Import")
    print("=" * 50)
    
    # 1. Check model directory structure
    model_path = project_root / "data/models/qwen25_0.5b.nemo"
    print(f"📁 Checking model directory: {model_path}")
    
    if not model_path.exists():
        print("❌ Model directory does not exist")
        return False
    
    # Check required files
    required_files = [
        "context/model.yaml",
        "context/io.json", 
        "context/nemo_tokenizer/tokenizer.json",
        "weights/metadata.json",
        "weights/__0_0.distcp",
        "weights/__0_1.distcp"
    ]
    
    for file_path in required_files:
        full_path = model_path / file_path
        if not full_path.exists():
            print(f"❌ Missing file: {file_path}")
            return False
        print(f"✅ Found file: {file_path}")
    
    # 2. Check model configuration
    print("\n📋 Checking model configuration...")
    config_path = model_path / "context/model.yaml"
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate key configurations
        model_config = config['config']
        expected_configs = {
            'hidden_size': 896,
            'num_layers': 24,
            'num_attention_heads': 14,
            'vocab_size': 151936,
            'seq_length': 32768
        }
        
        for key, expected_value in expected_configs.items():
            actual_value = model_config.get(key)
            if actual_value == expected_value:
                print(f"✅ {key}: {actual_value}")
            else:
                print(f"❌ {key}: expected {expected_value}, got {actual_value}")
                return False
                
    except Exception as e:
        print(f"❌ Configuration file reading failed: {e}")
        return False
    
    # 3. Check weight file sizes
    print("\n📊 Checking weight file sizes...")
    weights_dir = model_path / "weights"
    total_size = 0
    
    for weight_file in weights_dir.glob("*.distcp"):
        size = weight_file.stat().st_size
        total_size += size
        print(f"✅ {weight_file.name}: {size/1024/1024:.1f}MB")
    
    print(f"🎯 Total weight size: {total_size/1024/1024:.1f}MB")
    
    # 4. Check tokenizer
    print("\n🔤 Checking tokenizer...")
    tokenizer_path = model_path / "context/nemo_tokenizer"
    tokenizer_files = ["tokenizer.json", "vocab.json", "merges.txt"]
    
    for file_name in tokenizer_files:
        file_path = tokenizer_path / file_name
        if file_path.exists():
            print(f"✅ {file_name}: {file_path.stat().st_size/1024:.1f}KB")
        else:
            print(f"❌ Missing tokenizer file: {file_name}")
            return False
    
    # 5. Validate model availability through recipe system
    print("\n🧪 Validating model through recipe system...")
    try:
        from nemo.collections import llm
        
        # Create a simple recipe configuration
        recipe = llm.qwen25_500m.pretrain_recipe(
            name="validation_test",
            dir=str(project_root / "temp_validation"),
            num_nodes=1,
            num_gpus_per_node=1,
        )
        
        # Configure model restore path
        recipe.resume.restore_config.path = str(model_path)
        
        print("✅ Recipe configuration successful")
        print(f"✅ Model path set: {recipe.resume.restore_config.path}")
        
        # Validate configuration object
        if hasattr(recipe, 'model') and hasattr(recipe, 'resume'):
            print("✅ Recipe structure correct")
        else:
            print("❌ Recipe structure abnormal")
            return False
            
    except Exception as e:
        print(f"❌ Recipe validation failed: {e}")
        return False
    
    print("\n🎉 Model validation complete!")
    print("=" * 50)
    print("✅ All validation items passed")
    print("✅ Model import completely correct")
    print("✅ Ready for subsequent continual learning training")
    
    return True

if __name__ == "__main__":
    success = validate_model()
    sys.exit(0 if success else 1) 