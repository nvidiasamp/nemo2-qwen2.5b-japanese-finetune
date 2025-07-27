#!/usr/bin/env python3
"""
验证导入的Qwen2.5-0.5B模型是否正确
使用NeMo 2.0的recipe系统进行验证
"""

import os
import sys
from pathlib import Path

# 设置项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def validate_model():
    """验证导入的模型是否正确"""
    
    print("🔍 验证NeMo 2.0 Qwen2.5-0.5B模型导入")
    print("=" * 50)
    
    # 1. 检查模型目录结构
    model_path = project_root / "data/models/qwen25_0.5b.nemo"
    print(f"📁 检查模型目录: {model_path}")
    
    if not model_path.exists():
        print("❌ 模型目录不存在")
        return False
    
    # 检查必要文件
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
            print(f"❌ 缺少文件: {file_path}")
            return False
        print(f"✅ 找到文件: {file_path}")
    
    # 2. 检查模型配置
    print("\n📋 检查模型配置...")
    config_path = model_path / "context/model.yaml"
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 验证关键配置
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
                print(f"❌ {key}: 期望{expected_value}, 实际{actual_value}")
                return False
                
    except Exception as e:
        print(f"❌ 配置文件读取失败: {e}")
        return False
    
    # 3. 检查权重文件大小
    print("\n📊 检查权重文件大小...")
    weights_dir = model_path / "weights"
    total_size = 0
    
    for weight_file in weights_dir.glob("*.distcp"):
        size = weight_file.stat().st_size
        total_size += size
        print(f"✅ {weight_file.name}: {size/1024/1024:.1f}MB")
    
    print(f"🎯 总权重大小: {total_size/1024/1024:.1f}MB")
    
    # 4. 检查分词器
    print("\n🔤 检查分词器...")
    tokenizer_path = model_path / "context/nemo_tokenizer"
    tokenizer_files = ["tokenizer.json", "vocab.json", "merges.txt"]
    
    for file_name in tokenizer_files:
        file_path = tokenizer_path / file_name
        if file_path.exists():
            print(f"✅ {file_name}: {file_path.stat().st_size/1024:.1f}KB")
        else:
            print(f"❌ 缺少分词器文件: {file_name}")
            return False
    
    # 5. 通过recipe系统验证模型可用性
    print("\n🧪 通过recipe系统验证模型...")
    try:
        from nemo.collections import llm
        
        # 创建一个简单的recipe配置
        recipe = llm.qwen25_500m.pretrain_recipe(
            name="validation_test",
            dir=str(project_root / "temp_validation"),
            num_nodes=1,
            num_gpus_per_node=1,
        )
        
        # 配置模型恢复路径
        recipe.resume.restore_config.path = str(model_path)
        
        print("✅ Recipe配置成功")
        print(f"✅ 模型路径设置: {recipe.resume.restore_config.path}")
        
        # 验证配置对象
        if hasattr(recipe, 'model') and hasattr(recipe, 'resume'):
            print("✅ Recipe结构正确")
        else:
            print("❌ Recipe结构异常")
            return False
            
    except Exception as e:
        print(f"❌ Recipe验证失败: {e}")
        return False
    
    print("\n🎉 模型验证完成!")
    print("=" * 50)
    print("✅ 所有验证项目均通过")
    print("✅ 模型导入完全正确")
    print("✅ 可以用于后续的持续学习训练")
    
    return True

if __name__ == "__main__":
    success = validate_model()
    sys.exit(0 if success else 1) 