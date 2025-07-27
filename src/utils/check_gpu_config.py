#!/usr/bin/env python3
"""
GPU配置检查脚本
验证6000Ada GPU的状态和配置
"""

import torch
import logging

def setup_logging():
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('gpu_check.log')
        ]
    )
    return logging.getLogger(__name__)

def check_gpu_status():
    """检查GPU状态"""
    logger = setup_logging()
    
    logger.info("=== GPU配置检查 ===")
    
    # 检查CUDA可用性
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA可用性: {cuda_available}")
    
    if not cuda_available:
        logger.error("CUDA不可用，无法进行GPU训练")
        return False
    
    # 检查GPU数量
    gpu_count = torch.cuda.device_count()
    logger.info(f"可用GPU数量: {gpu_count}")
    
    # 检查GPU详细信息
    for i in range(gpu_count):
        logger.info(f"=== GPU {i} 详细信息 ===")
        
        # GPU名称
        gpu_name = torch.cuda.get_device_name(i)
        logger.info(f"GPU {i} 名称: {gpu_name}")
        
        # 显存信息
        memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        memory_free = memory_total - memory_reserved
        
        logger.info(f"GPU {i} 总显存: {memory_total:.2f} GB")
        logger.info(f"GPU {i} 已分配显存: {memory_allocated:.2f} GB")
        logger.info(f"GPU {i} 已预留显存: {memory_reserved:.2f} GB")
        logger.info(f"GPU {i} 可用显存: {memory_free:.2f} GB")
        
        # 检查是否是6000Ada
        if "6000 Ada" in gpu_name or "RTX 6000 Ada" in gpu_name:
            logger.info(f"✓ 发现6000Ada GPU: {gpu_name}")
        
        # 计算配置
        compute_capability = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i} 计算能力: {compute_capability.major}.{compute_capability.minor}")
        logger.info(f"GPU {i} 多处理器数量: {compute_capability.multi_processor_count}")
    
    # 设置默认GPU为GPU 0
    if gpu_count > 0:
        torch.cuda.set_device(0)
        current_device = torch.cuda.current_device()
        logger.info(f"当前默认GPU设备: {current_device}")
        
        # 简单的GPU测试
        logger.info("=== GPU功能测试 ===")
        try:
            # 创建测试张量
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.matmul(test_tensor, test_tensor.t())
            logger.info("✓ GPU矩阵运算测试通过")
            
            # 清理显存
            del test_tensor, result
            torch.cuda.empty_cache()
            logger.info("✓ 显存清理完成")
            
        except Exception as e:
            logger.error(f"✗ GPU功能测试失败: {str(e)}")
            return False
    
    logger.info("=== GPU检查完成 ===")
    return True

def check_training_readiness():
    """检查训练准备状态"""
    logger = logging.getLogger(__name__)
    
    logger.info("=== 训练准备状态检查 ===")
    
    # 检查PyTorch版本
    pytorch_version = torch.__version__
    logger.info(f"PyTorch版本: {pytorch_version}")
    
    # 检查CUDA版本
    cuda_version = torch.version.cuda
    logger.info(f"CUDA版本: {cuda_version}")
    
    # 检查cuDNN版本
    cudnn_version = torch.backends.cudnn.version()
    logger.info(f"cuDNN版本: {cudnn_version}")
    
    # 检查是否启用了cuDNN
    cudnn_enabled = torch.backends.cudnn.enabled
    logger.info(f"cuDNN启用状态: {cudnn_enabled}")
    
    # 建议配置
    logger.info("=== 单GPU训练建议配置 ===")
    logger.info("建议配置:")
    logger.info("- 设备数量: 1 (GPU 0)")
    logger.info("- 微批次大小: 2 (适应6000Ada 48GB显存)")
    logger.info("- 全局批次大小: 8")
    logger.info("- 序列长度: 2048")
    logger.info("- 混合精度训练: 推荐使用fp16")
    
    return True

if __name__ == "__main__":
    success = check_gpu_status()
    if success:
        check_training_readiness()
        print("\n✓ GPU检查完成，可以开始单GPU训练")
    else:
        print("\n✗ GPU检查失败，请检查CUDA环境") 