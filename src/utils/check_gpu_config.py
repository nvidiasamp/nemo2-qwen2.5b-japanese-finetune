#!/usr/bin/env python3
"""
GPU configuration check script
Verify the status and configuration of 6000Ada GPU
"""

import torch
import logging

def setup_logging():
    """Setup logging"""
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
    """Check GPU status"""
    logger = setup_logging()
    
    logger.info("=== GPU Configuration Check ===")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA availability: {cuda_available}")
    
    if not cuda_available:
        logger.error("CUDA not available, cannot proceed with GPU training")
        return False
    
    # Check GPU count
    gpu_count = torch.cuda.device_count()
    logger.info(f"Available GPU count: {gpu_count}")
    
    # Check detailed GPU information
    for i in range(gpu_count):
        logger.info(f"=== GPU {i} Details ===")
        
        # GPU name
        gpu_name = torch.cuda.get_device_name(i)
        logger.info(f"GPU {i} name: {gpu_name}")
        
        # Memory information
        memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        memory_free = memory_total - memory_reserved
        
        logger.info(f"GPU {i} total memory: {memory_total:.2f} GB")
        logger.info(f"GPU {i} allocated memory: {memory_allocated:.2f} GB")
        logger.info(f"GPU {i} reserved memory: {memory_reserved:.2f} GB")
        logger.info(f"GPU {i} available memory: {memory_free:.2f} GB")
        
        # Check if it's 6000Ada
        if "6000 Ada" in gpu_name or "RTX 6000 Ada" in gpu_name:
            logger.info(f"✓ Found 6000Ada GPU: {gpu_name}")
        
        # Compute capability
        compute_capability = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i} compute capability: {compute_capability.major}.{compute_capability.minor}")
        logger.info(f"GPU {i} multiprocessor count: {compute_capability.multi_processor_count}")
    
    # Set default GPU to GPU 0
    if gpu_count > 0:
        torch.cuda.set_device(0)
        current_device = torch.cuda.current_device()
        logger.info(f"Current default GPU device: {current_device}")
        
        # Simple GPU test
        logger.info("=== GPU Function Test ===")
        try:
            # Create test tensor
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.matmul(test_tensor, test_tensor.t())
            logger.info("✓ GPU matrix computation test passed")
            
            # Clean memory
            del test_tensor, result
            torch.cuda.empty_cache()
            logger.info("✓ Memory cleanup completed")
            
        except Exception as e:
            logger.error(f"✗ GPU function test failed: {str(e)}")
            return False
    
    logger.info("=== GPU Check Complete ===")
    return True

def check_training_readiness():
    """Check training readiness status"""
    logger = logging.getLogger(__name__)
    
    logger.info("=== Training Readiness Check ===")
    
    # Check PyTorch version
    pytorch_version = torch.__version__
    logger.info(f"PyTorch version: {pytorch_version}")
    
    # Check CUDA version
    cuda_version = torch.version.cuda
    logger.info(f"CUDA version: {cuda_version}")
    
    # Check cuDNN version
    cudnn_version = torch.backends.cudnn.version()
    logger.info(f"cuDNN version: {cudnn_version}")
    
    # Check if cuDNN is enabled
    cudnn_enabled = torch.backends.cudnn.enabled
    logger.info(f"cuDNN enabled status: {cudnn_enabled}")
    
    # Recommended configuration
    logger.info("=== Single GPU Training Recommended Configuration ===")
    logger.info("Recommended settings:")
    logger.info("- Device count: 1 (GPU 0)")
    logger.info("- Micro batch size: 2 (fits 6000Ada 48GB memory)")
    logger.info("- Global batch size: 8")
    logger.info("- Sequence length: 2048")
    logger.info("- Mixed precision training: recommend using fp16")
    
    return True

if __name__ == "__main__":
    success = check_gpu_status()
    if success:
        check_training_readiness()
        print("\n✓ GPU check completed, ready to start single GPU training")
    else:
        print("\n✗ GPU check failed, please check CUDA environment") 