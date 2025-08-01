import torch
import gc
import logging

logger = logging.getLogger(__name__)

def optimize_gpu_memory():
    """Optimize GPU memory usage for RTX 3060 6GB"""
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Set memory fraction (use 90% of available memory)
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass
            
        logger.info(f"GPU Memory optimized. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def clear_gpu_cache():
    """Clear GPU cache between operations"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class GPUMemoryMonitor:
    def __enter__(self):
        if torch.cuda.is_available():
            self.start_memory = torch.cuda.memory_allocated()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            memory_used = (end_memory - self.start_memory) / 1024**2
            logger.info(f"GPU Memory used in operation: {memory_used:.1f} MB")