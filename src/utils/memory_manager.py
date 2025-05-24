import torch
import gc
import psutil
import logging
from typing import Optional
from config import Config

class MemoryManager:
    def __init__(self):
        self.peak_memory = 0
        self.logger = logging.getLogger(__name__)
        
    def clear_gpu_cache(self):
        """Clear GPU cache to prevent memory leaks"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            # MPS doesn't have empty_cache, but we can force garbage collection
            gc.collect()
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage statistics"""
        stats = {
            'cpu_percent': psutil.virtual_memory().percent,
            'cpu_available_gb': psutil.virtual_memory().available / (1024**3),
        }
        
        if torch.cuda.is_available():
            stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        elif torch.backends.mps.is_available():
            stats['mps_allocated_gb'] = torch.mps.current_allocated_memory() / (1024**3)
        
        return stats
    
    def monitor_memory(self, operation: str = ""):
        """Monitor memory during operations"""
        before = self.get_memory_usage()
        
        def cleanup():
            self.clear_gpu_cache()
            after = self.get_memory_usage()
            
            if Config.DEVICE in ['cuda', 'mps']:
                gpu_key = 'gpu_allocated_gb' if Config.DEVICE == 'cuda' else 'mps_allocated_gb'
                if gpu_key in after and after[gpu_key] > self.peak_memory:
                    self.peak_memory = after[gpu_key]
                    if after[gpu_key] > 8.0:  # Warning if >8GB GPU usage
                        self.logger.warning(f"High GPU memory usage: {after[gpu_key]:.2f}GB during {operation}")
        
        return cleanup
    
    def optimize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor memory usage"""
        if tensor.device != Config.DEVICE:
            tensor = tensor.to(Config.DEVICE, non_blocking=True)
        
        # Use half precision for inference if supported
        if Config.DEVICE == 'cuda' and tensor.dtype == torch.float32:
            return tensor.half()
        
        return tensor
    
    def force_cleanup(self):
        """Force aggressive memory cleanup"""
        self.clear_gpu_cache()
        gc.collect()
        
    def get_memory_stats(self) -> dict:
        """Alias for get_memory_usage for compatibility"""
        return self.get_memory_usage()

# Global memory manager instance
memory_manager = MemoryManager()