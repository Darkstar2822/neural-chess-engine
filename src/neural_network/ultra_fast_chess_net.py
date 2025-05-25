import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import torch.jit
from config import Config

class MobileInvertedBottleneck(nn.Module):
    """Efficient MobileNet-style inverted bottleneck block for chess"""
    def __init__(self, in_channels: int, out_channels: int, expansion: int = 4, stride: int = 1):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        layers = []
        # Pointwise expansion
        if expansion != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise compression
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class SqueezeExcitation(nn.Module):
    """Lightweight attention mechanism"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)

class UltraFastChessNet(nn.Module):
    """Ultra-optimized chess neural network for maximum speed"""
    
    def __init__(self, 
                 input_planes: int = 16,
                 base_filters: int = 64,  # Reduced from 256
                 num_blocks: int = 8,     # Reduced from 19
                 use_se: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_planes = input_planes
        self.base_filters = base_filters
        
        # Lightweight input processing
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_planes, base_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU6(inplace=True)
        )
        
        # Efficient backbone with mobile blocks
        self.backbone = nn.ModuleList()
        current_channels = base_filters
        
        for i in range(num_blocks):
            # Keep consistent channel sizes
            if i < num_blocks // 2:
                out_channels = base_filters
            else:
                out_channels = base_filters + 32
            
            block = MobileInvertedBottleneck(
                in_channels=current_channels,
                out_channels=out_channels,
                expansion=3 if i < 4 else 2  # Lower expansion for later blocks
            )
            self.backbone.append(block)
            current_channels = out_channels
            
            # Add SE attention every 3rd block
            if use_se and (i + 1) % 3 == 0:
                self.backbone.append(SqueezeExcitation(out_channels))
        
        final_channels = current_channels
        
        # Ultra-compact policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(final_channels, 16, 1, bias=False),  # Massive reduction
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(dropout),
            nn.Flatten(),
            nn.Linear(16 * 64, 4096 + 64)  # 4096 normal moves + 64 underpromotions
        )
        
        # Ultra-compact value head
        self.value_head = nn.Sequential(
            nn.Conv2d(final_channels, 8, 1, bias=False),   # Even smaller
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        
        # Initialize weights for faster convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for faster training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input processing
        x = self.input_conv(x)
        
        # Backbone processing
        for layer in self.backbone:
            x = layer(x)
        
        # Dual heads
        policy = self.policy_head(x)
        policy = F.log_softmax(policy, dim=1)
        
        value = self.value_head(x)
        
        return policy, value
    
    @torch.jit.export
    def predict_fast(self, state_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """JIT-optimized inference method"""
        self.eval()
        with torch.no_grad():
            if len(state_tensor.shape) == 3:
                state_tensor = state_tensor.unsqueeze(0)
            
            policy, value = self.forward(state_tensor)
            policy = torch.exp(policy)
            
            return policy.squeeze(), value.squeeze()
    
    def save_model(self, filepath: str):
        """Save with optimization metadata"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_planes': self.input_planes,
                'base_filters': self.base_filters,
                'architecture': 'UltraFastChessNet',
                'optimized': True,
                'version': '2.0'
            },
            'model_size_mb': self.get_model_size_mb(),
            'parameter_count': self.count_parameters()
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, device: Optional[str] = None):
        """Load with automatic device optimization"""
        if device is None:
            device = Config.DEVICE
        
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        model = cls(
            input_planes=config['input_planes'],
            base_filters=config['base_filters']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Optimize for inference
        model.eval()
        if device != 'cpu':
            model = torch.jit.script(model)  # JIT compilation
        
        return model
    
    def count_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    
    def get_flops(self, input_shape: Tuple[int, int, int, int] = (1, 16, 8, 8)) -> int:
        """Estimate FLOPs for given input shape"""
        def conv_flops(in_channels, out_channels, kernel_size, input_size):
            return in_channels * out_channels * kernel_size * kernel_size * input_size[0] * input_size[1]
        
        def linear_flops(in_features, out_features):
            return in_features * out_features
        
        total_flops = 0
        # This is a simplified estimation - would need actual profiling for accuracy
        total_flops += conv_flops(16, 64, 9, (8, 8))  # Input conv
        total_flops += conv_flops(64, 96, 9 * 8, (8, 8))  # Approximate backbone
        total_flops += linear_flops(16 * 64, 4160)  # Policy head
        total_flops += linear_flops(8 * 1, 32) + linear_flops(32, 1)  # Value head
        
        return total_flops

class ModelFactory:
    """Factory for creating optimized models based on requirements"""
    
    @staticmethod
    def create_ultra_fast(strength: str = 'medium') -> UltraFastChessNet:
        """Create ultra-fast model with different strength levels"""
        configs = {
            'minimal': {'base_filters': 32, 'num_blocks': 4, 'dropout': 0.05},
            'fast': {'base_filters': 48, 'num_blocks': 6, 'dropout': 0.08},
            'medium': {'base_filters': 64, 'num_blocks': 8, 'dropout': 0.1},
            'strong': {'base_filters': 96, 'num_blocks': 12, 'dropout': 0.12}
        }
        
        config = configs.get(strength, configs['medium'])
        return UltraFastChessNet(**config)
    
    @staticmethod
    def create_quantized(model: UltraFastChessNet) -> torch.nn.Module:
        """Create quantized version for even faster inference"""
        model.eval()
        
        # Dynamic quantization for linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear}, 
            dtype=torch.qint8
        )
        
        return quantized_model
    
    @staticmethod
    def benchmark_model(model: nn.Module, input_shape: Tuple[int, int, int, int] = (1, 16, 8, 8), 
                       num_runs: int = 100) -> dict:
        """Benchmark model performance"""
        import time
        
        model.eval()
        device = next(model.parameters()).device
        
        # Warmup
        dummy_input = torch.randn(input_shape, device=device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        
        return {
            'avg_inference_time_ms': avg_time,
            'throughput_fps': 1000 / avg_time,
            'model_size_mb': model.get_model_size_mb() if hasattr(model, 'get_model_size_mb') else 0,
            'parameter_count': sum(p.numel() for p in model.parameters())
        }

# Performance optimizations for training
class FastTrainingMixin:
    """Mixin for faster training optimizations"""
    
    @staticmethod
    def setup_fast_training(model: nn.Module, device: str) -> nn.Module:
        """Setup model for fastest possible training"""
        model = model.to(device)
        
        # Enable optimizations
        if device == 'cuda':
            # Enable mixed precision
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
        elif device == 'mps':
            # MPS optimizations
            torch.backends.mps.enabled = True
            
        # JIT compile if possible
        try:
            model = torch.jit.script(model)
        except Exception:
            pass  # Fallback to eager mode
            
        return model
    
    @staticmethod
    def get_optimal_batch_size(model: nn.Module, device: str, input_shape: Tuple[int, int, int, int]) -> int:
        """Find optimal batch size for given model and device"""
        if device == 'cpu':
            return 32
        
        # Binary search for optimal batch size
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        optimal_batch = 1
        
        for batch_size in batch_sizes:
            try:
                dummy_input = torch.randn(batch_size, *input_shape[1:], device=device)
                with torch.no_grad():
                    _ = model(dummy_input)
                optimal_batch = batch_size
            except RuntimeError:  # OOM
                break
        
        return optimal_batch