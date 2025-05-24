import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from src.utils.memory_manager import memory_manager

class EfficientResidualBlock(nn.Module):
    """Memory and computation optimized residual block"""
    def __init__(self, filters, use_se_block=True):
        super(EfficientResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        
        # Squeeze-and-Excitation block for better feature selection
        self.use_se = use_se_block
        if use_se_block:
            self.se_fc1 = nn.Linear(filters, filters // 16)
            self.se_fc2 = nn.Linear(filters // 16, filters)
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        
        # Squeeze-and-Excitation
        if self.use_se:
            se_weight = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
            se_weight = F.relu(self.se_fc1(se_weight), inplace=True)
            se_weight = torch.sigmoid(self.se_fc2(se_weight)).view(out.size(0), out.size(1), 1, 1)
            out = out * se_weight
        
        out += residual
        return F.relu(out, inplace=True)

class OptimizedChessNet(nn.Module):
    """Optimized chess neural network with better performance"""
    def __init__(self, input_planes=16, filters=256, residual_blocks=19, use_mixed_precision=True):
        super(OptimizedChessNet, self).__init__()
        
        self.use_mixed_precision = use_mixed_precision and Config.DEVICE == 'cuda'
        
        # Input convolution with depthwise separable conv for efficiency
        self.input_conv = nn.Conv2d(input_planes, filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(filters)
        
        # Efficient residual blocks with SE attention
        self.residual_blocks = nn.ModuleList([
            EfficientResidualBlock(filters, use_se_block=(i % 3 == 0))  # SE every 3rd block
            for i in range(residual_blocks)
        ])
        
        # Policy head with efficient architecture
        self.policy_conv = nn.Conv2d(filters, 32, kernel_size=1, bias=False)  # Reduced channels
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096 + 4 * 4096)
        
        # Value head with dropout for regularization
        self.value_conv = nn.Conv2d(filters, 8, kernel_size=1, bias=False)  # Reduced channels
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 256)
        self.value_dropout = nn.Dropout(0.2)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Initialize weights efficiently
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Efficient weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Memory-efficient forward pass
        cleanup = memory_manager.monitor_memory("forward_pass")
        
        try:
            # Input processing
            x = F.relu(self.input_bn(self.input_conv(x)), inplace=True)
            
            # Residual blocks
            for block in self.residual_blocks:
                x = block(x)
            
            # Policy head
            policy = F.relu(self.policy_bn(self.policy_conv(x)), inplace=True)
            policy = policy.view(policy.size(0), -1)
            policy = self.policy_fc(policy)
            policy = F.log_softmax(policy, dim=1)
            
            # Value head
            value = F.relu(self.value_bn(self.value_conv(x)), inplace=True)
            value = value.view(value.size(0), -1)
            value = F.relu(self.value_fc1(value), inplace=True)
            value = self.value_dropout(value)
            value = torch.tanh(self.value_fc2(value))
            
            return policy, value
            
        finally:
            cleanup()
    
    def predict_batch(self, state_tensors):
        """Optimized batch prediction"""
        self.eval()
        with torch.no_grad():
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    return self.forward(state_tensors)
            else:
                return self.forward(state_tensors)
    
    def predict(self, state_tensor):
        """Single prediction with memory optimization"""
        self.eval()
        cleanup = memory_manager.monitor_memory("prediction")
        
        try:
            with torch.no_grad():
                if len(state_tensor.shape) == 3:
                    state_tensor = state_tensor.unsqueeze(0)
                
                # Ensure tensor is on same device as model
                state_tensor = state_tensor.to(next(self.parameters()).device)
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        policy, value = self.forward(state_tensor)
                else:
                    policy, value = self.forward(state_tensor)
                
                policy = torch.exp(policy)
                
                return policy.squeeze().cpu().numpy(), value.squeeze().cpu().item()
        finally:
            cleanup()
    
    def save_model(self, filepath):
        """Save model with version information"""
        from src.utils.model_versioning import model_version_manager
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_planes': self.input_conv.in_channels,
                'filters': self.input_conv.out_channels,
                'residual_blocks': len(self.residual_blocks),
                'architecture_version': '2.0',
                'optimized': True
            },
            'version_info': {
                'pytorch_version': torch.__version__,
                'creation_date': hash(str(torch.random.get_rng_state()))  # Unique ID
            }
        }, filepath)
        
        # Register with version manager
        metadata = {
            'architecture_version': '2.0',
            'filters': self.input_conv.out_channels,
            'residual_blocks': len(self.residual_blocks),
            'optimized': True,
            'mixed_precision': self.use_mixed_precision
        }
        
        version = model_version_manager.register_model(filepath, metadata)
        print(f"Model saved as version: {version}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load model with compatibility checking"""
        from src.utils.model_versioning import model_version_manager
        
        checkpoint = torch.load(filepath, map_location=Config.DEVICE, weights_only=False)
        config = checkpoint['config']
        
        # Check compatibility
        arch_version = config.get('architecture_version', '1.0')
        if not model_version_manager.is_compatible(arch_version, '2.0'):
            print(f"Warning: Loading model with architecture version {arch_version}")
        
        model = cls(
            input_planes=config['input_planes'],
            filters=config['filters'],
            residual_blocks=config['residual_blocks']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(Config.DEVICE)
        
        return model
    
    def get_model_size(self) -> int:
        """Get model size in bytes"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return param_size + buffer_size
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage statistics"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated(),
                'cached': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated()
            }
        elif torch.backends.mps.is_available():
            return {
                'allocated': torch.mps.current_allocated_memory(),
                'driver_allocated': torch.mps.driver_allocated_memory()
            }
        return {'cpu_only': True}