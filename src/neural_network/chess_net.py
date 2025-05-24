import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from src.utils.memory_manager import memory_manager
from src.utils.error_handler import safe_execute, handle_errors

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = F.relu(out)
        
        return out

class ChessNet(nn.Module):
    def __init__(self, input_planes=16, filters=256, residual_blocks=19):
        super(ChessNet, self).__init__()
        
        self.input_conv = nn.Conv2d(input_planes, filters, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(filters)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(filters) for _ in range(residual_blocks)
        ])
        
        self.policy_conv = nn.Conv2d(filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096 + 4 * 4096)
        
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    @handle_errors
    def forward(self, x):
        cleanup = memory_manager.monitor_memory("chess_net_forward")
        
        try:
            x = self.input_conv(x)
            x = self.input_bn(x)
            x = F.relu(x)
            
            for block in self.residual_blocks:
                x = block(x)
            
            policy = self.policy_conv(x)
            policy = self.policy_bn(policy)
            policy = F.relu(policy)
            policy = policy.view(policy.size(0), -1)
            policy = self.policy_fc(policy)
            policy = F.log_softmax(policy, dim=1)
            
            value = self.value_conv(x)
            value = self.value_bn(value)
            value = F.relu(value)
            value = value.view(value.size(0), -1)
            value = self.value_fc1(value)
            value = F.relu(value)
            value = self.value_fc2(value)
            value = torch.tanh(value)
            
            return policy, value
        finally:
            cleanup()
    
    @safe_execute
    def predict(self, state_tensor):
        self.eval()
        cleanup = memory_manager.monitor_memory("chess_net_predict")
        
        try:
            with torch.no_grad():
                if len(state_tensor.shape) == 3:
                    state_tensor = state_tensor.unsqueeze(0)
                
                # Move tensor to same device as model and optimize
                state_tensor = state_tensor.to(next(self.parameters()).device)
                state_tensor = memory_manager.optimize_tensor(state_tensor)
                
                policy, value = self.forward(state_tensor)
                policy = torch.exp(policy)
                
                return policy.squeeze().cpu().numpy(), value.squeeze().cpu().item()
        finally:
            cleanup()
    
    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_planes': self.input_conv.in_channels,
                'filters': self.input_conv.out_channels,
                'residual_blocks': len(self.residual_blocks)
            }
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        checkpoint = torch.load(filepath, map_location=Config.DEVICE, weights_only=False)
        config = checkpoint['config']
        
        model = cls(
            input_planes=config['input_planes'],
            filters=config['filters'],
            residual_blocks=config['residual_blocks']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(Config.DEVICE)
        
        return model