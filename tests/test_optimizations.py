#!/usr/bin/env python3
"""
Comprehensive tests for all optimization components
"""

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

import torch
import numpy as np
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from src.utils.memory_manager import MemoryManager
from src.utils.error_handler import safe_execute, handle_errors
from src.utils.model_versioning import ModelVersionManager
from src.neural_network.optimized_chess_net import OptimizedChessNet

class TestMemoryManager:
    """Test memory management functionality"""
    
    def setup_method(self):
        self.memory_manager = MemoryManager()
    
    def test_memory_monitoring(self):
        """Test memory monitoring context manager"""
        cleanup = self.memory_manager.monitor_memory("test_operation")
        assert callable(cleanup)
        
        # Should not raise exception
        cleanup()
    
    def test_tensor_optimization(self):
        """Test tensor optimization"""
        tensor = torch.randn(100, 100)
        optimized = self.memory_manager.optimize_tensor(tensor)
        
        # Should return a tensor
        assert isinstance(optimized, torch.Tensor)
        assert optimized.shape == tensor.shape
    
    def test_force_cleanup(self):
        """Test force cleanup functionality"""
        # Should not raise exception
        self.memory_manager.force_cleanup()
    
    def test_get_memory_stats(self):
        """Test memory statistics"""
        stats = self.memory_manager.get_memory_stats()
        assert isinstance(stats, dict)

class TestErrorHandler:
    """Test error handling functionality"""
    
    def test_safe_execute_decorator(self):
        """Test safe_execute decorator with successful function"""
        @safe_execute
        def successful_function(x, y):
            return x + y
        
        result = successful_function(2, 3)
        assert result == 5
    
    def test_safe_execute_with_exception(self):
        """Test safe_execute decorator with failing function"""
        @safe_execute(return_value=None)
        def failing_function():
            raise ValueError("Test error")
        
        result = failing_function()
        assert result is None
    
    def test_handle_errors_decorator(self):
        """Test handle_errors decorator"""
        @handle_errors
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
    
    def test_handle_errors_with_exception(self):
        """Test handle_errors decorator with exception"""
        @handle_errors
        def failing_function():
            raise RuntimeError("Test error")
        
        # Should not raise exception due to error handling
        result = failing_function()
        assert result is None

class TestModelVersioning:
    """Test model versioning functionality"""
    
    def setup_method(self):
        self.version_manager = ModelVersionManager()
    
    def test_register_model(self):
        """Test model registration"""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            filepath = tmp.name
            torch.save({'test': 'data'}, filepath)
        
        try:
            metadata = {'version': '1.0', 'test': True}
            version = self.version_manager.register_model(filepath, metadata)
            assert isinstance(version, str)
        finally:
            os.unlink(filepath)
    
    def test_compatibility_check(self):
        """Test version compatibility checking"""
        # Test compatible versions
        assert self.version_manager.is_compatible('1.0', '1.1')
        assert self.version_manager.is_compatible('2.0', '2.0')
        
        # Test incompatible versions (major version difference)
        assert not self.version_manager.is_compatible('1.0', '2.0')
    
    def test_get_model_info(self):
        """Test getting model information"""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            filepath = tmp.name
            torch.save({'test': 'data'}, filepath)
        
        try:
            metadata = {'version': '1.0'}
            version = self.version_manager.register_model(filepath, metadata)
            
            info = self.version_manager.get_model_info(version)
            assert info is not None
            assert 'version' in info['metadata']
        finally:
            os.unlink(filepath)

class TestOptimizedChessNet:
    """Test optimized neural network"""
    
    def setup_method(self):
        self.model = OptimizedChessNet(
            input_planes=12,
            filters=64,  # Smaller for testing
            residual_blocks=3
        )
    
    def test_model_creation(self):
        """Test model instantiation"""
        assert isinstance(self.model, OptimizedChessNet)
        assert hasattr(self.model, 'input_conv')
        assert hasattr(self.model, 'residual_blocks')
        assert hasattr(self.model, 'policy_fc')
        assert hasattr(self.model, 'value_fc2')
    
    def test_forward_pass(self):
        """Test forward pass"""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 12, 8, 8)
        
        policy, value = self.model(input_tensor)
        
        assert policy.shape[0] == batch_size
        assert value.shape == (batch_size, 1)
        assert torch.all(torch.abs(value) <= 1)  # tanh output should be in [-1, 1]
    
    def test_prediction(self):
        """Test single prediction"""
        input_tensor = torch.randn(12, 8, 8)
        
        policy_probs, value = self.model.predict(input_tensor)
        
        assert isinstance(policy_probs, torch.Tensor)
        assert isinstance(value, float)
        assert abs(value) <= 1
        assert torch.all(policy_probs >= 0)  # Probabilities should be non-negative
    
    def test_model_size(self):
        """Test model size calculation"""
        size = self.model.get_model_size()
        assert isinstance(size, int)
        assert size > 0
    
    def test_memory_usage(self):
        """Test memory usage reporting"""
        usage = self.model.get_memory_usage()
        assert isinstance(usage, dict)

class TestOptimizedTrainer:
    """Test optimized trainer"""
    
    def setup_method(self):
        self.model = OptimizedChessNet(
            input_planes=12,
            filters=32,  # Very small for testing
            residual_blocks=2
        )
        
        # Use standard trainer for now
        from src.neural_network.trainer import ChessNetTrainer
        self.trainer = ChessNetTrainer(self.model)
    
    def test_trainer_creation(self):
        """Test trainer instantiation"""
        assert hasattr(self.trainer, 'model')
        assert hasattr(self.trainer, 'optimizer')
    
    def test_batch_training(self):
        """Test batch training"""
        # Create dummy training data
        states = torch.randn(2, 12, 8, 8)
        policies = torch.randn(2, 4096 + 4 * 4096)  # Match expected policy size
        values = torch.randn(2)
        
        # Should not raise exception
        stats = self.trainer.train_batch(states, policies, values)
        
        assert isinstance(stats, dict)
        assert 'policy_loss' in stats
        assert 'value_loss' in stats

class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_optimized_training_pipeline_import(self):
        """Test that optimized training pipeline can be imported"""
        try:
            from optimized_training import OptimizedTrainingPipeline
            # Should not raise ImportError
            assert OptimizedTrainingPipeline is not None
        except ImportError as e:
            if HAS_PYTEST:
                pytest.fail(f"Could not import OptimizedTrainingPipeline: {e}")
            else:
                raise AssertionError(f"Could not import OptimizedTrainingPipeline: {e}")
    
    def test_memory_manager_integration(self):
        """Test memory manager integration with neural network"""
        from src.utils.memory_manager import memory_manager
        
        # Test that memory manager is available globally
        assert memory_manager is not None
        
        # Test monitoring
        cleanup = memory_manager.monitor_memory("test")
        assert callable(cleanup)
        cleanup()
    
    def test_error_handling_integration(self):
        """Test error handling integration"""
        from src.neural_network.chess_net import ChessNet
        
        # Create model and test that decorated methods exist
        model = ChessNet()
        
        # These methods should be decorated with error handlers
        assert hasattr(model, 'forward')
        assert hasattr(model, 'predict')

def run_manual_tests():
    """Run manual tests that require user observation"""
    print("🧪 Running manual optimization tests...")
    
    print("\n1. Testing memory manager...")
    from src.utils.memory_manager import memory_manager
    
    # Test memory monitoring
    cleanup = memory_manager.monitor_memory("manual_test")
    print("   ✅ Memory monitoring started")
    
    # Create some tensors
    tensors = [torch.randn(100, 100) for _ in range(10)]
    print("   ✅ Created test tensors")
    
    # Optimize tensors
    optimized = [memory_manager.optimize_tensor(t) for t in tensors]
    print("   ✅ Optimized tensors")
    
    # Cleanup
    cleanup()
    memory_manager.force_cleanup()
    print("   ✅ Memory cleanup completed")
    
    print("\n2. Testing error handling...")
    
    @safe_execute(default_return="fallback")
    def test_function(should_fail=False):
        if should_fail:
            raise ValueError("Test error")
        return "success"
    
    # Test successful execution
    result1 = test_function(False)
    print(f"   ✅ Successful execution: {result1}")
    
    # Test error handling
    result2 = test_function(True)
    print(f"   ✅ Error handled gracefully: {result2}")
    
    print("\n3. Testing optimized neural network...")
    
    # Create optimized model
    model = OptimizedChessNet(input_planes=12, filters=64, residual_blocks=3)
    model.to('cpu')  # Force CPU for testing to avoid device issues
    print(f"   ✅ Created optimized model ({model.get_model_size():,} bytes)")
    
    # Test forward pass
    input_tensor = torch.randn(1, 12, 8, 8)
    policy, value = model(input_tensor)
    print(f"   ✅ Forward pass: policy {policy.shape}, value {value.shape}")
    
    # Test prediction
    policy_probs, value_scalar = model.predict(input_tensor.squeeze(0))
    print(f"   ✅ Prediction: value={value_scalar:.3f}")
    
    print("\n4. Testing standard trainer with optimized model...")
    
    from src.neural_network.trainer import ChessNetTrainer
    trainer = ChessNetTrainer(model)
    print("   ✅ Created trainer")
    
    # Test batch training with dummy data
    states = torch.randn(2, 12, 8, 8)
    policies = torch.randn(2, 4096 + 4 * 4096)
    values = torch.randn(2)
    
    stats = trainer.train_batch(states, policies, values)
    print(f"   ✅ Training batch: {stats}")
    
    print("\n🎉 All manual tests completed successfully!")

if __name__ == "__main__":
    # Run manual tests
    run_manual_tests()
    
    # Run pytest if available
    if HAS_PYTEST:
        try:
            pytest.main([__file__, "-v"])
        except Exception as e:
            print(f"\n⚠️  pytest failed: {e}")
    else:
        print("\n⚠️  pytest not available, running basic tests only")
        
        # Run basic tests manually
        print("\n🧪 Running basic tests...")
        
        # Test memory manager
        test_mem = TestMemoryManager()
        test_mem.setup_method()
        test_mem.test_memory_monitoring()
        test_mem.test_tensor_optimization()
        print("   ✅ Memory manager tests passed")
        
        # Test error handler
        test_err = TestErrorHandler()
        test_err.test_safe_execute_decorator()
        test_err.test_handle_errors_decorator()
        print("   ✅ Error handler tests passed")
        
        # Test optimized model
        test_model = TestOptimizedChessNet()
        test_model.setup_method()
        test_model.test_model_creation()
        test_model.test_forward_pass()
        test_model.test_prediction()
        print("   ✅ Optimized model tests passed")
        
        print("\n🎉 All basic tests completed successfully!")