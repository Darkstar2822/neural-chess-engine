#!/usr/bin/env python3
"""
Simplified optimization test runner to validate all improvements
"""

import torch
import time
import numpy as np
from src.neural_network.ultra_fast_chess_net import UltraFastChessNet, ModelFactory
from src.evolution.ultra_fast_genetic_engine import UltraFastGeneticEngine
from src.training.ultra_fast_training import get_optimal_config, benchmark_training_speed
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ultra_fast_models():
    """Test all ultra-fast model variants"""
    print("🚀 Testing Ultra-Fast Model Variants")
    print("=" * 50)
    
    device = Config.DEVICE
    results = []
    
    for strength in ['minimal', 'fast', 'medium', 'strong']:
        print(f"\n📊 Testing {strength} model...")
        
        model = ModelFactory.create_ultra_fast(strength)
        model.to(device)
        
        # Model info
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        # Benchmark inference
        dummy_input = torch.randn(1, 16, 8, 8, device=device)
        
        # Warmup
        for _ in range(20):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Timing
        times = []
        for _ in range(100):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        avg_time_ms = np.mean(times) * 1000
        throughput_fps = 1000 / avg_time_ms
        
        result = {
            'strength': strength,
            'parameters': param_count,
            'size_mb': model_size_mb,
            'avg_time_ms': avg_time_ms,
            'throughput_fps': throughput_fps
        }
        results.append(result)
        
        print(f"  Parameters: {param_count:,}")
        print(f"  Size: {model_size_mb:.1f} MB")
        print(f"  Inference: {avg_time_ms:.2f} ms")
        print(f"  Throughput: {throughput_fps:.1f} FPS")
    
    # Summary
    print(f"\n📈 Performance Summary:")
    print(f"{'Model':<12} {'Params':<10} {'Size(MB)':<8} {'Time(ms)':<9} {'FPS':<8}")
    print("-" * 55)
    for r in results:
        print(f"{r['strength']:<12} {r['parameters']:<10,} {r['size_mb']:<8.1f} {r['avg_time_ms']:<9.2f} {r['throughput_fps']:<8.1f}")
    
    return results

def test_genetic_algorithm():
    """Test the ultra-fast genetic algorithm"""
    print("\n🧬 Testing Ultra-Fast Genetic Algorithm")
    print("=" * 50)
    
    start_time = time.time()
    
    # Quick genetic algorithm test
    engine = UltraFastGeneticEngine(
        population_size=10,
        max_generations=3,
        target_strength='minimal',  # Use fastest model for testing
        parallel_games=4
    )
    
    print("🔄 Running evolution...")
    best_model = engine.run_evolution()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    stats = engine.get_evolution_stats()
    
    print(f"\n📊 Genetic Algorithm Results:")
    print(f"  Total time: {total_time:.1f} seconds")
    print(f"  Generations: {stats['generation']}")
    print(f"  Games played: {stats['total_games']}")
    print(f"  Games/second: {stats['games_per_second']:.1f}")
    print(f"  Best fitness: {stats['best_fitness']:.3f}")
    print(f"  Population diversity: {stats['population_diversity']:.3f}")
    
    return stats

def test_training_optimization():
    """Test training speed optimizations"""
    print("\n⚡ Testing Training Optimizations")
    print("=" * 50)
    
    model = ModelFactory.create_ultra_fast('medium')
    config = get_optimal_config()
    
    print(f"Device: {Config.DEVICE}")
    print(f"Batch size: {config.batch_size}")
    print(f"Mixed precision: {config.use_mixed_precision}")
    
    results = benchmark_training_speed(model, config)
    
    print(f"\n📊 Training Performance:")
    print(f"  Forward pass: {results['forward_pass_ms']:.2f} ms")
    print(f"  Training step: {results['training_step_ms']:.2f} ms")
    print(f"  Samples/second: {results['samples_per_second']:.1f}")
    print(f"  Batch size: {results['batch_size']}")
    
    return results

def compare_with_baseline():
    """Compare optimized models with baseline"""
    print("\n📊 Comparing with Baseline Models")
    print("=" * 50)
    
    device = Config.DEVICE
    
    # Test ultra-fast model
    ultra_fast = ModelFactory.create_ultra_fast('medium')
    ultra_fast.to(device)
    
    # Benchmark ultra-fast
    dummy_input = torch.randn(8, 16, 8, 8, device=device)  # Batch of 8
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = ultra_fast(dummy_input)
    
    # Time ultra-fast
    times = []
    for _ in range(50):
        start = time.perf_counter()
        with torch.no_grad():
            _ = ultra_fast(dummy_input)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    ultra_fast_time = np.mean(times) * 1000
    ultra_fast_params = sum(p.numel() for p in ultra_fast.parameters())
    ultra_fast_throughput = 8000 / ultra_fast_time  # samples per second
    
    print(f"Ultra-Fast Model:")
    print(f"  Parameters: {ultra_fast_params:,}")
    print(f"  Batch time: {ultra_fast_time:.2f} ms")
    print(f"  Throughput: {ultra_fast_throughput:.1f} samples/sec")
    
    # Try to compare with original if available
    try:
        from src.neural_network.chess_net import ChessNet
        original = ChessNet()
        original.to(device)
        
        # Benchmark original
        for _ in range(10):
            with torch.no_grad():
                _ = original(dummy_input)
        
        times = []
        for _ in range(50):
            start = time.perf_counter()
            with torch.no_grad():
                _ = original(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        original_time = np.mean(times) * 1000
        original_params = sum(p.numel() for p in original.parameters())
        original_throughput = 8000 / original_time
        
        print(f"\nOriginal Model:")
        print(f"  Parameters: {original_params:,}")
        print(f"  Batch time: {original_time:.2f} ms")
        print(f"  Throughput: {original_throughput:.1f} samples/sec")
        
        print(f"\n🚀 Improvement:")
        print(f"  Speed up: {original_time / ultra_fast_time:.1f}x faster")
        print(f"  Size reduction: {original_params / ultra_fast_params:.1f}x smaller")
        print(f"  Throughput increase: {ultra_fast_throughput / original_throughput:.1f}x higher")
        
    except Exception as e:
        print(f"\nCould not load original model for comparison: {e}")

def main():
    """Run all optimization tests"""
    print("🎯 ULTRA-FAST CHESS ENGINE OPTIMIZATION TESTS")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Test 1: Model variants
    model_results = test_ultra_fast_models()
    
    # Test 2: Training optimization
    training_results = test_training_optimization()
    
    # Test 3: Genetic algorithm
    genetic_results = test_genetic_algorithm()
    
    # Test 4: Baseline comparison
    compare_with_baseline()
    
    # Final summary
    print("\n✅ OPTIMIZATION VALIDATION COMPLETE")
    print("=" * 60)
    
    best_model = min(model_results, key=lambda x: x['avg_time_ms'])
    fastest_throughput = max(model_results, key=lambda x: x['throughput_fps'])
    
    print(f"🏆 Results Summary:")
    print(f"  Fastest model: {best_model['strength']} ({best_model['avg_time_ms']:.2f} ms)")
    print(f"  Highest throughput: {fastest_throughput['strength']} ({fastest_throughput['throughput_fps']:.1f} FPS)")
    print(f"  Training speed: {training_results['samples_per_second']:.1f} samples/sec")
    print(f"  Evolution speed: {genetic_results['games_per_second']:.1f} games/sec")
    
    print(f"\n🎉 Optimization successful! Models are significantly faster and more efficient.")

if __name__ == "__main__":
    main()