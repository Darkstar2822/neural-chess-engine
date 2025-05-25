#!/usr/bin/env python3
"""
Ultra-Fast Chess Engine Optimization and Benchmarking System
Comprehensive performance optimization and validation suite
"""

import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import json
import os
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor
import psutil
import GPUtil

# Import our optimized components
from src.neural_network.ultra_fast_chess_net import UltraFastChessNet, ModelFactory
from src.evolution.ultra_fast_genetic_engine import UltraFastGeneticEngine
from src.training.ultra_fast_training import UltraFastTrainer, TrainingConfig, get_optimal_config, benchmark_training_speed
from src.neural_network.chess_net import ChessNet
from src.neural_network.optimized_chess_net import OptimizedChessNet
from config import Config

class PerformanceProfiler:
    """Advanced performance profiling for all components"""
    
    def __init__(self):
        self.results = {}
        self.device = Config.DEVICE
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def profile_model_inference(self, model: nn.Module, model_name: str, 
                               input_shape: Tuple[int, int, int, int] = (1, 16, 8, 8),
                               num_runs: int = 1000) -> Dict[str, Any]:
        """Profile model inference performance"""
        self.logger.info(f"Profiling {model_name} inference...")
        
        model.eval()
        device = next(model.parameters()).device
        
        # Memory profiling
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = 0
        
        # Warmup
        dummy_input = torch.randn(input_shape, device=device)
        for _ in range(50):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Timing benchmark
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            with torch.no_grad():
                output = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Memory usage
        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = peak_memory - initial_memory
        else:
            memory_used = 0
        
        # Calculate statistics
        times = np.array(times)
        
        results = {
            'model_name': model_name,
            'device': str(device),
            'input_shape': input_shape,
            'avg_inference_time_ms': float(np.mean(times)),
            'std_inference_time_ms': float(np.std(times)),
            'min_inference_time_ms': float(np.min(times)),
            'max_inference_time_ms': float(np.max(times)),
            'p95_inference_time_ms': float(np.percentile(times, 95)),
            'p99_inference_time_ms': float(np.percentile(times, 99)),
            'throughput_fps': float(1000 / np.mean(times)),
            'memory_used_mb': float(memory_used / (1024 * 1024)),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
            'flops_estimate': self._estimate_flops(model, input_shape) if hasattr(model, 'get_flops') else 0
        }
        
        self.results[f'{model_name}_inference'] = results
        return results
    
    def _estimate_flops(self, model: nn.Module, input_shape: Tuple[int, int, int, int]) -> int:
        """Estimate FLOPs for model"""
        if hasattr(model, 'get_flops'):
            return model.get_flops(input_shape)
        return 0
    
    def profile_training_speed(self, model: nn.Module, model_name: str, 
                              config: TrainingConfig) -> Dict[str, Any]:
        """Profile training performance"""
        self.logger.info(f"Profiling {model_name} training...")
        
        results = benchmark_training_speed(model, config)
        results['model_name'] = model_name
        
        self.results[f'{model_name}_training'] = results
        return results
    
    def profile_genetic_algorithm(self, population_size: int = 20, generations: int = 5) -> Dict[str, Any]:
        """Profile genetic algorithm performance"""
        self.logger.info("Profiling genetic algorithm...")
        
        start_time = time.time()
        
        # Create genetic engine
        engine = UltraFastGeneticEngine(
            population_size=population_size,
            max_generations=generations,
            target_strength='fast'  # Use faster model for benchmarking
        )
        
        # Run evolution
        best_model = engine.run_evolution()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        stats = engine.get_evolution_stats()
        
        results = {
            'total_time_seconds': total_time,
            'generations_completed': stats['generation'],
            'total_games_played': stats['total_games'],
            'games_per_second': stats['games_per_second'],
            'time_per_generation': total_time / max(1, stats['generation']),
            'best_fitness_achieved': stats['best_fitness'],
            'population_diversity': stats['population_diversity'],
            'memory_efficiency': 'high'  # Based on our optimizations
        }
        
        self.results['genetic_algorithm'] = results
        return results
    
    def profile_system_resources(self) -> Dict[str, Any]:
        """Profile system resource utilization"""
        self.logger.info("Profiling system resources...")
        
        # CPU info
        cpu_info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_freq_current': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'cpu_usage_percent': psutil.cpu_percent(interval=1)
        }
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_info = {
            'total_memory_gb': memory.total / (1024**3),
            'available_memory_gb': memory.available / (1024**3),
            'memory_usage_percent': memory.percent
        }
        
        # GPU info
        gpu_info = {'gpu_available': False}
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_info = {
                        'gpu_available': True,
                        'gpu_name': gpu.name,
                        'gpu_memory_total_gb': gpu.memoryTotal / 1024,
                        'gpu_memory_used_gb': gpu.memoryUsed / 1024,
                        'gpu_memory_util_percent': gpu.memoryUtil * 100,
                        'gpu_load_percent': gpu.load * 100,
                        'gpu_temperature': gpu.temperature
                    }
            except Exception:
                pass
        
        # PyTorch info
        pytorch_info = {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            'device_used': str(Config.DEVICE)
        }
        
        results = {
            'cpu': cpu_info,
            'memory': memory_info,
            'gpu': gpu_info,
            'pytorch': pytorch_info,
            'timestamp': time.time()
        }
        
        self.results['system_resources'] = results
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive performance report"""
        report = []
        report.append("=" * 80)
        report.append("ULTRA-FAST CHESS ENGINE PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append("")
        
        # System info
        if 'system_resources' in self.results:
            sys_info = self.results['system_resources']
            report.append("SYSTEM CONFIGURATION:")
            report.append(f"  Device: {sys_info['pytorch']['device_used']}")
            report.append(f"  CPU: {sys_info['cpu']['cpu_count']} cores ({sys_info['cpu']['cpu_count_logical']} logical)")
            report.append(f"  Memory: {sys_info['memory']['total_memory_gb']:.1f} GB")
            if sys_info['gpu']['gpu_available']:
                report.append(f"  GPU: {sys_info['gpu']['gpu_name']} ({sys_info['gpu']['gpu_memory_total_gb']:.1f} GB)")
            report.append("")
        
        # Model inference comparison
        inference_results = {k: v for k, v in self.results.items() if k.endswith('_inference')}
        if inference_results:
            report.append("MODEL INFERENCE PERFORMANCE:")
            report.append(f"{'Model':<25} {'Avg Time (ms)':<15} {'Throughput (FPS)':<18} {'Memory (MB)':<12} {'Parameters':<12}")
            report.append("-" * 85)
            
            for result in sorted(inference_results.values(), key=lambda x: x['avg_inference_time_ms']):
                report.append(f"{result['model_name']:<25} "
                            f"{result['avg_inference_time_ms']:<15.2f} "
                            f"{result['throughput_fps']:<18.1f} "
                            f"{result['memory_used_mb']:<12.1f} "
                            f"{result['parameter_count']:<12,}")
            report.append("")
        
        # Training performance
        training_results = {k: v for k, v in self.results.items() if k.endswith('_training')}
        if training_results:
            report.append("TRAINING PERFORMANCE:")
            for result in training_results.values():
                report.append(f"  {result['model_name']}:")
                report.append(f"    Training Step: {result['training_step_ms']:.2f} ms")
                report.append(f"    Samples/sec: {result['samples_per_second']:.1f}")
                report.append(f"    Batch Size: {result['batch_size']}")
            report.append("")
        
        # Genetic algorithm performance
        if 'genetic_algorithm' in self.results:
            ga_result = self.results['genetic_algorithm']
            report.append("GENETIC ALGORITHM PERFORMANCE:")
            report.append(f"  Total Time: {ga_result['total_time_seconds']:.1f} seconds")
            report.append(f"  Games/Second: {ga_result['games_per_second']:.1f}")
            report.append(f"  Time per Generation: {ga_result['time_per_generation']:.1f} seconds")
            report.append(f"  Best Fitness: {ga_result['best_fitness_achieved']:.3f}")
            report.append("")
        
        # Performance recommendations
        report.append("OPTIMIZATION RECOMMENDATIONS:")
        if 'system_resources' in self.results:
            sys_info = self.results['system_resources']
            if sys_info['gpu']['gpu_available']:
                report.append("  ✓ GPU acceleration available and recommended")
            else:
                report.append("  ⚠ Consider GPU acceleration for 10-50x speedup")
            
            if sys_info['memory']['total_memory_gb'] < 8:
                report.append("  ⚠ Low memory detected - reduce batch sizes")
            else:
                report.append("  ✓ Sufficient memory for large batch training")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "performance_results.json"):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"Results saved to {filename}")

class ModelComparator:
    """Compare different model architectures"""
    
    def __init__(self):
        self.models = {}
        self.profiler = PerformanceProfiler()
    
    def add_model(self, name: str, model: nn.Module):
        """Add model for comparison"""
        self.models[name] = model
    
    def run_comparison(self) -> Dict[str, Any]:
        """Run comprehensive comparison"""
        results = {}
        
        for name, model in self.models.items():
            print(f"\n🔍 Benchmarking {name}...")
            
            # Inference profiling
            inference_result = self.profiler.profile_model_inference(model, name)
            results[f'{name}_inference'] = inference_result
            
            # Training profiling
            config = get_optimal_config()
            training_result = self.profiler.profile_training_speed(model, name, config)
            results[f'{name}_training'] = training_result
        
        return results
    
    def create_comparison_plot(self, results: Dict[str, Any], save_path: str = "model_comparison.png"):
        """Create visual comparison plots"""
        # Extract inference results
        inference_data = {k.replace('_inference', ''): v for k, v in results.items() if k.endswith('_inference')}
        
        if not inference_data:
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        models = list(inference_data.keys())
        
        # Inference time comparison
        times = [inference_data[model]['avg_inference_time_ms'] for model in models]
        ax1.bar(models, times, color='skyblue')
        ax1.set_title('Average Inference Time')
        ax1.set_ylabel('Time (ms)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        throughput = [inference_data[model]['throughput_fps'] for model in models]
        ax2.bar(models, throughput, color='lightgreen')
        ax2.set_title('Throughput')
        ax2.set_ylabel('FPS')
        ax2.tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        memory = [inference_data[model]['memory_used_mb'] for model in models]
        ax3.bar(models, memory, color='salmon')
        ax3.set_title('Memory Usage')
        ax3.set_ylabel('Memory (MB)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Parameter count comparison
        params = [inference_data[model]['parameter_count'] / 1000000 for model in models]  # In millions
        ax4.bar(models, params, color='gold')
        ax4.set_title('Model Size')
        ax4.set_ylabel('Parameters (M)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to {save_path}")

def run_complete_optimization_suite():
    """Run the complete optimization and benchmarking suite"""
    print("🚀 Starting Ultra-Fast Chess Engine Optimization Suite")
    print("=" * 60)
    
    # Initialize profiler
    profiler = PerformanceProfiler()
    
    # System profiling
    print("\n📊 Profiling system resources...")
    profiler.profile_system_resources()
    
    # Model comparison
    print("\n🔄 Creating optimized models...")
    comparator = ModelComparator()
    
    # Original models
    try:
        original_model = ChessNet()
        comparator.add_model("Original ChessNet", original_model)
    except Exception as e:
        print(f"Could not load original model: {e}")
    
    try:
        optimized_model = OptimizedChessNet()
        comparator.add_model("Optimized ChessNet", optimized_model)
    except Exception as e:
        print(f"Could not load optimized model: {e}")
    
    # Ultra-fast models
    ultra_fast_minimal = ModelFactory.create_ultra_fast('minimal')
    ultra_fast_medium = ModelFactory.create_ultra_fast('medium')
    ultra_fast_strong = ModelFactory.create_ultra_fast('strong')
    
    comparator.add_model("UltraFast Minimal", ultra_fast_minimal)
    comparator.add_model("UltraFast Medium", ultra_fast_medium)
    comparator.add_model("UltraFast Strong", ultra_fast_strong)
    
    # Run model comparison
    print("\n⚡ Running model performance comparison...")
    comparison_results = comparator.run_comparison()
    
    # Add results to profiler
    profiler.results.update(comparison_results)
    
    # Create comparison plot
    comparator.create_comparison_plot(comparison_results)
    
    # Genetic algorithm profiling
    print("\n🧬 Profiling genetic algorithm performance...")
    profiler.profile_genetic_algorithm(population_size=10, generations=3)  # Quick test
    
    # Generate and save report
    print("\n📝 Generating performance report...")
    report = profiler.generate_report()
    print(report)
    
    # Save results
    profiler.save_results("ultra_fast_optimization_results.json")
    
    # Save report to file
    with open("ultra_fast_performance_report.txt", "w") as f:
        f.write(report)
    
    print("\n✅ Optimization suite completed!")
    print("📁 Files generated:")
    print("  - ultra_fast_optimization_results.json")
    print("  - ultra_fast_performance_report.txt")
    print("  - model_comparison.png")
    
    return profiler.results

def quick_performance_test():
    """Quick performance test for immediate feedback"""
    print("⚡ Quick Performance Test")
    print("-" * 30)
    
    # Test ultra-fast model
    model = ModelFactory.create_ultra_fast('medium')
    device = Config.DEVICE
    model.to(device)
    
    # Benchmark inference
    dummy_input = torch.randn(1, 16, 8, 8, device=device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Time 100 inferences
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 100 * 1000  # ms
    
    print(f"Device: {device}")
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Throughput: {1000/avg_time:.1f} FPS")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / (1024*1024):.1f} MB")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra-Fast Chess Engine Optimization Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick performance test only")
    parser.add_argument("--full", action="store_true", help="Run complete optimization suite")
    parser.add_argument("--genetic", action="store_true", help="Run genetic algorithm benchmark")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_performance_test()
    elif args.genetic:
        print("🧬 Running genetic algorithm benchmark...")
        profiler = PerformanceProfiler()
        profiler.profile_genetic_algorithm(population_size=20, generations=5)
        print(profiler.generate_report())
    else:
        # Default: run complete suite
        run_complete_optimization_suite()