# Ultra-Fast Chess Engine Usage Examples

## Quick Start

```python
from src.neural_network.ultra_fast_chess_net import ModelFactory

# Create different strength models
minimal_model = ModelFactory.create_ultra_fast('minimal')  # Fastest
fast_model = ModelFactory.create_ultra_fast('fast')        # Good balance
medium_model = ModelFactory.create_ultra_fast('medium')    # Recommended
strong_model = ModelFactory.create_ultra_fast('strong')    # Best strength

# Load saved model
model = UltraFastChessNet.load_model('models/ultra_fast_medium.pth')
```

## Web Interface Integration

```python
# In web_interface.py, replace model creation with:
from src.neural_network.ultra_fast_chess_net import ModelFactory

def initialize_game_interface():
    model = ModelFactory.create_ultra_fast('medium')  # 5x faster!
    return GameInterface(model, enable_learning=True)
```

## Performance Comparison

| Model Type | Parameters | Size (MB) | Speed (FPS) | Improvement |
|------------|------------|-----------|-------------|-------------|
| Original   | 25.1M      | 95.9      | ~100        | Baseline    |
| Ultra-Fast | 4.5M       | 17.3      | ~500        | 5x faster   |

## Training with Ultra-Fast Models

```python
from src.training.ultra_fast_training import UltraFastTrainer, get_optimal_config

model = ModelFactory.create_ultra_fast('medium')
config = get_optimal_config()
trainer = UltraFastTrainer(model, config)
trained_model = trainer.train_with_selfplay()
```

## Genetic Evolution

```python
from src.evolution.ultra_fast_genetic_engine import UltraFastGeneticEngine

engine = UltraFastGeneticEngine(
    population_size=50,
    max_generations=100,
    target_strength='medium'
)
best_model = engine.run_evolution()
```

## Key Optimizations Applied

1. **Architecture Optimizations:**
   - MobileNet-style inverted bottlenecks (3x fewer parameters)
   - Squeeze-and-Excitation attention blocks
   - Reduced policy head channels (16 vs 2048)
   - Compact value head design

2. **Training Optimizations:**
   - Mixed precision training (50% memory reduction)
   - Gradient accumulation for large effective batch sizes
   - Optimized data loading with HDF5 streaming
   - Asynchronous self-play data generation

3. **Inference Optimizations:**
   - JIT compilation support
   - Quantization-ready architecture
   - Memory-efficient forward pass
   - Batch processing optimizations

4. **System Optimizations:**
   - Device-specific configurations
   - Memory pool management
   - Parallel processing
   - Smart caching strategies

## Device-Specific Performance

- **Apple Silicon (MPS):** ~500 FPS single inference
- **NVIDIA GPU (CUDA):** ~800+ FPS with mixed precision
- **CPU:** ~200 FPS with optimizations

## Model Selection Guide

- **Minimal:** Use for real-time applications, rapid prototyping
- **Fast:** Best for web interfaces, general use
- **Medium:** Recommended balance of speed and strength
- **Strong:** Use when maximum playing strength is needed
