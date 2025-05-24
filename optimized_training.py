#!/usr/bin/env python3
"""
Optimized Chess Engine Training Script

This script uses all the optimization improvements including:
- Memory management with automatic cleanup
- Error handling with graceful recovery
- Mixed precision training (when available)
- Optimized neural network architecture
- Model versioning and compatibility checking
- Parallel self-play game generation
"""

import torch
import numpy as np
import os
from datetime import datetime
from pathlib import Path

from config import Config
from src.neural_network.optimized_chess_net import OptimizedChessNet, OptimizedTrainer
from src.neural_network.trainer import ChessNetTrainer
from src.training.parallel_selfplay import FastTrainingManager
from src.utils.memory_manager import memory_manager
from src.utils.error_handler import safe_execute, handle_errors
from src.utils.model_versioning import model_version_manager

class OptimizedTrainingPipeline:
    def __init__(self, use_optimized_model: bool = True):
        self.use_optimized = use_optimized_model
        self.device = Config.DEVICE
        self.generation = 0
        
        print(f"🚀 Initializing Optimized Training Pipeline")
        print(f"   Device: {self.device}")
        print(f"   Optimized Architecture: {use_optimized_model}")
        print(f"   Mixed Precision: {self.device == 'cuda'}")
        
        self._setup_directories()
        self._initialize_model()
        
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            'data/models/optimized',
            'logs/optimized',
            'data/training/optimized'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @handle_errors
    def _initialize_model(self):
        """Initialize or load the model"""
        if self.use_optimized:
            # Try to load latest optimized model
            latest_model = self._find_latest_model('data/models/optimized')
            if latest_model:
                print(f"📁 Loading optimized model: {latest_model}")
                self.model = OptimizedChessNet.load_model(latest_model)
                self.trainer = OptimizedTrainer(self.model, self.device)
            else:
                print("🆕 Creating new optimized model")
                self.model = OptimizedChessNet(
                    input_channels=12,
                    hidden_size=256,
                    num_blocks=10
                ).to(self.device)
                self.trainer = OptimizedTrainer(self.model, self.device)
        else:
            # Fallback to standard model with optimizations
            from src.neural_network.chess_net import ChessNet
            latest_model = self._find_latest_model('data/models')
            if latest_model:
                print(f"📁 Loading standard model: {latest_model}")
                self.model = ChessNet.load_model(latest_model)
            else:
                print("🆕 Creating new standard model")
                self.model = ChessNet().to(self.device)
            
            self.trainer = ChessNetTrainer(self.model)
        
        self.training_manager = FastTrainingManager(self.model)
        
        print(f"✅ Model initialized with {self._count_parameters():,} parameters")
        print(f"   Memory usage: {self._format_memory_usage()}")
    
    def _find_latest_model(self, directory: str) -> str:
        """Find the latest model file in directory"""
        try:
            model_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
            if model_files:
                latest = max(model_files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
                return os.path.join(directory, latest)
        except:
            pass
        return None
    
    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _format_memory_usage(self) -> str:
        """Format memory usage info"""
        if hasattr(self.model, 'get_memory_usage'):
            usage = self.model.get_memory_usage()
            if 'allocated' in usage:
                return f"{usage['allocated'] / 1024**2:.1f}MB"
        return "N/A"
    
    @handle_errors
    def training_iteration(self, games_per_iteration: int = 50, epochs_per_iteration: int = 5):
        """Run one complete training iteration"""
        print(f"\n🎯 Training Generation {self.generation}")
        print(f"   Games per iteration: {games_per_iteration}")
        print(f"   Training epochs: {epochs_per_iteration}")
        
        cleanup = memory_manager.monitor_memory("training_iteration")
        
        try:
            # 1. Generate training data through self-play
            print("🎮 Generating self-play games...")
            training_data = self.training_manager.train_iteration(games_per_iteration)
            
            if not training_data[0].size:  # Check if we got any data
                print("⚠️  No training data generated, skipping iteration")
                return False
            
            print(f"   Generated {len(training_data[0])} training examples")
            
            # 2. Train the model
            print("🧠 Training neural network...")
            if self.use_optimized:
                # Use optimized trainer
                batch_data = self._prepare_optimized_batch(training_data)
                training_stats = self.trainer.train_batch(batch_data)
                print(f"   Training stats: {training_stats}")
            else:
                # Use standard trainer
                training_history = self.trainer.train(training_data, epochs_per_iteration)
                final_losses = training_history[-1] if training_history else {}
                print(f"   Final losses: {final_losses}")
            
            # 3. Save the model
            self._save_model()
            
            # 4. Memory cleanup
            memory_manager.force_cleanup()
            
            self.generation += 1
            print(f"✅ Completed generation {self.generation - 1}")
            
            return True
            
        except Exception as e:
            print(f"❌ Training iteration failed: {e}")
            return False
        finally:
            cleanup()
    
    def _prepare_optimized_batch(self, training_data):
        """Prepare training data for optimized trainer"""
        states, policies, values = training_data
        
        # Convert to proper format for optimized trainer
        batch_data = []
        batch_size = min(len(states), 32)  # Process in smaller batches
        
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i + batch_size]
            batch_policies = policies[i:i + batch_size]
            batch_values = values[i:i + batch_size]
            
            batch_data.append({
                'positions': [torch.FloatTensor(s) for s in batch_states],
                'policies': [torch.FloatTensor(p) for p in batch_policies],
                'values': batch_values.tolist() if hasattr(batch_values, 'tolist') else list(batch_values)
            })
        
        return batch_data
    
    @safe_execute
    def _save_model(self):
        """Save the current model with versioning"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.use_optimized:
            model_path = f"data/models/optimized/optimized_model_gen_{self.generation}_{timestamp}.pt"
        else:
            model_path = f"data/models/standard_model_gen_{self.generation}_{timestamp}.pt"
        
        self.model.save_model(model_path)
        print(f"💾 Model saved: {model_path}")
        
        # Keep only the 5 most recent models to save space
        self._cleanup_old_models()
    
    def _cleanup_old_models(self):
        """Remove old model files to save disk space"""
        try:
            model_dir = 'data/models/optimized' if self.use_optimized else 'data/models'
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
            
            if len(model_files) > 5:
                # Sort by creation time and remove oldest
                model_files.sort(key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
                for old_file in model_files[:-5]:
                    os.remove(os.path.join(model_dir, old_file))
                    print(f"🗑️  Removed old model: {old_file}")
        except Exception as e:
            print(f"⚠️  Could not cleanup old models: {e}")
    
    @handle_errors
    def run_training(self, max_generations: int = 100, games_per_generation: int = 50):
        """Run the complete training pipeline"""
        print(f"\n🚀 Starting Optimized Training Pipeline")
        print(f"   Max generations: {max_generations}")
        print(f"   Games per generation: {games_per_generation}")
        print(f"   Target examples: {max_generations * games_per_generation}")
        
        success_count = 0
        failure_count = 0
        
        for generation in range(max_generations):
            print(f"\n{'='*60}")
            print(f"GENERATION {generation + 1}/{max_generations}")
            print(f"{'='*60}")
            
            # Run training iteration
            success = self.training_iteration(games_per_generation)
            
            if success:
                success_count += 1
                print(f"🎉 Generation {generation + 1} completed successfully")
            else:
                failure_count += 1
                print(f"💥 Generation {generation + 1} failed")
                
                # Stop if too many consecutive failures
                if failure_count >= 3:
                    print("❌ Too many failures, stopping training")
                    break
            
            # Print progress summary
            if (generation + 1) % 5 == 0:
                print(f"\n📊 Progress Summary:")
                print(f"   Successful generations: {success_count}")
                print(f"   Failed generations: {failure_count}")
                print(f"   Success rate: {success_count/(success_count + failure_count)*100:.1f}%")
                print(f"   Memory usage: {self._format_memory_usage()}")
        
        print(f"\n🏁 Training completed!")
        print(f"   Total successful generations: {success_count}")
        print(f"   Total failed generations: {failure_count}")

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Chess Engine Training')
    parser.add_argument('--generations', type=int, default=20, help='Number of training generations')
    parser.add_argument('--games-per-gen', type=int, default=30, help='Games per generation')
    parser.add_argument('--use-optimized', action='store_true', default=True, help='Use optimized architecture')
    parser.add_argument('--fallback', action='store_true', help='Use standard architecture as fallback')
    
    args = parser.parse_args()
    
    # Initialize training pipeline
    use_optimized = args.use_optimized and not args.fallback
    pipeline = OptimizedTrainingPipeline(use_optimized_model=use_optimized)
    
    # Run training
    pipeline.run_training(
        max_generations=args.generations,
        games_per_generation=args.games_per_gen
    )

if __name__ == "__main__":
    main()