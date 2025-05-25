import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.amp import autocast, GradScaler
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Iterator
import time
import logging
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import pickle
import h5py
from queue import Queue
import threading

from src.neural_network.ultra_fast_chess_net import UltraFastChessNet, ModelFactory, FastTrainingMixin
from src.engine.chess_game import ChessGame
from src.engine.neural_player import NeuralPlayer
from config import Config

@dataclass
class TrainingConfig:
    """Configuration for ultra-fast training"""
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    max_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_accumulation_steps: int = 4
    use_mixed_precision: bool = True
    use_gradient_clipping: bool = True
    max_gradient_norm: float = 1.0
    
    # Data generation
    selfplay_games_per_iteration: int = 1000
    max_game_length: int = 200
    temperature: float = 1.0
    exploration_noise: float = 0.25
    
    # Optimization
    optimizer: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau'
    warmup_epochs: int = 5
    
    # Performance
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Validation
    validation_split: float = 0.1
    validation_frequency: int = 5  # Validate every N epochs

class FastChessDataset(IterableDataset):
    """Streaming dataset for chess positions to reduce memory usage"""
    
    def __init__(self, data_source: str, chunk_size: int = 10000, cache_size: int = 50000):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.cache_size = cache_size
        self.cache = []
        self.cache_index = 0
        
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate over data with caching for performance"""
        if os.path.exists(self.data_source):
            with h5py.File(self.data_source, 'r') as f:
                positions = f['positions']
                policies = f['policies']
                values = f['values']
                
                total_samples = len(positions)
                indices = np.random.permutation(total_samples)
                
                for i in range(0, total_samples, self.chunk_size):
                    chunk_indices = indices[i:i + self.chunk_size]
                    
                    pos_chunk = torch.from_numpy(positions[chunk_indices]).float()
                    pol_chunk = torch.from_numpy(policies[chunk_indices]).float()
                    val_chunk = torch.from_numpy(values[chunk_indices]).float()
                    
                    for j in range(len(pos_chunk)):
                        yield pos_chunk[j], pol_chunk[j], val_chunk[j]

class SelfPlayDataGenerator:
    """Ultra-fast self-play data generation with streaming"""
    
    def __init__(self, model: UltraFastChessNet, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = Config.DEVICE
        
        # Data storage
        self.data_buffer = []
        self.max_buffer_size = 100000
        
        # Performance tracking
        self.games_generated = 0
        self.positions_generated = 0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_game_fast(self, model1: UltraFastChessNet, model2: UltraFastChessNet = None) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Generate training data from a single game"""
        if model2 is None:
            model2 = model1
        
        game = ChessGame()
        player1 = NeuralPlayer(model1, "Player1")
        player2 = NeuralPlayer(model2, "Player2")
        
        game_data = []
        move_count = 0
        
        while not game.is_game_over() and move_count < self.config.max_game_length:
            # Get current position state
            state_planes = game.get_state_planes().numpy()
            
            # Get current player
            current_player = player1 if game.current_player() else player2
            
            # Get policy and value predictions
            state_tensor = torch.from_numpy(state_planes).float().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                policy_logits, value = current_player.model(state_tensor)
                policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                value_pred = value.cpu().item()
            
            # Apply exploration noise
            if np.random.random() < self.config.exploration_noise:
                legal_moves = game.get_legal_moves()
                if legal_moves:
                    # Add Dirichlet noise to policy
                    noise = np.random.dirichlet([0.3] * len(policy_probs))
                    policy_probs = 0.75 * policy_probs + 0.25 * noise
            
            # Make move
            move = current_player.get_move(game, temperature=self.config.temperature)
            if not move:
                break
            
            game.make_move(move)
            
            # Store training data
            game_data.append((state_planes, policy_probs, value_pred))
            move_count += 1
        
        # Assign final game result to all positions
        final_result = game.get_game_result() if game.is_game_over() else 0.0
        
        # Update values with actual game result
        training_data = []
        for state, policy, _ in game_data:
            training_data.append((state, policy, final_result))
        
        self.games_generated += 1
        self.positions_generated += len(training_data)
        
        return training_data
    
    def generate_batch_parallel(self, num_games: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Generate multiple games in parallel"""
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            
            for _ in range(num_games):
                future = executor.submit(self.generate_game_fast, self.model)
                futures.append(future)
            
            # Collect results
            all_data = []
            for future in futures:
                try:
                    game_data = future.result(timeout=30)  # 30 second timeout per game
                    all_data.extend(game_data)
                except Exception as e:
                    self.logger.warning(f"Game generation failed: {e}")
            
            return all_data
    
    def save_data_hdf5(self, data: List[Tuple[np.ndarray, np.ndarray, float]], filename: str):
        """Save training data in efficient HDF5 format"""
        if not data:
            return
        
        positions = np.array([item[0] for item in data])
        policies = np.array([item[1] for item in data])
        values = np.array([item[2] for item in data])
        
        with h5py.File(filename, 'w') as f:
            f.create_dataset('positions', data=positions, compression='gzip', compression_opts=9)
            f.create_dataset('policies', data=policies, compression='gzip', compression_opts=9)
            f.create_dataset('values', data=values, compression='gzip', compression_opts=9)
            
            # Add metadata
            f.attrs['num_samples'] = len(data)
            f.attrs['games_generated'] = self.games_generated
            f.attrs['timestamp'] = time.time()

class UltraFastTrainer:
    """Ultra-optimized training pipeline for chess neural networks"""
    
    def __init__(self, model: UltraFastChessNet, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = Config.DEVICE
        
        # Setup model for fast training
        self.model = FastTrainingMixin.setup_fast_training(self.model, self.device)
        
        # Optimizer setup
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision setup
        self.scaler = GradScaler() if self.config.use_mixed_precision else None
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Data generation
        self.data_generator = SelfPlayDataGenerator(self.model, config)
        
        # Metrics tracking
        self.training_history = {
            'loss': [],
            'policy_loss': [],
            'value_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with optimal parameters"""
        if self.config.optimizer == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:  # SGD
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
                nesterov=True
            )
    
    def _setup_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler"""
        if self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.max_epochs // 3,
                gamma=0.1
            )
        else:  # plateau
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
    
    def compute_loss(self, positions: torch.Tensor, target_policies: torch.Tensor, 
                    target_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss with optimal numerical stability"""
        # Forward pass
        if self.config.use_mixed_precision:
            with autocast(device_type=self.device):
                policy_logits, value_pred = self.model(positions)
                
                # Policy loss (cross-entropy)
                policy_loss = F.cross_entropy(policy_logits, target_policies)
                
                # Value loss (MSE)
                value_loss = F.mse_loss(value_pred.squeeze(), target_values)
                
                # Combined loss
                total_loss = policy_loss + value_loss
        else:
            policy_logits, value_pred = self.model(positions)
            policy_loss = F.cross_entropy(policy_logits, target_policies)
            value_loss = F.mse_loss(value_pred.squeeze(), target_values)
            total_loss = policy_loss + value_loss
        
        return total_loss, policy_loss, value_loss
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch with all optimizations"""
        self.model.train()
        epoch_start = time.time()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        # Gradient accumulation
        accumulated_loss = 0.0
        
        for batch_idx, (positions, policies, values) in enumerate(dataloader):
            positions = positions.to(self.device, non_blocking=True)
            policies = policies.to(self.device, non_blocking=True)
            values = values.to(self.device, non_blocking=True)
            
            # Compute loss
            loss, policy_loss, value_loss = self.compute_loss(positions, policies, values)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulated_loss += loss.item()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_mixed_precision:
                    # Gradient clipping
                    if self.config.use_gradient_clipping:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.max_gradient_norm
                        )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard training
                    if self.config.use_gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config.max_gradient_norm
                        )
                    
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                accumulated_loss = 0.0
            
            # Accumulate metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
            
            # Progress logging
            if batch_idx % 100 == 0:
                self.logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        # Scheduler step
        if self.config.scheduler != 'plateau':
            self.scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'epoch_time': epoch_time,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validation with minimal overhead"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for positions, policies, values in dataloader:
                positions = positions.to(self.device, non_blocking=True)
                policies = policies.to(self.device, non_blocking=True)
                values = values.to(self.device, non_blocking=True)
                
                loss, _, _ = self.compute_loss(positions, policies, values)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Scheduler step for plateau
        if self.config.scheduler == 'plateau':
            self.scheduler.step(avg_loss)
        
        return avg_loss
    
    def train_with_selfplay(self) -> UltraFastChessNet:
        """Complete training loop with integrated self-play"""
        self.logger.info("Starting ultra-fast training with self-play...")
        self.logger.info(f"Device: {self.device}, Batch size: {self.config.batch_size}")
        
        start_time = time.time()
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Generate fresh training data
            self.logger.info(f"Epoch {epoch}: Generating {self.config.selfplay_games_per_iteration} games...")
            training_data = self.data_generator.generate_batch_parallel(
                self.config.selfplay_games_per_iteration
            )
            
            if not training_data:
                self.logger.warning("No training data generated, skipping epoch")
                continue
            
            # Save data
            data_file = f"temp_training_data_epoch_{epoch}.h5"
            self.data_generator.save_data_hdf5(training_data, data_file)
            
            # Create dataloader
            dataset = FastChessDataset(data_file)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                persistent_workers=self.config.persistent_workers,
                prefetch_factor=self.config.prefetch_factor
            )
            
            # Train epoch
            train_metrics = self.train_epoch(dataloader)
            
            # Validation
            val_loss = None
            if epoch % self.config.validation_frequency == 0:
                # Create validation data
                val_data = self.data_generator.generate_batch_parallel(
                    self.config.selfplay_games_per_iteration // 10
                )
                if val_data:
                    val_file = f"temp_val_data_epoch_{epoch}.h5"
                    self.data_generator.save_data_hdf5(val_data, val_file)
                    val_dataset = FastChessDataset(val_file)
                    val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch_size)
                    val_loss = self.validate(val_dataloader)
                    os.remove(val_file)
            
            # Update training history
            for key, value in train_metrics.items():
                self.training_history[key].append(value)
            
            # Logging
            epoch_time = time.time() - epoch_start
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            self.logger.info(f"Train Loss: {train_metrics['loss']:.4f}, LR: {train_metrics['learning_rate']:.6f}")
            if val_loss:
                self.logger.info(f"Val Loss: {val_loss:.4f}")
            
            # Early stopping
            current_loss = val_loss if val_loss else train_metrics['loss']
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience_counter = 0
                self._save_checkpoint('best_model.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Cleanup
            os.remove(data_file)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        
        return self.model
    
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/{filename}"
        
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'config': self.config.__dict__
        }, checkpoint_path)
    
    def load_checkpoint(self, filename: str):
        """Load training checkpoint"""
        checkpoint_path = f"checkpoints/{filename}"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.training_history = checkpoint['training_history']
            
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")

# Utility functions for training optimization
def get_optimal_config() -> TrainingConfig:
    """Get optimized training configuration based on device"""
    device = Config.DEVICE
    
    if device == 'cuda':
        return TrainingConfig(
            batch_size=1024,
            learning_rate=0.002,
            gradient_accumulation_steps=2,
            num_workers=8,
            use_mixed_precision=True
        )
    elif device == 'mps':
        return TrainingConfig(
            batch_size=512,
            learning_rate=0.001,
            gradient_accumulation_steps=4,
            num_workers=4,
            use_mixed_precision=False  # MPS doesn't support mixed precision yet
        )
    else:  # CPU
        return TrainingConfig(
            batch_size=256,
            learning_rate=0.0005,
            gradient_accumulation_steps=8,
            num_workers=2,
            use_mixed_precision=False
        )

def benchmark_training_speed(model: UltraFastChessNet, config: TrainingConfig) -> Dict[str, float]:
    """Benchmark training speed"""
    device = Config.DEVICE
    model = model.to(device)
    
    # Create dummy data
    batch_size = config.batch_size
    dummy_positions = torch.randn(batch_size, 16, 8, 8, device=device)
    dummy_policies = torch.randn(batch_size, 4160, device=device)
    dummy_values = torch.randn(batch_size, device=device)
    
    # Warmup
    for _ in range(10):
        policy, value = model(dummy_positions)
    
    # Benchmark forward pass
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    for _ in range(100):
        policy, value = model(dummy_positions)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    forward_time = (time.time() - start_time) / 100
    
    # Benchmark training step
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    for _ in range(50):
        optimizer.zero_grad()
        policy, value = model(dummy_positions)
        loss = nn.functional.mse_loss(policy, dummy_policies) + nn.functional.mse_loss(value, dummy_values)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize() if device == 'cuda' else None
    training_time = (time.time() - start_time) / 50
    
    return {
        'forward_pass_ms': forward_time * 1000,
        'training_step_ms': training_time * 1000,
        'samples_per_second': batch_size / training_time,
        'device': device,
        'batch_size': batch_size
    }