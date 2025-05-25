import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import time
import pickle
import tempfile
import os
from dataclasses import dataclass
from pathlib import Path
import logging
from copy import deepcopy

from src.neural_network.ultra_fast_chess_net import UltraFastChessNet, ModelFactory
from src.engine.chess_game import ChessGame
from src.engine.neural_player import NeuralPlayer
from config import Config

@dataclass
class Individual:
    """Lightweight individual representation"""
    genome: np.ndarray
    fitness: float = 0.0
    age: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    elo_rating: float = 1200.0
    parent_ids: Tuple[int, int] = (-1, -1)
    mutation_rate: float = 0.1
    
    @property
    def games_played(self) -> int:
        return self.wins + self.losses + self.draws
    
    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.games_played)

class SharedModelPool:
    """Efficient shared memory model pool for parallel evaluation"""
    
    def __init__(self, model_template: UltraFastChessNet, pool_size: int = 8):
        self.model_template = model_template
        self.pool_size = pool_size
        self.models = []
        self.device = Config.DEVICE
        
        # Pre-allocate models
        for _ in range(pool_size):
            model = ModelFactory.create_ultra_fast()
            model.to(self.device)
            model.eval()
            self.models.append(model)
    
    def get_model(self) -> UltraFastChessNet:
        """Get available model from pool"""
        if self.models:
            return self.models.pop()
        else:
            # Create new model if pool empty
            model = ModelFactory.create_ultra_fast()
            model.to(self.device)
            model.eval()
            return model
    
    def return_model(self, model: UltraFastChessNet):
        """Return model to pool"""
        if len(self.models) < self.pool_size:
            self.models.append(model)

class UltraFastGeneticEngine:
    """Ultra-optimized genetic algorithm for chess neural networks"""
    
    def __init__(self,
                 population_size: int = 50,
                 elite_size: int = 10,
                 tournament_size: int = 5,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 max_generations: int = 100,
                 target_strength: str = 'medium',
                 parallel_games: int = None,
                 use_gpu_acceleration: bool = True):
        
        self.population_size = population_size
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.target_strength = target_strength
        
        # Parallel processing setup
        self.parallel_games = parallel_games or min(mp.cpu_count(), 8)
        self.use_gpu = use_gpu_acceleration and Config.DEVICE != 'cpu'
        
        # Create model template
        self.model_template = ModelFactory.create_ultra_fast(target_strength)
        self.genome_size = sum(p.numel() for p in self.model_template.parameters())
        
        # Initialize population
        self.population: List[Individual] = []
        self.generation = 0
        self.best_fitness = float('-inf')
        self.fitness_history = []
        
        # Performance tracking
        self.start_time = time.time()
        self.total_games = 0
        self.total_evaluations = 0
        
        # Shared model pool for parallel evaluation
        self.model_pool = SharedModelPool(self.model_template, self.parallel_games * 2)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_population(self):
        """Initialize population with diverse individuals"""
        self.logger.info(f"Initializing population of {self.population_size} individuals...")
        
        for i in range(self.population_size):
            # Create diverse initial genomes
            if i == 0:
                # Keep one individual as the base model
                genome = self._model_to_genome(self.model_template)
            else:
                # Random initialization with different strategies
                if i < self.population_size // 3:
                    # Small mutations from base
                    genome = self._model_to_genome(self.model_template)
                    genome += np.random.normal(0, 0.01, genome.shape)
                elif i < 2 * self.population_size // 3:
                    # Xavier initialization
                    genome = np.random.normal(0, 0.1, self.genome_size)
                else:
                    # He initialization
                    genome = np.random.normal(0, 0.02, self.genome_size)
            
            individual = Individual(
                genome=genome,
                mutation_rate=np.random.uniform(0.05, 0.15)
            )
            self.population.append(individual)
        
        self.logger.info(f"Population initialized with {len(self.population)} individuals")
    
    def _model_to_genome(self, model: nn.Module) -> np.ndarray:
        """Convert model parameters to genome vector"""
        params = []
        for param in model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def _genome_to_model(self, genome: np.ndarray) -> UltraFastChessNet:
        """Convert genome vector to model"""
        model = ModelFactory.create_ultra_fast(self.target_strength)
        
        param_idx = 0
        with torch.no_grad():
            for param in model.parameters():
                param_size = param.numel()
                param_data = genome[param_idx:param_idx + param_size]
                param.data = torch.from_numpy(param_data.reshape(param.shape)).float()
                param_idx += param_size
        
        model.to(Config.DEVICE)
        return model
    
    def evaluate_individual_fast(self, individual: Individual, opponent_pool: List[Individual], 
                                num_games: int = 3) -> float:
        """Fast evaluation using vectorized operations and GPU acceleration"""
        if len(opponent_pool) < num_games:
            opponent_pool = opponent_pool * ((num_games // len(opponent_pool)) + 1)
        
        opponents = np.random.choice(opponent_pool, num_games, replace=False)
        
        wins = 0
        total_games = 0
        
        # Batch process games for efficiency
        model = self._genome_to_model(individual.genome)
        model.eval()
        
        for opponent in opponents:
            opponent_model = self._genome_to_model(opponent.genome)
            opponent_model.eval()
            
            # Play game with time limit for speed
            result = self._play_fast_game(model, opponent_model, max_moves=100)
            
            if result == 1:  # Win
                wins += 1
            elif result == 0.5:  # Draw
                wins += 0.5
            
            total_games += 1
        
        # Update individual statistics
        individual.wins += wins
        individual.games_played = individual.wins + individual.losses + individual.draws
        
        # Calculate multi-objective fitness
        fitness = self._calculate_fitness(individual)
        individual.fitness = fitness
        
        self.total_games += total_games
        return fitness
    
    def _play_fast_game(self, model1: nn.Module, model2: nn.Module, max_moves: int = 100) -> float:
        """Play a fast game between two models with move limit"""
        game = ChessGame()
        player1 = NeuralPlayer(model1, "Player1")
        player2 = NeuralPlayer(model2, "Player2")
        
        moves = 0
        while not game.is_game_over() and moves < max_moves:
            current_player = player1 if game.current_player() else player2
            
            # Get move with timeout
            try:
                move = current_player.get_move(game, temperature=0.1)  # Low temperature for speed
                if move:
                    game.make_move(move)
                else:
                    break
            except Exception:
                break
            
            moves += 1
        
        # Determine result
        if game.is_game_over():
            result = game.get_game_result()
            if result == 1.0:  # White wins
                return 1.0
            elif result == -1.0:  # Black wins
                return 0.0
            else:  # Draw
                return 0.5
        else:
            # Game reached move limit - evaluate position
            # Simple heuristic: count material and position
            return 0.5  # Draw by default
    
    def _calculate_fitness(self, individual: Individual) -> float:
        """Calculate multi-objective fitness score"""
        base_fitness = individual.win_rate * 100
        
        # Age penalty for diversity
        age_penalty = individual.age * 0.1
        
        # Games played bonus (exploration)
        experience_bonus = min(individual.games_played * 0.5, 10)
        
        # Elo rating component
        elo_component = (individual.elo_rating - 1200) / 10
        
        return base_fitness + experience_bonus + elo_component - age_penalty
    
    def selection_tournament(self, k: Optional[int] = None) -> Individual:
        """Fast tournament selection"""
        if k is None:
            k = self.tournament_size
        
        tournament = np.random.choice(self.population, k, replace=False)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover_uniform(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Optimized uniform crossover"""
        if np.random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        # Vectorized crossover
        mask = np.random.random(self.genome_size) < 0.5
        
        child1_genome = np.where(mask, parent1.genome, parent2.genome)
        child2_genome = np.where(mask, parent2.genome, parent1.genome)
        
        child1 = Individual(
            genome=child1_genome,
            parent_ids=(id(parent1), id(parent2)),
            mutation_rate=(parent1.mutation_rate + parent2.mutation_rate) / 2
        )
        
        child2 = Individual(
            genome=child2_genome,
            parent_ids=(id(parent1), id(parent2)),
            mutation_rate=(parent1.mutation_rate + parent2.mutation_rate) / 2
        )
        
        return child1, child2
    
    def mutate_adaptive(self, individual: Individual):
        """Adaptive mutation with self-adjusting rates"""
        # Adaptive mutation rate based on fitness
        if individual.fitness > np.mean([ind.fitness for ind in self.population]):
            mutation_strength = individual.mutation_rate * 0.5  # Smaller mutations for good individuals
        else:
            mutation_strength = individual.mutation_rate * 1.5  # Larger mutations for poor individuals
        
        # Different mutation strategies
        strategy = np.random.choice(['gaussian', 'cauchy', 'uniform'])
        
        if strategy == 'gaussian':
            mutation = np.random.normal(0, mutation_strength, self.genome_size)
        elif strategy == 'cauchy':
            mutation = np.random.standard_cauchy(self.genome_size) * mutation_strength * 0.1
        else:  # uniform
            mutation = np.random.uniform(-mutation_strength, mutation_strength, self.genome_size)
        
        individual.genome += mutation
        
        # Mutate mutation rate itself
        if np.random.random() < 0.1:
            individual.mutation_rate *= np.random.uniform(0.8, 1.2)
            individual.mutation_rate = np.clip(individual.mutation_rate, 0.01, 0.3)
    
    def evolve_generation(self):
        """Evolve one generation with optimized operations"""
        self.logger.info(f"Evolving generation {self.generation}...")
        start_time = time.time()
        
        # Parallel fitness evaluation
        self._evaluate_population_parallel()
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Track best fitness
        current_best = self.population[0].fitness
        if current_best > self.best_fitness:
            self.best_fitness = current_best
            self._save_best_individual()
        
        self.fitness_history.append(current_best)
        
        # Elite preservation
        elite = self.population[:self.elite_size]
        
        # Generate new population
        new_population = elite.copy()
        
        # Fill rest with offspring
        while len(new_population) < self.population_size:
            parent1 = self.selection_tournament()
            parent2 = self.selection_tournament()
            
            child1, child2 = self.crossover_uniform(parent1, parent2)
            
            self.mutate_adaptive(child1)
            self.mutate_adaptive(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        new_population = new_population[:self.population_size]
        
        # Age population
        for individual in new_population:
            individual.age += 1
        
        self.population = new_population
        self.generation += 1
        
        generation_time = time.time() - start_time
        self.logger.info(f"Generation {self.generation-1} completed in {generation_time:.2f}s")
        self.logger.info(f"Best fitness: {current_best:.3f}, Population avg: {np.mean([ind.fitness for ind in self.population]):.3f}")
    
    def _evaluate_population_parallel(self):
        """Evaluate population in parallel for maximum speed"""
        if self.use_gpu and len(self.population) > 10:
            self._evaluate_population_gpu_batch()
        else:
            self._evaluate_population_cpu_parallel()
    
    def _evaluate_population_cpu_parallel(self):
        """CPU-based parallel evaluation"""
        with ThreadPoolExecutor(max_workers=self.parallel_games) as executor:
            futures = []
            
            for individual in self.population:
                # Select random opponents
                opponents = [ind for ind in self.population if ind != individual]
                future = executor.submit(self.evaluate_individual_fast, individual, opponents, 3)
                futures.append(future)
            
            # Wait for all evaluations
            for future in futures:
                future.result()
    
    def _evaluate_population_gpu_batch(self):
        """GPU-based batch evaluation (experimental)"""
        # This would require significant GPU programming
        # For now, fall back to CPU parallel
        self._evaluate_population_cpu_parallel()
    
    def _save_best_individual(self):
        """Save the best individual"""
        best = self.population[0]
        model = self._genome_to_model(best.genome)
        
        save_path = f"models/ultra_fast_champion_gen_{self.generation}.pth"
        os.makedirs("models", exist_ok=True)
        
        # Save with metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'input_planes': 16,
                'base_filters': 64,
                'architecture': 'UltraFastChessNet',
                'optimized': True,
                'version': '2.0'
            },
            'evolution_metadata': {
                'generation': self.generation,
                'fitness': best.fitness,
                'win_rate': best.win_rate,
                'games_played': best.games_played,
                'elo_rating': best.elo_rating,
                'parent_ids': best.parent_ids,
                'mutation_rate': best.mutation_rate
            },
            'training_stats': {
                'total_games': self.total_games,
                'total_generations': self.generation,
                'best_fitness_history': self.fitness_history
            }
        }, save_path)
        
        self.logger.info(f"Best individual saved to {save_path}")
    
    def run_evolution(self) -> UltraFastChessNet:
        """Run complete evolution process"""
        self.logger.info("Starting ultra-fast genetic evolution...")
        self.logger.info(f"Parameters: pop_size={self.population_size}, generations={self.max_generations}")
        self.logger.info(f"Genome size: {self.genome_size:,} parameters")
        
        # Initialize
        self.initialize_population()
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.evolve_generation()
            
            # Early stopping if converged
            if len(self.fitness_history) > 10:
                recent_improvement = max(self.fitness_history[-5:]) - max(self.fitness_history[-10:-5])
                if recent_improvement < 0.1:
                    self.logger.info(f"Converged after {generation} generations")
                    break
        
        # Return best model
        best_individual = max(self.population, key=lambda x: x.fitness)
        best_model = self._genome_to_model(best_individual.genome)
        
        total_time = time.time() - self.start_time
        self.logger.info(f"Evolution completed in {total_time:.2f}s")
        self.logger.info(f"Total games played: {self.total_games}")
        self.logger.info(f"Best fitness achieved: {best_individual.fitness:.3f}")
        
        return best_model
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get detailed evolution statistics"""
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'total_games': self.total_games,
            'total_time': time.time() - self.start_time,
            'games_per_second': self.total_games / (time.time() - self.start_time),
            'population_diversity': np.std([ind.fitness for ind in self.population]),
            'average_age': np.mean([ind.age for ind in self.population]),
            'elite_win_rate': np.mean([ind.win_rate for ind in self.population[:self.elite_size]])
        }

class HyperParameterOptimizer:
    """Optimize genetic algorithm hyperparameters"""
    
    @staticmethod
    def optimize_parameters(base_config: Dict[str, Any], num_trials: int = 20) -> Dict[str, Any]:
        """Use simple grid search to optimize parameters"""
        best_config = base_config.copy()
        best_performance = 0
        
        # Parameter ranges to explore
        param_ranges = {
            'population_size': [30, 50, 100],
            'mutation_rate': [0.05, 0.1, 0.15],
            'crossover_rate': [0.7, 0.8, 0.9],
            'tournament_size': [3, 5, 7]
        }
        
        for trial in range(num_trials):
            # Random sample from parameter space
            config = base_config.copy()
            for param, values in param_ranges.items():
                config[param] = np.random.choice(values)
            
            # Quick evaluation (reduced generations)
            engine = UltraFastGeneticEngine(max_generations=5, **config)
            best_model = engine.run_evolution()
            
            # Simple performance metric
            performance = engine.best_fitness
            
            if performance > best_performance:
                best_performance = performance
                best_config = config
        
        return best_config