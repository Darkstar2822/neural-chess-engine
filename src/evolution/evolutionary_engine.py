"""
Core Evolutionary Engine for Chess Neural Network Evolution
Implements population-based evolution with multi-objective optimization
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import json
from pathlib import Path
import logging
from datetime import datetime

from ..neural_network.chess_net import ChessNet
from ..neural_network.optimized_chess_net import OptimizedChessNet
from ..engine.neural_player import NeuralPlayer
from ..engine.chess_game import ChessGame


@dataclass
class Individual:
    """Represents an individual in the population"""
    model: nn.Module
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    glicko_rating: float = 1500.0
    glicko_rd: float = 350.0  # Rating deviation
    glicko_vol: float = 0.06  # Volatility
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    tournament_results: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        self.id = f"gen{self.generation}_{hash(str(self.model.state_dict()))}"[:16]


class GlickoRatingSystem:
    """Glicko-2 rating system for chess engine evaluation"""
    
    def __init__(self, tau: float = 0.5):
        self.tau = tau  # System constant (volatility change)
        self.epsilon = 0.000001  # Convergence tolerance
    
    def g(self, rd: float) -> float:
        """Glicko g function"""
        return 1.0 / np.sqrt(1.0 + 3.0 * (rd / np.pi) ** 2)
    
    def E(self, mu: float, mu_j: float, rd_j: float) -> float:
        """Expected score function"""
        return 1.0 / (1.0 + np.exp(-self.g(rd_j) * (mu - mu_j)))
    
    def update_rating(self, individual: Individual, opponents: List[Individual], 
                     results: List[float]) -> Tuple[float, float, float]:
        """
        Update Glicko-2 rating based on tournament results
        
        Args:
            individual: Player being updated
            opponents: List of opponents faced
            results: List of results (1.0=win, 0.5=draw, 0.0=loss)
        
        Returns:
            New (rating, rd, volatility) tuple
        """
        # Convert to Glicko-2 scale
        mu = (individual.glicko_rating - 1500) / 173.7178
        phi = individual.glicko_rd / 173.7178
        sigma = individual.glicko_vol
        
        if not opponents:
            # No games played - increase uncertainty
            phi_new = np.sqrt(phi**2 + sigma**2)
            return individual.glicko_rating, phi_new * 173.7178, sigma
        
        # Calculate v (estimated variance)
        v_inv = 0
        delta = 0
        
        for opponent, result in zip(opponents, results):
            mu_j = (opponent.glicko_rating - 1500) / 173.7178
            phi_j = opponent.glicko_rd / 173.7178
            
            g_phi_j = self.g(phi_j * 173.7178)
            E_val = self.E(mu, mu_j, phi_j * 173.7178)
            
            v_inv += g_phi_j**2 * E_val * (1 - E_val)
            delta += g_phi_j * (result - E_val)
        
        v = 1.0 / v_inv if v_inv > 0 else float('inf')
        
        # Update volatility using iterative method
        sigma_new = self._update_volatility(sigma, delta, phi, v)
        
        # Update rating and RD
        phi_star = np.sqrt(phi**2 + sigma_new**2)
        phi_new = 1.0 / np.sqrt(1.0/phi_star**2 + 1.0/v)
        mu_new = mu + phi_new**2 * (delta / v)
        
        # Convert back to original scale
        rating_new = mu_new * 173.7178 + 1500
        rd_new = phi_new * 173.7178
        
        return rating_new, rd_new, sigma_new
    
    def _update_volatility(self, sigma: float, delta: float, phi: float, v: float) -> float:
        """Update volatility using iterative algorithm"""
        a = np.log(sigma**2)
        
        def f(x):
            ex = np.exp(x)
            return (ex * (delta**2 - phi**2 - v - ex)) / (2 * (phi**2 + v + ex)**2) - (x - a) / self.tau**2
        
        # Find bounds
        A = a
        if delta**2 > phi**2 + v:
            B = np.log(delta**2 - phi**2 - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            B = a - k * self.tau
        
        # Illinois algorithm
        fA = f(A)
        fB = f(B)
        
        while abs(B - A) > self.epsilon:
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            
            if fC * fB < 0:
                A, fA = B, fB
            else:
                fA /= 2
                
            B, fB = C, fC
        
        return np.exp(A / 2)


class MultiObjectiveFitness:
    """Multi-objective fitness evaluation with Pareto dominance"""
    
    def __init__(self, objectives: List[str] = None):
        self.objectives = objectives or [
            'win_rate', 'game_length', 'piece_efficiency', 
            'positional_strength', 'tactical_sharpness'
        ]
        self.weights = {obj: 1.0 for obj in self.objectives}
    
    def evaluate_individual(self, individual: Individual, 
                          tournament_results: List[Dict]) -> Dict[str, float]:
        """Evaluate all fitness objectives for an individual"""
        scores = {}
        
        # Win rate (higher is better)
        scores['win_rate'] = self._calculate_win_rate(tournament_results)
        
        # Average game length (moderate is better - not too quick, not too long)
        scores['game_length'] = self._calculate_game_length_score(tournament_results)
        
        # Piece efficiency (material advantage relative to moves)
        scores['piece_efficiency'] = self._calculate_piece_efficiency(tournament_results)
        
        # Positional strength (based on final positions)
        scores['positional_strength'] = self._calculate_positional_strength(tournament_results)
        
        # Tactical sharpness (ability to find forcing moves)
        scores['tactical_sharpness'] = self._calculate_tactical_sharpness(tournament_results)
        
        return scores
    
    def pareto_dominance_rank(self, population: List[Individual]) -> List[int]:
        """Assign Pareto dominance ranks to population"""
        n = len(population)
        domination_count = [0] * n  # How many individuals dominate this one
        dominated_individuals = [[] for _ in range(n)]  # Who this individual dominates
        fronts = [[]]
        
        # Compare all pairs
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(population[i], population[j]):
                        dominated_individuals[i].append(j)
                    elif self._dominates(population[j], population[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        # Build subsequent fronts
        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_individuals[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            front_idx += 1
            fronts.append(next_front)
        
        # Assign ranks
        ranks = [0] * n
        for rank, front in enumerate(fronts[:-1]):  # Last front is empty
            for i in front:
                ranks[i] = rank
        
        return ranks
    
    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Check if ind1 dominates ind2 (better in all objectives)"""
        better_in_any = False
        
        for objective in self.objectives:
            score1 = ind1.fitness_scores.get(objective, 0)
            score2 = ind2.fitness_scores.get(objective, 0)
            
            if score1 < score2:
                return False  # ind1 is worse in this objective
            elif score1 > score2:
                better_in_any = True
        
        return better_in_any
    
    def _calculate_win_rate(self, results: List[Dict]) -> float:
        """Calculate win rate from tournament results"""
        if not results:
            return 0.0
        
        total_score = sum(result.get('score', 0) for result in results)
        return total_score / len(results)
    
    def _calculate_game_length_score(self, results: List[Dict]) -> float:
        """Score based on game length - prefer moderate length games"""
        if not results:
            return 0.0
        
        lengths = [result.get('moves', 50) for result in results]
        avg_length = np.mean(lengths)
        
        # Optimal range: 30-60 moves
        if 30 <= avg_length <= 60:
            return 1.0
        elif avg_length < 30:
            return avg_length / 30.0
        else:
            return max(0.1, 1.0 - (avg_length - 60) / 40.0)
    
    def _calculate_piece_efficiency(self, results: List[Dict]) -> float:
        """Calculate piece efficiency score"""
        if not results:
            return 0.0
        
        efficiencies = []
        for result in results:
            material_advantage = result.get('final_material_advantage', 0)
            moves = result.get('moves', 1)
            efficiency = material_advantage / moves if moves > 0 else 0
            efficiencies.append(max(0, efficiency))
        
        return np.mean(efficiencies)
    
    def _calculate_positional_strength(self, results: List[Dict]) -> float:
        """Evaluate positional play strength"""
        # Placeholder - would need position evaluation
        return np.random.uniform(0.3, 0.7)  # TODO: Implement proper evaluation
    
    def _calculate_tactical_sharpness(self, results: List[Dict]) -> float:
        """Evaluate tactical playing strength"""
        # Placeholder - would analyze move quality
        return np.random.uniform(0.3, 0.7)  # TODO: Implement proper evaluation


class EvolutionaryEngine:
    """Main evolutionary engine for chess neural network evolution"""
    
    def __init__(self, population_size: int = 50, max_generations: int = 100,
                 tournament_size: int = 8, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        
        self.glicko_system = GlickoRatingSystem()
        self.fitness_evaluator = MultiObjectiveFitness()
        self.population: List[Individual] = []
        self.generation = 0
        
        self.logger = logging.getLogger(__name__)
        
    def initialize_population(self, base_model_path: str = None) -> None:
        """Initialize the population with random variations"""
        self.logger.info(f"Initializing population of size {self.population_size}")
        
        for i in range(self.population_size):
            if base_model_path and i == 0:
                # First individual is the base model
                model = ChessNet.load_model(base_model_path)
            else:
                # Create random variations
                model = ChessNet()
                if base_model_path:
                    base_model = ChessNet.load_model(base_model_path)
                    self._apply_random_mutation(model, base_model)
            
            individual = Individual(model=model, generation=0)
            self.population.append(individual)
    
    def evolve(self) -> List[Individual]:
        """Main evolutionary loop"""
        self.logger.info("Starting evolutionary process")
        
        for generation in range(self.max_generations):
            self.generation = generation
            self.logger.info(f"Generation {generation + 1}/{self.max_generations}")
            
            # Evaluate population through tournaments
            self._evaluate_population()
            
            # Update Glicko ratings
            self._update_ratings()
            
            # Calculate multi-objective fitness
            self._calculate_fitness_scores()
            
            # Selection and reproduction
            if generation < self.max_generations - 1:
                self.population = self._create_next_generation()
            
            # Log progress
            self._log_generation_stats()
        
        return self.population
    
    def _evaluate_population(self) -> None:
        """Run tournaments to evaluate individuals"""
        self.logger.info("Running population tournaments")
        
        # Clear previous tournament results
        for individual in self.population:
            individual.tournament_results = []
        
        # Run multiple tournament rounds
        for round_num in range(3):  # Multiple rounds for better evaluation
            self._run_tournament_round()
    
    def _run_tournament_round(self) -> None:
        """Run one round of tournaments"""
        import random
        
        # Pair up individuals for games
        shuffled_pop = self.population.copy()
        random.shuffle(shuffled_pop)
        
        # Use process pool for parallel game execution
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for i in range(0, len(shuffled_pop) - 1, 2):
                player1 = shuffled_pop[i]
                player2 = shuffled_pop[i + 1]
                
                future = executor.submit(self._play_game, player1, player2)
                futures.append((future, player1, player2))
            
            # Collect results
            for future, player1, player2 in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    self._record_game_result(player1, player2, result)
                except Exception as e:
                    self.logger.error(f"Game execution failed: {e}")
    
    def _play_game(self, player1: Individual, player2: Individual) -> Dict:
        """Play a game between two individuals"""
        game = ChessGame()
        neural_player1 = NeuralPlayer(player1.model)
        neural_player2 = NeuralPlayer(player2.model)
        
        move_count = 0
        max_moves = 150
        
        while not game.is_game_over() and move_count < max_moves:
            current_player = neural_player1 if game.current_player() else neural_player2
            
            try:
                move = current_player.get_move(game)
                if move in game.get_legal_moves():
                    game.make_move(move)
                    move_count += 1
                else:
                    # Invalid move - random fallback
                    legal_moves = game.get_legal_moves()
                    if legal_moves:
                        game.make_move(np.random.choice(legal_moves))
                        move_count += 1
                    else:
                        break
            except Exception as e:
                self.logger.warning(f"Move generation failed: {e}")
                break
        
        # Determine result  
        game_result_val = game.get_game_result()
        if game_result_val == 1.0:
            result = 'white_wins'
        elif game_result_val == -1.0:
            result = 'black_wins'
        else:
            result = 'draw'
        
        return {
            'moves': move_count,
            'result': result,
            'final_position': game.get_fen(),
            'final_material_advantage': self._calculate_material_advantage(game)
        }
    
    def _record_game_result(self, player1: Individual, player2: Individual, 
                           game_result: Dict) -> None:
        """Record the result of a game"""
        result = game_result['result']
        
        # Convert result to scores
        if result == 'white_wins':
            score1, score2 = 1.0, 0.0
        elif result == 'black_wins':
            score1, score2 = 0.0, 1.0
        else:  # draw
            score1, score2 = 0.5, 0.5
        
        # Record for both players
        player1.tournament_results.append({
            'opponent_id': player2.id,
            'score': score1,
            'moves': game_result['moves'],
            'final_material_advantage': game_result.get('final_material_advantage', 0)
        })
        
        player2.tournament_results.append({
            'opponent_id': player1.id,
            'score': score2,
            'moves': game_result['moves'],
            'final_material_advantage': -game_result.get('final_material_advantage', 0)
        })
    
    def _update_ratings(self) -> None:
        """Update Glicko ratings for all individuals"""
        for individual in self.population:
            if not individual.tournament_results:
                continue
            
            opponents = []
            results = []
            
            for result in individual.tournament_results:
                opponent = next((p for p in self.population if p.id == result['opponent_id']), None)
                if opponent:
                    opponents.append(opponent)
                    results.append(result['score'])
            
            if opponents:
                new_rating, new_rd, new_vol = self.glicko_system.update_rating(
                    individual, opponents, results
                )
                individual.glicko_rating = new_rating
                individual.glicko_rd = new_rd
                individual.glicko_vol = new_vol
    
    def _calculate_fitness_scores(self) -> None:
        """Calculate multi-objective fitness scores"""
        for individual in self.population:
            scores = self.fitness_evaluator.evaluate_individual(
                individual, individual.tournament_results
            )
            individual.fitness_scores = scores
    
    def _create_next_generation(self) -> List[Individual]:
        """Create the next generation through selection and reproduction"""
        # Calculate Pareto ranks
        ranks = self.fitness_evaluator.pareto_dominance_rank(self.population)
        
        # Select parents based on Pareto rank and Glicko rating
        parents = self._select_parents(ranks)
        
        # Create offspring
        offspring = []
        while len(offspring) < self.population_size:
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            child = self._crossover(parent1, parent2)
            
            if np.random.random() < self.mutation_rate:
                self._mutate(child)
            
            offspring.append(child)
        
        return offspring[:self.population_size]
    
    def _select_parents(self, ranks: List[int]) -> List[Individual]:
        """Select parents based on Pareto rank and rating"""
        # Sort by rank (lower is better) then by rating (higher is better)
        ranked_pop = [(individual, rank) for individual, rank in zip(self.population, ranks)]
        ranked_pop.sort(key=lambda x: (x[1], -x[0].glicko_rating))
        
        # Select top 50% as potential parents
        num_parents = max(2, len(self.population) // 2)
        return [individual for individual, _ in ranked_pop[:num_parents]]
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Create offspring through neural network crossover"""
        child_model = ChessNet()
        
        # Simple layer-wise crossover
        with torch.no_grad():
            for (name1, param1), (name2, param2), (name_child, param_child) in zip(
                parent1.model.named_parameters(),
                parent2.model.named_parameters(), 
                child_model.named_parameters()
            ):
                # Random layer selection
                if np.random.random() < 0.5:
                    param_child.copy_(param1)
                else:
                    param_child.copy_(param2)
        
        child = Individual(
            model=child_model,
            generation=self.generation + 1,
            parent_ids=[parent1.id, parent2.id]
        )
        
        return child
    
    def _mutate(self, individual: Individual) -> None:
        """Apply mutation to an individual"""
        mutation_strength = 0.01
        
        with torch.no_grad():
            for param in individual.model.parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * mutation_strength
                    param.add_(noise)
        
        individual.mutation_history.append(f"gaussian_{mutation_strength}")
    
    def _apply_random_mutation(self, target_model: nn.Module, source_model: nn.Module) -> None:
        """Apply random mutation to create initial population diversity"""
        mutation_strength = 0.05
        
        with torch.no_grad():
            for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
                if target_param.requires_grad:
                    noise = torch.randn_like(source_param) * mutation_strength
                    target_param.copy_(source_param + noise)
    
    def _calculate_material_advantage(self, game: ChessGame) -> float:
        """Calculate material advantage from board state"""
        piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}  # pawn, knight, bishop, rook, queen, king
        
        white_material = 0
        black_material = 0
        
        import chess
        for square in chess.SQUARES:
            piece = game.board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        return white_material - black_material
    
    def _log_generation_stats(self) -> None:
        """Log statistics for current generation"""
        ratings = [ind.glicko_rating for ind in self.population]
        win_rates = [ind.fitness_scores.get('win_rate', 0) for ind in self.population]
        
        self.logger.info(f"Generation {self.generation + 1} stats:")
        self.logger.info(f"  Avg rating: {np.mean(ratings):.1f} ± {np.std(ratings):.1f}")
        self.logger.info(f"  Best rating: {np.max(ratings):.1f}")
        self.logger.info(f"  Avg win rate: {np.mean(win_rates):.3f}")
        self.logger.info(f"  Best win rate: {np.max(win_rates):.3f}")