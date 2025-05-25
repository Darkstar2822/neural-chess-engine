"""
Stockfish-Killer: Advanced Co-Evolutionary Chess Engine
Designed to beat Stockfish and other top engines through specialized population evolution
"""

import numpy as np
import torch
import torch.nn as nn
import chess
import chess.engine
import chess.pgn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging
from datetime import datetime
import asyncio
import concurrent.futures
from pathlib import Path
import json
import random
from enum import Enum

from .enhanced_evolutionary_engine import EnhancedEvolutionaryEngine, EvolvableIndividual
from .neuroevolution import NetworkGenome, EvolvableChessNet
from ..engine.neural_player import NeuralPlayer
from ..engine.chess_game import ChessGame


class PlayingStyle(Enum):
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    POSITIONAL = "positional"
    TACTICAL = "tactical"
    ENDGAME_SPECIALIST = "endgame"


@dataclass
class ChessEngineRating:
    """Rating and performance data for chess engines"""
    name: str
    elo_rating: int
    wins: int = 0
    losses: int = 0
    draws: int = 0
    
    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws
    
    @property
    def win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.wins / self.total_games
    
    @property
    def score_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.total_games


@dataclass
class SpecializedIndividual(EvolvableIndividual):
    """Individual specialized for specific playing style"""
    playing_style: PlayingStyle = PlayingStyle.POSITIONAL
    style_strength: float = 1.0  # How strongly it exhibits this style
    anti_engine_training: List[str] = field(default_factory=list)  # Engines it was trained against
    engine_performance: Dict[str, ChessEngineRating] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        self.specialize_for_style()
    
    def specialize_for_style(self):
        """Apply style-specific modifications to the network"""
        if not hasattr(self, 'style_applied'):
            self._apply_style_bias()
            self.style_applied = True
    
    def _apply_style_bias(self):
        """Apply neural network biases for playing style"""
        with torch.no_grad():
            # Style-specific parameter adjustments
            style_multipliers = {
                PlayingStyle.AGGRESSIVE: self._aggressive_bias,
                PlayingStyle.DEFENSIVE: self._defensive_bias,
                PlayingStyle.POSITIONAL: self._positional_bias,
                PlayingStyle.TACTICAL: self._tactical_bias,
                PlayingStyle.ENDGAME_SPECIALIST: self._endgame_bias
            }
            
            style_multipliers[self.playing_style]()
    
    def _aggressive_bias(self):
        """Bias toward aggressive, attacking play"""
        # Increase weights for tactical patterns, attacking moves
        if hasattr(self.model, 'policy_head'):
            # Boost weights for center control and piece activity
            self.model.policy_head[0].weight *= (1.0 + 0.1 * self.style_strength)
    
    def _defensive_bias(self):
        """Bias toward solid, defensive play"""
        # Increase weights for defensive patterns, king safety
        if hasattr(self.model, 'value_head'):
            # Conservative evaluation adjustments
            self.model.value_head[0].bias *= (1.0 - 0.05 * self.style_strength)
    
    def _positional_bias(self):
        """Bias toward long-term positional play"""
        # Balanced approach with slight positional preference
        pass  # Default behavior is already somewhat positional
    
    def _tactical_bias(self):
        """Bias toward tactical, combinative play"""
        # Increase sensitivity to tactical patterns
        if hasattr(self.model, 'policy_head'):
            # Sharpen policy distribution for tactical clarity
            self.model.policy_head[-1].weight *= (1.0 + 0.15 * self.style_strength)
    
    def _endgame_bias(self):
        """Bias toward endgame expertise"""
        # Specialized endgame knowledge
        if hasattr(self.model, 'value_head'):
            # More precise endgame evaluation
            self.model.value_head[-1].weight *= (1.0 + 0.2 * self.style_strength)


class StockfishKillerEngine:
    """Advanced co-evolutionary engine designed to beat Stockfish"""
    
    def __init__(self, 
                 total_population_size: int = 150,
                 num_specialists: int = 5,
                 generations_per_cycle: int = 10,
                 stockfish_path: str = "/opt/homebrew/bin/stockfish"):
        
        self.total_population_size = total_population_size
        self.num_specialists = num_specialists
        self.generations_per_cycle = generations_per_cycle
        self.stockfish_path = stockfish_path
        
        # Specialized populations
        self.populations = {}
        self.population_sizes = self._calculate_population_sizes()
        
        # Performance tracking
        self.generation = 0
        self.stockfish_performance = {}
        self.engine_benchmark_results = {}
        
        # Tournament settings
        self.time_control = {"time": 10.0, "increment": 0.1}  # 10 seconds + 0.1 increment
        self.tournament_games_per_matchup = 6  # 3 as white, 3 as black
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Stockfish-Killer Engine initialized")
    
    def _calculate_population_sizes(self) -> Dict[PlayingStyle, int]:
        """Distribute population across playing styles"""
        base_size = self.total_population_size // self.num_specialists
        remainder = self.total_population_size % self.num_specialists
        
        sizes = {}
        styles = list(PlayingStyle)[:self.num_specialists]
        
        for i, style in enumerate(styles):
            sizes[style] = base_size + (1 if i < remainder else 0)
        
        return sizes
    
    def initialize_specialized_populations(self):
        """Initialize all specialized populations"""
        self.logger.info("Initializing specialized populations for Stockfish domination")
        
        for style, size in self.population_sizes.items():
            self.logger.info(f"Creating {style.value} population: {size} individuals")
            
            population = []
            for i in range(size):
                # Create specialized individual
                genome = NetworkGenome()
                individual = SpecializedIndividual(
                    genome=genome,
                    model=EvolvableChessNet(genome),
                    playing_style=style,
                    style_strength=np.random.uniform(0.7, 1.3),
                    generation=0
                )
                population.append(individual)
            
            self.populations[style] = population
        
        total_created = sum(len(pop) for pop in self.populations.values())
        self.logger.info(f"Total population created: {total_created} specialized individuals")
    
    async def benchmark_against_stockfish(self, individual: SpecializedIndividual,
                                        stockfish_level: int = 8,
                                        num_games: int = 6) -> ChessEngineRating:
        """Benchmark individual against Stockfish"""
        self.logger.info(f"Benchmarking {individual.playing_style.value} vs Stockfish Level {stockfish_level}")
        
        rating = ChessEngineRating(name=f"Stockfish-{stockfish_level}", elo_rating=2000 + stockfish_level * 100)
        
        # Configure Stockfish
        transport, engine = await chess.engine.popen_uci(self.stockfish_path)
        await engine.configure({"Skill Level": stockfish_level, "Threads": 1})
        
        try:
            for game_num in range(num_games):
                # Alternate colors
                our_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK
                
                result = await self._play_engine_game(individual, engine, our_color)
                
                if result == "win":
                    rating.wins += 1
                elif result == "loss":
                    rating.losses += 1
                else:
                    rating.draws += 1
                
                self.logger.info(f"Game {game_num + 1}/{num_games}: {result}")
        
        finally:
            await engine.quit()
        
        # Update individual's performance record
        individual.engine_performance[rating.name] = rating
        
        self.logger.info(f"Benchmark complete: {rating.wins}W-{rating.losses}L-{rating.draws}D "
                        f"(Score: {rating.score_rate:.3f})")
        
        return rating
    
    async def _play_engine_game(self, individual: SpecializedIndividual, 
                              engine: chess.engine.UciProtocol,
                              our_color: chess.Color) -> str:
        """Play a single game against an engine"""
        game = ChessGame()
        neural_player = NeuralPlayer(individual.model)
        
        move_count = 0
        max_moves = 200
        
        while not game.is_game_over() and move_count < max_moves:
            if game.current_player() == our_color:
                # Our move
                try:
                    move = neural_player.get_move(game, temperature=0.1)
                    if move and move in game.get_legal_moves():
                        game.make_move(move)
                    else:
                        # Fallback to random legal move
                        legal_moves = game.get_legal_moves()
                        if legal_moves:
                            game.make_move(random.choice(legal_moves))
                        else:
                            break
                except Exception as e:
                    self.logger.warning(f"Neural player error: {e}")
                    legal_moves = game.get_legal_moves()
                    if legal_moves:
                        game.make_move(random.choice(legal_moves))
                    else:
                        break
            else:
                # Engine move
                try:
                    result = await engine.play(
                        game.board, 
                        chess.engine.Limit(time=self.time_control["time"])
                    )
                    if result.move:
                        game.make_move(result.move)
                    else:
                        break
                except Exception as e:
                    self.logger.warning(f"Engine error: {e}")
                    break
            
            move_count += 1
        
        # Determine result from our perspective
        game_result = game.get_game_result()
        
        if game_result is None:
            return "draw"
        elif (game_result > 0 and our_color == chess.WHITE) or (game_result < 0 and our_color == chess.BLACK):
            return "win"
        elif game_result == 0:
            return "draw"
        else:
            return "loss"
    
    def evolve_populations(self):
        """Evolve all specialized populations through competition"""
        self.logger.info(f"Evolution cycle {self.generation}: Evolving specialized populations")
        
        # Inter-population tournament
        self._run_inter_population_tournament()
        
        # Intra-population evolution
        for style, population in self.populations.items():
            self.logger.info(f"Evolving {style.value} population")
            evolved_pop = self._evolve_single_population(population, style)
            self.populations[style] = evolved_pop
        
        # Cross-population genetic exchange (limited)
        self._cross_population_exchange()
        
        self.generation += 1
    
    def _run_inter_population_tournament(self):
        """Tournament between different specialized populations"""
        self.logger.info("Running inter-population tournament")
        
        styles = list(self.populations.keys())
        results_matrix = np.zeros((len(styles), len(styles)))
        
        # Round-robin between populations
        for i, style1 in enumerate(styles):
            for j, style2 in enumerate(styles):
                if i != j:
                    # Sample representatives from each population
                    pop1_sample = random.sample(self.populations[style1], min(3, len(self.populations[style1])))
                    pop2_sample = random.sample(self.populations[style2], min(3, len(self.populations[style2])))
                    
                    wins = 0
                    total_games = 0
                    
                    for ind1 in pop1_sample:
                        for ind2 in pop2_sample:
                            result = self._play_individuals(ind1, ind2)
                            if result == "win":
                                wins += 1
                            total_games += 1
                    
                    results_matrix[i, j] = wins / total_games if total_games > 0 else 0.5
        
        # Update fitness based on inter-population performance
        for i, style in enumerate(styles):
            avg_performance = np.mean(results_matrix[i, :])
            for individual in self.populations[style]:
                individual.fitness_scores["inter_population_performance"] = avg_performance
    
    def _play_individuals(self, ind1: SpecializedIndividual, ind2: SpecializedIndividual) -> str:
        """Play game between two individuals"""
        game = ChessGame()
        player1 = NeuralPlayer(ind1.model)
        player2 = NeuralPlayer(ind2.model)
        
        move_count = 0
        max_moves = 150
        
        while not game.is_game_over() and move_count < max_moves:
            current_player = player1 if game.current_player() else player2
            
            try:
                move = current_player.get_move(game, temperature=0.2)
                if move and move in game.get_legal_moves():
                    game.make_move(move)
                else:
                    legal_moves = game.get_legal_moves()
                    if legal_moves:
                        game.make_move(random.choice(legal_moves))
                    else:
                        break
            except:
                legal_moves = game.get_legal_moves()
                if legal_moves:
                    game.make_move(random.choice(legal_moves))
                else:
                    break
            
            move_count += 1
        
        result = game.get_game_result()
        if result is None or result == 0:
            return "draw"
        elif result > 0:
            return "win"  # ind1 (white) wins
        else:
            return "loss"  # ind1 (white) loses
    
    def _evolve_single_population(self, population: List[SpecializedIndividual], 
                                style: PlayingStyle) -> List[SpecializedIndividual]:
        """Evolve a single specialized population"""
        # Create enhanced evolutionary engine for this population
        engine = EnhancedEvolutionaryEngine(
            population_size=len(population),
            max_generations=1,  # Single generation step
            use_topology_evolution=True
        )
        
        # Set the population
        engine.population = population
        
        # Evaluate and evolve
        engine._evaluate_population()
        engine._update_ratings()
        engine._calculate_fitness_scores()
        
        # Create next generation
        next_gen = engine._create_next_generation()
        
        # Ensure proper specialization for new individuals
        for individual in next_gen:
            if isinstance(individual, SpecializedIndividual):
                individual.playing_style = style
                individual.specialize_for_style()
        
        return next_gen
    
    def _cross_population_exchange(self, exchange_rate: float = 0.05):
        """Limited genetic exchange between populations"""
        self.logger.info("Cross-population genetic exchange")
        
        styles = list(self.populations.keys())
        
        for i, style1 in enumerate(styles):
            for j, style2 in enumerate(styles[i+1:], i+1):
                if random.random() < exchange_rate:
                    # Exchange top individuals
                    pop1 = self.populations[style1]
                    pop2 = self.populations[style2]
                    
                    # Find best individuals
                    best1 = max(pop1, key=lambda x: x.glicko_rating)
                    best2 = max(pop2, key=lambda x: x.glicko_rating)
                    
                    # Create hybrid offspring
                    from .neuroevolution import NeuroevolutionEngine
                    neuro_engine = NeuroevolutionEngine()
                    
                    hybrid_genome = neuro_engine.crossover(best1.genome, best2.genome)
                    hybrid = SpecializedIndividual(
                        genome=hybrid_genome,
                        model=EvolvableChessNet(hybrid_genome),
                        playing_style=random.choice([style1, style2]),
                        generation=self.generation
                    )
                    
                    # Replace weakest individual in target population
                    target_style = hybrid.playing_style
                    weakest = min(self.populations[target_style], key=lambda x: x.glicko_rating)
                    weakest_idx = self.populations[target_style].index(weakest)
                    self.populations[target_style][weakest_idx] = hybrid
    
    async def stockfish_gauntlet(self, num_levels: int = 5) -> Dict[int, float]:
        """Run gauntlet against multiple Stockfish levels"""
        self.logger.info(f"Running Stockfish gauntlet: Levels 1-{num_levels}")
        
        gauntlet_results = {}
        
        for level in range(1, num_levels + 1):
            self.logger.info(f"Gauntlet Level {level}")
            level_results = []
            
            # Test each population's best individual
            for style, population in self.populations.items():
                best_individual = max(population, key=lambda x: x.glicko_rating)
                
                rating = await self.benchmark_against_stockfish(
                    best_individual, 
                    stockfish_level=level,
                    num_games=4
                )
                
                level_results.append(rating.score_rate)
                self.logger.info(f"{style.value} vs Stockfish {level}: {rating.score_rate:.3f}")
            
            gauntlet_results[level] = np.mean(level_results)
            self.logger.info(f"Average performance vs Stockfish {level}: {gauntlet_results[level]:.3f}")
            
            # Early stopping if we're consistently losing
            if gauntlet_results[level] < 0.1:
                self.logger.warning(f"Poor performance at level {level}, stopping gauntlet")
                break
        
        return gauntlet_results
    
    def get_best_individual(self) -> SpecializedIndividual:
        """Get the overall best individual across all populations"""
        all_individuals = []
        for population in self.populations.values():
            all_individuals.extend(population)
        
        # Score by combined rating and Stockfish performance
        def composite_score(individual):
            base_score = individual.glicko_rating
            
            # Bonus for Stockfish performance
            stockfish_bonus = 0
            for rating in individual.engine_performance.values():
                if "Stockfish" in rating.name:
                    stockfish_bonus += rating.score_rate * 500
            
            return base_score + stockfish_bonus
        
        return max(all_individuals, key=composite_score)
    
    def save_champion(self, filename: str = None):
        """Save the best individual as our Stockfish-beating champion"""
        champion = self.get_best_individual()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stockfish_killer_champion_{timestamp}.pth"
        
        # Save model
        torch.save({
            'model_state_dict': champion.model.state_dict(),
            'genome': champion.genome,
            'playing_style': champion.playing_style.value,
            'generation': champion.generation,
            'glicko_rating': champion.glicko_rating,
            'engine_performance': champion.engine_performance,
            'fitness_scores': champion.fitness_scores
        }, filename)
        
        self.logger.info(f"Champion saved: {filename}")
        self.logger.info(f"Style: {champion.playing_style.value}")
        self.logger.info(f"Rating: {champion.glicko_rating:.1f}")
        
        return filename
    
    async def training_cycle(self, cycles: int = 10, stockfish_test_frequency: int = 3):
        """Complete training cycle to beat Stockfish"""
        self.logger.info(f"Starting training cycle: {cycles} cycles to beat Stockfish")
        
        # Initialize populations
        self.initialize_specialized_populations()
        
        best_stockfish_performance = 0.0
        
        for cycle in range(cycles):
            self.logger.info(f"=== Training Cycle {cycle + 1}/{cycles} ===")
            
            # Evolve populations
            self.evolve_populations()
            
            # Test against Stockfish periodically
            if (cycle + 1) % stockfish_test_frequency == 0:
                gauntlet_results = await self.stockfish_gauntlet(num_levels=3)
                avg_performance = np.mean(list(gauntlet_results.values()))
                
                self.logger.info(f"Cycle {cycle + 1} Stockfish Performance: {avg_performance:.3f}")
                
                if avg_performance > best_stockfish_performance:
                    best_stockfish_performance = avg_performance
                    champion_file = self.save_champion()
                    self.logger.info(f"New best performance! Saved: {champion_file}")
                
                # If we're beating Stockfish consistently, test higher levels
                if avg_performance > 0.55:
                    self.logger.info("Strong performance detected! Testing higher Stockfish levels...")
                    extended_gauntlet = await self.stockfish_gauntlet(num_levels=8)
                    max_level_beaten = max([level for level, score in extended_gauntlet.items() if score > 0.5], default=0)
                    self.logger.info(f"Highest Stockfish level beaten: {max_level_beaten}")
        
        # Final champion save
        final_champion = self.save_champion()
        self.logger.info(f"Training complete! Final champion: {final_champion}")
        
        return final_champion