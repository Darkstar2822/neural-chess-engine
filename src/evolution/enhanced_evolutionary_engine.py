"""
Enhanced Evolutionary Engine with Neuroevolution Capabilities
Integrates topology evolution with the existing evolutionary framework
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging
from datetime import datetime
import copy

from .evolutionary_engine import (
    Individual, GlickoRatingSystem, MultiObjectiveFitness, EvolutionaryEngine
)
from .neuroevolution import (
    NeuroevolutionEngine, NetworkGenome, EvolvableChessNet, NodeType, ActivationType
)
from ..engine.neural_player import NeuralPlayer
from ..engine.chess_game import ChessGame


@dataclass
class EvolvableIndividual(Individual):
    """Individual with evolvable neural network topology"""
    genome: NetworkGenome = None
    topology_complexity: float = 0.0
    innovation_score: float = 0.0
    
    def __post_init__(self):
        if self.genome is None:
            self.genome = NetworkGenome()
        
        # Create PyTorch model from genome
        if self.model is None:
            self.model = EvolvableChessNet(self.genome)
        
        super().__post_init__()
    
    def calculate_complexity(self) -> float:
        """Calculate topology complexity metrics"""
        num_nodes = len([n for n in self.genome.nodes.values() if n.node_type == NodeType.HIDDEN])
        num_connections = len([c for c in self.genome.connections.values() if c.enabled])
        
        # Complexity score (normalized)
        self.topology_complexity = (num_nodes * 0.1) + (num_connections * 0.01)
        return self.topology_complexity
    
    def calculate_innovation(self) -> float:
        """Calculate innovation score based on unique structures"""
        # Count different activation types
        activations = set(n.activation for n in self.genome.nodes.values() if n.node_type == NodeType.HIDDEN)
        
        # Count layer depth
        max_layer = max((n.layer for n in self.genome.nodes.values()), default=0)
        
        # Innovation score
        self.innovation_score = len(activations) * 0.2 + max_layer * 0.1
        return self.innovation_score


class EnhancedEvolutionaryEngine(EvolutionaryEngine):
    """Enhanced evolutionary engine with neuroevolution capabilities"""
    
    def __init__(self, population_size: int = 50, max_generations: int = 100,
                 tournament_size: int = 8, mutation_rate: float = 0.1,
                 use_topology_evolution: bool = True):
        
        super().__init__(population_size, max_generations, tournament_size, mutation_rate)
        
        self.use_topology_evolution = use_topology_evolution
        self.neuroevolution_engine = NeuroevolutionEngine(population_size)
        
        # Enhanced fitness with topology objectives
        self.fitness_evaluator = EnhancedMultiObjectiveFitness()
        
        # Topology evolution parameters
        self.topology_mutation_rate = 0.3
        self.speciation_enabled = True
        self.novelty_search_enabled = True
        
        # Novelty archive for diversity
        self.novelty_archive: List[NetworkGenome] = []
        self.archive_size = 100
        
        self.logger = logging.getLogger(__name__)
    
    def initialize_population(self, base_model_path: str = None) -> None:
        """Initialize population with evolvable topologies"""
        self.logger.info(f"Initializing evolvable population of size {self.population_size}")
        
        if self.use_topology_evolution:
            # Create genomes using neuroevolution engine
            genomes = self.neuroevolution_engine.create_initial_population()
            
            for i, genome in enumerate(genomes):
                individual = EvolvableIndividual(
                    genome=genome,
                    model=EvolvableChessNet(genome),
                    generation=0
                )
                individual.calculate_complexity()
                individual.calculate_innovation()
                self.population.append(individual)
        else:
            # Fall back to standard initialization
            super().initialize_population(base_model_path)
    
    def _calculate_fitness_scores(self) -> None:
        """Calculate enhanced multi-objective fitness scores"""
        for individual in self.population:
            # Standard fitness
            scores = self.fitness_evaluator.evaluate_individual(
                individual, individual.tournament_results
            )
            
            # Add topology-specific objectives
            if isinstance(individual, EvolvableIndividual):
                scores['topology_complexity'] = individual.calculate_complexity()
                scores['innovation_score'] = individual.calculate_innovation()
                scores['novelty_score'] = self._calculate_novelty(individual)
            
            individual.fitness_scores = scores
    
    def _calculate_novelty(self, individual: EvolvableIndividual) -> float:
        """Calculate novelty score for diversity maintenance"""
        if not self.novelty_search_enabled or not isinstance(individual, EvolvableIndividual):
            return 0.0
        
        # Calculate behavioral signature
        signature = self._get_behavioral_signature(individual)
        
        # Find k-nearest neighbors in archive + population
        all_signatures = []
        
        # Add archive signatures
        for archived_genome in self.novelty_archive:
            archived_individual = EvolvableIndividual(
                model=EvolvableChessNet(archived_genome),
                genome=archived_genome
            )
            all_signatures.append(self._get_behavioral_signature(archived_individual))
        
        # Add current population signatures
        for pop_individual in self.population:
            if pop_individual != individual and isinstance(pop_individual, EvolvableIndividual):
                all_signatures.append(self._get_behavioral_signature(pop_individual))
        
        if not all_signatures:
            return 1.0
        
        # Calculate distances to k-nearest neighbors
        k = min(15, len(all_signatures))
        distances = []
        
        for other_signature in all_signatures:
            distance = np.linalg.norm(np.array(signature) - np.array(other_signature))
            distances.append(distance)
        
        distances.sort()
        novelty = np.mean(distances[:k])
        
        return novelty
    
    def _get_behavioral_signature(self, individual: EvolvableIndividual) -> List[float]:
        """Extract behavioral signature from individual's gameplay"""
        # Simple signature based on topology and recent performance
        signature = []
        
        # Topology features
        num_nodes = len([n for n in individual.genome.nodes.values() if n.node_type == NodeType.HIDDEN])
        num_connections = len([c for c in individual.genome.connections.values() if c.enabled])
        max_layer = max((n.layer for n in individual.genome.nodes.values()), default=0)
        
        signature.extend([num_nodes / 100.0, num_connections / 1000.0, max_layer / 10.0])
        
        # Performance features
        recent_results = individual.tournament_results[-5:] if len(individual.tournament_results) >= 5 else individual.tournament_results
        
        if recent_results:
            avg_score = np.mean([r.get('score', 0) for r in recent_results])
            avg_moves = np.mean([r.get('moves', 50) for r in recent_results])
            material_efficiency = np.mean([r.get('final_material_advantage', 0) for r in recent_results])
            
            signature.extend([avg_score, avg_moves / 100.0, material_efficiency / 10.0])
        else:
            signature.extend([0.0, 0.5, 0.0])
        
        return signature
    
    def _update_novelty_archive(self):
        """Update novelty archive with diverse individuals"""
        # Add novel individuals to archive
        for individual in self.population:
            if isinstance(individual, EvolvableIndividual):
                novelty = self._calculate_novelty(individual)
                
                # Add to archive if sufficiently novel
                if novelty > 0.5 and len(self.novelty_archive) < self.archive_size:
                    self.novelty_archive.append(copy.deepcopy(individual.genome))
                elif novelty > 0.8 and len(self.novelty_archive) >= self.archive_size:
                    # Replace least novel in archive
                    self.novelty_archive.pop(0)
                    self.novelty_archive.append(copy.deepcopy(individual.genome))
    
    def _create_next_generation(self) -> List[Individual]:
        """Create next generation with topology evolution"""
        if not self.use_topology_evolution:
            return super()._create_next_generation()
        
        # Speciation for topology evolution
        evolvable_pop = [ind for ind in self.population if isinstance(ind, EvolvableIndividual)]
        
        if self.speciation_enabled and evolvable_pop:
            evolvable_genomes = [ind.genome for ind in evolvable_pop]
            self.neuroevolution_engine.speciate(evolvable_genomes)
        
        # Calculate Pareto ranks
        ranks = self.fitness_evaluator.pareto_dominance_rank(self.population)
        
        # Select parents
        parents = self._select_parents(ranks)
        evolvable_parents = [p for p in parents if isinstance(p, EvolvableIndividual)]
        
        # Create offspring
        offspring = []
        
        while len(offspring) < self.population_size:
            if len(evolvable_parents) >= 2 and np.random.random() < 0.8:
                # Topology crossover and mutation
                parent1, parent2 = np.random.choice(evolvable_parents, 2, replace=False)
                
                # Crossover genomes
                child_genome = self.neuroevolution_engine.crossover(parent1.genome, parent2.genome)
                
                # Mutate topology
                if np.random.random() < self.topology_mutation_rate:
                    child_genome = self.neuroevolution_engine.mutate(child_genome)
                
                # Create individual
                child = EvolvableIndividual(
                    genome=child_genome,
                    model=EvolvableChessNet(child_genome),
                    generation=self.generation + 1,
                    parent_ids=[parent1.id, parent2.id]
                )
                
                offspring.append(child)
                
            elif evolvable_parents:
                # Use topology evolution even for remaining slots
                parent1 = np.random.choice(evolvable_parents)
                child_genome = copy.deepcopy(parent1.genome)
                
                # Apply mutation
                if np.random.random() < self.topology_mutation_rate:
                    child_genome = self.neuroevolution_engine.mutate(child_genome)
                
                child = EvolvableIndividual(
                    genome=child_genome,
                    model=EvolvableChessNet(child_genome),
                    generation=self.generation + 1,
                    parent_ids=[parent1.id]
                )
                
                offspring.append(child)
        
        # Update novelty archive
        self._update_novelty_archive()
        
        return offspring[:self.population_size]
    
    def _log_generation_stats(self) -> None:
        """Log enhanced statistics including topology metrics"""
        super()._log_generation_stats()
        
        # Topology statistics
        evolvable_pop = [ind for ind in self.population if isinstance(ind, EvolvableIndividual)]
        
        if evolvable_pop:
            complexities = [ind.topology_complexity for ind in evolvable_pop]
            innovations = [ind.innovation_score for ind in evolvable_pop]
            
            avg_nodes = np.mean([
                len([n for n in ind.genome.nodes.values() if n.node_type == NodeType.HIDDEN])
                for ind in evolvable_pop
            ])
            
            avg_connections = np.mean([
                len([c for c in ind.genome.connections.values() if c.enabled])
                for ind in evolvable_pop
            ])
            
            self.logger.info(f"Topology stats:")
            self.logger.info(f"  Avg nodes: {avg_nodes:.1f}")
            self.logger.info(f"  Avg connections: {avg_connections:.1f}")
            self.logger.info(f"  Avg complexity: {np.mean(complexities):.3f}")
            self.logger.info(f"  Avg innovation: {np.mean(innovations):.3f}")
            self.logger.info(f"  Species count: {len(self.neuroevolution_engine.species)}")
            self.logger.info(f"  Novelty archive size: {len(self.novelty_archive)}")


class EnhancedMultiObjectiveFitness(MultiObjectiveFitness):
    """Enhanced fitness evaluation with topology objectives"""
    
    def __init__(self):
        objectives = [
            'win_rate', 'game_length', 'piece_efficiency', 
            'positional_strength', 'tactical_sharpness',
            'topology_complexity', 'innovation_score', 'novelty_score'
        ]
        super().__init__(objectives)
    
    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Enhanced dominance with topology considerations"""
        # Standard multi-objective dominance
        better_in_any = False
        
        for objective in self.objectives:
            score1 = ind1.fitness_scores.get(objective, 0)
            score2 = ind2.fitness_scores.get(objective, 0)
            
            # For complexity, prefer moderate values (not too simple, not too complex)
            if objective == 'topology_complexity':
                optimal_complexity = 0.5
                score1 = 1.0 - abs(score1 - optimal_complexity)
                score2 = 1.0 - abs(score2 - optimal_complexity)
            
            if score1 < score2:
                return False
            elif score1 > score2:
                better_in_any = True
        
        return better_in_any