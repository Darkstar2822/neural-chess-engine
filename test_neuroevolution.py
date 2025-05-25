"""
Test the enhanced neuroevolution capabilities
"""

import sys
sys.path.append('.')

import logging
import torch
from pathlib import Path
from src.evolution.enhanced_evolutionary_engine import EnhancedEvolutionaryEngine

def test_neuroevolution():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("Starting enhanced neuroevolution test...")
    
    # Initialize enhanced evolutionary engine
    engine = EnhancedEvolutionaryEngine(
        population_size=6,  # Small for quick test
        max_generations=3,  # Just 3 generations
        tournament_size=3,
        mutation_rate=0.2,
        use_topology_evolution=True
    )
    
    print("Initializing evolvable population...")
    engine.initialize_population()
    
    print(f"Population initialized with {len(engine.population)} evolvable individuals")
    
    # Show initial topology diversity
    for i, individual in enumerate(engine.population):
        if hasattr(individual, 'genome'):
            num_nodes = len([n for n in individual.genome.nodes.values() 
                           if n.node_type.value == 'hidden'])
            num_connections = len([c for c in individual.genome.connections.values() 
                                 if c.enabled])
            print(f"Individual {i}: {num_nodes} hidden nodes, {num_connections} connections")
    
    # Run evolution
    print("Running neuroevolution...")
    final_population = engine.evolve()
    
    print(f"Evolution complete! Final population size: {len(final_population)}")
    
    # Show final results
    for i, individual in enumerate(final_population):
        complexity = individual.fitness_scores.get('topology_complexity', 0)
        innovation = individual.fitness_scores.get('innovation_score', 0)
        novelty = individual.fitness_scores.get('novelty_score', 0)
        win_rate = individual.fitness_scores.get('win_rate', 0)
        
        print(f"Individual {i}: WR={win_rate:.3f}, "
              f"Complex={complexity:.3f}, Innov={innovation:.3f}, Novel={novelty:.3f}")

if __name__ == "__main__":
    test_neuroevolution()