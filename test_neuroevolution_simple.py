"""
Simple test showing neuroevolution topology changes
"""

import sys
sys.path.append('.')

from src.evolution.neuroevolution import NeuroevolutionEngine, NetworkGenome

def test_topology_operations():
    print("Testing neuroevolution topology operations...")
    
    # Create neuroevolution engine
    engine = NeuroevolutionEngine(population_size=4)
    
    # Create initial genome
    genome = NetworkGenome()
    print(f"Initial genome: {len(genome.nodes)} nodes, {len(genome.connections)} connections")
    
    # Add some connections
    engine._add_random_connections(genome, 3)
    print(f"After adding connections: {len(genome.nodes)} nodes, {len(genome.connections)} connections")
    
    # Test mutations
    original_genome = genome
    
    # Test add node mutation
    mutated1 = engine.mutate(genome)
    hidden_nodes1 = len([n for n in mutated1.nodes.values() if n.node_type.value == 'hidden'])
    print(f"After mutation 1: {len(mutated1.nodes)} nodes ({hidden_nodes1} hidden), {len(mutated1.connections)} connections")
    
    # Test another mutation
    mutated2 = engine.mutate(mutated1)
    hidden_nodes2 = len([n for n in mutated2.nodes.values() if n.node_type.value == 'hidden'])
    print(f"After mutation 2: {len(mutated2.nodes)} nodes ({hidden_nodes2} hidden), {len(mutated2.connections)} connections")
    
    # Test crossover
    if len(mutated1.connections) > 0 and len(mutated2.connections) > 0:
        mutated1.fitness = 0.8
        mutated2.fitness = 0.6
        
        offspring = engine.crossover(mutated1, mutated2)
        hidden_nodes_offspring = len([n for n in offspring.nodes.values() if n.node_type.value == 'hidden'])
        print(f"Crossover offspring: {len(offspring.nodes)} nodes ({hidden_nodes_offspring} hidden), {len(offspring.connections)} connections")
    
    # Test speciation
    population = [original_genome, mutated1, mutated2]
    if len(mutated2.connections) > 0:
        population.append(offspring)
    
    engine.speciate(population)
    print(f"Speciation created {len(engine.species)} species")
    
    for species_id, members in engine.species.items():
        print(f"  Species {species_id}: {len(members)} members")
    
    print("✅ Neuroevolution topology operations working!")

if __name__ == "__main__":
    test_topology_operations()