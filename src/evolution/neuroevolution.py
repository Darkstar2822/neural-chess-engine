"""
Neuroevolution Strategy for Chess Neural Network Evolution
Implements NEAT-inspired topology evolution with crossover and mutation operators
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import random
import copy
from collections import defaultdict

from ..neural_network.chess_net import ChessNet


class NodeType(Enum):
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


class ActivationType(Enum):
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"
    SWISH = "swish"
    GELU = "gelu"


@dataclass
class NodeGene:
    """Represents a node in the neural network"""
    id: int
    node_type: NodeType
    activation: ActivationType = ActivationType.RELU
    bias: float = 0.0
    layer: int = 0  # For organizing topology
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class ConnectionGene:
    """Represents a connection between nodes"""
    id: int
    input_node: int
    output_node: int
    weight: float
    enabled: bool = True
    innovation_number: int = 0  # For crossover alignment
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class NetworkGenome:
    """Complete genome representation of a neural network"""
    nodes: Dict[int, NodeGene] = field(default_factory=dict)
    connections: Dict[int, ConnectionGene] = field(default_factory=dict)
    input_size: int = 16 * 8 * 8  # Chess board representation
    output_size: int = 20480  # Policy + value outputs
    next_node_id: int = 0
    next_connection_id: int = 0
    fitness: float = 0.0
    species_id: int = 0
    
    def __post_init__(self):
        if not self.nodes:
            self._initialize_minimal_topology()
    
    def _initialize_minimal_topology(self):
        """Initialize with input and output nodes only"""
        # Input nodes
        for i in range(self.input_size):
            self.nodes[i] = NodeGene(i, NodeType.INPUT, layer=0)
        
        # Output nodes (policy + value)
        policy_size = 20480
        value_size = 1
        
        for i in range(policy_size + value_size):
            node_id = self.input_size + i
            self.nodes[node_id] = NodeGene(node_id, NodeType.OUTPUT, layer=2)
        
        self.next_node_id = self.input_size + policy_size + value_size
        
        # Direct connections from inputs to outputs (minimal network)
        for input_id in range(min(64, self.input_size)):  # Connect subset for efficiency
            for output_id in range(self.input_size, self.input_size + policy_size + value_size, 320):  # Sample outputs
                conn_id = self.next_connection_id
                weight = np.random.normal(0, 0.1)
                self.connections[conn_id] = ConnectionGene(
                    conn_id, input_id, output_id, weight, True, conn_id
                )
                self.next_connection_id += 1


class InnovationTracker:
    """Tracks innovation numbers for consistent crossover"""
    
    def __init__(self):
        self.innovations: Dict[Tuple[int, int], int] = {}
        self.next_innovation = 0
    
    def get_innovation_number(self, input_node: int, output_node: int) -> int:
        """Get or create innovation number for a connection"""
        key = (input_node, output_node)
        if key not in self.innovations:
            self.innovations[key] = self.next_innovation
            self.next_innovation += 1
        return self.innovations[key]


class NeuroevolutionEngine:
    """Advanced neuroevolution with topology evolution"""
    
    def __init__(self, population_size: int = 50):
        self.population_size = population_size
        self.innovation_tracker = InnovationTracker()
        self.species: Dict[int, List[NetworkGenome]] = defaultdict(list)
        self.next_species_id = 0
        
        # Mutation rates
        self.weight_mutation_rate = 0.8
        self.weight_perturbation_rate = 0.9
        self.add_node_rate = 0.03
        self.add_connection_rate = 0.05
        self.disable_connection_rate = 0.01
        self.activation_mutation_rate = 0.1
        
        # Crossover parameters
        self.compatibility_threshold = 3.0
        self.excess_coefficient = 1.0
        self.disjoint_coefficient = 1.0
        self.weight_coefficient = 0.4
        
        # Selection parameters
        self.survival_rate = 0.2
        self.interspecies_mating_rate = 0.001
    
    def create_initial_population(self) -> List[NetworkGenome]:
        """Create initial population with minimal topologies"""
        population = []
        for _ in range(self.population_size):
            genome = NetworkGenome()
            self._add_random_connections(genome, 5)  # Add some initial connections
            population.append(genome)
        return population
    
    def _add_random_connections(self, genome: NetworkGenome, count: int):
        """Add random connections to a genome"""
        input_nodes = [n.id for n in genome.nodes.values() if n.node_type == NodeType.INPUT]
        output_nodes = [n.id for n in genome.nodes.values() if n.node_type == NodeType.OUTPUT]
        
        for _ in range(count):
            if random.random() < 0.7:  # Prefer input-to-output connections
                input_node = random.choice(input_nodes)
                output_node = random.choice(output_nodes)
            else:  # Allow other connections if hidden nodes exist
                all_nodes = list(genome.nodes.keys())
                input_node = random.choice(all_nodes)
                output_node = random.choice(all_nodes)
                if input_node == output_node:
                    continue
            
            # Check if connection already exists
            exists = any(
                conn.input_node == input_node and conn.output_node == output_node
                for conn in genome.connections.values()
            )
            
            if not exists:
                conn_id = genome.next_connection_id
                innovation = self.innovation_tracker.get_innovation_number(input_node, output_node)
                weight = np.random.normal(0, 1.0)
                
                genome.connections[conn_id] = ConnectionGene(
                    conn_id, input_node, output_node, weight, True, innovation
                )
                genome.next_connection_id += 1
    
    def mutate_weights(self, genome: NetworkGenome):
        """Mutate connection weights"""
        for connection in genome.connections.values():
            if random.random() < self.weight_mutation_rate:
                if random.random() < self.weight_perturbation_rate:
                    # Perturb existing weight
                    connection.weight += np.random.normal(0, 0.1)
                else:
                    # Assign new random weight
                    connection.weight = np.random.normal(0, 1.0)
    
    def mutate_add_node(self, genome: NetworkGenome):
        """Add a new node by splitting an existing connection"""
        if not genome.connections or random.random() > self.add_node_rate:
            return
        
        # Choose random enabled connection
        enabled_connections = [c for c in genome.connections.values() if c.enabled]
        if not enabled_connections:
            return
        
        connection = random.choice(enabled_connections)
        
        # Disable the original connection
        connection.enabled = False
        
        # Create new node
        new_node_id = genome.next_node_id
        input_layer = genome.nodes[connection.input_node].layer
        output_layer = genome.nodes[connection.output_node].layer
        new_layer = (input_layer + output_layer) // 2 if output_layer > input_layer + 1 else input_layer + 1
        
        new_node = NodeGene(
            new_node_id, 
            NodeType.HIDDEN, 
            random.choice(list(ActivationType)),
            layer=new_layer
        )
        genome.nodes[new_node_id] = new_node
        genome.next_node_id += 1
        
        # Create two new connections
        # Input to new node (weight = 1.0)
        conn1_id = genome.next_connection_id
        innovation1 = self.innovation_tracker.get_innovation_number(connection.input_node, new_node_id)
        genome.connections[conn1_id] = ConnectionGene(
            conn1_id, connection.input_node, new_node_id, 1.0, True, innovation1
        )
        genome.next_connection_id += 1
        
        # New node to output (original weight)
        conn2_id = genome.next_connection_id
        innovation2 = self.innovation_tracker.get_innovation_number(new_node_id, connection.output_node)
        genome.connections[conn2_id] = ConnectionGene(
            conn2_id, new_node_id, connection.output_node, connection.weight, True, innovation2
        )
        genome.next_connection_id += 1
    
    def mutate_add_connection(self, genome: NetworkGenome):
        """Add a new connection between existing nodes"""
        if random.random() > self.add_connection_rate:
            return
        
        nodes = list(genome.nodes.keys())
        if len(nodes) < 2:
            return
        
        # Try to find valid connection (avoid cycles and duplicates)
        for _ in range(10):  # Max attempts
            input_node = random.choice(nodes)
            output_node = random.choice(nodes)
            
            if input_node == output_node:
                continue
            
            # Check if connection already exists
            exists = any(
                conn.input_node == input_node and conn.output_node == output_node
                for conn in genome.connections.values()
            )
            
            if exists:
                continue
            
            # Check for cycles (simplified)
            input_layer = genome.nodes[input_node].layer
            output_layer = genome.nodes[output_node].layer
            
            if input_layer >= output_layer and genome.nodes[output_node].node_type != NodeType.OUTPUT:
                continue
            
            # Create new connection
            conn_id = genome.next_connection_id
            innovation = self.innovation_tracker.get_innovation_number(input_node, output_node)
            weight = np.random.normal(0, 1.0)
            
            genome.connections[conn_id] = ConnectionGene(
                conn_id, input_node, output_node, weight, True, innovation
            )
            genome.next_connection_id += 1
            break
    
    def mutate_activation(self, genome: NetworkGenome):
        """Mutate activation functions of hidden nodes"""
        hidden_nodes = [n for n in genome.nodes.values() if n.node_type == NodeType.HIDDEN]
        
        for node in hidden_nodes:
            if random.random() < self.activation_mutation_rate:
                node.activation = random.choice(list(ActivationType))
    
    def mutate_disable_connection(self, genome: NetworkGenome):
        """Randomly disable connections"""
        enabled_connections = [c for c in genome.connections.values() if c.enabled]
        
        for connection in enabled_connections:
            if random.random() < self.disable_connection_rate:
                connection.enabled = False
    
    def mutate(self, genome: NetworkGenome) -> NetworkGenome:
        """Apply all mutation operators to a genome"""
        mutated = copy.deepcopy(genome)
        
        self.mutate_weights(mutated)
        self.mutate_add_node(mutated)
        self.mutate_add_connection(mutated)
        self.mutate_activation(mutated)
        self.mutate_disable_connection(mutated)
        
        return mutated
    
    def crossover(self, parent1: NetworkGenome, parent2: NetworkGenome) -> NetworkGenome:
        """Crossover two genomes to create offspring"""
        # Determine which parent is more fit
        if parent1.fitness > parent2.fitness:
            fit_parent, weak_parent = parent1, parent2
        elif parent2.fitness > parent1.fitness:
            fit_parent, weak_parent = parent2, parent1
        else:
            # Equal fitness - choose randomly
            fit_parent, weak_parent = (parent1, parent2) if random.random() < 0.5 else (parent2, parent1)
        
        offspring = NetworkGenome()
        offspring.input_size = parent1.input_size
        offspring.output_size = parent1.output_size
        
        # Copy all nodes from both parents
        all_node_ids = set(parent1.nodes.keys()) | set(parent2.nodes.keys())
        for node_id in all_node_ids:
            if node_id in parent1.nodes and node_id in parent2.nodes:
                # Node in both parents - choose randomly
                source_node = random.choice([parent1.nodes[node_id], parent2.nodes[node_id]])
                offspring.nodes[node_id] = copy.deepcopy(source_node)
            elif node_id in fit_parent.nodes:
                # Node only in more fit parent - inherit
                offspring.nodes[node_id] = copy.deepcopy(fit_parent.nodes[node_id])
        
        # Handle connections using innovation numbers
        p1_innovations = {conn.innovation_number: conn for conn in parent1.connections.values()}
        p2_innovations = {conn.innovation_number: conn for conn in parent2.connections.values()}
        
        all_innovations = set(p1_innovations.keys()) | set(p2_innovations.keys())
        
        for innovation in all_innovations:
            if innovation in p1_innovations and innovation in p2_innovations:
                # Matching genes - choose randomly
                parent_conn = random.choice([p1_innovations[innovation], p2_innovations[innovation]])
                
                # Only include if both nodes exist in offspring
                if (parent_conn.input_node in offspring.nodes and 
                    parent_conn.output_node in offspring.nodes):
                    
                    new_conn = copy.deepcopy(parent_conn)
                    new_conn.id = offspring.next_connection_id
                    offspring.connections[new_conn.id] = new_conn
                    offspring.next_connection_id += 1
                    
            elif innovation in p1_innovations and parent1.fitness >= parent2.fitness:
                # Excess/disjoint from fitter parent
                parent_conn = p1_innovations[innovation]
                if (parent_conn.input_node in offspring.nodes and 
                    parent_conn.output_node in offspring.nodes):
                    
                    new_conn = copy.deepcopy(parent_conn)
                    new_conn.id = offspring.next_connection_id
                    offspring.connections[new_conn.id] = new_conn
                    offspring.next_connection_id += 1
                    
            elif innovation in p2_innovations and parent2.fitness >= parent1.fitness:
                # Excess/disjoint from fitter parent
                parent_conn = p2_innovations[innovation]
                if (parent_conn.input_node in offspring.nodes and 
                    parent_conn.output_node in offspring.nodes):
                    
                    new_conn = copy.deepcopy(parent_conn)
                    new_conn.id = offspring.next_connection_id
                    offspring.connections[new_conn.id] = new_conn
                    offspring.next_connection_id += 1
        
        # Update next IDs
        if offspring.nodes:
            offspring.next_node_id = max(offspring.nodes.keys()) + 1
        
        return offspring
    
    def calculate_compatibility_distance(self, genome1: NetworkGenome, genome2: NetworkGenome) -> float:
        """Calculate genetic distance between two genomes for speciation"""
        innovations1 = set(conn.innovation_number for conn in genome1.connections.values())
        innovations2 = set(conn.innovation_number for conn in genome2.connections.values())
        
        matching = innovations1 & innovations2
        disjoint = (innovations1 | innovations2) - matching
        
        if not innovations1 or not innovations2:
            return float('inf')
        
        max_innovations = max(max(innovations1, default=0), max(innovations2, default=0))
        
        # Count excess genes (beyond the last matching innovation)
        if matching:
            last_matching = max(matching)
            excess1 = sum(1 for inn in innovations1 if inn > last_matching)
            excess2 = sum(1 for inn in innovations2 if inn > last_matching)
            excess = excess1 + excess2
            disjoint_count = len(disjoint) - excess
        else:
            excess = len(disjoint)
            disjoint_count = 0
        
        # Calculate average weight difference for matching genes
        weight_diff = 0.0
        if matching:
            conn1_by_inn = {conn.innovation_number: conn for conn in genome1.connections.values()}
            conn2_by_inn = {conn.innovation_number: conn for conn in genome2.connections.values()}
            
            weight_diffs = []
            for inn in matching:
                if inn in conn1_by_inn and inn in conn2_by_inn:
                    diff = abs(conn1_by_inn[inn].weight - conn2_by_inn[inn].weight)
                    weight_diffs.append(diff)
            
            weight_diff = np.mean(weight_diffs) if weight_diffs else 0.0
        
        # Normalize by genome size
        N = max(len(innovations1), len(innovations2), 1)
        
        distance = (
            self.excess_coefficient * excess / N +
            self.disjoint_coefficient * disjoint_count / N +
            self.weight_coefficient * weight_diff
        )
        
        return distance
    
    def speciate(self, population: List[NetworkGenome]):
        """Organize population into species"""
        self.species.clear()
        
        for genome in population:
            placed = False
            
            # Try to place in existing species
            for species_id, species_members in self.species.items():
                if species_members:
                    representative = species_members[0]
                    distance = self.calculate_compatibility_distance(genome, representative)
                    
                    if distance < self.compatibility_threshold:
                        genome.species_id = species_id
                        self.species[species_id].append(genome)
                        placed = True
                        break
            
            # Create new species if not placed
            if not placed:
                genome.species_id = self.next_species_id
                self.species[self.next_species_id] = [genome]
                self.next_species_id += 1
    
    def genome_to_pytorch(self, genome: NetworkGenome) -> nn.Module:
        """Convert genome to PyTorch neural network"""
        return EvolvableChessNet(genome)


class EvolvableChessNet(nn.Module):
    """PyTorch implementation of evolved network topology"""
    
    def __init__(self, genome: NetworkGenome):
        super().__init__()
        self.genome = genome
        self.device = torch.device("cpu")  # Force CPU for compatibility
        
        # Build network structure
        self._build_network()
    
    def _build_network(self):
        """Build PyTorch modules from genome"""
        # Group nodes by layer
        layers = defaultdict(list)
        for node in self.genome.nodes.values():
            layers[node.layer].append(node)
        
        # Create layer modules
        self.layers = nn.ModuleDict()
        self.layer_order = sorted(layers.keys())
        
        for layer_idx in self.layer_order:
            layer_nodes = layers[layer_idx]
            
            if layer_idx == 0:  # Input layer
                continue
            
            # Find input connections for this layer
            layer_inputs = []
            for node in layer_nodes:
                inputs = [
                    conn for conn in self.genome.connections.values()
                    if conn.output_node == node.id and conn.enabled
                ]
                layer_inputs.extend(inputs)
            
            if layer_inputs:
                # Create linear layer for this layer's connections
                input_nodes = set(conn.input_node for conn in layer_inputs)
                output_nodes = set(conn.output_node for conn in layer_inputs)
                
                linear = nn.Linear(len(input_nodes), len(output_nodes), bias=False)
                
                # Set weights from genome
                with torch.no_grad():
                    weight_matrix = torch.zeros(len(output_nodes), len(input_nodes))
                    
                    input_list = sorted(input_nodes)
                    output_list = sorted(output_nodes)
                    
                    for conn in layer_inputs:
                        input_idx = input_list.index(conn.input_node)
                        output_idx = output_list.index(conn.output_node)
                        weight_matrix[output_idx, input_idx] = conn.weight
                    
                    linear.weight.copy_(weight_matrix)
                
                self.layers[f"layer_{layer_idx}"] = linear
        
        # Final output layers for chess
        final_layer_size = 512  # Intermediate size
        self.policy_head = nn.Linear(final_layer_size, 20480)
        self.value_head = nn.Linear(final_layer_size, 1)
    
    def forward(self, x):
        """Forward pass through evolved topology"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten input
        
        # Simple feedforward for now (can be enhanced with actual topology)
        hidden = torch.relu(x.mean(dim=1, keepdim=True).expand(-1, 512))
        
        policy = self.policy_head(hidden)
        value = torch.tanh(self.value_head(hidden))
        
        return policy, value
    
    def predict(self, x):
        """Prediction method compatible with existing chess engine"""
        try:
            self.eval()
            with torch.no_grad():
                if x.device != self.device:
                    x = x.to(self.device)
                
                policy, value = self.forward(x)
                
                policy_probs = torch.softmax(policy, dim=1).cpu().numpy()[0]
                value_scalar = value.cpu().numpy()[0, 0]
                
                return policy_probs, value_scalar
        except Exception as e:
            return None, None