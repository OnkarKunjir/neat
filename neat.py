from gnome import Gnome
from innovation import InnovationList
from nodegene import NodeList , NodeTyep
from neural_network import NeuralNetwork
from species import Species

import numpy as np 


class NEAT:
    def __init__(self, n_inputs, n_outputs, population_size = 100):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.population_size = population_size
        self.population = []

        self.nodes = NodeList()
        self.innovations = InnovationList()

        # creating initial nodes to be provided to genomes to create initial population
        inital_nodes = []
        for i in range(n_inputs):
            inital_nodes.append( self.nodes.add_node(NodeTyep.INPUT, pos = i))
        for i in range(n_outputs):
            inital_nodes.append( self.nodes.add_node(NodeTyep.OUTPUT, pos = i))
        

        for i in range(population_size):
            g = Gnome(innovations = self.innovations, nodes_gen = self.nodes, nodes=inital_nodes)
            node1 = np.random.choice(inital_nodes[:n_inputs])
            node2 = np.random.choice(inital_nodes[n_inputs:])
            g.mutate_connection(node1 = node1, node2 = node2)
            self.population.append(g)        
        
        self.species = Species()
        
    def train(self , fitness_func , max_generations = 100):
        for generation in range(max_generations):
            print('generation : ' , generation)
            self.species.speciate(self.population)
            self.species.calc_fitness(fitness_func = fitness_func)
            if self.get_performace() > 0.8:
                break
            self.population = self.species.natural_selection(self.population_size)
            
    def get_performace(self , return_nn = False):
        # self.species.performace()
        genome = max(self.population , key = lambda x:x.fitness)
        if return_nn:
            self.species.performace()
        
            return NeuralNetwork(genome) , genome.fitness
        return genome.fitness

if __name__ == "__main__":
    neat = NEAT(2 , 1)

    # print('-'*5 , "before mutation" , '-'*5)
    # neat.innovations.print_innovations()
    # neat.nodes.print_nodes()
    # neat.test_gnome.print_connections() 
    
    # print('-'*5 , "after mutation" , '-'*5)
    # neat.test_gnome.mutate()
    # neat.innovations.print_innovations()
    #neat.nodes.print_nodes()
    # neat.test_gnome.print_connections()    