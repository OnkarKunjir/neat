from gnome import Gnome
from innovation import InnovationList
from nodegene import NodeList , NodeTyep
from neural_network import NeuralNetwork
from species import Species

from copy import deepcopy
import numpy as np 


class NEAT:
    def __init__(self, n_inputs, n_outputs, population_size = 100, c1 = 1, c2 = 1, c3 = 0.4, delta_t = 3):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.population_size = population_size
        self.population = []
        
        self.n_species = 0
        self.species = []

        self.nodes = NodeList()
        self.innovations = InnovationList()
        self.number_generation_allowed_to_not_improve = 20

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.delta_t = delta_t
        self.max_tries_to_search = 3
        
        self.best_of_each_generation = {}
        self.generation = 0

        # creating initial nodes to be provided to genomes to create initial population
        self.inital_nodes = []
        for i in range(n_inputs):
            self.inital_nodes.append( self.nodes.add_node(NodeTyep.INPUT, pos = i))
        for i in range(n_outputs):
            self.inital_nodes.append( self.nodes.add_node(NodeTyep.OUTPUT, pos = i))
    
    def epoch(self, fitness_func):
        self.population[:] = []
        total_average = sum(s.average_fitness for s in self.species)
        for s in self.species:
            s.spawns_required = int(round(self.population_size * s.average_fitness / total_average))

        species = []
        
        for s in self.species:
            # remove the species which are not improving 
            if s.generations_not_improved < self.number_generation_allowed_to_not_improve and s.spawns_required > 0:
                species.append(s)
        
        self.species[:] = species

        for s in self.species:
            for m in s.members:
                if np.random.rand() < 0.25:
                    m.mutate()

        for s in self.species:
            # performing cross over
            # process of natural selection
            k = max(1 , round(len(s.members)*s.survival_rate))
            mating_pool = s.members[:k] # members list is sorted every time fitness is caluclated (descending order)
            s.members[:] = []
            # s.members = s.members[:k]
            while len(s.members) < s.spawns_required:
                # if species has less members then crossover two randomly selected parents
                n = min(len(mating_pool), self.max_tries_to_search)
                g1 = self.tournament_selection(mating_pool, n)
                g2 = self.tournament_selection(mating_pool, n)
                g1.mutate()
                g2.mutate()
                child = Gnome.crossover(g1 , g2)
                # child.mutate()
                s.add_member(child)
        
        
        for s in self.species:
            self.population.extend(s.members)
            s.members[:] = []
            s.age += 1
        
        
        while len(self.population) < self.population_size:
            # create new genomes if population size is less
            g = Gnome(innovations = self.innovations, nodes_gen = self.nodes, nodes=self.inital_nodes)
            node1 = np.random.choice(self.inital_nodes[:self.n_inputs])
            node2 = np.random.choice(self.inital_nodes[self.n_inputs:])
            g.mutate_connection(node1 = node1, node2 = node2)
            self.population.append(g)
        
        for i in self.population:
            # catagorize population in species
            matched = False
            for s in self.species:
                delta = self.compactibility(i , s.representative)
                if delta < self.delta_t:
                    s.add_member(i)
                    matched = True
                    break
            if not matched:
                s = Species(self.n_species, i)
                self.n_species += 1
                self.species.append(s)

        #calculate fitness of each member in species
        self.evaluate(fitness_func = fitness_func)

        # delete species with no members
        self.species[:] = filter(lambda s: len(s.members) > 0, self.species)

        for s in self.species:
            s.members.sort(key = lambda x : x.fitness , reverse = True)
            s.representative = np.random.choice(s.members)
            s.average_fitness = sum([i.fitness for i in s.members ])/len(s.members)
            if s.leader.fitness < s.members[0].fitness:
                s.leader = deepcopy(s.members[0])
            else:
                s.generations_not_improved += 1
    
    def evaluate(self, fitness_func = None):
        if fitness_func:
            for s in self.species:
                for genome in s.members:
                    f = fitness_func(NeuralNetwork(genome))
                    genome.fitness = f 

    def compactibility(self , gnome1 , gnome2):
        # returns compactibility value (delta)
        E = D = W = 0
        len1 = len(gnome1.connection_genes)
        len2 = len(gnome2.connection_genes)
        E = np.abs(len1 - len2)
        
        if len1 > len2:
            N = len1
            for i in range(len1):
                temp = gnome2.get_connection(gnome1.connection_genes[i].innovation_number)
                if temp:
                    W += np.abs(gnome1.connection_genes[i].weight - temp.weight)
                else:
                    D += 1
        else:
            N = len2
            for i in range(len2):
                temp = gnome1.get_connection(gnome2.connection_genes[i].innovation_number)
                if temp:
                    W += np.abs(gnome2.connection_genes[i].weight - temp.weight)
                else:
                    D += 1
        
        delta = (self.c1*E)/N + (self.c2*D)/N + self.c3*W
        return delta

            
    def tournament_selection(self, genomes, number_to_compare):
        champion = None
        for _ in range(number_to_compare):
            # g = genomes[randint(0,len(genomes)-1)]
            g = np.random.choice(genomes)
            if champion == None or g.fitness > champion.fitness:
                champion = g
        return champion

    def get_best_genome(self):
        leaders = [i.leader for i in self.species]
        return max(leaders , key = lambda x : x.fitness)

    def train(self, fitness_func, max_generations = 30, max_fitness = 1):
        self.generation = 0
        for i in range(max_generations):
            self.epoch(fitness_func)
            best_genome = self.get_best_genome()
            print(best_genome.fitness)
            self.best_of_each_generation[self.generation] = NeuralNetwork(best_genome)

            if best_genome.fitness >= max_fitness:
                print(fitness_func(NeuralNetwork(best_genome) , return_output=True))

                break
            self.generation += 1
        return self.best_of_each_generation

if __name__ == "__main__":
    neat = NEAT(2 , 1)