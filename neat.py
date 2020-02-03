from gnome import Genome
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

        # for s in self.species:
        #     for m in s.members:
        #         if np.random.rand() < 0.25:
        #             m.mutate()

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
                # g1.mutate()
                # g2.mutate()
                child = Genome.crossover(g1 , g2)
                # if len(child.connection_genes) == 0:
                #     continue
                    # child = Genome(innovations = self.innovations, nodes_gen = self.nodes, n_inputs = self.n_inputs , n_outputs = self.n_outputs)
                child.mutate()
                s.add_member(child)
                
        
        
        for s in self.species:
            self.population.extend(s.members)
            s.members[:] = []
            s.age += 1
        
        
        while len(self.population) < self.population_size:
            # create new genomes if population size is less
            g = Genome(innovations = self.innovations, nodes_gen = self.nodes, n_inputs = self.n_inputs , n_outputs = self.n_outputs)
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
            average_fitness = sum([i.fitness for i in s.members ])/len(s.members)
            if average_fitness <= s.average_fitness:
                s.generations_not_improved += 1

            s.average_fitness = average_fitness
            s.leader = deepcopy(s.members[0])
        # for s in self.species:
        #     s.members.sort(key = lambda x : x.fitness , reverse = True)
        #     s.representative = np.random.choice(s.members)
        #     s.average_fitness = sum([i.fitness for i in s.members ])/len(s.members)
        #     if s.leader.fitness < s.members[0].fitness:
        #         s.leader = deepcopy(s.members[0])
        #     else:
        #         s.generations_not_improved += 1
    
    def evaluate(self, fitness_func = None):
        if fitness_func:
            for s in self.species:
                for genome in s.members:
                    f = fitness_func(NeuralNetwork(genome))
                    # print(f)
                    genome.fitness = f 

    def compactibility(self , genome1 , genome2):
        E = D = W = 0
        conn1 = genome1.connection_genes
        conn2 = genome2.connection_genes
        
        conn1.sort(key = lambda x: x.innovation_number)
        conn2.sort(key = lambda x: x.innovation_number)
        len1 = len(conn1)
        len2 = len(conn2)

        n1 = n2 = 0
        w_len = 1
        while n1 < len1 and n2 < len2:
            c1 = conn1[n1] if n1 < len1 else None
            c2 = conn2[n2] if n2 < len2 else None
            if c1 and c2:
                if c1.innovation_number == c2.innovation_number:
                    W += np.abs( c1.weight - c2.weight )
                    w_len += 1
                    n1 += 1
                    n2 += 1
                elif c1.innovation_number < c2.innovation_number:
                    D += 1
                    n1 += 1
                else:
                    D += 1
                    n2 += 1
            elif c2 == None and c1:
                E += 1
                n1 += 1
            elif c1 == None and c2:
                E += 1
                n2 += 1

        N = max(len1 , len2)
        W /= w_len
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
            print('Generation : ' , self.generation , '\t' , best_genome.fitness)
            self.best_of_each_generation[self.generation] = NeuralNetwork(best_genome)

            # if best_genome.fitness >= max_fitness:
            #     return NeuralNetwork(best_genome)
            self.generation += 1
        return NeuralNetwork(best_genome)

if __name__ == "__main__":
    neat = NEAT(2 , 1)