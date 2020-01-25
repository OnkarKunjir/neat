import numpy as np
from neural_network import NeuralNetwork

class Species:
    # class to classify gnomes into various species
    def __init__(self, c1 = 1, c2 = 1, c3 = 0.4, delta_t = 3):
        self.n_species = 0
        self.species = {}

        self.representatives = {}   # representive of each species

        # constants to calculate the compactibility value
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.delta_t = delta_t

    def get_representatives(self , id):
        representative = self.representatives.get(id) 
        if representative == None:
            representative = np.random.choice(self.species[id])
            self.representatives[id] = representative
        return representative

    def update_representatives(self):
        # update representatives of all species 
        for i in list(self.species.keys()):
            self.representatives[i] = np.random.choice( self.species[i] )
            
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
        

    def speciate(self, population):
        s = 0
        self.n_species = 0
        self.species = {}

        if self.n_species == 0:
            # if there are no species
            self.n_species += 1
            self.species[self.n_species] = [population[0]]
            s = 1

        for i in range(s , len(population)):
            found_match = False
            for j in list(self.species.keys()):
                # representive = np.random.choice(self.species[j])
                representive = self.get_representatives(j)
                delta = self.compactibility(population[i] , representive)
                # print(delta)
                if delta < self.delta_t:
                    self.species[j].append(population[i])
                    found_match = True
                    break
            if not found_match:
                # if no match found make new species
                self.n_species += 1
                self.species[self.n_species] = [population[i]]
        
        self.update_representatives()

    def calc_fitness(self, fitness_func = None):
        if fitness_func:
            # population = []
            # for i in self.species.values():
            #     population += i

            for species in self.species.keys():
                for genome in self.species[species]:
                    f = fitness_func(NeuralNetwork(genome))
                    # deltas = [self.compactibility(genome , i) for i in population]
                    # den = 1
                    # for d in deltas:
                    #     if d < self.delta_t:
                    #         den += 1
                    genome.fitness = f 
    

    def performace(self):
        for i in self.species.keys():
            genomes = self.species[i]
            avg_fitness = 0
            n = 0
            for gnome in genomes:
                avg_fitness += gnome.fitness
                n += 1
            print('species : ' , int(i) , ' avarage_fitness : ' , avg_fitness/n)

    def natural_selection(self, population_size):
        population = []
        selection_prob = {}
        species_keys = list(self.species.keys())
        
        for i in species_keys:
            self.species[i].sort(key = lambda x : x.fitness)
            n = len(self.species[i])
            selection_prob[i] =  np.arange(1 , 1+n) / (n*(n+1)/2)
        

        for i in range(population_size):
            key = np.random.choice(species_keys)
            gnome = np.random.choice(self.species[key] , p = selection_prob[key])
            gnome.mutate()
            population.append(gnome)
        
        return population