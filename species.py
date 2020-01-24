import numpy as np
class Species:
    def __init__(self, population, c1 = 1, c2 = 1, c3 = 0.4, delta_t = 3):
        self.n_species = 0
        self.species = {}

        self.representatives = {}

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.delta_t = delta_t
    
    def get_representatives(self , id):
        representative = self.representatives.get(id) 
        if representative:
            representative = np.random.choice(self.species[id])
            self.representatives[id] = representative
        return representative

    def update_representatives(self):
        for i in list(self.species.keys()):
            self.representatives[i] = np.random.choice( self.species[i] )
            
    def compactibility(self , gnome1 , gnome2):
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
        if self.n_species == 0:
            self.n_species += 1
            self.species[self.n_species] = [population[0]]
            s = 1

        for i in range(s , len(population)):
            found_match = False
            for j in list(self.species.keys()):
                representive = np.random.choice(self.species[j])
                delta = self.compactibility(population[i] , representive)
                # print(delta)
                if delta < self.delta_t:
                    self.species[j].append(population[i])
                    found_match = True
                    break
            if not found_match:
                self.n_species += 1
                self.species[self.n_species] = [population[i]]
        
        self.update_representatives()

    def calc_fitness(self):
        # TODO: calculate fitness of each species
        pass