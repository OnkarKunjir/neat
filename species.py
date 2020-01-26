import numpy as np
from neural_network import NeuralNetwork
from copy import deepcopy

class Species:
    def __init__(self, species_id, first_member):
        self.species_id = id
        self.members = []

        self.leader = deepcopy(first_member)
        self.leader_old = None
        self.representative = first_member
        self.generations_not_improved = 0
        self.age = 0
        self.spawns_required = 0 # number_to_spawn, spawn_amount
        self.max_fitness = 0.0
        self.average_fitness = 0.0

        self.young_age_threshhold = 10
        self.young_age_fitness_bonus = 1.3
        self.old_age_threshold = 50
        self.old_age_fitness_penalty = 0.7

        self.survival_rate = 0.2

        self.add_member(first_member)
    
    def add_member(self, member):
        member.species_id = self.species_id
        self.members.append(member)