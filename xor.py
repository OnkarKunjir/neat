from neat import NEAT
import numpy as np
data = [[0 , 0] , [0 , 1] , [1 , 0] , [1 , 1]]
labels = [0 , 1 , 1 , 0]

def fitness(nn , return_output = False):
    loss = 0
    outputs = []
    for i in range(4):
        outputs.append( nn.feed_forward(data[i]) )
        loss += np.sqrt( (outputs[-1][0] - labels[i])**2 )
    score = 4 - loss
    if return_output:
        return (score**2 , outputs)
    return score**2
    
if __name__ == "__main__":
    neat = NEAT(2 , 1, 150)
    best_gnomes = neat.train(fitness, 100, 15)
    print(fitness(best_gnomes , True))
    best_gnomes.genome.print_connections()