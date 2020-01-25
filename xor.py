from neat import NEAT
import numpy as np
data = [[0 , 0] , [0 , 1] , [1 , 0] , [1 , 1]]
labels = [0 , 1 , 1 , 0]

def fitness(nn):
    loss = 0
    for i in range(4):
        output = nn.feed_forward(data[i])
        loss += np.abs(output[0] - labels[i])
    return 1 - loss/4


if __name__ == "__main__":
    neat = NEAT(2 , 1)
    neat.train(fitness , max_generations = 1)
