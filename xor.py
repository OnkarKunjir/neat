from neat import NEAT
import numpy as np
data = [[0 , 0] , [0 , 1] , [1 , 0] , [1 , 1]]
labels = [0 , 1 , 1 , 0]

def fitness(nn , return_output = False):
    loss = 0
    output = []
    for i in range(4):
        output.append(nn.feed_forward(data[i]))
        # output = nn.feed_forward(data[i])
        loss += np.abs(output[-1][0] - labels[i])
        # loss += np.abs(np.round(output[-1][0]) - labels[i])
    
    fitness = 1 - loss/4
    if not return_output:
        return fitness
    return (fitness , output)


if __name__ == "__main__":
    neat = NEAT(2 , 1, 130)
    best_gnomes = neat.train(fitness, 50, 0.86)

    # for i in range(10):
        # neat.epoch(fitness)
    # neat.train(fitness , max_generations = 20)
    # l = fitness(neat.get_performace(return_nn=True)[0] , True)
    # print('fitness' , l)
