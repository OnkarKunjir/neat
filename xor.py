from neat import NEAT
import numpy as np
data = [[0 , 0] , [0 , 1] , [1 , 0] , [1 , 1]]
labels = [0 , 0 , 0 , 0]

def fitness(nn , return_output = False):
    loss = 0
    output = []
    for i in range(4):
        output.append(nn.feed_forward(data[i]))
        # output = nn.feed_forward(data[i])
        loss += np.abs(output[-1][0] - labels[i])
        # loss += np.abs(np.round(output[-1][0]) - labels[i])
    if not return_output:
        return 1 - loss/4
    return (1-loss/4 , output)


if __name__ == "__main__":
    neat = NEAT(2 , 1)
    neat.train(fitness , max_generations = 10)
    l = fitness(neat.get_performace(return_nn=True)[0] , True)
    print('loss' , l)
