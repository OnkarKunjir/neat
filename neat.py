from gnome import Gnome
from innovation import InnovationList
from nodegene import NodeList , NodeTyep

import numpy as np 


class NEAT:
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.nodes = NodeList()
        self.innovations = InnovationList()

        # creating initial nodes to be provided to genomes to create initial population
        inital_nodes = []
        for i in range(n_inputs):
            inital_nodes.append( self.nodes.add_node(NodeTyep.INPUT) )
        for i in range(n_outputs):
            inital_nodes.append( self.nodes.add_node(NodeTyep.OUTPUT) )
        
        self.test_gnome = Gnome(innovations = self.innovations, nodes_gen = self.nodes, nodes=inital_nodes, connection_mutation=1, node_mutation=1)

if __name__ == "__main__":
    neat = NEAT(2 , 1)

    print('-'*5 , "before mutation" , '-'*5)
    neat.innovations.print_innovations()
    #neat.nodes.print_nodes()
    # neat.test_gnome.print_connections() 
    
    print('-'*5 , "after mutation" , '-'*5)
    neat.test_gnome.mutate()
    neat.innovations.print_innovations()
    #neat.nodes.print_nodes()
    # neat.test_gnome.print_connections()    