import numpy as np

class ConnectionGene:
    def __init__(self ,in_node_id ,out_node_id , innovation_number , enable = True , weight = None  ):
        self.in_node_id = in_node_id
        self.out_node_id = out_node_id
        self.weight = weight
        self.enable = enable
        self.innovation_number = innovation_number
        
        if self.weight == None:
            self.weight = np.random.randn()
            # self.weight = np.random.normal(3)

    def print_connection(self):
        print("in node: ", self.in_node_id)
        print("out node: ", self.out_node_id)
        print("weight: ", self.weight)
        print("enabled: ", self.enable)
        print("innovation number: ", self.innovation_number)
        print("_"*10)