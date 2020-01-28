import numpy as np

class NeuronType:
    INPUT, OUTPUT, HIDEEN, BIAS = range(4)

class Neuron:
    def __init__(self, id, neuron_type, pos = 0):
        self.id = id
        self.neuron_type = neuron_type
        self.value = 0
        self.pos = pos
        # contains dict neuron id : weight of link
        self.input_links = []

class NeuralNetwork:
    def __init__(self , genome):
        self.genome = genome
        self.layers_info = {}
        self.neurons = []
        self.input_neurons = []
        self.output_neurons = []

        # calculating number of layers
        nodes = self.genome.nodes
        nodes.sort(key = lambda x: x.layer)
        
        self.neurons = [ Neuron(i.id , i.node_type , i.pos) for i in nodes]
        self.input_neurons = list( filter(lambda x: x.neuron_type == NeuronType.INPUT , self.neurons) )
        self.output_neurons = list( filter(lambda x : x.neuron_type == NeuronType.OUTPUT , self.neurons) )

    def sigmoid(self , x):
        return 1/(1+np.exp(-4.9*x))

    def feed_forward(self , inputs):
        # sort input and output list
        self.input_neurons.sort(key = lambda x: x.pos)
        self.output_neurons.sort(key = lambda x: x.pos)

        input_len = len(inputs)
        connections = self.genome.connection_genes

        for i in range(input_len):
            self.input_neurons[i].value = inputs[i]
            
        for neuron in self.neurons[input_len:]:
            if neuron.neuron_type != NeuronType.BIAS:
                in_connections = list(filter( lambda x: x.out_node_id == neuron.id and x.enable, connections ))
                bias = 0
                neuron_value = 0
                for connection in in_connections:
                    in_neuron = list( filter( lambda x: x.id == connection.in_node_id , self.neurons ) )[0]
                    if in_neuron.neuron_type == NeuronType.BIAS:
                        bias += connection.weight
                    else:
                        neuron_value += connection.weight * in_neuron.value
                neuron.value = self.sigmoid(neuron_value+bias)
        output = [i.value for i in self.output_neurons]
        return output
    