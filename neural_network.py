import numpy as np

class NeuronType:
    INPUT, OUTPUT, HIDEEN, BIAS = range(4)

class Neuron:
    def __init__(self, id, neuron_type, bias, pos = 0):
        self.id = id
        self.neuron_type = neuron_type
        self.value = 0
        self.pos = pos
        # contains dict neuron id : weight of link
        self.input_links = []
        self.bias = bias

class NeuralNetwork:
    def __init__(self , gnome):
        self.gnome = gnome
        self.layers_info = {}
        self.neurons = {}
        self.input_neurons_id = []
        self.output_neurons_id = []

        # calculating number of layers
        nodes = self.gnome.nodes
        nodes.sort(key = lambda x: x.layer)
        
        for i in nodes:
            if i.layer not in self.layers_info.keys():
                self.layers_info[i.layer] = 0
            self.layers_info[i.layer] += 1
            self.neurons[i.id] = Neuron(i.id, i.node_type, i.bias, i.pos)

            if i.node_type == NeuronType.INPUT:
                self.input_neurons_id.append(i.id)
            elif i.node_type == NeuronType.OUTPUT:
                self.output_neurons_id.append(i.id)
            
        for neuron in self.neurons.values():
            neuron.input_links = list(filter( lambda x: x.out_node_id == neuron.id and x.enable , self.gnome.connection_genes))
        self.input_neurons_id.sort(key = lambda x: self.neurons[x].pos)
        self.output_neurons_id.sort(key = lambda x: self.neurons[x].pos)

        for i in range(100):
            self.feed_forward([1,1])

    def sigmoid(self , x):
        return 1/(1+np.exp(-x))

    def feed_forward(self , inputs):
        
        # performing the computation
        n_inputs = len(inputs)
        for i in range(n_inputs):
            self.neurons[self.input_neurons_id[i]].value = inputs[i]

        for neuron in list(self.neurons.values())[n_inputs : ]:
            total = 0
            for connection in neuron.input_links:
                total += connection.weight * self.neurons[connection.in_node_id].value
            neuron.value = self.sigmoid(total + neuron.bias)
            
        output = [ self.neurons[i].value for i in self.output_neurons_id] 
        return output