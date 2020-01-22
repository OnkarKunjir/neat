import numpy as np

class NeuronType:
    INPUT, OUTPUT, HIDEEN = range(3)

class Neuron:
    def __init__(self, id, neuron_type):
        self.id = id
        self.neuron_type = neuron_type
        self.value = 0

        # contains dict neuron id : weight of link
        self.input_links = []

class NeuralNetwork:
    def __init__(self , gnome):
        self.gnome = gnome
        self.layers_info = {}
        self.neurons = {}
        
        self.gnome.mutate_node()
        # calculating number of layers
        nodes = self.gnome.nodes
        nodes.sort(key = lambda x: x.layer)
        
        for i in nodes:
            if i.layer not in self.layers_info.keys():
                self.layers_info[i.layer] = 0
            self.layers_info[i.layer] += 1
            self.neurons[i.id] = Neuron(i.id, i.node_type)
            
        for neuron in self.neurons.values():
            neuron.input_links = list(filter( lambda x: x.out_node_id == neuron.id and x.enable , self.gnome.connection_genes))

        self.feed_forward([0 , 1])
    def feed_forward(self , inputs):
        
        # performing the computation
        i_input = 0
        output = []
        for neuron in self.neurons.values():
            if neuron.neuron_type == NeuronType.INPUT:
                neuron.value = inputs[i_input]
                i_input += 1
            
            else:
                # TODO: add activation function, returning output in proper order
                
                total = 0
                for connection in neuron.input_links:
                    total += connection.weight * self.neurons[connection.in_node_id].value
                neuron.value = total
                if neuron.neuron_type == NeuronType.OUTPUT:
                    output.append(neuron.value)
        return output