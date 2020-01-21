# TODO: add bias node type

class NodeTyep:
    INPUT , OUTPUT , HIDDEN = range(3)
class Node:
    def __init__(self , id , node_type , layer = None):
        self.id = id
        self.node_type = node_type
        self.layer = layer

        if self.layer == None:
            if node_type == NodeTyep.INPUT:
                self.layer = 0
            elif node_type == NodeTyep.OUTPUT:
                self.layer = 1

class NodeList:
    def __init__(self):
        self.global_node_id = 0
        self.nodes = []
    
    def add_node(self, node_type, layer = None):
        self.global_node_id += 1
        node = Node(id = self.global_node_id, node_type = node_type, layer = layer)
        self.nodes.append(node)
        return node
    
    def get_node(self , id):
        # returns node object from nodes list
        return self.nodes[id-1]

    def print_nodes(self):
        for i in self.nodes:
            print("-"*10)
            print("id: " , i.id)
            print("type: " , i.node_type)
            print("layer: ", i.layer)