class InnovationType:
    CONNECTION , ADD_NODE = range(2)

class Innovation:
    def __init__(self, in_node_id, out_node_id, innovation_number, innovation_type, inserted_node_id = None):
        self.in_node_id = in_node_id 
        self.out_node_id = out_node_id
        self.innovation_number = innovation_number
        self.innovation_type = innovation_type
        self.inserted_node_id = inserted_node_id    # if innovation type is ADD_NODE then specify inserted nodes id

class InnovationList:
    def __init__(self):
        self.innovations = []
        self.global_innovation_number = 0
    
    def exists_innovation(self , in_node_id , out_node_id , innovation_type):
        for i in self.innovations:
            if i.innovation_type == innovation_type and i.in_node_id == in_node_id and i.out_node_id == out_node_id:
                return i
        return None

    def get_innovation(self, in_node_id, out_node_id, innovation_type , inserted_node_id = None):
        # returns innovation object if found in innovations list
        # else creates new innovation appends to list and returns new innovation object
        for i in self.innovations:
            if i.in_node_id == in_node_id and i.out_node_id == out_node_id and i.innovation_type == innovation_type:
                return i
        # create new object not found
        innovation = Innovation(in_node_id = in_node_id , out_node_id = out_node_id , innovation_number = self.global_innovation_number, innovation_type = innovation_type, inserted_node_id = inserted_node_id)
        self.innovations.append(innovation)
        self.global_innovation_number += 1
        return innovation

    def print_innovations(self):
        types = ["connection" , "add_node" , "disable"]
        for i in self.innovations:
            print("-"*10)
            print("in node: " , i.in_node_id)
            print("out node: " , i.out_node_id)
            print("innovation number: " , i.innovation_number)
            print("type: ", types[i.innovation_type])
            print("inserted node: ", i.inserted_node_id)
    