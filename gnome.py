import numpy as np
from connectiongene import ConnectionGene
from innovation import InnovationType
from nodegene import NodeTyep

from copy import deepcopy
#TODO add mutation without recursion

class Gnome:
	def __init__(self, innovations, nodes_gen, nodes = None, weight_mutation  = 0.8, weight_reset_prob = 0.1, connection_mutation = 0.05, node_mutation = 0.03, attempt_to_find_unlinked_nodes = 5):
		self.connection_genes = []	# list of all connection in gnome
		self.nodes = nodes	# list of generated nodes

		self.innovations = innovations	#InnovationList object
		self.node_gen = nodes_gen	#NodeList object

		self.fitness = 0
		# parameters for mutation
		self.weight_mutation = weight_mutation	
		self.weight_reset_prob = weight_reset_prob
		self.connection_mutation = connection_mutation
		self.node_mutation = node_mutation
		self.attempt_to_find_unlinked_nodes = attempt_to_find_unlinked_nodes
		
		# self.mutate_connection() # creating initial connection

	def is_connected(self, node1, node2):
		#function to find if two nodes are connected or not
		for i in self.connection_genes:
			if i.in_node_id == node1 and i.out_node_id == node2:
				return True
		return False
	
	def find_node(self , id):
		for i in self.nodes:
			if i.id == id:
				return i
		return None

	def mutate_connection(self, node1 = None, node2 = None):
		#function which adds new connection between 2 unconnected nodes and updates connection gene
		if node1:
			if not self.is_connected(node1.id , node2.id) and (node1.layer < node2.layer):
				innovation_number = self.innovations.get_innovation(node1.id, node2.id, innovation_type = InnovationType.CONNECTION).innovation_number
				self.connection_genes.append( ConnectionGene(in_node_id = node1.id, out_node_id=node2.id, enable = True, innovation_number = innovation_number) )
				return True
				
		for i in range(self.attempt_to_find_unlinked_nodes):
			node1 = np.random.choice(self.nodes)
			node2 = np.random.choice(self.nodes)
			if not self.is_connected(node1.id , node2.id) and (node1.layer < node2.layer):
				innovation_number = self.innovations.get_innovation(node1.id, node2.id, innovation_type = InnovationType.CONNECTION).innovation_number
				self.connection_genes.append( ConnectionGene(in_node_id = node1.id, out_node_id=node2.id, enable = True, innovation_number = innovation_number) )
				return True
		return False
	
	def mutate_node(self):
		connection = np.random.choice(self.connection_genes)
		in_node = self.find_node(connection.in_node_id)
		out_node = self.find_node(connection.out_node_id)

		# get innovation to insert node if not create new innovation
		innovation = self.innovations.get_innovation(in_node_id = in_node.id, out_node_id = out_node.id, innovation_type = InnovationType.ADD_NODE)
		inserted_node = None
		if innovation.inserted_node_id == None:
			# create new node to be inserted
			inserted_node = self.node_gen.add_node(node_type = NodeTyep.HIDDEN, layer = (in_node.layer + out_node.layer)/2)
			innovation.inserted_node_id = inserted_node.id
			connection.enable = False
		
		else:
			# get old node from innovation
			inserted_node = self.node_gen.get_node(innovation.inserted_node_id)
		self.nodes.append(inserted_node)
		#adding connection between in_node and inserted node
		i = self.innovations.get_innovation(in_node_id = in_node.id, out_node_id = inserted_node.id, innovation_type = InnovationType.CONNECTION)
		self.connection_genes.append(ConnectionGene(in_node_id = in_node.id, out_node_id = inserted_node.id, enable = True, innovation_number = i.innovation_number, weight = 1))
		
		#adding connection between inserted node and out_node
		i = self.innovations.get_innovation(in_node_id = inserted_node.id, out_node_id = out_node.id, innovation_type = InnovationType.CONNECTION)
		self.connection_genes.append(ConnectionGene(in_node_id = inserted_node.id, out_node_id = out_node.id, enable = True, innovation_number = i.innovation_number, weight = connection.weight))
		
		

	def mutate(self):
		# function to mutate gnome
		if np.random.rand() < self.connection_mutation:
			self.mutate_connection()
		if np.random.rand() < self.node_mutation:
			self.mutate_node()

		for i in self.connection_genes:
			# update weight
			if np.random.rand() < self.weight_mutation:
				if np.random.rand() < self.weight_reset_prob:
					i.weight = np.random.randn()
				else:
					i.weight += np.random.randn()
		

	def print_connections(self):
		# function to print connection_genes list
		for i in self.connection_genes:
			i.print_connection()

	def get_connection(self , innovation_number):
		for i in self.connection_genes:
			if i.innovation_number == innovation_number:
				return i
		return False

	@staticmethod
	def crossover(gnome1 , gnome2):
		child = None
		gnome1.connection_genes.sort(key = lambda x : x.innovation_number)
		gnome2.connection_genes.sort(key = lambda x : x.innovation_number)

		connections = None

		nodes = list(set().union(gnome1.nodes , gnome2.nodes))
		child = Gnome(gnome1.innovations, gnome1.node_gen, nodes = nodes)	 

		if len(gnome1.connection_genes) > len(gnome2.connection_genes):
			child.connection_genes = deepcopy(gnome1.connection_genes)
			connections = deepcopy(gnome2.connection_genes)
		else:
			child.connection_genes = deepcopy(gnome2.connection_genes)
			connections = deepcopy(gnome1.connection_genes)

		for connection in connections:
			temp = child.get_connection(connection.innovation_number)
			if temp:
				temp.weight = (temp.weight + connection.weight)/2
			else:
				child.connection_genes.append(connection)
		return child