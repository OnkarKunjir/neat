import numpy as np
from connectiongene import ConnectionGene
from innovation import InnovationType
from nodegene import NodeTyep

from copy import deepcopy
#TODO add mutation without recursion

class Genome:
	def __init__(self, innovations, nodes_gen, n_inputs, n_outputs, nodes = None, weight_mutation  = 0.8, weight_reset_prob = 0.1, connection_mutation = 0.05, node_mutation = 0.03, attempt_to_find_unlinked_nodes = 5):
		self.connection_genes = []	# list of all connection in geneome
		self.nodes = nodes	# list of generated nodes
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs

		self.innovations = innovations	#InnovationList object
		self.node_gen = nodes_gen	#NodeList object

		self.fitness = 0
		# parameters for mutation
		self.weight_mutation = weight_mutation	
		self.weight_reset_prob = weight_reset_prob
		self.connection_mutation = connection_mutation
		self.node_mutation = node_mutation
		self.attempt_to_find_unlinked_nodes = attempt_to_find_unlinked_nodes
		
		self.species_id = None

		self.std_dev = 3
		self.weight_range = (-10 , 10)
		
		if self.nodes == None:
			self.nodes = []
			# adding input nodes
			for i in range(self.n_inputs):
				n = self.node_gen.get_node(i)
				if n == None:
					n = self.node_gen.add_node(NodeTyep.INPUT , pos = i)
				self.nodes.append(n)
			# adding output nodes
			for i in range(self.n_inputs , self.n_inputs+self.n_outputs):
				n = self.node_gen.get_node(i)
				if n == None:
					n = self.node_gen.add_node(NodeTyep.OUTPUT , pos = i)
				self.nodes.append(n)
			
			# adding bias node
			n = self.node_gen.get_node(self.n_inputs + self.n_outputs)
	
			if n == None:
				n = self.node_gen.add_node(NodeTyep.BIAS , layer = 0)			
			self.nodes.append(n)

			# creating random connection between input and output nodes
			# node1 = np.random.choice(self.nodes[:self.n_inputs])
			# node2 = np.random.choice(self.nodes[self.n_inputs:self.n_inputs+self.n_outputs])
			# self.add_connection(node1 , node2)

			# make fully connected nn
			input_nodes = self.nodes[:self.n_inputs]
			bias = self.nodes[-1]
			output_nodes = self.nodes[self.n_inputs:self.n_inputs+self.n_outputs]

			for i in input_nodes+[bias]:
				for j in output_nodes:
					self.add_connection(i , j)
			
	def is_connected(self, node1, node2):
		#function to find if two nodes are connected or not
		for i in self.connection_genes:
			if i.in_node_id == node1 and i.out_node_id == node2:
				return True
		return False
	
	def add_connection(self , node1 , node2):
		# not allowing recurrent connections
		if not self.is_connected(node1.id , node2.id) and (node1.layer < node2.layer):
			innovation_number = self.innovations.get_innovation(node1.id, node2.id, innovation_type = InnovationType.CONNECTION).innovation_number
			self.connection_genes.append( ConnectionGene(in_node_id = node1.id, out_node_id=node2.id, enable = True, innovation_number = innovation_number , weight = np.random.normal(0 , self.std_dev)) )
			return True
		return False

	def mutate_connection(self):
		# function which adds new connection between 2 unconnected nodes and updates connection gene	
		for i in range(self.attempt_to_find_unlinked_nodes):
			node1 = np.random.choice(self.nodes)
			node2 = np.random.choice(self.nodes[self.n_inputs:])
			if self.add_connection(node1 , node2):
				break		
	
	def add_node(self):
		connection = None
		while connection == None:
			connection = np.random.choice(self.connection_genes)
			in_node = self.node_gen.get_node(connection.in_node_id)
			if in_node.node_type == NodeTyep.BIAS:
				connection = None

		in_node = self.node_gen.get_node(connection.in_node_id)
		out_node = self.node_gen.get_node(connection.out_node_id)
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
		# function to mutate geneome
		if np.random.rand() < self.connection_mutation:
			self.mutate_connection()
		if np.random.rand() < self.node_mutation:
			self.add_node()

		for i in self.connection_genes:
			# update weight of the connection
			if np.random.rand() < self.weight_mutation:
				if np.random.rand() < self.weight_reset_prob:
					i.weight = np.random.normal(0 , self.std_dev)
				else:
					i.weight += np.random.normal(0 , self.std_dev)
					# i.weight = np.clip(i.weight , self.weight_range[0] , self.weight_range[1])
			
	def get_input_nodes(self):
		return list(filter(lambda x : x.node_type == NodeTyep.INPUT , self.nodes))
	
	def get_output_nodes(self):
		return list(filter(lambda x: x.node_type == NodeTyep.OUTPUT , self.nodes))

	
	@staticmethod
	def crossover(genome1 , genome2):
		innovations = genome1.innovations
		node_gen = genome1.node_gen
		n_inputs = genome1.n_inputs
		n_outputs = genome1.n_outputs

		child = Genome(innovations = innovations , nodes_gen = node_gen , n_inputs = n_inputs , n_outputs = n_outputs )
		
		connection1 = genome1.connection_genes
		connection2 = genome2.connection_genes

		connection1.sort(key = lambda x: x.innovation_number)
		connection2.sort(key = lambda x: x.innovation_number)

		# max_innovation = max(connection1 + connection2 , key = lambda x: x.innovation_number)
	
		
		len1 = len(connection1)
		len2 = len(connection2)

		n1 = n2 = 0
		selected_nodes = [i.id for i in child.nodes]
		selected_innovations = [i.innovation_number for i in child.connection_genes]

		# finding matching genes
		while n1 < len1 or n2 < len2:
			c1 = connection1[n1] if n1 < len1 else None
			c2 = connection2[n2] if n2 < len2 else None
			selected_connection = None

			if c1 and c2:
				if c1.innovation_number == c2.innovation_number:
					selected_connection = deepcopy(np.random.choice([c1 , c2]))
					if not (c1.enable and c2.enable):
						if np.random.rand() < 0.75:
							selected_connection.enable = False
					n1 += 1
					n2 += 1
					
				elif c1.innovation_number < c2.innovation_number:
					if genome1.fitness > genome1.fitness:
						selected_connection = c1
					n1 += 1
				else:
					if genome2.fitness > genome1.fitness:
						selected_connection = c2
					n2 += 1
			elif c2 == None and c1:
				if genome1.fitness > genome2.fitness:
					selected_connection = c1
				n1 += 1
			elif c1 == None and c2:
				if genome2.fitness > genome1.fitness:
					selected_connection = c2
				n2 += 1
			
			if selected_connection and selected_connection.innovation_number not in selected_innovations:
				child.connection_genes.append(deepcopy(selected_connection))
				selected_innovations.append(selected_connection.innovation_number)
				
				if selected_connection.in_node_id not in selected_nodes:
					child.nodes.append( node_gen.get_node(selected_connection.in_node_id) )
					selected_nodes.append(selected_connection.in_node_id)

				if selected_connection.out_node_id not in selected_nodes:
					child.nodes.append( node_gen.get_node(selected_connection.out_node_id) )
					selected_nodes.append(selected_connection.out_node_id)
			else:
				selected_connection = None

		return child

	

	def print_connections(self):
		# function to print connection_genes list
		for i in self.connection_genes:
			i.print_connection()

	def print_nodes(self):
		for i in self.nodes:
			print('id : ' , i.id)
			print('type : ' , i.node_type)
			print('layer : ' , i.layer)
			print('pos : ' , i.pos)
			print('_'*10)
