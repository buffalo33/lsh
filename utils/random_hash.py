import numpy as np
from pyparsing import null_debug_action
    
class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.projections = np.random.randn(self.hash_size, inp_dimensions)
        #print("projections")
        #print(self.projections)

    # Compute the hash of a vector by multiplying the vector with the transpose of the matrix composed with random vectors.    
    def generate_hash(self, inp_vector):
        bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
        return ''.join(bools.astype('str'))

    # Associate in the hash_table dictonary the hash of the vector as key and a label (for example the user name) as value.
    def __setitem__(self, inp_vec, label):

        #print("type(label)")
        #print(type(label))

        hash_value = self.generate_hash(inp_vec)
        self.hash_table[hash_value] = self.hash_table.get(hash_value, list()) + [label]
        return hash_value
        
    # Get all the label in the same bucket as the input inp_vec.
    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_value, [])