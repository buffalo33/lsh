import random_hash as rh
import numpy as np

class LSH:
    def __init__(self, num_tables, hash_size, inp_dimensions,signatures=dict(),filled_buckets=False):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_tables = dict()
        self.filled_buckets = filled_buckets # turn to true once the buckets have been filled
        self.signatures = signatures
        for i in range(self.num_tables):
            self.hash_tables[i] = rh.HashTable(self.hash_size, self.inp_dimensions)
        #if self.filled_buckets == False:
        #    for i in range(self.num_tables):
        #        self.hash_tables[i] = rh.HashTable(self.hash_size, self.inp_dimensions)
        #else:
        #    for i in range(self.num_tables):
        #        self.hash_tables[i] = dict()


    # Put the vector inp_vec in the buckets for all hash functions contained in hash_tables.
    def __setitem__(self, inp_vec, inp_vec_idx):
        if self.filled_buckets == False:
            #print("not using signatures")
            #print("self.filled_buckets == False")
            signature = ''
            #print("inp_vec")
            #print(inp_vec)
            for table in self.hash_tables:
                hash_value = self.hash_tables[table].__setitem__(inp_vec,inp_vec_idx)
                #if inp_vec_idx == 100:
                #    print("hash_value")
                #    print(hash_value)
                signature += hash_value
            self.signatures[inp_vec_idx] = signature
        else:
            for bandIdx in range(self.num_tables):
                #print("using signatures")
                #print("bandIdx")
                #print(bandIdx)
                #print("self.hash_tables")
                #print(self.hash_tables)
                #print("type(inp_vec_idx)")
                #print(type(inp_vec_idx))

                hash_value = self.signatures[inp_vec_idx][self.hash_size*bandIdx:self.hash_size*(bandIdx+1)]

                self.hash_tables[bandIdx].hash_table[hash_value] = self.hash_tables[bandIdx].hash_table.get(hash_value, list()) + [inp_vec_idx]
    
    # For a vector index, look for each hash which vector indexes are contained in the same bucket.
    def __getitem__(self, inp_vec,inp_vec_idx):
        results = list()
        if self.filled_buckets == False:
            #print("not using signatures")
            for table in self.hash_tables:
                results.extend(self.hash_tables[table].__getitem__(inp_vec))
        else:
            for bandIdx in range(self.num_tables):
                hash_value = self.signatures[inp_vec_idx][self.hash_size*bandIdx:self.hash_size*(bandIdx+1)]

                #print("bandIdx")
                #print(bandIdx)
                #print("self.hash_tables")
                #print(self.hash_tables)
                #print("self.hash_tables[bandIdx].hash_table")
                #print(self.hash_tables[bandIdx].hash_table)
                #print("hash")
                #print(self.signatures[inp_vec_idx][self.hash_size*bandIdx:self.hash_size*(bandIdx+1)])
                #print("bucket values for 0")
                #print(self.hash_tables[bandIdx].hash_table['0'])
                #print("bucket")
                #print(self.hash_tables[bandIdx].hash_table[self.signatures[inp_vec_idx][self.hash_size*bandIdx:self.hash_size*(bandIdx+1)]])
                results.extend(str(667))

                results.extend(self.hash_tables[bandIdx].hash_table[self.signatures[inp_vec_idx][self.hash_size*bandIdx:self.hash_size*(bandIdx+1)]])

            #if inp_vec_idx == 201:
            #    print("neighbours for 201")
            #    print(list(set(results)))
        return list(set(results))