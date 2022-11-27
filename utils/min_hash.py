import numpy as np
import collections

from numpy.random import permutation

class MinHash(object):
    
    def __init__(self,vector_size,n_rows=None,n_bands=None,seed=123,signatures=None) -> None:
        """
        either permutations are already defined or we input the number of permutations we want
        the object takes as input the matrix threshold_rating_array on which it will compute signatures
        """
        if seed:
            np.random.seed(seed)
        self.n_rows=n_rows
        self.n_bands=n_bands
        self.n_permutations=self.n_rows*self.n_bands
        self.vector_size=vector_size
        self.permutations=[np.random.permutation(self.vector_size) for k in range(self.n_permutations)]
        self.hashes= {k:None for k in range(n_bands)}  #collections.defaultdict(dict) # key = band_index, value = dictionnary of hashes (one bucket) , value.key = similar items in a bucket
        self.filled_buckets=False # turn to true once the buckets have been filled
        if not signatures:
            self.signatures={} # the full signature for each user
        else : 
            self.signatures=signatures
            
    def __emptyhashes__(self):
        """
        Empties hashes object when we want to try different values of nbands and nrows
        """
        self.hashes= {k:None for k in range(self.n_bands)}
        
    def __changebands__(self, n_bands,n_rows):
        self.n_bands=n_bands
        self.n_rows=n_rows
    
    def __setitem__(self,user_vector,user_index):
        """set a user hashes to the hashes dictionnary"""
        full_sig=self.__generate_full_signature__(user_vector=user_vector)
        splited_signature=self.__create_signature_partition__(full_sig)
        for slice_index in range(self.n_bands): # going through all the bands
            if (self.hashes[slice_index] is None):
                # if the hash has never been seen, we initialise an entry in the dict:
                # create the entry and add the user
                self.hashes[slice_index]={}
                # [splited_signature[slice_index]]=[user_index]
            if (not splited_signature[slice_index] in self.hashes[slice_index]):
                self.hashes[slice_index][splited_signature[slice_index]]=[user_index]
            else :
                self.hashes[slice_index][splited_signature[slice_index]]=self.hashes[slice_index][splited_signature[slice_index]]\
                 + [user_index]
        self.signatures[user_index]=full_sig
        
    def __getitem_singleband__(self,user_vector,slice_index):
        """
        get items that are similar in a specific slice (band) for a specific user
        """
        hash_value=self.__generate_band_hash__(user_vector=user_vector,
                            band_id=slice_index)
        return self.hashes[slice_index].get(hash_value, [])
    
    def __getitem__(self,user_vector,user_id):
        similar_items=[]
        for slice_index in range(self.n_bands):
            simi=self.__getitem_singleband__(user_vector=user_vector,slice_index=slice_index)
            if simi==[]:
                continue
            similar_items.extend(simi)
        return list(set(similar_items))

    def __generate_band_hash__(self,user_vector,band_id):
        band_hash=[]
        permutations=self.permutations[self.n_rows*band_id: (band_id +1)*self.n_rows]
        for permutation in permutations:
            band_hash.append(self.__generate_single_hash__(user_vector,permutation))
        return ','.join([str(el) for el in band_hash])


    def __generate_full_signature__(self,user_vector):
        full_signature=[]
        for permutation in self.permutations:
            hash=self.__generate_single_hash__(user_vector=user_vector,permutation=permutation)
            full_signature.append(hash)
        return full_signature

    def __create_signature_partition__(self,signature):
        """
        splits a signature into bands
        """
        splited_signature=[]
        for band_index in range(self.n_bands):
            bande=signature[self.n_rows*band_index:(band_index+1)*self.n_rows]
            splited_signature.append(','.join([str(el) for el in bande]))
        return splited_signature
        
    def __generate_single_hash__(self, user_vector,permutation):
        """
        Find the first non null value following the permutation order
        """
        for index in permutation:
            if user_vector[index]:
                return index