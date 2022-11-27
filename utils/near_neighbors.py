from cmath import isnan
import numpy as np
import random_hash_tables as rht
import math as m
import preprocess as prep
import time

# For each vector and for each hash function, place the index of the vector as label in the corresponding hash buckets.
def fill_buckets(vectors,lsh,verbose=False,return_time=False):
    time_init=time.time()
    lenVect = len(vectors)
    for vectorIdx in range(lenVect):
        if verbose:
            print(f"Filling buckets for user {vectorIdx+1}/{lenVect}",end="\r")
        lsh.__setitem__(vectors[vectorIdx],vectorIdx)
    lsh.filled_buckets=True
    
    if verbose:
        print("")
    
    time_filled=int((time.time()-time_init)*100)/100
    
    if verbose:
        print(f"Filled buckets in {time_filled} seconds")
    
    if return_time:
        return time_filled
    return 0

# For a vector index, look for each hash which vector indexes are contained in the same bucket.
def search_similar_vectors(vectors,vectorIdx,lsh):
    time_init=time.time()
    """
    returns similar found users that are not the user himself
    """
    #print("vectorIdx")
    #print(vectorIdx)
    #print("vectors[vectorIdx]")
    #print(vectors[vectorIdx])

    vector = vectors[vectorIdx]
    all_neighbors = lsh.__getitem__(vector,vectorIdx)
    #print("")
    time_filled=int((time.time()-time_init)*100)/100
    #print(f"Get similar vectors in {time_filled} seconds")
    return list(set(all_neighbors)-set([vectorIdx]))

def pearson(vectors,idxU,idxV):
    """
    optimised pearson with numpy"""
    x1 = vectors[idxU]
    x2 = vectors[idxV]
    non_null_indexes = set.intersection(set(np.nonzero(x1)[0]),set(np.nonzero(x2)[0]))
    non_null_indexes = np.array(sorted(list(non_null_indexes)))
    # print("\nEEE",non_null_indexes)
    if len(non_null_indexes)==0:
        return 0

    mu1=sum(x1)/len(x1)
    mu2=sum(x2)/len(x2)
    numerator_vector=(x1[non_null_indexes]-mu1)*(x2[non_null_indexes]-mu2)
    numerator = np.sum(numerator_vector)

    denom_x1=m.sqrt(np.sum((x1[non_null_indexes]-mu1)**2))
    denom_x2=m.sqrt(np.sum((x2[non_null_indexes]-mu2)**2))
    return min(1,numerator/(denom_x1 * denom_x2))

# Compute the Pearson correlation coefficient for two vectors.
def pearson2(vectors,idxU,idxV):
    nbCol = np.shape(vectors)[1]
    
    intersectionItems = []
    ratedU = set()
    ratedV = set()
    for i in range(nbCol):
        if (vectors[idxU][i] > 0.):
            ratedU.add(i)
        if (vectors[idxV][i] > 0.):
            ratedV.add(i)
    intersectionItems = ratedU.intersection(ratedV)

    if intersectionItems == set():
        return 0

    muU = 0.
    muV = 0.
    numerator = 0.
    rootU = 0.
    rootV = 0.

    for idx in ratedU:
        muU += vectors[idxU][idx]
    muU = muU / len(ratedU)
    
    for idx in ratedV:
        muV += vectors[idxV][idx]
    muV = muV / len(ratedV)

    for idx in intersectionItems:
        numerator += (vectors[idxU][idx] - muU)*(vectors[idxV][idx] - muV)

    for idx in intersectionItems:
        rootU += (vectors[idxU][idx] - muU)**2 
    rootU = m.sqrt(rootU)

    for idx in intersectionItems:
        rootV += (vectors[idxV][idx] - muV)**2
    rootV = m.sqrt(rootV)

    sim = numerator/(rootU * rootV)
    return sim

def jaccard(vectors,vectorIdx,neighborIdx):
    vector = vectors[vectorIdx]
    neighbor = vectors[neighborIdx]
    set_vector=set(np.where(vector == True)[0])
    set_neighbor=set(np.where(neighbor == True)[0])
    inter_len=len(set.intersection(set_vector,set_neighbor))
    union_len=len(set_vector)+len(set_neighbor)-inter_len
    return inter_len/union_len

# Compute the predicted coefficient of the matrix using the near neighbors.
def near_neighbor_prediction(preprocessing,vectorIdx,itemIdx,lsh,sim_function_label,computed_similarities=[]):
    vectors = preprocessing.threshold_ratings_array
    ratings_vectors = preprocessing.ratings_array
    if lsh.filled_buckets==False:
        fill_buckets(vectors,lsh)

    #print(lsh.hash_tables[0])

    similar_vectors_str = search_similar_vectors(vectors,vectorIdx,lsh)
    similar_vectors = [int(element) for element in similar_vectors_str]

    prediction = 0
    sum_sim = 0
    for neighborIdx in similar_vectors:
        similarity = 0.
        if sim_function_label == "pearson":
            similarity = pearson(ratings_vectors,vectorIdx,neighborIdx)
        elif sim_function_label == "jaccard":
            similarity = jaccard(vectors,vectorIdx,neighborIdx)
        elif sim_function_label == "pearson_computed":
            similarity = computed_similarities[vectorIdx][neighborIdx]

        rating = ratings_vectors[neighborIdx][itemIdx]
        if rating > 0:
            prediction += similarity * ratings_vectors[neighborIdx][itemIdx]
            sum_sim += similarity
    
    if sum_sim == 0.:
        return 0

    prediction = prediction/sum_sim
    return prediction/2.

def generate_vectors_uniform(nbUsers, nbItems, lackPortion):
    vectors = np.random.rand(nbUsers,nbItems)
    vectors *= 5.

    for userIdx in range(nbUsers):
        for itemIdx in range(nbItems):
            vectors[userIdx][itemIdx] = round(vectors[userIdx][itemIdx],1)
    return vectors

def find_best_band_for_row(bandRange,row,vectors,userIdx,itemIdx):
    errors = []

    for nbBands in range(bandRange):
        lsh = rht.LSH(nbBands,row,len(vectors[0]))
        prediction = near_neighbor_prediction(vectors,userIdx,itemIdx,lsh)
        error = (vectors[userIdx][itemIdx] - prediction)**2 
        errors.append(error)
    
    best_band = min(errors)

    return (best_band,errors.index(best_band),errors)

def predict_rating(user_index, movie_index, similarity_matrix,rating_matrix,normalisation_factor=1):
    """
    normalisation_factor is required as we mapped the ratings from [0.5,5] to [1,10] 
    --> return the rating in the original format
    
    """
    user_similarities=similarity_matrix[user_index]
    # because the matrix is symetric and we stored only the upper part, to keep
    # user similarities we have to concat two arrays
    user_part_before= similarity_matrix[:,user_index][:user_index]
    user_part_after=similarity_matrix[user_index][user_index:]
    user_similarities=np.concatenate((user_part_before,user_part_after))
    # some simi are set to nan (not computed) --> transform them to 0
    np.nan_to_num(user_similarities,copy=False) 
    movie_ratings_array = rating_matrix[:,movie_index]
    users_simi_sum=user_similarities.sum()
    if users_simi_sum==0:
        return 0
    predicted_rating=normalisation_factor*np.dot(user_similarities,movie_ratings_array)/(users_simi_sum)
    if np.isnan(predicted_rating):
        return 0
    return predicted_rating
    

def generate_similarity_matrix(threshold_array,lsh,verbose=False):
        # generate simi matricx
        similarity_matrix=np.empty(shape=(threshold_array.shape[0],threshold_array.shape[0]))
        similarity_matrix[:]=np.nan
        nb_users=threshold_array.shape[0]
        for user2_index,user2_vector in enumerate(threshold_array):
            if verbose:
                print(f"Going through user {user2_index+1}/{nb_users}",end="\\r")
            similar_vectors= search_similar_vectors(
                                            vectors=threshold_array,
                                            vectorIdx=user2_index,
                                            lsh=lsh
                                        )
            for similar_vector_index in similar_vectors:
                id_1=min(similar_vector_index,user2_index)
                id_2=max(similar_vector_index,user2_index)
                # if the similarity hasn\'t been computed yet
                if np.isnan(similarity_matrix[id_1][id_2]):
                    similarity_matrix[id_1][id_2]=jaccard(vectors=threshold_array,vectorIdx=id_1,neighborIdx=id_2)
        return similarity_matrix
    
def generate_bruteforce_jaccard_matrix(threshold_array_train,verbose=True):
    similarity_matrix_bruteforce_jac=np.empty(shape=(threshold_array_train.shape[0],threshold_array_train.shape[0]))
    similarity_matrix_bruteforce_jac[:]=np.nan
    # setting matrix values to nan by default and computing the simi only if the one in the upper diag hasn't been computed ye
    for user2_index,user2_vector in enumerate(threshold_array_train):
        for user1_index,user1_vector in enumerate(threshold_array_train[:user2_index,:]):
            if verbose:
                print(f"Going through user_2 = {user2_index}/{threshold_array_train.shape[0]} user_1 = {user1_index}/{threshold_array_train.shape[0]}",end="\r")

            similarity_matrix_bruteforce_jac[user1_index][user2_index]=jaccard(
                vectors=threshold_array_train,vectorIdx=user1_index,neighborIdx=user2_index)
        # np.nan_to_num(similarity_matrix_bruteforce_jac,copy=False)
    if verbose:
        print(f"\nThe maximum similarity for Jaccart is {np.nanmax(similarity_matrix_bruteforce_jac)}")

    return similarity_matrix_bruteforce_jac