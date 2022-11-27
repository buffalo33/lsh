import random_hash_tables as rht
import near_neighbors as nn

def find_best_band_for_row(bandRange,rows,preprocessing,userIdx,itemIdx,lsh_label,sim_function_label):
    errors = []
    vectors = preprocessing.threshold_ratings_array
    ratings_vectors = preprocessing.ratings_array


    for nbBands in range(1,bandRange+1):
        error = 0
        nbSamples = 100
        for sampleIdx in range(1,nbSamples):
            lsh = rht.LSH(nbBands,rows,len(vectors[0]))
            prediction = nn.near_neighbor_prediction(preprocessing,userIdx,itemIdx,lsh,sim_function_label)
            error = (ratings_vectors[userIdx][itemIdx] - prediction)**2
        error = error/nbSamples 
        errors.append(error)

    best_error = min(errors)
    best_band = errors.index(best_error)


    return (best_error,best_band,errors)