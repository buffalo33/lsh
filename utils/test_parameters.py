import random
import random_hash_tables as rht
import near_neighbors as nn
import numpy as np
import copy

def generate_signatures_tables(nbTables,vectors,lenSignatures=0):
    (nbUsers,nbItems) = vectors.shape

    signaturesArray = []

    for i in range(nbTables):
        lsh = rht.LSH(1,lenSignatures,nbItems,filled_buckets=False)
        if lsh.filled_buckets==False:
            nn.fill_buckets(vectors,lsh)
        #print("lsh.signatures 100-102")
        #print(lsh.signatures[100])
        #print(lsh.signatures[101])
        #print(lsh.signatures[102])
        signaturesArray.append(copy.deepcopy(lsh.signatures))
        #print([signaturesArray[j][100] for j in range(i)])

    #print([signaturesArray[i][100] for i in range(nbTables)])
    return signaturesArray

def generate_test_data_set(preprocessing):
    vectors = preprocessing.threshold_ratings_array
    ratings_vectors = preprocessing.ratings_array

    (nbUsers,nbItems) = ratings_vectors.shape
    test_data_set = []

    for userIdx in range(nbUsers):
        ratedItemMax = random.randint(0,nbItems)
        itemIdx = 0
        choosenItem = 0
        while (itemIdx < nbItems - 1) and (choosenItem < ratedItemMax):
            rate = ratings_vectors[userIdx][itemIdx]
            if rate > 0:
                choosenItem = itemIdx
            itemIdx += 1
        test_data_set.append([userIdx,itemIdx,ratings_vectors[userIdx][choosenItem]])

        ratings_vectors[userIdx][choosenItem] = 0
        vectors[userIdx][choosenItem] = 0
    return np.array(test_data_set)

def compute_similarities(preprocessing):
    ratings_vectors = preprocessing.ratings_array
    (nbUsers,nbItems) = ratings_vectors.shape

    pearson_matrix = np.zeros((nbUsers,nbUsers))

    for userIdxA in range(nbUsers):
      for userIdxB in range(userIdxA + 1):
        similarity = nn.pearson(ratings_vectors,userIdxA,userIdxB)
        pearson_matrix[userIdxA][userIdxB] = similarity
        pearson_matrix[userIdxB][userIdxA] = similarity

    return pearson_matrix

def test_band_range(nbTables,preprocessing,signaturesArray,test_data_set,lenSignatures=0,computed_similarities=[]):
    vectors = preprocessing.threshold_ratings_array
    
    (nbUsers,nbItems) = vectors.shape

    meanGlobalErrorArray = []

    nbRows = 1
    while nbRows <= lenSignatures:
        nbBands = int(lenSignatures/nbRows)
        print("nbRows")
        print(nbRows)

        lshArray = []
        
        for i in range(nbTables):
            lsh = rht.LSH(nbBands,nbRows,nbItems,signatures=signaturesArray[i],filled_buckets=True)
            nn.fill_buckets(vectors,lsh,verbose=False,return_time=False)
            lshArray.append(lsh)

        #print("signatures for 201")
        #for lsh in lshArray:
        #    print(lsh.signatures[201])

        mseArray = []

        for test_data in test_data_set:

            (userIdx,itemIdx,realValue) = test_data
            userIdx = int(userIdx)
            itemIdx = int(itemIdx)
            
            predictions = []

            for lsh in lshArray:
                #print(lsh.signatures[userIdx])
                #predictions.append(nn.near_neighbor_prediction(preprocessing,userIdx,itemIdx,lsh,"pearson"))
                predictions.append(nn.near_neighbor_prediction(preprocessing,userIdx,itemIdx,lsh,"pearson_computed",computed_similarities))
            
            #print("predictions")
            #print(predictions)
            #print("realValue/2.")
            #print(realValue/2.)


            predictions = np.array(predictions)
            predictions = (predictions - realValue/2.) ** 2

            mse = (1./len(predictions)) * sum(predictions)

            mseArray.append(mse)

        mseArray = np.array(mseArray)
        meanGlobalError = (1./len(mseArray)) * sum(mseArray) 
        meanGlobalErrorArray.append(meanGlobalError)

        nbRows *= 2

    bestError = min(meanGlobalErrorArray)
    bestRows = meanGlobalErrorArray.index(bestError)

    return (bestError,bestRows,meanGlobalErrorArray)

