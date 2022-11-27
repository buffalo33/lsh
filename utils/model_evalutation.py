import numpy as np
from near_neighbors import predict_rating


def compute_mse(infered_ratings,test_ratings):
    vec_size=infered_ratings.shape[0]
    return np.sum((infered_ratings-test_ratings)**2)/vec_size

def generate_infered_ratings_array(test_ids,similarity_matrix,rating_array_copy,normalisation_factor=1):
    infered_ratings=[]
    for user_id,dic_user in test_ids.items():
        movie_id=dic_user["movie_id"]
        infered_rating=predict_rating(user_id,movie_id,similarity_matrix,rating_array_copy,normalisation_factor=1)
        infered_ratings.append(infered_rating)
    return np.array(infered_ratings)


def complete_mse_computation(test_ids,similarity_matrix,rating_array_copy,normalisation_factor=1):
    test_ratings_array=np.array([el["movie_rating"] for el in test_ids.values()])
    infered_ratings=generate_infered_ratings_array(
            test_ids,similarity_matrix,rating_array_copy,normalisation_factor=1
        )
    computed_mse=compute_mse(infered_ratings,test_ratings_array)
    return computed_mse