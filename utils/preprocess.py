import pandas as pd
import numpy as np
from math import isnan

from psutil import users

class Preprocess(object):

    def __init__(self,df_path) -> None:
        self.df_path=df_path

    def preprocess_ratings_df(df_path,rating_threshold=3):
        """
        don't use because using pandas is heavy
        """
        # read input ratings dataframe
        ratings_df=pd.read_csv(df_path)
        ratings_df.drop("timestamp",inplace=True,axis=1)
        input_df=ratings_df.pivot_table(values='rating', index="userId", columns='movieId').rename_axis(None, axis=1).rename_axis(None, axis=0)
        if rating_threshold:
            np_df=input_df.to_numpy()
            # put 1 values where rating above threshold
            np_df=np.where(np_df>=rating_threshold,1,np.nan)
            final_df=pd.DataFrame(np_df,index=input_df.index,columns=input_df.columns)
            # remove all candidates that have only rated films below the threshold (they don't add any information)
            init_users_nb=final_df.shape[0]
            nb_evaluations=final_df.apply(lambda row : sum([k for k in row if not isnan(k)]),axis=1)
            candidates_to_drop=nb_evaluations.index[nb_evaluations<1]
            final_df.drop(candidates_to_drop,axis=0,inplace=True)
            print(f"After removing the irrelevent users, there are {final_df.shape[0]} users left : removed {init_users_nb - final_df.shape[0]}")
            return final_df
        return input_df

    def preprocess_ratings_np(self,rating_threshold=3):
        """
        use this model
        """
        # careful !! the ratings are transposed as ratings from 1 to 10 in the ratings_array
        # possible to use directly numpy to read the csv file
        # genfromtxt(large_dataset_path, delimiter=',',skip_header=1)
        ratings_df=pd.read_csv(self.df_path)
        nb_films=ratings_df.movieId.unique().shape[0]
        nb_users=ratings_df.userId.unique().shape[0]
        dataset_np=ratings_df.to_numpy()
        # creating the mapping of users from new id to initial id (when there is a missing user for instance)
        users_mapping={}
        movies_mapping={}

        for k,user in enumerate(sorted(ratings_df.userId.unique())):
            users_mapping[k]=user

        for k,movie in enumerate(sorted(ratings_df.movieId.unique())):
            movies_mapping[k]=movie

        reverted_movies_mapping={v:k for k,v in movies_mapping.items()}
        reverted_users_mapping={v:k for k,v in users_mapping.items()}

        # creating the arrays
        threshold_ratings_array=np.zeros(shape=(nb_users,nb_films),dtype=bool)
        ratings_array=np.zeros(shape=(nb_users,nb_films),dtype=np.int8)
        for row in dataset_np:
            # if rating above threshold
            user=int(row[0])
            movie=int(row[1])
            rating=int(2*row[2])
            # creating the complete rating matrix
            ratings_array[reverted_users_mapping[user],reverted_movies_mapping[movie]]=rating
            # creating the ratings matrix with threshold
            if rating>rating_threshold: 
                threshold_ratings_array[reverted_users_mapping[user],reverted_movies_mapping[movie]]=True

        # drop users who only saw one film or who didn't score any film above the threshold

        # drop users that only scored one film
        non_zeros=np.count_nonzero(ratings_array,axis=1)
        indexes_to_drop_init_df=np.where(non_zeros==1)[0]
        self.indexes_to_drop_init_df=indexes_to_drop_init_df

        # dropping all users that haven't rated movies above threshold
        indexes_to_remove_threshold=np.where(np.all(np.invert(threshold_ratings_array),axis=1))[0]
        indexes_to_drop=list(indexes_to_drop_init_df)+list(indexes_to_remove_threshold)
        indexes_to_drop.sort()

        print(f"Removing {len(indexes_to_drop)} irrelevent users")
        self.indexes_to_drop=indexes_to_drop
        for k in indexes_to_drop:
            # removing the keys that are not anymore relevent
            previous_key=users_mapping[k]
            del users_mapping[k]
            del reverted_users_mapping[previous_key]

        new_users_mapping={}
        # settings the index to the right value (we deleted some rows)
        for index,key in enumerate(users_mapping.keys()):
                new_users_mapping[index]=users_mapping[key]
        users_mapping=new_users_mapping
        del new_users_mapping
        reverted_users_mapping={v:k for k,v in users_mapping.items()}

        indexes_to_keep=list(set([k for k in range(ratings_array.shape[0])]) - set(indexes_to_drop))
        indexes_to_keep.sort()
        self.indexes_to_keep=indexes_to_keep
        ratings_array= np.delete(ratings_array, indexes_to_drop, axis=0) # ratings_array[indexes_to_keep,:]
        threshold_ratings_array= np.delete(threshold_ratings_array, indexes_to_drop, axis=0)# threshold_ratings_array[indexes_to_keep,:]
        self.threshold_ratings_array=threshold_ratings_array
        self.ratings_array=ratings_array
        self.users_mapping=users_mapping
        self.movies_mapping=movies_mapping
        self.reverted_movies_mapping=reverted_movies_mapping
        self.reverted_users_mapping=reverted_users_mapping

def create_train_test(threshold_rating_array,full_rating_array,seed=None):
    """
    retreive some samples on which to compute prediction and return the iniaitl arrays without the dataaa to predict
    """
    if seed:
        np.random.seed(seed)
    test_ids={}
    nb_users=threshold_rating_array.shape[0]
    threshold_array_copy=np.copy(threshold_rating_array)
    full_rating_array_copy=np.copy(full_rating_array)
    for user_id,user_row in enumerate(threshold_rating_array):
        print(f"User {user_id+1}/{nb_users} " ,end="\r")
        random_movie_ids=np.where(user_row)[0]
        movie_id=np.random.choice(random_movie_ids)
        movie_rating=full_rating_array[user_id][movie_id]
        # remove the ratings in the initial dataframes
        threshold_rating_array[user_id][movie_id]=False
        full_rating_array_copy[user_id][movie_id]=0
        test_ids[user_id]={'movie_id':movie_id,
                            "movie_rating":movie_rating}
    return test_ids,threshold_array_copy,full_rating_array_copy