import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def sim_rec(user_data, item_data, favorites_test, index, top_n = 5):
    S = cosine_similarity(user_data[index], item_data).flatten()
    related_docs_indices = [i for i in S.argsort()[::-1] if i != index]
    return [(index, S[index]) for index in related_docs_indices][0:top_n]

def user_lookup(user_name, user_data):
    user_id = dfuser.iloc[dfuser['name']==user_name]
    return user_data.iloc[user_id]

def overlap():
    pass

if __name__ == '__main__':
    path = '../data/rec_input/'
    user_data1 = pd.read_json('../data/user_data1')#complete user data
    item_test_data1 = pd.read_json(path+'gl_item_test_data1')#syntehtic new posts
    dfuser = pd.read_csv('data/usernames.txt',sep='\t', header=1, parse_dates=['joindate'], skiprows=0, index_col='userid')
    # fav_posts_test = pd.read_json(path+'gl_fav_posts_test')#actuals to compare
    index = user_lookup(user_name, user_data1)#look up the index of a user

    for index, score in sim_rec(user_data1,item_test_data1, index):
        print score, item_test_data1[index]
