import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def sim_rec(user_data, item_data, index, top_n = 10):
    """Calculates up post recommendations for an individual user
    """
    s = cosine_similarity(user_data[user_data.index==index], item_data).flatten()
    post_indexes = s.argsort()[::-1][0:top_n]
    return [item_data.index[i] for i in post_indexes]

def user_lookup(user_name, user_data):
    user_id = dfuser.iloc[dfuser['name']==user_name]
    return user_data.iloc[user_id]

if __name__ == '__main__':
    path = '../data/rec_input/'
    user_data1 = pd.read_json('../data/user_data1')#complete user data
    item_test_data1 = pd.read_json(path+'gl_item_test_data1')#syntehtic new posts
    dfuser = pd.read_csv('data/usernames.txt',sep='\t', header=1, parse_dates=['joindate'], skiprows=0, index_col='userid')
    fav_posts_test = pd.read_json(path+'gl_fav_posts_test')#actuals to compare
    user_name = 'Jaclyn'
    index = user_lookup(user_name, user_data1)#look up the index of a user
    sim_rec(user_data1,item_test_data1,index)


# : d[215374]
# Out[97]:
# Int64Index([152591, 29218, 83530, 143448, 80804, 123224, 33947, 32388, 66082,
#             1014],
#            dtype='int64')
