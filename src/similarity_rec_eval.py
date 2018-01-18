import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def sim_rec(user_data, item_data, top_n=10):
    s = cosine_similarity(user_data, item_data)
    post_indexes = [i.argsort()[::-1][0:top_n] for i in s]
    return {u: item_data.index[i] for u, i in zip(user_data.index,post_indexes)}

def overlap(recs, actuals):


if __name__ == '__main__':
    path = '../data/rec_input/'
    user_data1 = pd.read_json('../data/user_data1')#complete user data
    item_test_data1 = pd.read_json(path+'gl_item_test_data1')#syntehtic new posts
    fav_posts_test = pd.read_json(path+'gl_fav_posts_test')#actuals to compare
    recs = sim_rec(user_data1,item_test_data1, top_n=100)
    overlap()
