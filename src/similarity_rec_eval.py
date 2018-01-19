import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score
from collections import defaultdict

def sim_rec(user_data, item_data, top_n=10):
    """
    Calculates similarity for multiple users,
    ouputs a dictionary of recommended posts
    """
    s = cosine_similarity(user_data, item_data)
    post_indexes = [i.argsort()[::-1][0:top_n] for i in s]
    return {u: item_data.index[i] for u, i in zip(user_data.index,post_indexes)}

def metrics(recs, actuals):
    act = {u: list(actuals[actuals['faver']==u]['target']) for u in set(actuals['faver'])}
    precision = {}
    recall = {}
    for u in act.keys():
        tp = len([p for p in act[u] if p in recs[u]])
        precision[u] = tp / float(len(set(recs[u])))
        recall[u] = tp / float(len(set(act[u])))
    return (np.mean([i for i in precision.values()])*100,
            np.mean([i for i in recall.values()])*100)

if __name__ == '__main__':
    path = '../data/rec_input/current/'
    user_data = pd.read_json(path+'user_data1')#complete user data
    item_test_data = pd.read_json(path+'gl_item_test_data1')#syntehtic new posts
    fav_posts_test = pd.read_json(path+'gl_fav_posts_test')#actuals to compare

    # item_train_data = pd.read_json(path+'gl_item_train_data1')
    # item_train_data.set_index('postid',inplace=True)
    # item_train_data.index.rename(None,inplace=True)
    # item_train_data.columns = item_train_data.columns.map(int)

    # fav_posts_train = pd.read_json(path+'gl_fav_posts_train')#actuals to compare

    item_data = item_test_data.append(item_train_data)
    fav_posts = fav_posts_test.append(fav_posts_train)

    perf = {}
    for n in [5,10,50,100,1000]:
        recs = sim_rec(user_data,item_data, top_n=n)#run the function above or load a file
        # random_recs = {u: random.sample(set(item_data.index),n) for u in user_data.index}
        perf[n] = metrics(recs, fav_posts)

    print(perf)
