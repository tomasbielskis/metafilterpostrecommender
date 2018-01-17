import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
from collections import defaultdict

item_train_data = pd.read_json('../data/rec_input/gl_item_train_data1')

user_train_data = pd.read_json('../data/rec_input/gl_user_train_data1')


S = cosine_similarity(user_train_data,item_train_data)




if __name__ == '__main__':
    # inputdfs = defaultdict(pd.DataFrame)
    path = '../data/rec_input/'
    # files = os.listdir(path)
    # for file in files:
    #     inputdfs[os.path.basename(file)] = pd.read_json(path+file)

    user_com_feat_train1 = pd.read_json(path+'user_com_feat_train1')
    user_fav_comment_features_train1 = pd.read_json(path+'user_fav_comment_features_train1')
    user_post_feat_train1 = pd.read_json(path+'user_post_feat_train1')
    user_fav_post_features_train1 = pd.read_json(path+'user_fav_post_features_train1')
    user_com_feat_test1 = pd.read_json(path+'user_com_feat_test1')
    user_fav_comment_features_test1 = pd.read_json(path+'user_fav_comment_features_test1')
    user_post_feat_test1 = pd.read_json(path+'user_post_feat_test1')
    user_fav_post_features_test1 = pd.read_json(path+'user_fav_post_features_test1')

    gl_item_test_data1 = pd.read_json(path+'gl_item_test_data1')#syntehtic new posts
    gl_fav_posts_test = pd.read_json(path+'gl_fav_posts_test')#actuals to compare
