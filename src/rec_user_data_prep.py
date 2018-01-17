import numpy as np
import pandas as pd

if __name__ == '__main__':
    path = '../data/rec_input/'
    user_post_feat_train1 = pd.read_json(path+'user_post_feat_train1')
    user_post_feat_test1 = pd.read_json(path+'user_post_feat_test1')
    user_data = user_post_feat_train1.append(user_post_feat_test1)

    user_com_feat_train1 = pd.read_json(path+'user_com_feat_train1')
    user_data = user_data.append(user_com_feat_train1)

    user_com_feat_test1 = pd.read_json(path+'user_com_feat_test1')
    user_data = user_data.append(user_com_feat_test1)

    user_fav_post_features_train1 = pd.read_json(path+'user_fav_post_features_train1')
    user_data = user_data.append(user_fav_post_features_train1.drop('postid',axis=1))

    user_fav_comment_features_train1 = pd.read_json(path+'user_fav_comment_features_train1')
    user_data = user_data.append(user_fav_comment_features_train1.drop('commentid',axis=1))

    user_fav_comment_features_test1 = pd.read_json(path+'user_fav_comment_features_test1')
    user_data = user_data.append(user_fav_comment_features_test1.drop('commentid',axis=1))

    user_fav_post_features_test1 = pd.read_json(path+'user_fav_post_features_test1')
    user_data = user_data.append(user_fav_post_features_test1.drop('postid',axis=1))

    user_data.groupby('userid').mean()
    user_data.to_json('../data/user_train_data1')
