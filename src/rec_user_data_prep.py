import numpy as np
import pandas as pd

if __name__ == '__main__':
    path = '../data/rec_input/'
    user_post_feat_train2 = pd.read_json(path+'user_post_feat_train2')
    user_post_feat_test2 = pd.read_json(path+'user_post_feat_test2')
    user_data = user_post_feat_train2.append(user_post_feat_test2)

    user_com_feat_train2 = pd.read_json(path+'user_com_feat_train2')
    user_data = user_data.append(user_com_feat_train2)

    user_com_feat_test2 = pd.read_json(path+'user_com_feat_test2')
    user_data = user_data.append(user_com_feat_test2)

    user_fav_post_features_train2 = pd.read_json(path+'user_fav_post_features_train2')
    user_data = user_data.append(user_fav_post_features_train2.drop('postid',axis=1))

    user_fav_comment_features_train2 = pd.read_json(path+'user_fav_comment_features_train2')
    user_data = user_data.append(user_fav_comment_features_train2.drop('commentid',axis=1))

    user_fav_comment_features_test2 = pd.read_json(path+'user_fav_comment_features_test2')
    user_data = user_data.append(user_fav_comment_features_test2.drop('commentid',axis=1))

    user_fav_post_features_test2 = pd.read_json(path+'user_fav_post_features_test2')
    user_data = user_data.append(user_fav_post_features_test2.drop('postid',axis=1))

    user_data = user_data.groupby('userid').mean()
    user_data.to_json('../data/user_data2')
