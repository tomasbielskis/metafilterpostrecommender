"""NEEDS TO BE RUN IN gl-env (python 2.7)"""

import pandas as pd
import numpy as np
import pickle
import graphlab

def gl_rec():

    rec1 = graphlab.recommender.factorization_recommender.create(
                graphlab.SFrame(train_data),
                user_id='faver',
                item_id='postid',
                target='rating',
                binary_target=True)
    rec2 = graphlab.recommender.factorization_recommender.create(
                graphlab.SFrame(train_data),
                user_id='faver',
                item_id='postid',
                target='rating',
                binary_target=True,
                item_data=graphlab.SFrame(item_train_data))
    rec3 = graphlab.recommender.factorization_recommender.create(
                graphlab.SFrame(train_data),
                user_id='faver',
                item_id='postid',
                target='rating',
                binary_target=True,
                user_data=graphlab.SFrame(user_train_data))
    rec4 = graphlab.recommender.factorization_recommender.create(
                graphlab.SFrame(train_data),
                user_id='faver',
                item_id='postid',
                target='rating',
                binary_target=True,
                item_data=graphlab.SFrame(item_train_data),
                user_data=graphlab.SFrame(user_train_data))

    rec1.evaluate_precision_recall(graphlab.SFrame(test_data))
    rec2.evaluate_precision_recall(graphlab.SFrame(test_data))
    rec3.evaluate_precision_recall(graphlab.SFrame(test_data))
    rec4.evaluate_precision_recall(graphlab.SFrame(test_data))

    graphlab.recommender.util.compare_models(graphlab.SFrame(test_data),
                [rec1, rec2, rec3, rec4],
                metric='precision_recall')

if __name__ == '__main__':
    path = '../data/rec_input/'
    train_data = pd.read_json(path+'gl_fav_train')
    test_data = pd.read_json(path+'gl_fav_test')
    item_train_data = pd.read_json(path+'gl_item_train_data1')
    # item_test_data = pd.read_json(path+'gl_item_test_data1')
    user_train_data = pd.read_json(path+'gl_user_train_data1')
    # user_test_data = pd.read_json(path+'gl_user_test_data1')
    train_data['rating'] = 1
    train_data.columns = [u'faver', u'postid', u'rating']
    test_data.columns = [u'faver', u'postid']
    user_train_data['faver'] = user_train_data.index.map(int)

    gl_rec()
