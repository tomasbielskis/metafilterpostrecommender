"""NEEDS TO BE RUN IN gl-env (python 2.7)"""

import pandas as pd
import numpy as np
import pickle
import graphlab

train_data = pd.read_json('../data/gl_fav_input_df')
item_data = pd.read_json('../data/gl_item_data_input')
train_data['rating'] = 1

train_data.columns = [u'faver', u'postid', u'rating']

rec1 = graphlab.recommender.factorization_recommender.create(
            graphlab.SFrame(train_data),
            user_id='faver',
            item_id='postid',
            target='rating',
            binary_target=True,
            item_data=graphlab.SFrame(item_data))
rec2 = graphlab.recommender.factorization_recommender.create(
            graphlab.SFrame(train_data),
            user_id='faver',
            item_id='postid',
            target='rating',
            binary_target=True)

graphlab.recommender.util.compare_models(graphlab.SFrame(train_data), [rec1, rec2])
graphlab.recommender.util.compare_models(graphlab.SFrame(train_data), [rec1, rec2],metric='precision_recall')

rec1:
8.157690078901613e-06
rec2:
6.362300720061738e-08
