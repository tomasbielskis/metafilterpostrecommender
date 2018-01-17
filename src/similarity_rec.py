import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

item_train_data = pd.read_json('../data/rec_input/gl_item_train_data1')
user_train_data = pd.read_json('../data/rec_input/gl_user_train_data1')

S = cosine_similarity(user_train_data,item_train_data)
