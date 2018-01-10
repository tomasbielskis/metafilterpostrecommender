from time import time
import pandas as pd
import numpy as np
import pickle

def process_favorites_data():
    # Load favorites for 2017
    dffavorites = pd.read_csv('../data/favorites2017.csv',
                              parse_dates=['datestamp'],
                              index_col='faveid')
    # Select only the posts data
    dffavorites = dffavorites[dffavorites['type'].isin([1, 3, 5])]
    # Drop unnecessary columns
    dffavorites = dffavorites[['faver', 'target']]
    dffavorites.to_json('../data/gl_fav_input_df')

    # need to add lookups for train test plit

def process_item_data():
    # Load post_text for the index
    posttext = pd.read_json('../data/parsedtext/posttext')[0]
    posttext.index = posttext.index.map(str)

    # Load text feature models
    model_path = '../data/stemmedpostsonlymodels/'
    # tfidf_vectorizer = pickle.load(open(model_path+'tfidfvectorizer', 'rb'))
    # tf_vectorizer = pickle.load(open(model_path+'tfvectorizer', 'rb'))
    tfidf = pickle.load(open(model_path+'tfidfmodel', 'rb'))
    # tf = pickle.load(open(model_path+'tfmodel', 'rb'))
    nmf1 = pickle.load(open(model_path+'NMF Frobenius norm', 'rb'))
    # nmf2 = pickle.load(open(model_path+'NMF Kullback-Leibler', 'rb'))
    # lda = pickle.load(open(model_path+'LDA', 'rb'))
    W = nmf1.transform(tfidf)
    H = nmf1.components_
    item_data = pd.DataFrame(W)
    item_data['postid'] = posttext.index
    item_data.to_json('../data/gl_item_data_input')

    # need to add train test split

    # incorporate other metadata on items such as number of comments, number of favorites, recency

def process_user_data():
    # User posts
    # User comments



if __name__ == '__main__':
