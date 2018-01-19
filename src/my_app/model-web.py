from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import pickle
from nltk.stem.snowball import SnowballStemmer
import multiprocessing
import threading
import os

def stemming(post):
    """A function for stemming the words in a list of posts"""
    stemmer = SnowballStemmer("english")
    post = [stemmer.stem(word) for word in post.split(" ")]
    post = " ".join(post)
    return post

def stem_parrallel(pool_size, data):
    """The parallel job for stemming. Takes the number of cores and the dataset"""
    pool = multiprocessing.Pool(pool_size)
    results = pool.map(stemming, data)
    pool.close()
    pool.join()
    return results

def run_stemmer(path, string_list):
    string_list = stem_parrallel(pool_size, string_list)
    string_list = pd.Series(string_list)
    return string_list

def tfidf_vect(path, string_list):
    # Use tf-idf features for NMF.
    tfidf = tfidf_vectorizer.transform(string_list)
    return tfidf

def NMF_Frob(path, tfidf, tfidf_vectorizer):
    # Transform with NMF model
    nmf = nmf1.transform(tfidf)
    return nmf

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
    # Setting up the parameters
    n_features = 50000 #max features used in NMF including unigrams and bigrams
    n_components = 50 #latent features
    n_top_words = 20 #for printing model output
    pool_size = 4 #number of processors on the machine for stemming

    #if form input
    data = #form input
    data_stemmed = run_stemmer(output_path, data) #change to pd.read_json if already present
    # Load models
    model_path = '../data/nlp/train/'
    tfidf_vectorizer = pickle.load(open(model_path+'tfidfvectorizer', 'rb'))
    nmf1 = pickle.load(open(model_path+'NMF_Frobenius', 'rb'))
    tfidf = tfidf_vect(output_path, data_stemmed)
    user_data = NMF_Frob(output_path, tfidf, tfidf_vectorizer)
    index = 0

    #if existing user:
    user_data = pd.read_json('../data/user_data1')#complete user data
    dfuser = pd.read_csv('data/usernames.txt',sep='\t', header=1, parse_dates=['joindate'], skiprows=0, index_col='userid')
    user_name = 'Jaclyn' # from form on site
    index = user_lookup(user_name, user_data1)#look up the index of a user

# Available posts to recommend, needs to be prepared outside
    item_test_data1 = pd.read_json(path+'gl_item_test_data1')#syntehtic new posts
    # item_train_data = pd.read_json(path+'gl_item_train_data1')
    # item_train_data.set_index('postid',inplace=True)
    # item_train_data.index.rename(None,inplace=True)
    # item_train_data.columns = item_train_data.columns.map(int)
    item_data = item_test_data#.append(item_train_data)

    sim_rec(user_data,item_data,index)
