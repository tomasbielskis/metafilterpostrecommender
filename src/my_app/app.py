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
from flask import Flask, request, render_template

app = Flask(__name__)

# home page
@app.route('/')
def index():
    return render_template('index.html', title='Metafilter.com post recommender')

# Form page to submit text
@app.route('/submission_page/')
def submission_page():
    return '''
        <form action="/post_recommender" method='POST' >
            <input type="text" name="user_input" />
            <input type="submit" />
        </form>
        '''

# My word counter app
@app.route('/post_recommender', methods=['POST'] )
def post_recommender():
    user_name = str(request.form['user_input'])
    page = 'These are some posts that you might like: {0}'
    return page.format(recommend(user_name))

def sim_rec(user_data, item_data, index, top_n = 10):
    """Calculates post recommendations for an individual user
    """
    s = cosine_similarity(user_data[user_data.index==index], item_data).flatten()
    post_indexes = s.argsort()[::-1][0:top_n]
    return [item_data.index[i] for i in post_indexes]

def user_lookup(user_name):
    dfuser = pd.read_csv('data/usernames.txt',sep='\t', header=1, parse_dates=['joindate'], skiprows=0, index_col='userid')
    return dfuser[dfuser['name']==user_name].index[0]

def recommend(user_name):
    index = user_lookup(user_name)#look up the index of a user
    return sim_rec(user_data, item_data, index,top_n=10)

if __name__ == '__main__':
    user_data = pd.read_json('data/user_data2')#complete user data
    item_data = pd.read_json('data/gl_item_test_data2')#syntehtic new posts

    app.run(host='0.0.0.0', port=8080, debug=True)



#
# def stemming(post):
#     """A function for stemming the words in a list of posts"""
#     stemmer = SnowballStemmer("english")
#     post = [stemmer.stem(word) for word in post.split(" ")]
#     post = " ".join(post)
#     return post
#
# def stem_parrallel(pool_size, data):
#     """The parallel job for stemming. Takes the number of cores and the dataset"""
#     pool = multiprocessing.Pool(pool_size)
#     results = pool.map(stemming, data)
#     pool.close()
#     pool.join()
#     return results
#
# def run_stemmer(path, string_list):
#     string_list = stem_parrallel(pool_size, string_list)
#     string_list = pd.Series(string_list)
#     return string_list
#
# def tfidf_vect(path, string_list):
#     # Use tf-idf features for NMF.
#     tfidf = tfidf_vectorizer.transform(string_list)
#     return tfidf
#
# def NMF_Kullback(path, tfidf, tfidf_vectorizer):
#     # Transform with NMF model
#     nmf = nmf2.transform(tfidf)
#     return nmf
