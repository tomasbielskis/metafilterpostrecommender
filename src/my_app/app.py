from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import pickle
from nltk.stem.snowball import SnowballStemmer
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
            <input type="text" name="user_input1" />
            <input type="submit" />
        </form>
        '''

@app.route('/post_recommender', methods=['POST'] )
def post_recommender():
    user_string = str(request.form['user_input1'])
    # page = 'These are some posts that you might like: {0}'
    if user_string == 'test':
        return 'lame user name, try again'
    elif len(user_string.split(' ')) == 1:
        post_dict = recommend_user(user_string)
    else:
        post_dict = recommend_text(user_string)
    return render_template('predict.html',posts=post_dict)

def stemming(post):
    """A function for stemming the words in a list of posts"""
    stemmer = SnowballStemmer("english")
    post = [stemmer.stem(word) for word in post.split(" ")]
    post = " ".join(post)
    return post

def tfidf_vect(stemmed_string,tfidf_vectorizer):
    # Use tf-idf features for NMF.
    tfidf = tfidf_vectorizer.transform([stemmed_string])
    return tfidf

def NMF_Kullback(tfidf,nmf2):
    # Transform with NMF model
    nmf = nmf2.transform(tfidf)
    return nmf

def sim_rec(user_data, item_data, index, top_n = 10):
    """Calculates post recommendations for an individual user
    """
    if index in user_data.index:
        s = cosine_similarity(user_data[user_data.index==index], item_data).flatten()
        post_indexes = s.argsort()[::-1][0:top_n]
        return [item_data.index[i] for i in post_indexes]
    else:
        return 'No data on this user...'

def user_lookup(user_name):
    dfuser = pd.read_csv('data/usernames.txt',sep='\t', header=1, parse_dates=['joindate'], skiprows=0, index_col='userid')
    if user_name in set(dfuser['name']):
        return dfuser[dfuser['name']==user_name].index[0]
    else:
        return None

def post_lookup(post_list):
    post_titles = pd.read_csv('data/posttitles_mefi.txt',sep='\t', header=1, skiprows=0, index_col='postid', error_bad_lines=False)
    return {p: post_titles[post_titles.index==p]['title'].values[0] for p in post_list if len(post_titles[post_titles.index==p]['title']) >= 1}

def recommend_user(user_name):
    index = user_lookup(user_name)#look up the index of a user
    if index != None:
        posts = sim_rec(user_data, item_data, index,top_n=10)
        pt = post_lookup(posts)
        if len(pt) >=1:
            return post_lookup(posts)
        else:
            return 'User not found'

def recommend_text(string):
    data_stemmed = stemming(string) #change to pd.read_json if already present
    tfidf_vectorizer = pickle.load(open('data/tfidfvectorizer', 'rb'))
    nmf2 = pickle.load(open('data/NMF Kullback-Leibler', 'rb'))
    tfidf = tfidf_vect(data_stemmed, tfidf_vectorizer)
    nmf_features = NMF_Kullback(tfidf, nmf2)
    nmf_features = pd.DataFrame(nmf_features)
    postids = sim_rec(nmf_features, item_data, 0,top_n=10)
    return post_lookup(postids)

if __name__ == '__main__':
    user_data = pd.read_json('data/user_data2')#complete user data
    item_data = pd.read_json('data/item_data2')#syntehtic new posts

    app.run(host='0.0.0.0', port=8080, debug=True)
