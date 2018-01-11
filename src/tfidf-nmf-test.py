from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
from nltk.stem.snowball import SnowballStemmer
import multiprocessing
import threading
import os

def print_top_words(model, feature_names, n_top_words=20):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

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
    print("Stemming words...")
    t0 = time()
    string_list = stem_parrallel(pool_size, string_list)
    string_list = pd.Series(string_list)
    string_list.to_json(path+'stemmedposts')
    print("done in %0.3fs." % (time() - t0))
    return string_list

def tfidf_vect(path, string_list):
    # Use tf-idf features for NMF.
    print("Extracting tf-idf features for NMF...")
    t0 = time()
    tfidf = tfidf_vectorizer.transform(string_list)
    pickle.dump(tfidf, open(path+'tfidfmodel', 'wb'))#pickling a sparse matrix not the best, use scipy sparse save_npz
    print("done in %0.3fs." % (time() - t0))
    return tfidf

def tf_vect(path, string_list):
    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    t0 = time()
    tf = tf_vectorizer.transform(string_list)
    pickle.dump(tf, open(path+'tfmodel', 'wb'))#not use pickle
    print("done in %0.3fs." % (time() - t0))
    print()
    return tf

def NMF_Frob(path, tfidf, tfidf_vectorizer):
    # Fit the NMF model
    print("Transforming tf-idf features with NMF model (Frobenius norm), "
          "n_features=%d..."
          % (n_features))
    t0 = time()
    nmf = nmf1.transform(tfidf)
    pickle.dump(nmf, open(path+'NMF_Frobenius', 'wb'))
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in NMF model (Frobenius norm):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf1, tfidf_feature_names, n_top_words)

def NMF_Kullback(path, tfidf, tfidf_vectorizer):
    # Fit the NMF model
    print("Transforming tf-idf features with NMF model (generalized Kullback-Leibler divergence) "
          "n_features=%d..."
          % (n_features))
    t0 = time()
    nmf = nmf2.transform(tfidf)
    pickle.dump(nmf, open(path+'NMF Kullback-Leibler', 'wb'))
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf2, tfidf_feature_names, n_top_words)

def LDA_func(path, tf, tf_vectorizer):
    print("Transforming tf features with LDA model, "
          "n_features=%d..."
          % (n_features))
    t0 = time()
    lda = lda1.transform(tf)
    pickle.dump(lda, open(path+'LDA', 'wb'))
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda1, tf_feature_names, n_top_words)

if __name__ == '__main__':
    # Setting up the parameters
    # n_samples = 200000
    n_features = 50000 #max features used in NMF including unigrams and bigrams
    n_components = 50 #latent features
    n_top_words = 20 #for printing model output
    pool_size = 64 #number of processors on the machine for stemming
    # Set the path to save files
    output_path = '../data/nlp/test/'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # data = load_data() # load data function or other
    data = pd.read_json('../data/nlp/testdata',typ='series')
    data_stemmed = run_stemmer(output_path, data) #change to pd.read_json if already present
    # data_stemmed = pd.read_json('../data/stemmed')

    # Load models
    model_path = '../data/nlp/train/'
    tfidf_vectorizer = pickle.load(open(model_path+'tfidfvectorizer', 'rb'))
    tf_vectorizer = pickle.load(open(model_path+'tfvectorizer', 'rb'))
    nmf1 = pickle.load(open(model_path+'NMF_Frobenius', 'rb'))
    nmf2 = pickle.load(open(model_path+'NMF Kullback-Leibler', 'rb'))
    lda1 = pickle.load(open(model_path+'LDA', 'rb'))


    tfidf = tfidf_vect(output_path, data_stemmed) #change to pickle.laod if already present
    tf = tf_vect(output_path, data_stemmed) ##change to pickle.laod if already present

    NMF_Frob(output_path, tfidf, tfidf_vectorizer)
    NMF_Kullback(output_path, tfidf, tfidf_vectorizer)
    LDA_func(output_path, tf, tf_vectorizer)
