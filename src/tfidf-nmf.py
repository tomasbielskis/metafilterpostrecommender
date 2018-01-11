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

def load_data():
    print("Loading and splitting (train/test) dataset...")
    t0 = time()
    posttext = pd.read_json('../data/parsedtext/posttext')[0]#[0:10000]
    commenttext = pd.read_json('../data/parsedtext/commenttext')[0]#[0:10000]
    ptrain, ptest = train_test_split(posttext, test_size=0.2, random_state=42)
    ctrain, ctest = train_test_split(commenttext, test_size=0.2, random_state=42)

    train = ptrain.append(ctrain).reset_index()
    test = ptest.append(ctest).reset_index()
    ptrain.to_json('../data/nlp/ptraindata')
    ptest.to_json('../data/nlp/ptestdata')
    ctrain.to_json('../data/nlp/ctraindata')
    ctest.to_json('../data/nlp/ctestdata')

    train.to_json('../data/nlp/traindata')
    test.to_json('../data/nlp/testdata')
    print("done in %0.3fs." % (time() - t0))
    return train #switch to process the other ones

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
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       stop_words='english',
                                       ngram_range=(1,2))
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(string_list)
    pickle.dump(tfidf_vectorizer, open(path+'tfidfvectorizer', 'wb'))
    pickle.dump(tfidf, open(path+'tfidfmodel', 'wb'))#pickling a sparse matrix not the best, use scipy sparse save_npz
    print("done in %0.3fs." % (time() - t0))
    return tfidf, tfidf_vectorizer

def tf_vect(path, string_list):
    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    stop_words='english',
                                    ngram_range=(1, 2))
    t0 = time()
    tf = tf_vectorizer.fit_transform(string_list)
    pickle.dump(tf_vectorizer, open(path+'tfvectorizer', 'wb'))
    pickle.dump(tf, open(path+'tfmodel', 'wb'))#not use pickle
    print("done in %0.3fs." % (time() - t0))
    print()
    return tf, tf_vectorizer

def NMF_Frob(path, tfidf, tfidf_vectorizer):

    # Fit the NMF model
    print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
          "n_features=%d..."
          % (n_features))
    t0 = time()
    nmf = NMF(n_components=n_components, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)
    pickle.dump(nmf, open(path+'NMF_Frobenius', 'wb'))
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in NMF model (Frobenius norm):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)

def NMF_Kullback(path, tfidf, tfidf_vectorizer):
    # Fit the NMF model
    print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
          "tf-idf features, n_features=%d..."
          % (n_features))
    t0 = time()
    nmf = NMF(n_components=n_components, random_state=1,
              beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
              l1_ratio=.5).fit(tfidf)
    pickle.dump(nmf, open(path+'NMF Kullback-Leibler', 'wb'))
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)

def LDA_func(path, tf, tf_vectorizer):
    print("Fitting LDA models with tf features, "
          "n_features=%d..."
          % (n_features))
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    t0 = time()
    lda.fit(tf)
    pickle.dump(lda, open(path+'LDA', 'wb'))
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)

if __name__ == '__main__':
    # Setting up the parameters
    # n_samples = 200000
    n_features = 50000 #max features used in NMF including unigrams and bigrams
    n_components = 50 #latent features
    n_top_words = 20 #for printing model output
    pool_size = 32 #number of processors on the machine for stemming
    # Set the path to save files
    output_path = '../data/nlp/train/'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    data = load_data() # load data function or other
    data_stemmed = run_stemmer(output_path, data) #change to pd.read_json if already present
    # data_stemmed = pd.read_json('../data/stemmed')
    tfidf, tfidf_vectorizer = tfidf_vect(output_path, data_stemmed) #change to pickle.laod if already present
    tf, tf_vectorizer = tf_vect(output_path, data_stemmed) ##change to pickle.laod if already present

    NMF_Frob(output_path, tfidf, tfidf_vectorizer)
    NMF_Kullback(output_path, tfidf, tfidf_vectorizer)
    LDA_func(output_path, tf, tf_vectorizer)
