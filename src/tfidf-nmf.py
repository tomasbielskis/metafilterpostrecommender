from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pandas as pd
import numpy as np
import pickle
from nltk.stem.snowball import SnowballStemmer
import multiprocessing
import threading

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

if __name__ == '__main__':

    # n_samples = 200000
    n_features = 50000
    n_components = 50
    n_top_words = 20
    pool_size = 4

    print("Loading dataset...")
    t0 = time()
    posttext = pd.read_json('../data/parsedtext/posttext')[0][0:10000]
    commenttext = []#pd.read_json('../data/parsedtext/commenttext')[0]
    dataset = posttext.append(commenttext)

    data_samples = dataset#[:n_samples]
    print("done in %0.3fs." % (time() - t0))

    print("Stemming words...")
    t0 = time()
    data_samples = stem_parrallel(pool_size, data_samples)
    data_samples = pd.Series(data_samples)
    data_samples.to_json('../data/stemmedposts')
    print("done in %0.3fs." % (time() - t0))

    # Use tf-idf features for NMF.
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       stop_words='english',
                                       ngram_range=(1,2))
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    pickle.dump(tfidf_vectorizer, open('../data/tfidfvectorizer', 'wb'))
    pickle.dump(tfidf, open('../data/tfidfmodel', 'wb'))#pickling a sparse matrix not the best, use scipy sparse save_npz
    print("done in %0.3fs." % (time() - t0))

    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    stop_words='english',
                                    ngram_range=(1, 2))
    t0 = time()
    tf = tf_vectorizer.fit_transform(data_samples)
    pickle.dump(tf_vectorizer, open('../data/tfvectorizer', 'wb'))
    pickle.dump(tf, open('../data/tfmodel', 'wb'))#not use pickle
    print("done in %0.3fs." % (time() - t0))
    print()

    # Fit the NMF model
    print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
          "n_features=%d..."
          % (n_features))
    t0 = time()
    nmf = NMF(n_components=n_components, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)
    pickle.dump(nmf, open('../data/NMF Frobenius norm', 'wb'))
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in NMF model (Frobenius norm):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)

    # Fit the NMF model
    print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
          "tf-idf features, n_features=%d..."
          % (n_features))
    t0 = time()
    nmf = NMF(n_components=n_components, random_state=1,
              beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
              l1_ratio=.5).fit(tfidf)
    pickle.dump(nmf, open('../data/NMF Kullback-Leibler', 'wb'))
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)


    print("Fitting LDA models with tf features, "
          "n_features=%d..."
          % (n_features))
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    t0 = time()
    lda.fit(tf)
    pickle.dump(lda, open('../data/LDA', 'wb'))
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)
