from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from time import time
import pandas as pd
import numpy as np
import pickle

if __name__ == '__main__':

    model_path = '../data/nlp/train/'
    tfidf = pickle.load(open(model_path+'tfidfmodel', 'rb'))
    tf = pickle.load(open(model_path+'tfmodel', 'rb'))

    # nmf1 = pickle.load(open(model_path+'NMF_Frobenius', 'rb'))
    nmf2 = pickle.load(open(model_path+'NMF Kullback-Leibler', 'rb'))
    lda = pickle.load(open(model_path+'LDA', 'rb'))

    # W1 = nmf1.transform(tfidf)
    W2 = nmf2.transform(tfidf)
    W3 = lda.transform(tf)

    # pickle.dump(W1, open(model_path+'W1', 'wb'))
    pickle.dump(W2, open(model_path+'W2', 'wb'))
    pickle.dump(W3, open(model_path+'W3', 'wb'))




# """Collection of imports and functions to import and process data"""
# import pandas as pd
# import numpy as np
#
# def formatdate(df, datefield):
#     df[datefield] = pd.to_datetime(df[datefield],
#                                    format='%b %d %Y %H:%M:%S:%f%p')
#     return df
#
# if __name__ == '__main__':
#
#     # dffavorites = pd.read_csv('../data/favoritesdata.txt',sep='\t', header=1,
#     #                           parse_dates=['datestamp'], skiprows=0,
#     #                           index_col='faveid')
#
#     # dfusers = pd.read_csv('../data/usernames.txt',sep='\t', header=1,
#     #                       parse_dates=['joindate'], skiprows=0, index_col='userid')
#
#     dfposts = pd.read_csv('../data/postdata_mefi.txt',sep='\t', header=1,
#                           parse_dates=['datestamp'], skiprows=0, index_col='postid')
#
#     dfcomments = pd.read_csv('../data/commentdata_mefi.txt',sep='\t', header=1,
#                              parse_dates=['datestamp'], skiprows=0, index_col='postid')
#
#     posttext = pd.read_json('../data/parsedtext/posttext')[0]
#     commenttext = pd.read_json('../data/parsedtext/commenttext')[0]
#
#     for df in [dfposts, dfcomments, posttext, commenttext]:
#         df.index = df.index.map(str)
#
#     dfposts = dfposts.join(posttext, how='left')
#     dfcomments = dfcomments.join(commenttext, how='left')
#
#     dfposts = formatdate(dfposts, 'datestamp')
#     dfcomments = formatdate(dfcomments, 'datestamp')
#
