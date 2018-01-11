from time import time
import pandas as pd
import numpy as np
import pickle

def formatdate(df, datefield):
    df[datefield] = pd.to_datetime(df[datefield],
                                   format='%b %d %Y %H:%M:%S:%f%p')
    return df

def process_favorites_data():
    # Load favorites for 2017
    # dffavorites2017 = pd.read_csv('../data/favorites2017.csv',
    #                           parse_dates=['datestamp'],
    #                           index_col='faveid')
    # dffavorites2016 = pd.read_csv('../data/favorites2016.csv',
    #                           parse_dates=['datestamp'],
    #                           index_col='faveid')
    # dffavorites2015 = pd.read_csv('../data/favorites2015.csv',
    #                           parse_dates=['datestamp'],
    #                           index_col='faveid')
    # dffavorites2014 = pd.read_csv('../data/favorites2014.csv',
    #                           parse_dates=['datestamp'],
    #                           index_col='faveid')
    dffavorites = pd.read_csv('../data/favoritesdata.txt',sep='\t', header=1,
                              parse_dates=['datestamp'], skiprows=0,
                              index_col='faveid')
    dffavorites = formatdate(dffavorites,'datestamp')

    dffavorites = dffavorites[dffavorites['datestamp'].dt.year >= 2017]

    # Select only the posts data
    dffavorites = dffavorites[dffavorites['type'].isin([1, 3, 5])]
    # Drop unnecessary columns
    dffavorites = dffavorites[['faver', 'target']]

    ptraindata = pd.read_json('../data/nlp/ptraindata',typ='series')
    ptestdata = pd.read_json('../data/nlp/ptestdata', typ='series')

    dffavorites_train = dffavorites[dffavorites['target'].isin(ptraindata.index)]
    dffavorites_test = dffavorites[dffavorites['target'].isin(ptestdata.index)]

    dffavorites_train.to_json('../data/gl_fav_train')
    dffavorites_test.to_json('../data/gl_fav_test')

def process_item_data():
    ptraindata = pd.read_json('../data/nlp/ptraindata',typ='series')
    ctraindata = pd.read_json('../data/nlp/ctraindata',typ='series')

    # Load text feature models
    model_path = '../data/nlp/train/'
    # tfidf_vectorizer = pickle.load(open(model_path+'tfidfvectorizer', 'rb'))
    # tf_vectorizer = pickle.load(open(model_path+'tfvectorizer', 'rb'))
    tfidf = pickle.load(open(model_path+'tfidfmodel', 'rb'))
    # tf = pickle.load(open(model_path+'tfmodel', 'rb'))
    nmf1 = pickle.load(open(model_path+'NMF_Frobenius', 'rb'))
    # nmf2 = pickle.load(open(model_path+'NMF Kullback-Leibler', 'rb'))
    # lda = pickle.load(open(model_path+'LDA', 'rb'))
    posts = len(ptraindata)
    W1 = nmf1.transform(tfidf)
    item_data_train1 = pd.DataFrame(W1)[:posts+1]
    item_data_train1['postid'] = ptraindata.index
    item_data_train1.to_json('../data/gl_item_data_input1')
    # need same for test
    # incorporate other metadata on items such as number of comments, number of favorites, recency

# def process_user_data():
    dfposts = pd.read_csv('../data/postdata_mefi.txt',sep='\t', header=1,
                          parse_dates=['datestamp'], skiprows=0, index_col='postid')
    dfcomments = pd.read_csv('../data/commentdata_mefi.txt',sep='\t', header=1,
                             parse_dates=['datestamp'], skiprows=0, index_col='postid')
    user_post_features = dfposts[['postid','userid']].join(item_data_train1,on='postid',how='left')
    com_data_train1 = pd.DataFrame(W1)[posts:]
    com_data_train1['commentid'] = ctraindata.index
    user_comment_features = dfcomments[['commentid','userid']].join(com_data_train1,on='commentid', how='left')
    user_data = user_post_features.drop('postid').append(user_comment_features.drop('commentid')
    user_data = user_data.groupby('userid').mean()
    user_data.to_json('../data/gl_user_data_input1')

# for each user in dfposts, pull up the postid and find the nmf features for it
# for each user in dfcomments pull up the commentid and find nmf features for it
# then average all the nmf columns for that user

if __name__ == '__main__':
