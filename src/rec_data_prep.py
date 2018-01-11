from time import time
import pandas as pd
import numpy as np
import pickle

def formatdate(df, datefield):
    df[datefield] = pd.to_datetime(df[datefield],
                                   format='%b %d %Y %H:%M:%S:%f%p')
    return df

def process_favorites_data():
    dffavorites = pd.read_csv('../data/favoritesdata.txt',sep='\t', header=1,
                              parse_dates=['datestamp'], skiprows=0,
                              index_col='faveid')
    dffavorites = formatdate(dffavorites,'datestamp')
    # Filter the recent favorites
    dffavorites = dffavorites[dffavorites['datestamp'].dt.year >= 2014]
    # Select only the posts data (type = 1, 3, 5)
    dffavorites = dffavorites[dffavorites['type'].isin([1, 3, 5])]
    # Drop unnecessary columns
    dffavorites = dffavorites[['faver', 'target']]
    # Look up the train/test indexes
    ptraindata = pd.read_json('../data/nlp/ptraindata',typ='series')
    ptestdata = pd.read_json('../data/nlp/ptestdata', typ='series')
    dffavorites_train = dffavorites[dffavorites['target'].isin(ptraindata.index)]
    dffavorites_test = dffavorites[dffavorites['target'].isin(ptestdata.index)]
    dffavorites_train.to_json('../data/gl_fav_train')
    dffavorites_test.to_json('../data/gl_fav_test')

def process_item_train_data():
    # Grab post and comment train data for index
    ptraindata = pd.read_json('../data/nlp/ptraindata',typ='series')
    ctraindata = pd.read_json('../data/nlp/ctraindata',typ='series')

    # Load text feature models
    model_path = '../data/nlp/train/'
    tfidf = pickle.load(open(model_path+'tfidfmodel', 'rb'))
    tf = pickle.load(open(model_path+'tfmodel', 'rb'))
    nmf1 = pickle.load(open(model_path+'NMF_Frobenius', 'rb'))
    nmf2 = pickle.load(open(model_path+'NMF Kullback-Leibler', 'rb'))
    lda = pickle.load(open(model_path+'LDA', 'rb'))
    posts = len(ptraindata)
    W1 = nmf1.transform(tfidf)
    W2 = nmf2.transform(tfidf)
    W3 = lda.transform(tf)

    # Load user posts and user comments metadata from the mefi datadump
    dfposts = pd.read_csv('../data/postdata_mefi.txt',sep='\t', header=1,
                          parse_dates=['datestamp'], skiprows=0, index_col='postid')
    dfcomments = pd.read_csv('../data/commentdata_mefi.txt',sep='\t', header=1,
                             parse_dates=['datestamp'], skiprows=0, index_col='postid')

    i = 1
    # save to json the three text feature models for the posts (not comments)
    for w in [W1, W2, W3]:
        item_data_train = pd.DataFrame(w)[:posts]
        com_data_train = pd.DataFrame(w)[posts:]
        item_data_train['postid'] = ptraindata.index
        com_data_train['commentid'] = ctraindata.index
        item_data_train.to_json('../data/gl_item_train_data'+str(i))

        user_post_features = dfposts[['postid','userid']].join(item_data_train,on='postid',how='inner')
        user_comment_features = dfcomments[['commentid','userid']].join(com_data_train,on='commentid', how='inner')

        user_data = user_post_features.drop('postid').append(user_comment_features.drop('commentid')
        user_data = user_data.groupby('userid').mean()
        user_data.to_json('../data/gl_user_train_data'+str(i))
        i += 1

# for each user in dfposts, pull up the postid and find the nmf features for it
# for each user in dfcomments pull up the commentid and find nmf features for it
# then average all the nmf columns for that user

# need same for test
# incorporate other metadata on items such as number of comments, number of favorites, recency
# def process_user_data():


def process_item_test_data():
    # Grab the post and comment data for the index
    ptestdata = pd.read_json('../data/nlp/ptestdata',typ='series')
    ctestdata = pd.read_json('../data/nlp/ctestdata',typ='series')

    # Load text feature models
    model_path = '../data/nlp/test/'
    nmf1 = pickle.load(open(model_path+'NMF_Frobenius', 'rb'))
    nmf2 = pickle.load(open(model_path+'NMF Kullback-Leibler', 'rb'))
    lda = pickle.load(open(model_path+'LDA', 'rb'))
    posts = len(ptestdata)

    # Load user posts and user comments metadata from the mefi datadump
    dfposts = pd.read_csv('../data/postdata_mefi.txt',sep='\t', header=1,
                          parse_dates=['datestamp'], skiprows=0, index_col='postid')
    dfcomments = pd.read_csv('../data/commentdata_mefi.txt',sep='\t', header=1,
                             parse_dates=['datestamp'], skiprows=0, index_col='postid')

    i = 1
    # save to json the three text feature models for the posts (not comments)
    for w in [nmf1, nmf2, lda]:
        item_data_test = pd.DataFrame(w)[:posts]
        com_data_test = pd.DataFrame(w)[posts:]
        item_data_test['postid'] = ptestdata.index
        com_data_test['commentid'] = ctestdata.index
        item_data_test.to_json('../data/gl_item_test_data'+str(i))

        user_post_features = dfposts[['postid','userid']].join(item_data_test,on='postid',how='inner')
        user_comment_features = dfcomments[['commentid','userid']].join(com_data_test,on='commentid', how='inner')

        user_data = user_post_features.drop('postid').append(user_comment_features.drop('commentid')
        user_data = user_data.groupby('userid').mean()
        user_data.to_json('../data/gl_user_test_data'+str(i))
        i += 1

if __name__ == '__main__':
    process_favorites_data()
    process_item_train_data()
    process_item_test_data()