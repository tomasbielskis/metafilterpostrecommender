import pandas as pd
import numpy as np

def formatdate(df, datefield):
    df[datefield] = pd.to_datetime(df[datefield],
                                      format='%b %d %Y %H:%M:%S:%f%p')
    return df

if __name__ == '__main__':

    # dffavorites = pd.read_csv('../data/favoritesdata.txt',sep='\t', header=1,
    #                     parse_dates=['datestamp'], skiprows=0,
    #                     index_col='faveid')

    # dfusers = pd.read_csv('../data/usernames.txt',sep='\t', header=1,
    #                     parse_dates=['joindate'], skiprows=0, index_col='userid')

    dfposts = pd.read_csv('../data/postdata_mefi.txt',sep='\t', header=1,
                        parse_dates=['datestamp'], skiprows=0, index_col='postid')

    dfcomments = pd.read_csv('../data/commentdata_mefi.txt',sep='\t', header=1,
                        parse_dates=['datestamp'], skiprows=0, index_col='postid')

    posttext = pd.read_json('../data/parsedtext/posttext')
    commenttext = pd.read_json('../data/parsedtext/commenttext')

    for df in [dfposts, dfcomments, posttext, commenttext]:
        df.index = df.index.map(str)

    dfposts = dfposts.join(posttext, how='left')
    dfcomments = dfcomments.join(commenttext, how='left')

    dfposts = formatdate(dfposts, 'datestamp')
    dfcomments = formatdate(dfcomments, 'datestamp')
