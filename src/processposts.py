from collections import defaultdict
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import bs4
from pymongo import MongoClient
import multiprocessing

def extract_posts_comments():
    """extracts post and comment ids and text from the mefi pages stored as
    separate text files """
    posts = defaultdict(str)
    comments = defaultdict(str)
    commentids = []

    for file in os.listdir(path):
        with open(path+file,"r") as f:
            page = f.read()
        soup = BeautifulSoup(page, 'html.parser')
        commenttext = soup.find_all('div', class_ = 'comments')
        post_text = soup.find('div', attrs={'class': 'copy'})
        psmallcopy = post_text.find('span', class_='smallcopy')
        if psmallcopy:
            psmallcopy.decompose()
        posts[file] = post_text.get_text()
        commenttext
        for t in commenttext:
            commentid = t.previous_sibling
            if type(commentid) is not bs4.element.Tag:
                continue
            csmallcopy = t.find('span', class_='smallcopy')
            if csmallcopy:
                csmallcopy.decompose()
            comments[commentid['name']] = t.text

    postdf = pd.DataFrame.from_dict(posts,orient='index')
    commentdf = pd.DataFrame.from_dict(comments,orient='index')
    chunksize = 20000
    for i in range(chunksize,len(postdf),chunksize):
        postdf[i-chunksize:i,:].to_json('../data/posttext{}-{}'.format(i-chunksize,i))
    for i in range(chunksize,len(commentdf),chunksize):
        commentdf[i-chunksize:i,:].to_json('../data/commenttext{}-{}'.format(i-chunksize,i))



    # DB_NAME = "mefi"
    # COLLECTION_NAME1 = "posts"
    # COLLECTION_NAME2 = "comments"
    #
    # client = MongoClient()
    # db = client[DB_NAME]
    # coll1 = db[COLLECTION_NAME1]
    # coll2 = db[COLLECTION_NAME2]
    #
    # coll1.insert(posts)
    # coll2.insert(comments)

# def extract_parallel_concurrent(pool_size, file_list):
#     pool = multiprocessing.Pool(pool_size)
#     pool.map(extract_posts_comments, file_list)
#     pool.close()
#     pool.join()


if __name__ == '__main__':
    path = '../data/posts/'
    extract_posts_comments()
