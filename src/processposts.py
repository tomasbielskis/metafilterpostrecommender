from collections import defaultdict
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import bs4
from pymongo import MongoClient
import multiprocessing

def extract_posts_comments():
    posts = defaultdict(str)
    comments = defaultdict(str)
    commentids = []
    for file in os.listdir(path):
        with open(path+file,"r") as f:
            page = f.read()
        soup = BeautifulSoup(page, 'html.parser')
        commenttext = soup.find_all('div', class_ = 'comments')
        post_text = soup.find('div', attrs={'class': 'copy'})
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
    pd.DataFrame([posts]).to_json('../data/posttext')
    pd.DataFrame([comments]).to_json('../data/commenttext')

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
