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

    DB_NAME = "mefi"
    COLLECTION_NAME1 = "posts"
    COLLECTION_NAME2 = "comments"

    client = MongoClient()
    db = client[DB_NAME]
    coll1 = db[COLLECTION_NAME1]
    coll2 = db[COLLECTION_NAME2]

    coll1.remove({})
    coll2.remove({})

    for file in os.listdir(path):
        with open(path+file,"r") as f:
            page = f.read()
        soup = BeautifulSoup(page, 'html.parser')

        post_text = soup.find('div', attrs={'class': 'copy'})
        if post_text:
            psmallcopy = post_text.find('span', class_='smallcopy')
            if psmallcopy:
                psmallcopy.decompose()
            posts[file] = post_text.get_text()
            coll1.insert({'pid': file, 'ptext': post_text.get_text()},check_keys=False)

        commenttext = soup.find_all('div', class_ = 'comments')
        if commenttext:
            for t in commenttext:
                commentid = t.previous_sibling
                if type(commentid) is bs4.element.Tag:
                    csmallcopy = t.find('span', class_='smallcopy')
                    if csmallcopy:
                        csmallcopy.decompose()
                    comments[commentid['name']] = t.text
                    coll2.insert({'cid':commentid['name'], 'ctext': t.text},check_keys=False)

    postdf = pd.DataFrame.from_dict(posts,orient='index')
    commentdf = pd.DataFrame.from_dict(comments,orient='index')
    postdf.to_json('../data/posttext')
    commentdf.to_json('../data/commenttext')


if __name__ == '__main__':
    path = '../data/posts/'
    extract_posts_comments()
