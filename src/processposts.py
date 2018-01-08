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

    # Set up mongodb to collect the parsed data incrementally
    DB_NAME = "mefi"
    COLLECTION_NAME1 = "posts"
    COLLECTION_NAME2 = "comments"

    client = MongoClient()
    db = client[DB_NAME]
    coll1 = db[COLLECTION_NAME1]
    coll2 = db[COLLECTION_NAME2]

    # Clear mongodb in case it exists previously
    coll1.remove({})
    coll2.remove({})

    # Loop over raw text files produced by the scraper
    for file in os.listdir(path):
        with open(path+file,"r") as f:
            page = f.read()
        soup = BeautifulSoup(page, 'html.parser')

        # Parse post text and post ids
        post_text = soup.find('div', attrs={'class': 'copy'})
        if post_text:
            psmallcopy = post_text.find('span', class_='smallcopy')
            # Get rid of the smallcopy to avoid having to clean up later
            if psmallcopy:
                psmallcopy.decompose()
            # Add posts to a dictionary
            posts[file] = post_text.get_text()
            # Add posts to mongodb as a backup
            coll1.insert({'pid': file, 'ptext': post_text.get_text()},check_keys=False)

        # Parse text for the comments
        commenttext = soup.find_all('div', class_ = 'comments')
        if commenttext:
            for t in commenttext:
                # Capture the post ids from the previous name tag
                commentid = t.previous_sibling
                if type(commentid) is bs4.element.Tag:
                    # Clean out the smallcopy as it's unnecessary for our purposes
                    csmallcopy = t.find('span', class_='smallcopy')
                    if csmallcopy:
                        csmallcopy.decompose()
                    # Some name id's contain text and throw a KeyError, ignore those
                    try:
                        comments[commentid['name']] = t.text
                        coll2.insert({'cid':commentid['name'], 'ctext': t.text},check_keys=False)
                    except KeyError:
                        continue

    # Once the job completed successfully, record posts and comments as json files of pd data frames
    postdf = pd.DataFrame.from_dict(posts,orient='index')
    postdf.to_json('../data/posttext')

    commentdf = pd.DataFrame.from_dict(comments,orient='index')
    commentdf.to_json('../data/commenttext')


if __name__ == '__main__':
    path = '../data/posts/'
    extract_posts_comments()
