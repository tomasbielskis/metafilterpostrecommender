from collections import defaultdict
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import bs4

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
        post_text.find('span', class_='smallcopy').decompose()
        posts[file] = post_text.get_text()
        for t in commenttext:
            commentid = t.previous_sibling
            if type(commentid) is not bs4.element.Tag:
                continue
            t.find('span', class_='smallcopy').decompose()
            comments[commentid['name']] = t.text
    pd.DataFrame([posts]).to_json('/home/tomas/galvanize/capstone/data/posttext')
    pd.DataFrame([comments]).to_json('/home/tomas/galvanize/capstone/data/commenttext')

if __name__ == '__main__':
    path = '/home/tomas/galvanize/capstone/data/posts/'
    extract_posts_comments()
