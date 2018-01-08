import pandas as pd
import numpy as np
import os
import requests
from timeit import Timer
import multiprocessing
import threading
import boto

def getmfpage(postid):
    """passed a post_id saves the page content for the post as a json file in data/posts"""
    url = 'https://www.metafilter.com/'
    page = requests.get(url+str(postid)).content
    path = '../data/posts/'
    with open(path+str(postid),'wb') as f:
        f.write(page)

def scrape_sequential(n):
    """Sequential job, used only for testing and performance comparisson.
    Loops through n postsids and calls getmfpage for them"""
    for p in dfposts.index[0:n]:
        if p not in os.listdir('data/posts'):
            getmfpage(p)

def scrape_parallel_concurrent(pool_size, post_list):
    """The parrallel job for scraping. Takes the number of cores and a list of post ids"""
    pool = multiprocessing.Pool(pool_size)
    pool.map(getmfpage, post_list)
    pool.close()
    pool.join()

if __name__ == '__main__':

    # Get the complete list of postids from the metafilter infodump,
    # use download_infodump.py to get the post metadata archive
    dfposts = pd.read_csv('../data/postdata_mefi.txt',sep='\t', header=1,
                        parse_dates=['datestamp'], skiprows=0, index_col='postid')
    # Specify pool size based on the number of cores on the machine
    pool_size = 8

    # Use boto on the s3 bucket to check which posts have already been scraped
    conn = boto.connect_s3()
    b = conn.get_bucket('tomasbielskis-galvanizebucket')
    filenames = [f.name.strip('capstone/data/posts/') for f in b.list(prefix='capstone/data/posts/')]

    # Exclude the pages that have already been scraped
    post_list = [x for x in dfposts.index if str(x) not in filenames]

    # t = Timer(lambda: scrape_sequential(n))
    # print("Completed sequential in %s seconds." % t.timeit(1))

    # Run and time the scraper
    t = Timer(lambda: scrape_parallel_concurrent(pool_size,post_list))
    print("Completed using thrseads in %s seconds." % t.timeit(1))
