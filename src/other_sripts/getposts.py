import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

url = 'https://www.metafilter.com/'
params = ['171194']

def get_post_text(url, params):
    page = requests.get(url+params).content
    soup = BeautifulSoup(page, 'html.parser')
    post_text = soup.find('div', attrs={'class': 'copy'}).get_text()
    return post_text

def get_post_ids(file):
    '''returns list of post ids'''
    pass

def save_documents():
    pass

def getmfpages(params):
    """passed a list of post_ids returns the page content for those posts"""
    url = 'https://www.metafilter.com/'
    page = requests.get(url+params).content
