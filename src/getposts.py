import requests
from bs4 import BeautifulSoup

url = 'https://www.metafilter.com/'
params = '171194'

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
