import requests, zipfile, io
"""script to download the infodump and unzip the files into a specified directory

NOT TESTED YET, since I had downloaded the files before manually and they haven't been updated since

"""

def get_mefi_infodump(path):
    zip_url = 'http://mefi.us/infodump/infodump-all.zip'
    r = requests.get(zip_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)

if __name__ == '__main__':
    path = '../data/'
    get_mefi_infodump(path)
