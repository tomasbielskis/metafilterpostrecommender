import pickle
import os

def pickle_2(path):
    files = os.listdir(path)
    files.remove('stemmedposts')
    for file in files:
        old_pickle = pickle.load(open(path+file,'rb'))
        pickle.dump(old_pickle, open(path+"p2"+file,'wb'),protocol=2,encoding='latin1')

if __name__ == '__main__':
    path = '../data/stemmedpostsonlymodels/'
    pickle_2(path)
