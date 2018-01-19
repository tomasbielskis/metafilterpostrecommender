import pickle
import pandas as pd
from build_model import TextClassifier, get_data



with open('static/model.pkl', 'rb') as f:
    model = pickle.load(f)

X, y = get_data('data/articles.csv')

print("Accuracy:", model.score(X, y))
print("Predictions:", model.predict(X))
print(model.predict(['string string ball']))
