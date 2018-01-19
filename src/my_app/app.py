from flask import Flask, request, render_template
import pickle
import pandas as pd
from build_model import TextClassifier, get_data
app = Flask(__name__)

# home page
@app.route('/')
def index():
    return render_template('index.html', title='Topic Predictions!')

# Form page to submit text
@app.route('/submission_page/')
def submission_page():
    return '''
        <form action="/topic_predictor" method='POST' >
            <input type="text" name="user_input" />
            <input type="submit" />
        </form>
        '''

# My word counter app
@app.route('/topic_predictor', methods=['POST'] )
def topic_predictor():
    text = str(request.form['user_input'])
    with open('/home/tomas/galvanize/dsi-data-products/my_app/static/model.pkl', 'rb') as f:
        model = pickle.load(f)
    page = 'This is your topic {0}'
    return page.format(model.predict([text]))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
