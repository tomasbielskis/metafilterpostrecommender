from flask import Flask, request, render_template
import pickle
import pandas as pd
app = Flask(__name__)

# home page
@app.route('/')
def index():
    return render_template('index.html', title='Metafilter.com post recommender')

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
    user_name = str(request.form['user_input'])
    page = 'These are some posts that you might like: {0}'
    return page.format(model.predict([user_name]))

if __name__ == '__main__':
    with open('static/model.pkl', 'rb') as f:
        model = pickle.load(f)


    app.run(host='0.0.0.0', port=8080, debug=True)
