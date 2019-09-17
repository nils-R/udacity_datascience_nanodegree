import pandas as pd
from flask import Flask, render_template, request
import joblib
from sqlalchemy import create_engine
import sys, os

sys.path.append(os.getcwd()) # inelegant but required to unpickle model
import scripts.utils as utils

app = Flask(__name__)

# load data
df, _, _, _ = utils.load_data_from_db('data/DisasterResponse.db', 'messages')

# extract data needed for visuals
df_categories = df.loc[:, 'request':]

# load model
model = utils.load_model("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
   
    graphJSON, ids = utils.generate_plotly_graphs(df_categories)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df_categories.columns, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()