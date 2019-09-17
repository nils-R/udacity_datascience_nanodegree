import json
import pandas as pd
import plotly
import waitress
import joblib
from flask import Flask, render_template, request
from plotly.graph_objs import Bar, Heatmap, Table
from sqlalchemy import create_engine
import sys, os

sys.path.append(os.getcwd()) # inelegant but required to unpickle model

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
   
    # extract data needed for visuals
    df_categories = df.loc[:,'related':]

    ## 1st plot data
    cate_counts = df_categories.sum().sort_values(ascending=False).reset_index()
    cate_counts.columns = ['category', 'frequency']
    cate_names = cate_counts['category']

    ## 2nd plot data
    corr = df_categories[cate_names].corr()

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=cate_counts['category'],
                    y=cate_counts['frequency']
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    x=cate_names,
                    y=cate_names,
                    z=corr
                )
            ],

            'layout': {
                'title': "Correlation Between Message Categories",
                'yaxis': {
                    'title':"Message Category"
                },
                'xaxis': {
                    'title': "Message Category"
                }
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    #app.run(host='127.0.0.1', port=5000, debug=True)
    waitress.serve(app, host='127.0.0.1', port=5000)


if __name__ == '__main__':
    main()