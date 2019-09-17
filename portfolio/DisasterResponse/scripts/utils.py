import pickle

import nltk
import pandas as pd
from sqlalchemy import create_engine
import json
import plotly
from plotly.graph_objs import Bar, Heatmap, Table

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def tokenize(text, remove_stopwords=True):
    """
    Tokenizes a given text.
    Args:
        text: text string
        remove_stopwords: option to remove stopwords or not
    Returns:
        (array) clean_tokens: array of clean tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    stopw = set(stopwords.words('english')) if remove_stopwords else []

    cleaned_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens if token not in stopw]
    return cleaned_tokens


def load_data_from_db(dbfilepath, tablename='messages_train'):
    """
    Loads data from database
    Args:
        database_filepath: path to database
        tablename: name of table to read from
    Returns:
        (DataFrame) X: feature
        (DataFrame) Y: labels
        (array) category_names: column headers
    """
    engine = create_engine(f'sqlite:///{dbfilepath}')
    df = pd.read_sql_table(tablename, engine)
    X = df.message
    Y = df.loc[:, 'request':]
    category_names = Y.columns
    return df, X, Y, category_names


def load_model(model_filepath):
    """
    Load the model from a Python pickle
    Args:
        model_filepath: Path where to load the model
    """
    with open(model_filepath, 'rb') as f:
        return pickle.load(f)


def generate_plotly_graphs(df_categories):
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
    return graphJSON, ids


