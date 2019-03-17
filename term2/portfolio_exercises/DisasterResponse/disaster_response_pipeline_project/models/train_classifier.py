import sys
import pandas as pd
import numpy as np
import pickle

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def load_data(dbfilepath, tablename='messages'):
    '''
    input: (
        dbfilepath: path to database
        tablename: name of table to fetch
            )
    Loads data from sqlite database 
    output: (
        X: features dataframe
        y: target dataframe
        category_names: names of targets
        )
    '''
    engine = create_engine(f'sqlite:///{dbfilepath}')
    df = pd.read_sql_table(tablename, engine)
    X = df.message
    Y = df.loc[:, 'request':]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text, remove_stopwords=True):

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    stopw = set(stopwords.words('english')) if remove_stopwords else []

    cleaned_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens if token not in stopw]
    return cleaned_tokens


def build_model():
    pipeline = Pipeline([
    ('vect' , CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))      
    ])
    
    parameters = {
       #'vect__ngram_range': ((1, 1), (1, 2)),
       # 'tfidf__norm' : ['l2', None], 
       # 'tfidf__smooth_idf' : [True, False],
        'clf__estimator__n_estimators': [30]
    }

    model = GridSearchCV(pipeline, param_grid=parameters, cv=3)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    target_names = Y_test.columns.values
    
    # Calculate performance metrics for each category
    for i in range(len(category_names)):
        print(f'Category:, {category_names[i]} \n {classification_report(Y_test.iloc[:, i].values, Y_pred[:, i])}')
        print(f'Accuracy of {category_names[i]}: {accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i]):.2f}')
        
    print(f'\nBest Parameters: {model.best_params_}')


def save_model(model, model_filepath):
    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:3]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()