import sys, os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(os.getcwd()) # inelegant but works
from scripts.utils import tokenize, load_data_from_db


def build_model():
    """
    Builds classification model
    Returns:
        (object) model: GridSearchCV object
    """
    pipeline = Pipeline([
    ('vect' , CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
       #'vect__ngram_range': ((1, 1), (1, 2)),
       #'tfidf__norm' : ['l2', None],
       #'tfidf__smooth_idf' : [True, False],
        'clf__estimator__n_estimators': [30],
        'clf__estimator__max_depth': [30]
    }

    model = GridSearchCV(pipeline, param_grid=parameters, verbose=2, cv=2)
    return model


def make_predictions(model, X_test):
    """
    Use model object to make predictions
    :param model:
    :param X_test:
    :return:
    """
    return model.predict(X_test)


def evaluate_model(model, y_test, y_pred, category_names):
    """
    Evaluates the model against a test dataset
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        y_pred: Predicted labels
        category_names: String array of category names
    """

    # Calculate performance metrics for each category
    for i in range(len(category_names)):
        print(f'Category:, {category_names[i]} \n {classification_report(y_test.iloc[:, i].values, y_pred[:, i])}')
        print(f'Accuracy of {category_names[i]}: {accuracy_score(y_test.iloc[:, i].values, y_pred[:, i]):.2f}')
        
    print(f'\nBest Parameters: {model.best_params_}')


def save_model(model, model_filepath):
    """
    Save the model to a Python pickle
    Args:
        model: Trained model
        model_filepath: Path where to save the model
    """    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:3]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        _, X, y, category_names = load_data_from_db(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)

        print('Making predictions...')
        y_pred = make_predictions(model, X_test)
        
        print('Evaluating model...')
        evaluate_model(model, y_test, y_pred, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'scripts/train_classifier.py data/DisasterResponse.db models/classifier.pkl')


if __name__ == '__main__':
    main()