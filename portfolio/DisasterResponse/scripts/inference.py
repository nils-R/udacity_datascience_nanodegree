import sys, os
import pandas as pd

sys.path.append(os.getcwd()) # inelegant but works
from scripts.utils import load_data_from_db, load_model
from scripts.train_classifier import make_predictions


def main():
    if len(sys.argv) == 2:
        json_file_path = sys.argv[1]
        model = load_model(model_filepath='models/classifier.pkl')
        df = pd.read_json(json_file_path)
        X = df.message
        y_pred = make_predictions(model, X)
        _, _, _, category_names = load_data_from_db('data/DisasterResponse.db', 'messages_train')
        y_pred = pd.DataFrame(y_pred, columns = category_names)
        df_pred = pd.concat([df[['id', 'message']], y_pred], axis=1)
        return df_pred.to_json(orient='records')
    else:
        print('Please provide filepath to json-file')


if __name__ == '__main__':
    df_pred = main()