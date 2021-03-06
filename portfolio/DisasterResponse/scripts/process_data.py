import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages and categories datasets from the specified filepaths
    Args:
        messages_filepath: Filepath to the messages dataset
        categories_filepath: Filepath to the categories dataset
    Returns:
        (DataFrame) df: Merged Pandas dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Cleans the merged dataset
    Args:
        df: dataframe
    Returns:
        (DataFrame) df: Cleaned dataframe
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.loc[0]
    category_colnames = list(pd.Series(row).str.split('-', expand=True)[0])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str.slice(-1)
        categories[column] = categories[column].astype(int)
    
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df[~df.duplicated(subset=None, keep='first')]
    
    return df


def save_data(df, database_filepath, table_name):
    """
    Saves clean dataset into a sqlite database
    Args:
        df:  Cleaned dataframe
        database_filename: Name of the database file
        table_name:
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # make splits to be able to easily test inference module
        df_train_test = df.sample(frac=0.9)
        df_inference = df.loc[~df.index.isin(df_train_test.index), :'genre']
        save_data(df_inference, database_filepath, 'messages_inference')

        print('Cleaning data...')
        df_train_test = clean_data(df_train_test)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df_train_test, database_filepath, 'messages_train')
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()