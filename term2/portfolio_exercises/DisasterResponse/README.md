# Disaster Response Pipeline Project

### Project Overview

The purpose of this project is to build a model for an API that classifies disaster messages. It also includes a web app where an emergency worker can input a new message to get it classified.

### Project Components

This project contains three main parts: ETL Pipeline, Machine Learning Pipeline and Flask App. 

1. ETL Pipeline
   * **ETL Pipeline Preparation.ipynb**: This jupyter notebook shows the code and development of ETL pipeline.
   * **process_data.py**: This Python script loads the messages and categories datasets, merges and cleans them, begore storing the resulting dataset in a SQLite database. 
2. Machine Line Pipeline
   * **ML Pipeline Preparation.ipynb**: This Jupyther notebook shows the code and develoment of Machine Learning Pipeline.
   * **train_classifier.py**:  This Python script loads the data from a SQLite database. It then uses the data to train and tune a Machine Learning model using GridSearchCV. Lastly, it saves the best performing model as a pickle file. 
3. Flask App
   * The web app consists of two sections:
    - Graphs showing descriptive statistics of the dataset
    - An input field for classification of new messages

### Running Instructions

1. Run the following commands in the project's root directory to set up the database and model.
   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run the web app.
   `python run.py`
3. Go to http://0.0.0.0:3001/, or http://localhost:3001

### Screenshots of the web app

![image-1](image1)

![image-2](image2)

![image-3](image3)

