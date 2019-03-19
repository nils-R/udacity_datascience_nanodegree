## Disaster Response Pipeline Project

![Intro Pic](screenshots/classification.jpg)

### Table of Contents
1. [Project Overview](#project_overview)
2. [Project Components](#project_components)
	1. [ETL Pipeline](#etl_pipeline)
	2. [ML Pipeline](#ml_pipeline)
	3. [Web App](#app)
3. [Running Instructions](#running_instructions)
4. [Author](#author)
5. [License](#license)
6. [Acknowledgement](#acknowledgement)
7. [Screenshots](#screenshots)

<a name="project_overview"></a>
### Project Overview

The purpose of this project is to build a model for an API that classifies disaster messages. It also includes a web app where an emergency worker can input a new message to get it classified.

<a name="project_components"></a>
### Project Components

This project contains three main parts: ETL Pipeline, Machine Learning Pipeline and Flask App. 

<a name="etl_pipeline"></a>
1. ETL Pipeline
   * **ETL Pipeline Preparation.ipynb**: This jupyter notebook shows the code and development of ETL pipeline.
   * **process_data.py**: This Python script loads the messages and categories datasets, merges and cleans them, begore storing the resulting dataset in a SQLite database. 
<a name="ml_pipeline"></a>
2. Machine Line Pipeline
   * **ML Pipeline Preparation.ipynb**: This Jupyther notebook shows the code and develoment of Machine Learning Pipeline.
   * **train_classifier.py**:  This Python script loads the data from a SQLite database. It then uses the data to train and tune a Machine Learning model using GridSearchCV. Lastly, it saves the best performing model as a pickle file.
<a name="app"></a>
3. Flask App
   * The web app consists of two sections:
    - Graphs showing descriptive statistics of the dataset
    - An input field for classification of new messages

<a name="running_instructions"></a>
### Running Instructions

1. Run the following commands in the project's root directory to set up the database and model.
   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run the web app.
   `python run.py`
3. Go to http://0.0.0.0:3001/, or http://localhost:3001

<a name="author"></a>
### Authors

* [Nils Randau](https://github.com/nils-R)

<a name="license"></a>
### License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
### Acknowledgements

* [Udacity](https://www.udacity.com/) for providing the web app template
* [Figure Eight](https://www.figure-eight.com/) for providing the dataset

<a name="screenshots"></a>
### Screenshots of the web app

![classification](/images/classification.jpg)

![category_count](/images/category_counts.jpg)

![heatmap](/images/correlation.jpg)

