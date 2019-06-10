# Sparkify
## Project overview
The goal of this project is to build a customer churn prediction model for a ficticious music streaming service called "Sparkify". Based on historical user actions, the model
tries to predict whether it is likely that a user will cancel his/her service. 
The project is the capstone of [Udacity's](https://eu.udacity.com/) Data Science Nanodegree and a blogpost describing the modelling process can be found [here](https://medium.com/p/d6fba2f24e0e/edit)

### Dataset 
The dataset at hand contains user interactions with the Sparkify service, e.g. songs played, friends added and thumbs up/down. Each interaction also contains information about the user, such as gender and device used. The full dataset is available at Udacity Amazon s3 bucket: https://s3.amazonaws.com/video.udacity-data.com/topher/2018/December/5c1d6681_medium-sparkify-event-data/medium-sparkify-event-data.json

## Modelling
### Feature engineering
For each user in the dataset we add a binary target label, as well as a number of categorical and numerical input features, e.g:
* time since registration
* gender
* home state
* average session duration
* number of sessions
* number of visits to every page

### Model training and evaluation
We create two candidate datasets - one where page activity is summed, one where it is divided by number of sessions. For each dataset, we split them into train and test parts.
Thereafter a pipeline is defined and a number of candidate estimators are trained. We use AUROC to evaluate what the best performing model is and end up with a random forest estimator, achieving an AUROC of 0.821 on the test set.

## Implementation details
All work was done on an IBM Watson Studio cluster, utilizing Python 3.6 and Spark 2.3.
### Libraries used
* numPy 1.15.4
* pandas 0.24.1
* pySpark version 2.3.3
* matplotlib 3.0.2
* seaborn version 0.9.0
* sklearn 0.20.2
* ibmos2spark 1.0.1
* project_lib 1.6.1
* IPython 5.8.0

### Code
All code is contained in the sparkify jupyter notebook
