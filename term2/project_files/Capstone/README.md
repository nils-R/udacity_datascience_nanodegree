{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Sparkify\n",
    "## Project overview\n",
    "The goal of this project is to build a customer churn prediction model for a ficticious music streaming service called \"Sparkify\". Based on historical user actions, the model\n",
    "tries to predict whether it is likely that a user will cancel his/her service. \n",
    "The project is the capstone of [Udacity's](https://eu.udacity.com/) Data Science Nanodegree and a blogpost describing the modelling process can be found [here](https://medium.com/p/d6fba2f24e0e/edit)\n",
    "\n",
    "### Dataset \n",
    "The dataset at hand contains user interactions with the Sparkify service, e.g. songs played, friends added and thumbs up/down. Each interaction also contains information about the user, such as gender and device used. The full dataset is available at Udacity Amazon s3 bucket: https://s3.amazonaws.com/video.udacity-data.com/topher/2018/December/5c1d6681_medium-sparkify-event-data/medium-sparkify-event-data.json\n",
    "\n",
    "## Modelling\n",
    "### Feature engineering\n",
    "For each user in the dataset we add a binary target label, as well as a number of categorical and numerical input features, e.g:\n",
    "* time since registration\n",
    "* gender\n",
    "* home state\n",
    "* average session duration\n",
    "* number of sessions\n",
    "* number of visits to every page\n",
    "\n",
    "### Model training and evaluation\n",
    "We create two candidate datasets - one where page activity is summed, one where it is divided by number of sessions. For each dataset, we split them into train and test parts.\n",
    "Thereafter a pipeline is defined and a number of candidate estimators are trained. We use AUROC to evaluate what the best performing model is and end up with a random forest estimator, achieving an AUROC of 0.821 on the test set.\n",
    "\n",
    "## Implementation details\n",
    "All work was done on an IBM Watson Studio cluster, utilizing Python 3.6 and Spark 2.3.\n",
    "### Libraries used\n",
    "* numPy 1.15.4\n",
    "* pandas 0.24.1\n",
    "* pySpark version 2.3.3\n",
    "* matplotlib 3.0.2\n",
    "* seaborn version 0.9.0\n",
    "* sklearn 0.20.2\n",
    "* ibmos2spark 1.0.1\n",
    "* project_lib 1.6.1\n",
    "* IPython 5.8.0\n",
    "\n",
    "### Code\n",
    "All code is contained in the sparkify jupyter notebook\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "",
   "name": ""
  },
  "language_info": {
   "name": ""
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
