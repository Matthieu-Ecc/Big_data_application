#!/usr/bin/env python
# coding: utf-8


import pandas as pd

X_train = pd.read_csv("data/X_train_ready.csv")
X_test  = pd.read_csv("data/X_test_ready.csv")
y_train = pd.read_csv("data/y_train_ready.csv")
y_test  = pd.read_csv("data/y_test_ready.csv")



"""
    the goal of the Training-model.py file is to train the three differentes model.

    -we start by loading the dataset
    -we import the librarys
    -we loaded the models and make them fit to the data
    -and then we dump them in a model folder in order to save them for after

    
    
"""


X_train = X_train.drop(columns='Unnamed: 0')
X_test = X_test.drop(columns='Unnamed: 0')
y_train = y_train.drop(columns='Unnamed: 0')
y_test  = y_test.drop(columns='Unnamed: 0')


"""
# Fitting Random Forest classifier to the dataset
# import the regressor"""
from sklearn.ensemble import RandomForestClassifier
  
""" 
 # create regressor object"""

classifier = RandomForestClassifier(max_depth=10)

"""
# fit the regressor with x and y data """

classifier.fit(X_train, y_train) 

"""
# Fitting the xgboost 
"""
import xgboost as xgb
model=xgb.XGBClassifier(random_state=3,learning_rate=0.01)
model.fit(X_train, y_train)


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)


from joblib import dump
dump(model, 'model/Model_XGboost')
dump(classifier,'model/Model_tree_classifier')
dump(gbc,'model/gbc')

