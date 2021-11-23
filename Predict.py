#!/usr/bin/env python
# coding: utf-8

from os import path
import numpy
import pandas as pd

X = pd.read_csv("data/X_test_prod_ready.csv")

X = X.drop(columns='Unnamed: 0')

from joblib import load

def predictByXgboost(X : pd.DataFrame) -> numpy.ndarray:

    """Produce a prediction set with features in input and using the xgboost model """

    xgboost = load('model/Model_XGboost')

    prediction = xgboost.predict(X)
    return prediction

def predictByTreeClassifier(X : pd.DataFrame) -> numpy.ndarray:
    
    """Produce a prediction set with features in input and using the treeclassifier model """
    
    tree_classifier = load('model/Model_tree_classifier')

    prediction = tree_classifier.predict(X)
    return prediction

def predictByGBC(X : pd.DataFrame) -> numpy.ndarray:

    """Produce a prediction set with features in input and using the gradient boosting classifier model """

    Gradient_boosting_classifiers = load('model/gbc')

    prediction = Gradient_boosting_classifiers.predict(X)
    return prediction


result = predictByGBC(X)

print("result: ",result.sum())


df = pd.DataFrame(result)

df.to_csv(path_or_buf="result/prediction.csv")

