#!/usr/bin/env python
# coding: utf-8

import numpy
import pandas as pd

X = pd.read_csv("X_test_prod_ready")

X = X.drop(columns='Unnamed: 0')


from joblib import load

xgboost = load('model/Model_XGboost')

tree_classifier = load('model/Model_tree_classifier')

Gradient_boosting_classifiers = load('model/gbc')


def predictByXgboost(X : pd.DataFrame) -> numpy.ndarray:
    prediction = xgboost.predict(X)
    return prediction

def predictByTreeClassifier(X : pd.DataFrame) -> numpy.ndarray:
    prediction = tree_classifier.predict(X)
    return prediction

def predictByGDC(X : pd.DataFrame) -> numpy.ndarray:
    prediction = Gradient_boosting_classifiers.predict(X)
    return prediction



