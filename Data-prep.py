#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split

df_train = pd.read_csv("data/application_train.csv")
df_test = pd.read_csv("data/application_test.csv")

def DelNanColumns(df : pd.DataFrame,tsh : int) -> pd.DataFrame:
    cpt=0
    for i in df.isnull().sum():
        verif = i/df.shape[0]*100
        if verif>=tsh:
            print(verif)
            cpt = cpt+1
    
    df = df.drop(df.columns[df.apply(lambda col: col.isnull().sum()/df.shape[0]*100 > tsh)], axis=1)

    print(cpt," columns delted")


    return df


df_train = DelNanColumns(df_train,40)


df_train = df_train.dropna()



df_minus_y = df_train

y = df_train.TARGET

X = df_minus_y.drop('TARGET',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state= 3)


df_test = df_test[X_train.columns]

df_test_prod = df_test.dropna()


X_train.to_csv(path_or_buf="data/X_train")
X_test.to_csv(path_or_buf="data/X_test")
y_train.to_csv(path_or_buf="data/y_train")
y_test.to_csv(path_or_buf="data/y_test")
df_test_prod.to_csv(path_or_buf="data/X_test_prod")


