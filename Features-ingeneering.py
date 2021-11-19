#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


X_train = pd.read_csv("data/X_train.csv",index_col=False)
X_test  = pd.read_csv("data/X_test.csv",index_col=False)
y_train = pd.read_csv("data/y_train.csv",index_col=False)
y_test  = pd.read_csv("data/y_test.csv",index_col=False)
X_test_prod = pd.read_csv("data/X_test_prod.csv",index_col=False)


X_train = X_train.drop(columns=['Unnamed: 0','SK_ID_CURR'])
X_test  = X_test.drop(columns=['Unnamed: 0','SK_ID_CURR'])
X_test_prod = X_test_prod.drop(columns=['Unnamed: 0','SK_ID_CURR'])



y_train = y_train.drop(columns='Unnamed: 0')
y_test  = y_test.drop(columns='Unnamed: 0')



def EncodeDataFramelabel(df : pd.DataFrame) -> pd.DataFrame :

    # encode eache colone wich is not a number

    count=0
    for type in df.dtypes:
        if type != "int64" and type != "float64" :
            #print(type, count)
            df.iloc[:, count]= le.fit_transform(df.iloc[:, count])
        count = count+1
    return df  




X_train = EncodeDataFramelabel(X_train)
X_test = EncodeDataFramelabel(X_test)
X_test_prod = EncodeDataFramelabel(X_test_prod)



x = X_train.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_train = pd.DataFrame(x_scaled)

x_ = X_test.values
x__scaled = min_max_scaler.fit_transform(x_)
X_test = pd.DataFrame(x__scaled)

X_test_prod = pd.DataFrame(min_max_scaler.fit_transform(X_test_prod.values))

X_train.to_csv(path_or_buf="data/X_train_ready.csv")
X_test.to_csv(path_or_buf="data/X_test_ready.csv")

X_test_prod.to_csv(path_or_buf="data/X_test_prod_ready.csv")

y_train.to_csv(path_or_buf="data/y_train_ready.csv")
y_test.to_csv(path_or_buf="data/y_test_ready.csv")

