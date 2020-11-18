import pandas as pd
import numpy as np
import math
import pickle
import os.path
from os import path
from sklearn.linear_model import LinearRegression
import sklearn.svm
from load_data import load_data, format_data, check_answers


def split_data():
    if not path.exists("data/pickles/data_original.p"):
        load_data()

    data_original = pickle.load(open( "data/pickles/data_original.p", "rb" ))
    answers = pickle.load(open( "data/pickles/answers.p", "rb" ))

    df_original = format_data(data_original, training=False)

    X = []
    for index, row in df_original.iterrows():
        X.append(row[2:])

    df = format_data(data_original)

    X_train = []
    Y_train = []
    ids = []
    for index, row in df.iterrows():
        ids.append(row[0])
        X_train.append(row[2:])
        Y_train.append(row[1])
        
    return X, X_train, Y_train, answers


def BaselineModel(X, X_train, Y_train):
    lin_reg = LinearRegression(fit_intercept=False)
    reg = lin_reg.fit(X_train, Y_train)
    return reg