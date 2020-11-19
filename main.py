import os.path
import pandas as pd
import math
import matplotlib.pyplot as plt
from load_data import load_data, format_data, check_answers, group_together, elizabeth_known
from collab_filtering import setup_data, CollabFilteringModel
from baseline import split_data, BaselineModel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


import sys

model = int(sys.argv[1])


def c_filtering(option, percent, gridsearch):
    ids, data, answers = setup_data()
    algo = CollabFilteringModel(data, option, gridsearch)

    pairs = []
    preds = []
    for i in ids:
        prediction = algo.predict(1, i)
        val = prediction.est
        pairs.append((i, val))

    roomsNotDone = [x for x in pairs if x[0] not in elizabeth_known]
    length = len(roomsNotDone)
    percentage = percent / 100.0
    threshold = sorted(roomsNotDone, key = lambda x: x[1])[int((length - 1) * percentage)][1]
    for i in sorted(pairs, key = lambda x: x[0]):
        if i[1] >= threshold:
            preds.append(1)
        else:
            preds.append(0)


    stats = check_answers(preds, answers)

def baseline(percent):
    X, Y, X_train, Y_train, answers = split_data()
    reg = BaselineModel(X, X_train, Y_train)

    preds_val = reg.predict(X)

    
    ids = list(range(1,190))
    pairs=[]
    for i in ids:
        pairs.append((i, preds_val[i-1]))

    roomsNotDone = [x for x in pairs if x[0] not in elizabeth_known]
    length = len(roomsNotDone)
    percentage = percent / 100.0
    threshold = sorted(roomsNotDone, key = lambda x: x[1])[int((length - 1) * percentage)][1]
    preds = []
    for i in sorted(pairs, key = lambda x: x[0]):
        if i[1] >= threshold:
            preds.append(1)
        else:
            preds.append(0)

    roomsDone = [x for x in pairs if x[0] in elizabeth_known]
    known_preds = []
    limited_answers = []
    for i in sorted(roomsDone, key = lambda x: x[0]):
        limited_answers.append(Y[i[0]-1])
        known_preds.append(i[1])
    
    rms = sqrt(mean_squared_error(limited_answers, known_preds))
    print("rms: ", rms)
    mae = mean_absolute_error(limited_answers, known_preds)
    print("mae: ", mae)

    stats = check_answers(preds, answers)


if __name__ == '__main__':
    if model in [0, 1, 2, 3]:
        if (len(sys.argv) == 3 and model==0) or (len(sys.argv)==4):
            p = float(sys.argv[2])
            if model == 0:
                baseline(p)
            elif 1 <= p <= 100:
                g = int(sys.argv[3])
                c_filtering(model, p, g)
            else:
                print("Usage: <model, percent, gridsearch>, %s is not a valid percent." % (p))
        else:
            print("Usage: <model, percent, gridsearch>, please enter a valid percent and 0/1 boolean value for gridsearch.")
            print("Model Options: Baseline=0, KNNMeans=1, SVD=2, CoClustering=3")