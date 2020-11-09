import os.path
import pandas as pd
import math
from load_data import load_data, format_data, check_answers, group_together
from collab_filtering import setup_data, CollabFilteringModel
from baseline import split_data, BaselineModel
import sys

model = sys.argv[1]

def c_filtering():
    ids, data, answers = setup_data()
    algo = CollabFilteringModel(data)

    pairs = []
    preds = []
    for i in ids:
        prediction = algo.predict(1, i)
        val = prediction.est
        pairs.append((i, val))
        if val >= 7.7:
            preds.append(1)
        else:
            preds.append(0)
    check_answers(preds, answers)

def baseline():
    X, X_train, Y_train, answers = split_data()
    reg = BaselineModel(X, X_train, Y_train)

    preds_val = reg.predict(X)
    preds = []
    for p in preds_val:
        if p >= 50:
            preds.append(1)
        else:
            preds.append(0)
            
    check_answers(preds, answers)


if __name__ == '__main__':
    if model == "baseline":
        baseline()
    else:
        c_filtering()