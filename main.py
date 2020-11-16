import os.path
import pandas as pd
import math
from load_data import load_data, format_data, check_answers, group_together, elizabeth_known
from collab_filtering import setup_data, CollabFilteringModel
from baseline import split_data, BaselineModel
import sys

model = sys.argv[1]

def c_filtering(percent):
    ids, data, answers = setup_data()
    algo = CollabFilteringModel(data)

    pairs = []
    preds = []
    for i in ids:
        prediction = algo.predict(1, i)
        val = prediction.est
        pairs.append((i, val))
        '''
        if val >= 7.7:
            preds.append(1)
        else:
            preds.append(0)
        '''

    roomsNotDone = [x for x in pairs if x[0] not in elizabeth_known]
    length = len(roomsNotDone)
    percentage = percent / 100.0
    threshold = sorted(roomsNotDone, key = lambda x: x[1])[int((length - 1) * percentage)][1]
    for i in sorted(pairs, key = lambda x: x[0]):
        if i[1] >= threshold:
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
    elif model == 'k-nearest-neighbors':
        if len(sys.argv) == 3:
            p = float(sys.argv[2])
            if 0 <= p <= 100:
                c_filtering(p)
            else:
                print("Usage: <model, percent>, %s is not a valid percent." % (p))
        else:
            print("Usage: <model, percent>, please enter a valid percent.")