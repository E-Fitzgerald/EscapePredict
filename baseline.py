import pandas as pd
import numpy as np
import math
import pickle
import os.path
from os import path
from sklearn.linear_model import LinearRegression
import sklearn.svm

def check_answers(preds, answers):
    correct = 0
    false_pos = 0
    false_neg = 0
    total_one = 0
    total_zero = 0
    for i in range(len(preds)):
        if answers[i] == 1:
            total_one += 1
        else:
            total_zero += 1

        if preds[i] == answers[i]:
            correct += 1
        elif preds[i] == 0:
            false_neg += 1
        else:
            false_pos += 1
    return correct, false_pos, false_neg, total_one, total_zero

def format_data(data, training=True):
    abridged = []
    if training:
        for i in range(len(data)):
            if not math.isnan(data[i][1]):
                abridged.append(data[i])
        df = pd.DataFrame(data=abridged)
        df = df.fillna(0)
    else:
        for i in range(len(data)):
            abridged.append(data[i])
        df = pd.DataFrame(data=abridged)
        df = df.fillna(0)
    return df


if not path.exists("pickles/answers.p"):
    xl = pd.ExcelFile("EscapeTheProject.xlsx")
    df_original = pd.read_excel(xl, "OriginalData")
    df_scaled = pd.read_excel(xl, "ScaledData")
    df_binary = pd.read_excel(xl, "BinaryData")
    df_answers = pd.read_excel(xl, "Answers")

    data_original = []
    for index, row in df_original.iterrows():
        new_arr = [x for x in np.array(row["Lizzy":])]
        new_arr = [row["ID"]] + new_arr
        data_original.append(new_arr)
    pickle.dump(data_original, open( "pickles/data_original.p", "wb" ))

    data_scaled = []
    for index, row in df_scaled.iterrows():
        new_arr = [x for x in np.array(row["Lizzy":])]
        new_arr = [row["ID"]] + new_arr
        data_scaled.append(new_arr)
    pickle.dump(data_scaled, open( "pickles/data_scaled.p", "wb" ))

    data_binary = []
    for index, row in df_binary.iterrows():
        new_arr = [x for x in np.array(row["Lizzy":])]
        new_arr = [row["ID"]] + new_arr
        data_binary.append(new_arr)
    pickle.dump(data_binary, open( "pickles/data_binary.p", "wb" ))

    answers = []
    for index, row in df_answers.iterrows():
        answers.append(row["YN"])
    pickle.dump(answers, open( "pickles/answers.p", "wb" ))
else:
    data_original = pickle.load(open( "pickles/data_original.p", "rb" ))
    data_scaled = pickle.load(open( "pickles/data_scaled.p", "rb" ))
    data_binary = pickle.load(open( "pickles/data_binary.p", "rb" ))
    answers = pickle.load(open( "pickles/answers.p", "rb" ))



df_original = format_data(data_original, training=False)
df_scaled = format_data(data_scaled, training=False)
df_binary = format_data(data_binary, training=False)

X = []
for index, row in df_original.iterrows():
    X.append(row[2:])


df_original_training = format_data(data_original)
df_scaled_training = format_data(data_scaled)
df_binary_training = format_data(data_binary)

X_train = []
Y_train = []
ids = []
for index, row in df_original_training.iterrows():
    ids.append(row[0])
    X_train.append(row[2:])
    Y_train.append(row[1])

lin_reg = LinearRegression(fit_intercept=False)
reg = lin_reg.fit(X_train, Y_train)

preds_val = reg.predict(X)
preds = []
for p in preds_val:
    if p >= 44.5:
        preds.append(1)
    else:
        preds.append(0)
correct, false_pos, false_neg, yes, no = check_answers(preds, answers)
print("Percentage correct: ", correct/(yes+no))
print("Number of False Positives: ", false_pos)
print("Percentage of False that were correctly classified: ", 1-(false_pos/no))
print("Number of False Negatives: ", false_neg)
print("Percentage of True that were correctly classified: ", 1-(false_neg/yes))