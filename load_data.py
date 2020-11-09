import pandas as pd
import numpy as np
import pickle
import math

elizabeth_known = [4,14,30,32,37,60,86,114,137,141,144,146,150,158,166,170,176,177,178,181,182,183,184,185,186,187,188,189]

def load_data():
    xl = pd.ExcelFile("data/EscapeTheProject.xlsx")
    df_original = pd.read_excel(xl, "OriginalData")
    df_scaled = pd.read_excel(xl, "ScaledData")
    df_binary = pd.read_excel(xl, "BinaryData")
    df_answers = pd.read_excel(xl, "Answers")

    data_original = []
    for index, row in df_original.iterrows():
        new_arr = [x for x in np.array(row["Lizzy":])]
        new_arr = [row["ID"]] + new_arr
        data_original.append(new_arr)
    pickle.dump(data_original, open( "data/pickles/data_original.p", "wb" ))

    data_scaled = []
    for index, row in df_scaled.iterrows():
        new_arr = [x for x in np.array(row["Lizzy":])]
        new_arr = [row["ID"]] + new_arr
        data_scaled.append(new_arr)
    pickle.dump(data_scaled, open( "data/pickles/data_scaled.p", "wb" ))

    data_binary = []
    for index, row in df_binary.iterrows():
        new_arr = [x for x in np.array(row["Lizzy":])]
        new_arr = [row["ID"]] + new_arr
        data_binary.append(new_arr)
    pickle.dump(data_binary, open( "data/pickles/data_binary.p", "wb" ))

    answers = []
    for index, row in df_answers.iterrows():
        answers.append(row["YN"])
    pickle.dump(answers, open( "data/pickles/answers.p", "wb" ))


def format_data(data, training=True, fill_na=True):
    abridged = []
    if training:
        for i in range(len(data)):
            if not math.isnan(data[i][1]):
                abridged.append(data[i])
    else:
        for i in range(len(data)):
            abridged.append(data[i])
    
    df = pd.DataFrame(data=abridged)
    if fill_na:
        df = df.fillna(0)

    return df


def check_answers(preds, answers):
    correct = 0
    false_pos = 0
    false_neg = 0
    total_one = 0
    total_zero = 0

    for i in range(len(preds)):
        if i+1 not in elizabeth_known:
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
    

    print("================================================================================")
    print("Percentage correct: ", correct/(total_one+total_zero))
    print("---------------------------------------------------------------------------------")
    print("Number of rooms to not recommend: ", total_zero)
    print("Number of False Positives: ", false_pos)
    print("Percentage of False that were correctly classified: ", 1-(false_pos/total_zero))
    print("---------------------------------------------------------------------------------")
    print("Number of rooms to recommend: ", total_one)
    print("Number of False Negatives: ", false_neg)
    print("Percentage of True that were correctly classified: ", 1-(false_neg/total_one))
    print("================================================================================")

    return correct, false_pos, false_neg, total_one, total_zero

def group_together(df, ids, column):
    items = []
    users = []
    ratings = []

    for i in ids:
        row = i-1
        if not math.isnan(df[column][row]):
            items.append(i)
            users.append(column)
            ratings.append(df[column][row])
    
    return items, users, ratings