import os.path
from os import path
import pandas as pd
import pickle
import math
from load_data import load_data, format_data, check_answers, group_together
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans
from surprise import SVD
from surprise import CoClustering
from surprise.model_selection import GridSearchCV



def setup_data():
    if not path.exists("data/pickles/data_binary.p"):
        load_data()

    data_scaled = pickle.load(open( "data/pickles/data_scaled.p", "rb" ))
    answers = pickle.load(open( "data/pickles/answers.p", "rb" ))

    df_scaled = format_data(data_scaled, training=False, fill_na=False)

    ids = list(range(1, 190))
    items = []
    users = []
    ratings = []

    for i in range(1, 9):
        i, u, r = group_together(df_scaled, ids, i)
        items += i
        users += u
        ratings += r

    ratings_dict = {
        "item": items,
        "user": users,
        "rating": ratings,
    }

    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)

    return ids, data, answers


def CollabFilteringModel(data, option=1, gridsearch=True):
    
    if option==1:
        sim_options = {
            "name":  "pearson_baseline",
            "min_support": 2,
            "user_based": False,
        }

        if gridsearch:
            sim_options = {
            "name": ["msd", "cosine", "pearson", "pearson_baseline"],
            "min_support": [1, 2, 3, 4, 5],
            "user_based": [False, True],
            }
            param_grid = {"sim_options": sim_options}

            gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=3)
            gs.fit(data)

            print(gs.best_score["rmse"])
            print(gs.best_params["rmse"])
            print(gs.best_score["mae"])
            print(gs.best_params["mae"])

            sim_options = {
                "name":  gs.best_params["rmse"]["sim_options"]["name"],
                "min_support": gs.best_params["rmse"]["sim_options"]["min_support"],
                "user_based": gs.best_params["rmse"]["sim_options"]["user_based"],
            }

        algo = KNNWithMeans(sim_options=sim_options)

        trainingSet = data.build_full_trainset()

        algo.fit(trainingSet)

    elif option==2:
        n_epochs = 200
        lr_all = .002
        reg_all = .1
        
        if gridsearch:
            param_grid = {
                "n_epochs": [10, 200],
                "lr_all": [0.002, 0.1],
                "reg_all": [0.05, 0.9]
            }

            gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)
            gs.fit(data)

            print(gs.best_score["rmse"])
            print(gs.best_params["rmse"])
            print(gs.best_score["mae"])
            print(gs.best_params["mae"])

            n_epochs = gs.best_params["rmse"]["n_epochs"]
            lr_all = gs.best_params["rmse"]["lr_all"]
            reg_all = gs.best_params["rmse"]["reg_all"]
        

        
        algo = SVD(n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)

        trainingSet = data.build_full_trainset()

        algo.fit(trainingSet)
    else:
        n_cltr_u  = 3
        n_cltr_i  = 3
        n_epochs = 200
        
        if gridsearch:
            param_grid = {
                "n_epochs": [10, 200],
                "n_cltr_u": [2,3,4,5,6],
                "n_cltr_i": [2,3,4,5,6]
            }

            gs = GridSearchCV(CoClustering, param_grid, measures=["rmse", "mae"], cv=3)
            gs.fit(data)

            print(gs.best_score["rmse"])
            print(gs.best_params["rmse"])
            print(gs.best_score["mae"])
            print(gs.best_params["mae"])

            n_epochs = gs.best_params["rmse"]["n_epochs"]
            n_cltr_u = gs.best_params["rmse"]["n_cltr_u"]
            n_cltr_i = gs.best_params["rmse"]["n_cltr_i"]
        

        
        algo = CoClustering(n_cltr_u =n_cltr_u , n_epochs=n_epochs, n_cltr_i=n_cltr_i)

        trainingSet = data.build_full_trainset()

        algo.fit(trainingSet)

    return algo