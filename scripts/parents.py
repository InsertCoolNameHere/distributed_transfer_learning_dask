from random import sample
from time import time
import pandas as pd
import pymongo
from sklearn import ensemble
import numpy as np
import os
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import mean_squared_error
from math import sqrt

import dask.dataframe as dd

import random

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

import pickle

import dask
from dask import delayed
from dask.distributed import Client

DATASET = "macav2"

df_clusters = pd.read_csv("~/ucc-21/clusters-macav2.csv")
print(df_clusters.head())

time1 = time()
gk = df_clusters.groupby('cluster_id')
print(f'gk: {gk}')

parent_maps = {}
child_to_parent = {}

for name, group in gk:
    row = group[group.distance == group.distance.min()]
    row_max = group[group.distance == group.distance.max()]

    children = list(group.gis_join)
    distances = list(group.distance)

    dist_min = row['distance'].item()
    dist_max = row_max['distance'].item()

    pg = str(row['gis_join'].item())

    parent_index = children.index(pg)
    children.pop(parent_index)
    distances.pop(parent_index)

    inner_dict = {}
    inner_dict['dist_min'] = dist_min
    inner_dict['dist_max'] = dist_max
    inner_dict['children'] = children
    inner_dict['distances'] = distances

    parent_maps[pg] = inner_dict

    for c in children:
        child_to_parent[c] = pg

print(parent_maps)
print(child_to_parent)

# ----------------------------------------------------
# Constants and Helper Functions

sample_min = 0.05
sample_max = 0.25

query_collection = "macav2"

mongo_urls = [
    'mongodb://lattice-100:27018/',
    'mongodb://lattice-101:27018/',
    'mongodb://lattice-102:27018/',
    'mongodb://lattice-103:27018/',
    'mongodb://lattice-104:27018/'
]

mongo_db_name = "sustaindb"
query_fild = "gis_join"
train_test = 0.8

if DATASET == 'macav2':
    training_labels = ["min_surface_downwelling_shortwave_flux_in_air", "max_surface_downwelling_shortwave_flux_in_air",
                   "max_specific_humidity", "min_max_air_temperature", "max_max_air_temperature"]
    target_labels = ["max_min_air_temperature"]


# QUERY projection
client_projection = {}
for val in training_labels:
    client_projection[val] = 1
for val in target_labels:
    client_projection[val] = 1

# ----------------------------------------------------
# Modeling
saved_models = {}


# ACTUAL QUERYING
def query_sustaindb(query_gisjoin, sustain_db):
    sustain_collection = sustain_db[query_collection]
    client_query = {query_fild: query_gisjoin}
    query_results = list(sustain_collection.find(client_query, client_projection))
    return list(query_results)


# SAMPLE FROM QUERY RESULTS
def data_sampling(query_results, exhaustive, sample_percent=1):
    if exhaustive:
        all_data = query_results
    else:
        data_size = int(len(query_results) * sample_percent)
        all_data = sample(query_results, data_size)

    return pd.DataFrame(all_data)


# GET SAMPLE % BASED ON DISTANCE FROM CENTROID
def get_sample_percent(gis_join):
    parent_gis = child_to_parent[gis_join]
    inner_dict = parent_maps[parent_gis]
    d_max = inner_dict['dist_max']
    d_min = inner_dict['dist_min']
    children = inner_dict['children']
    distances = inner_dict['distances']

    my_index = children.index(gis_join)
    my_distance = distances[my_index]

    frac = (my_distance - d_min) / (d_max - d_min)

    perc = sample_min + (sample_max - sample_min) * frac

    perc *= 100
    perc = int(perc)
    perc = perc - (perc % 5)

    perc = perc / 100
    return perc


# GET PERCENTAGE DISTANCE FROM CENTROID
def get_distance_percentage(gis_join):
    parent_gis = child_to_parent[gis_join]
    inner_dict = parent_maps[parent_gis]
    d_max = inner_dict['dist_max']
    d_min = inner_dict['dist_min']
    children = inner_dict['children']
    distances = inner_dict['distances']

    my_index = children.index(gis_join)
    my_distance = distances[my_index]

    frac = (my_distance - d_min) / (d_max - d_min)

    return frac * 100


def exhaustive_training(X, Y, gis_join, algorithm):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    if algorithm == 'gb':
        param_grid = {'max_depth': [2, 3], 'min_samples_split': [15, 20, 50]}
        base_est = ensemble.GradientBoostingRegressor(random_state=0)
        sh = HalvingGridSearchCV(base_est, param_grid, cv=5, verbose=1,
                                 factor=2, resource='n_estimators', max_resources=600).fit(X, pd.Series.ravel(Y))
    elif algorithm == 'lr':
        param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}
        base_est = linear_model.LinearRegression()
        sh = HalvingGridSearchCV(base_est, param_grid, cv=5, verbose=1,
                                 factor=2, resource='n_samples', max_resources=600).fit(X, pd.Series.ravel(Y))
    else:
        print(f'Algorithm not supported: {algorithm}')
        return

    clf_best = sh.best_estimator_
    rmse = sqrt(mean_squared_error(pd.Series.ravel(y_test), clf_best.predict(X_test)))

    print("PARENT GISJOIN: ", gis_join, "RMSE:", rmse)
    return clf_best


def sampled_training(X, Y, gis_join, saved_models):
    parent_gis = child_to_parent[gis_join]
    clf = saved_models[parent_gis]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    clf.fit(X_train, pd.Series.ravel(y_train))

    rmse = sqrt(mean_squared_error(pd.Series.ravel(y_test), clf.predict(X_test)))

    print("CHILD GISJOIN: ", gis_join, "RMSE:", rmse)
    return clf


def train_gisjoin(gis_join, exhaustive=True, saved_models={}):
    _time1 = time()
    mongo_url = mongo_urls[random.randint(0, len(mongo_urls) - 1)]
    sustainclient = pymongo.MongoClient(mongo_url)
    sustain_db = sustainclient[mongo_db_name]

    sample_percent = 1
    if not exhaustive:
        # print("SAMPLED CHILD TRAINING.....")
        sample_percent = get_sample_percent(gis_join)

    # QUERY
    results = query_sustaindb(gis_join, sustain_db)

    df_sampled = data_sampling(results, exhaustive, sample_percent)

    Y = df_sampled.loc[:, target_labels]
    X = df_sampled.loc[:, training_labels]
    # print(X.shape, Y.shape)

    if exhaustive:
        clf = exhaustive_training(X, Y, gis_join, 'gb')
    else:
        clf = sampled_training(X, Y, gis_join, saved_models)

    # saved_models[gis_join] = clf
    _time2 = time()
    return gis_join, clf, _time2 - _time1
# ----------------------------------------------------


outputs = []
time1 = time()
# TRAINING PARENTS FIRST
for pk in parent_maps.keys():
    ret = delayed(train_gisjoin)(pk, True)
    outputs.append(ret)

futures = dask.persist(*outputs)  # trigger computation in the background
results = dask.compute(*futures)
# print(f'Time to train one GISJOIN: {time() - time1} s')
# ----------------------------------------------------
print(results)
print(f'No. of results: {len(results)}')

for sm in results:
    (gis_join, model, time_taken) = sm
    saved_models[gis_join] = model

print(saved_models)
pickle.dump(saved_models, open(f'parent_models_{DATASET}.pkl', 'wb'))

time2 = time()
print(f'Time Taken to build parent models: {time2 - time1} s')
