from random import sample
from time import time
import pandas as pd
import pymongo
from sklearn import ensemble
import numpy as np
import os
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import mean_squared_error
from math import sqrt

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

DATASET = "noaa_nam_2"
print(f'Dataset: {DATASET}')

# -------------------------------------------------------
# Constants and Helper functions
query_collection = DATASET
mongo_url = "mongodb://lattice-100:27018/"
mongo_db_name = "sustaindb"
query_fild = "gis_join"
sample_percent = 0.1
train_test = 0.8
feature_importance_percentage = 98
exhaustive_sample_percent = 0.0001

if DATASET == "macav2":
    training_labels = [
        "min_surface_downwelling_shortwave_flux_in_air",
        "max_surface_downwelling_shortwave_flux_in_air",
        "max_specific_humidity",
        "min_max_air_temperature",
        "max_max_air_temperature"
    ]
    target_labels = ["max_min_air_temperature"]
elif DATASET == "noaa_nam_2":
    training_labels = [
        "mean_sea_level_pressure_pascal",
        "surface_pressure_surface_level_pascal",
        "10_metre_u_wind_component_meters_per_second",
        "10_metre_v_wind_component_meters_per_second",
        "soil_temperature_kelvin"
    ]
    target_labels = ["pressure_pascal"]

# QUERY-RELATED
sustainclient = pymongo.MongoClient(mongo_url)
sustain_db = sustainclient[mongo_db_name]

# QUERY projection
client_projection = {}
for val in training_labels:
    client_projection[val] = 1
for val in target_labels:
    client_projection[val] = 1


def fancy_logging(msg, unique_id=""):
    print(unique_id, ":", "====================================")
    print(unique_id, ":", msg, ": TIME: ", time())

# -------------------------------------------------------
# Data Fetch
# ACTUAL QUERYING
def query_sustaindb(query_gisjoin):
    sustain_collection = sustain_db[query_collection]
    client_query = {query_fild: query_gisjoin}

    start_time = time()
    query_results = list(sustain_collection.find(client_query, client_projection))

    return list(query_results)


def queryall_sustaindb():
    sustain_collection = sustain_db[query_collection]
    client_query = {}

    start_time = time()
    query_results = list(sustain_collection.find(client_query, client_projection))

    return list(query_results)

#df = query_sustaindb('G3701310')
df = queryall_sustaindb()
print("1: ", len(df))
# -------------------------------------------------------
# Data Sampling
def data_sampling(query_results, exhaustive, sample_percent=1):
    if exhaustive:
        all_data = query_results
    else:
        data_size = int(len(query_results) * sample_percent)
        all_data = sample(query_results, data_size)

    return pd.DataFrame(all_data)

sampled_df = data_sampling(df, False, exhaustive_sample_percent)
Y = sampled_df.loc[:,target_labels]
X = sampled_df.loc[:, training_labels]
print(f'X.shape: {X.shape}, Y.shape: {Y.shape}')
# -------------------------------------------------------
# Data splitting into training and validation
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print(f'X_train.shape: {X_train.shape}')
print(f'X_test.shape: {X_test.shape}')
print(f'y_train.shape: {y_train.shape}')
print(f'y_test.shape: {y_test.shape}')
# -------------------------------------------------------
# Modeling
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
param_grid = {'max_depth': [2, 3], 'min_samples_split': [15, 20, 50]}
base_est = ensemble.RandomForestRegressor(random_state=0)
sh = HalvingGridSearchCV(base_est, param_grid, cv=5, verbose=3,
                         factor=2, resource='n_estimators', max_resources=600).fit(X, pd.Series.ravel(Y))

# THE BEST MODEL
clf_best = sh.best_estimator_

rmse = sqrt(mean_squared_error(pd.Series.ravel(y_test), clf_best.predict(X_test)))
print(f'rmse: {rmse}')
# -------------------------------------------------------
# Extract top featurs
feature_importance = clf_best.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.sum())
sorted_idx = np.argsort(feature_importance)
print(np.flip(sorted_idx), np.flip(feature_importance[sorted_idx]))

feature_importance = np.flip(feature_importance[sorted_idx])
sorted_idx=np.flip(sorted_idx)

print(f'sorted_idx: {sorted_idx}')
print(f'feature_importance: {feature_importance}')

# -------------------------------------------------------
# FIND N FOR WHICH IMPORTANCE % > feature-importance-percentage
def find_cumulative(lists, val_max):
    cu_list = []
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(1, length + 1)]

    print(cu_list)
    res = next(x for x, val in enumerate(cu_list)
               if val > val_max)
    return res

cut_off_indx = find_cumulative(feature_importance, feature_importance_percentage)

print("LAST INDEX: ", cut_off_indx)

chopped_indices = sorted_idx[0:cut_off_indx+1]

print(f'sorted_idx: {sorted_idx}')
print(f'chopped_indices: {chopped_indices}')
# -------------------------------------------------------
# Selected top columns
candidate_x_columns = list(X.columns)
candidate_y_columns = list(Y.columns)

print(f'candidate_x_columns: {candidate_x_columns}')
print(f'candidate_y_columns: {candidate_y_columns}')

selected_x_columns = [candidate_x_columns[i] for i in chopped_indices]
print(f'selected_x_columns: {selected_x_columns}')
# -------------------------------------------------------
# Training phase 2
chopped_projection = []
chopped_projection.extend(selected_x_columns)
chopped_projection.extend(candidate_y_columns)

print(chopped_projection)


def construct_chopped_query(chopped_projection, gis_join):
    # PROJECTION
    proj_d = {}
    proj_dict = {'$project': proj_d}

    # GROUP + AGGREGATION
    group_d = {}
    group_dict = {'$group': group_d}

    full_query = [proj_dict, group_dict]

    # PROJECTION PART
    for cp in chopped_projection:
        proj_d[cp] = "$" + str(cp)
    proj_d[gis_join] = "$" + str(gis_join)

    # GROUP PART
    group_d['_id'] = "$" + str(gis_join)
    for cp in chopped_projection:
        inner_dict = {}
        inner_dict["$avg"] = "$" + str(cp)
        group_d[cp] = inner_dict

    return full_query


agg_pipeline = construct_chopped_query(chopped_projection, query_fild)

sustain_collection = sustain_db[query_collection]
cur = sustain_collection.aggregate(agg_pipeline)
agg_results = list(cur)

print(len(agg_results))
print(f'agg_results[0]: {agg_results[0]}')
# -------------------------------------------------------
# Data staging for phase 2
phase2_df = pd.DataFrame(agg_results)
print(f'chopped_projection: {chopped_projection}')
df_importance = phase2_df.loc[:, chopped_projection]
print(df_importance.head())
# -------------------------------------------------------
# KMeans Clustering
num_clusters = int(sqrt(len(agg_results)))
print(f'num_clusters: {num_clusters}')

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


kmeans = KMeans(n_clusters=num_clusters).fit(df_importance)
centroids = kmeans.cluster_centers_
print(f'centroids: {centroids}')

# -------------------------------------------------------
from sklearn.metrics import pairwise_distances_argmin_min

df_ultimate = pd.DataFrame(columns=["gis_join", "cluster_id", "distance"])

for index, row in phase2_df.iterrows():
    input_x = row[chopped_projection]
    gis_join = row['_id']
    # print(input_x, gis_join)
    closest, d = pairwise_distances_argmin_min([np.array(input_x)], centroids)
    df_ultimate.loc[index] = [gis_join, closest[0], d[0]]

print(df_ultimate.head())
df_ultimate.to_csv(f'~/ucc-21/clusters-{DATASET}-1.csv')
# -------------------------------------------------------
