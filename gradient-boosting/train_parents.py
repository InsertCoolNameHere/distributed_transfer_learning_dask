#!/bin/python3.8

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
import pickle

import random

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

import constants
import util

time1 = time()

print('[*] Reading clusters.csv')
df_clusters = pd.read_csv("/tmp/clusters.csv")

gk = df_clusters.groupby('cluster_id')



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

    constants.parent_maps[pg] = inner_dict

    for c in children:
        constants.child_to_parent[c] = pg

print('[+] Parent Maps: ')
print(constants.parent_maps)

print('[+] Child to Parent Map: ')
print(constants.child_to_parent)



# Modeling
saved_models = {}

# ------------------------------------------------------------------------
# Separate child GISJoins based on sampling percetange/distance
sampling_perc_to_children_map = {}
for ck in constants.child_to_parent.keys():
    perc = util.get_distance_percentage(ck)
    sampling_perc_to_children_map[ck] = perc

keys_0_to_15 = []
keys_15_to_25 = []

for gis_join, perc in sampling_perc_to_children_map.items():
    if perc < 15:
        keys_0_to_15.append(gis_join)
    elif perc > 15:
        keys_15_to_25.append(gis_join)

print(f'keys_0_to_15: {len(keys_0_to_15)}')
pickle.dump(keys_0_to_15, open('keys_0_to_15.pkl', 'wb'))

print(f'keys_15_to_25: {len(keys_15_to_25)}')
pickle.dump(keys_15_to_25, open('keys_15_to_25.pkl', 'wb'))

# Init Dask
import dask
from dask import delayed
from dask.distributed import Client

client = Client('localhost:9010')

# Train Parent GISJoins
outputs = []

for pk in constants.parent_maps.keys():
    ret = delayed(util.train_gisjoin)(pk, True)
    outputs.append(ret)

futures = dask.persist(*outputs)  # trigger computation in the background
results = dask.compute(*futures)

print('[+] Printing results')
print(results)

for sm in results:
    (gis_join, model) = sm
    saved_models[gis_join] = model

print('[+] Printing saved (parent) models')
print(saved_models)

print('[*] Dumping parent models to pickle file')
pickle.dump(saved_models, open('parent_models.pkl', 'wb'))

print('[+] Training Parent GISJoins Completed!')
