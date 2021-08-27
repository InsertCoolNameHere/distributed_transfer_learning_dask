#!/bin/python3.8

import pickle
import dask
from dask import delayed
from dask.distributed import Client

from train_parents

saved_models = pickle.load(open('parent_models.pkl', 'rb'))

# client = Client('localhost:9010')
outputs2 = []
keys_0_to_15 = pickle.load(open('keys_0_to_15.pkl', 'rb'))

# TRAINING CHILDREN NEXT
# for ck in child_to_parent.keys():
for ck in keys_0_to_15:
    ret = delayed(train_parents.train_gisjoin)(ck, False, saved_models)
    outputs2.append(ret)

futures2 = dask.persist(*outputs2)  # trigger computation in the background
results2 = dask.compute(*futures2)

for sm in results2:
    (gis_join, model) = sm
    saved_models[gis_join] = model

print(saved_models)
