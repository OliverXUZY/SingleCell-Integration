#!/usr/bin/env python
# coding: utf-8

# In[51]:

# import gc
# import func.data as dt
import hdf5plugin
import joblib
import h5py

import sklearn as sk
import pandas as pd
import numpy as np
import pathlib as pl

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso

# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from scipy.stats import pearsonr
# from sklearn.metrics import mean_squared_error as mse
# from sklearn import datasets

# RAW_DATA_DIR = pl.Path("/staging/syan58/open-problems-multimodal/raw")
# PRO_DATA_DIR = pl.Path("/staging/syan58/open-problems-multimodal/processed")
# METHOD_DIR = pl.Path("/staging/syan58/open-problems-multimodal/methods")

RAW_DATA_DIR = pl.Path("D:/Data/open-problems-multimodal/raw")
PRO_DATA_DIR = pl.Path("D:/Data/open-problems-multimodal/processed")
METHOD_DIR = pl.Path("D:/OneDrive/OneDrive - UW-Madison/Kris/Code/SingleCell-Integration/methods")

DATA_NAME = "cite"
METHOD_NAME = "lasso"

def readH5pyFile(filename):
    with h5py.File(filename, "r") as f:
        a_group_key = list(f.keys())[0]
        data = list(f[a_group_key])
        group = f[a_group_key]      # returns as a h5py dataset object
        d = {}
        for i in list(group.keys()):
            d[i] = group[i][()]
    return d

# In[52]:

print("Loading data...")

train_input = readH5pyFile(RAW_DATA_DIR / f"train_{DATA_NAME}_inputs.h5")
train_target = readH5pyFile(RAW_DATA_DIR / f"train_{DATA_NAME}_targets.h5")

train_input_val = train_input['block0_values']
train_target_val = train_target['block0_values']

# In[ ]:
# Run standardization and PCA

print("Running PCA...")

p = 500
train_sdsc = StandardScaler()
train_input_sdsc = train_sdsc.fit_transform(train_input_val)
train_pca = PCA(n_components=p)

X_train = train_pca.fit_transform(train_input_sdsc)
y_train = train_target_val

# In[ ]:
# Grid Search

# param_grid = [{'alpha': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]}]
# model = Lasso()
# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)
# best_model = grid_search.best_estimator_


# In[]:

print("Fitting the model...")

model = Lasso(alpha = 0.01)
model.fit(X_train, y_train)

# Make prediction

print("Making prediction...")

test_input = readH5pyFile(RAW_DATA_DIR / f"test_{DATA_NAME}_inputs.h5")
test_input_val = test_input["block0_values"]

X_test = train_pca.transform(train_sdsc.transform(test_input_val))
y_pred = model.predict(X_test)

np.savetxt(PRO_DATA_DIR / f"test_{DATA_NAME}_{METHOD_NAME}_pred.csv", y_pred, delimiter=",")

# In[ ]:
# Save intermediate results
SAVE = False
if SAVE:
    print("Saving intermediate results...")
    joblib.dump(train_sdsc, METHOD_DIR / f"train_{DATA_NAME}_sdsc.m")
    joblib.dump(train_pca, METHOD_DIR / f"train_{DATA_NAME}_pca.m")
    # np.savetxt(PRO_DATA_DIR / f"train_{DATA_NAME}_pca.csv", X_train, delimiter=",")
    joblib.dump(model, METHOD_DIR / f"{DATA_NAME}_{METHOD_NAME}.m")