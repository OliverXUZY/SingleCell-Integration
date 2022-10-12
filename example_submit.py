#!/usr/bin/env python
# coding: utf-8

# In[51]:


import gc
import joblib
import func.data as dt
import h5py
import hdf5plugin

import sklearn as sk
import pandas as pd
import numpy as np
import pathlib as pl

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse

from sklearn import datasets
from sklearn.linear_model import Lasso

raw_data_dir = pl.Path("D:/Data/open-problems-multimodal/raw")
pro_data_dir = pl.Path("D:/Data/open-problems-multimodal/processed")
method_dir = pl.Path("D:/OneDrive/OneDrive - UW-Madison/Kris/Code/SingleCell-Integration/methods")

data_name = "multi"
method_name = "lasso"


# In[52]:


train_input = dt.readH5pyFile(raw_data_dir / f"train_{data_name}_inputs.h5")
train_target = dt.readH5pyFile(raw_data_dir / f"train_{data_name}_targets.h5")


# In[ ]:


train_input_val = train_input['block0_values']
train_target_val = train_target['block0_values']


# In[ ]:


print(train_input_val)
print(train_target_val)
print(train_input_val.shape)
print(train_target_val.shape)


# In[ ]:


# Run standardization and PCA
p = 500

train_sdsc = StandardScaler()
train_input_sdsc = train_sdsc.fit_transform(train_input_val)

train_pca = PCA(n_components=p)
X_train = train_pca.fit_transform(train_input_sdsc)

# X_train = pd.DataFrame(train_pca.transform(train_input_sdsc))


# In[ ]:


# save intermediate results

joblib.dump(train_sdsc, method_dir / f"train_{data_name}_sdsc.m")
joblib.dump(train_pca, method_dir / f"train_{data_name}_pca.m")

np.savetxt(pro_data_dir / f"train_{data_name}_pca.csv", X_train, delimiter=",")
# X_train.to_csv(data_dir / f"train_{data_name}_pca.csv")


# In[ ]:


y_train = train_target_val
print(X_train.shape)
print(y_train.shape)


# In[ ]:


param_grid = [{'alpha': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]}]
model = Lasso()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)


# In[ ]:


best_model = grid_search.best_estimator_
joblib.dump(best_model, method_dir / f"{data_name}_{method_name}.m")


# In[ ]:


print(best_model)
print(grid_search.cv_results_)


# In[ ]:


# Make prediction

test_input = dt.readH5pyFile(raw_data_dir / f"test_{data_name}_inputs.h5")


# In[ ]:


test_input_val = test_input["block0_values"]
print(test_input_val.shape)


# In[ ]:


X_test = train_pca.transform(train_sdsc.transform(test_input_val))
y_pred = best_model.predict(X_test)


# In[ ]:


print(y_pred.shape)


# In[47]:


np.savetxt(pro_data_dir / f"test_{data_name}_{method_name}_pred.csv", y_pred, delimiter=",")

