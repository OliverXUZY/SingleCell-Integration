import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

from sklearn.decomposition import IncrementalPCA

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

import prepro as pp
from prepro import SingleCellDataset, SingleCellDataset_test

print('sys.argv is', sys.argv)
print("current working directory is", os.getcwd())
# print("current dir contains", os.listdir())

DATA_DIR = "/staging/zxu444"
# FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

# FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
# FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
# FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

# FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
# FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")


# train_multi_target = pp.readH5pyFile(FP_MULTIOME_TRAIN_TARGETS) ## file2 is train_multi_targets
# print(len(multi_input))
# print(len(train_multi_target))

BATCH_SIZE = 1024
IPCA_N_COMPONENTS = 1000

train_multi = SingleCellDataset(FP_MULTIOME_TRAIN_INPUTS,FP_MULTIOME_TRAIN_TARGETS)
trainloader_multi = DataLoader(train_multi, batch_size=BATCH_SIZE)
print(f"len of train_multi is {len(train_multi)},  trainloader_multi is {len(trainloader_multi)}")

test_multi = SingleCellDataset(FP_MULTIOME_TEST_INPUTS)
testloader_multi = DataLoader(test_multi, batch_size=BATCH_SIZE)
print(f"len of test_multi is {len(test_multi)},  testloader_multi is {len(testloader_multi)}")


ipca = IncrementalPCA(n_components=IPCA_N_COMPONENTS, 
                    batch_size=BATCH_SIZE)

print(f"training start: test_ipca_ncom{IPCA_N_COMPONENTS}_batch{BATCH_SIZE}")
i = 0
# cell_ids, inputs, targets = next(iter(trainloader_multi))
for cell_ids, inputs, targets in trainloader_multi:
    if len(cell_ids) <= 1000:
        break
    ipca.fit(inputs)

print(ipca.components_.shape)
print(ipca.get_params())
filename = f"test_ipca_ncom{IPCA_N_COMPONENTS}_batch{BATCH_SIZE}.sav"
joblib.dump(ipca, f"{filename}")
