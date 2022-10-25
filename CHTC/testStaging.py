import sys
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

import prepro as pp
from func.prepro import SingleCellDataset, SingleCellDataset_test

print('sys.argv is', sys.argv)

file1 = sys.argv[1] ## test_cite_inputs

DATA_DIR = "open-problems-multimodal"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")



multi_input = pp.readH5pyFile(file1) ## file1 is train_multi_inputs
multi_target = pp.readH5pyFile(file2) ## file2 is train_multi_targets
res1 = 0  ## your first prediction matrix based on test_cite_inputs

res2 = 0  ## your second prediction matrix based on test_multi_inputs

# res1.to_csv("preMat_protein.csv.gz",compression='gzip')
# res2.to_csv("preMat_RNA.csv.gz",compression='gzip')

print(len(multi_input))
print(len(multi_target))

train_multi = SingleCellDataset(FP_MULTIOME_TRAIN_INPUTS,FP_MULTIOME_TRAIN_TARGETS)
trainloader_multi = DataLoader(train_multi, batch_size=1024)
print(f"len of test_multi is {len(train_multi)},  testloader_multi is {len(trainloader_multi)}")

test_multi = SingleCellDataset(FP_MULTIOME_TRAIN_INPUTS)
testloader_multi = DataLoader(test_multi, batch_size=1024)
print(f"len of test_multi is {len(test_multi)},  testloader_multi is {len(testloader_multi)}")
