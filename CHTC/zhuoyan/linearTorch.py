import os
import sys
os.chdir("/Users/zyxu/Documents/py/kris")
sys.path.append("/Users/zyxu/Documents/py/kris/")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from prepro import SingleCellDataset, SingleCellDataset_test
from model_predict import LinearRegression

DATA_DIR = "/staging/zxu444"
SUBMIT_DIR = "submit"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")

FP_MULTIOME_TRAIN_TARGETS_k_centers = "k_centers_targets.csv.gz"
FP_MULTIOME_TRAIN_weightPCA = os.path.join(DATA_DIR,"weightPCA.csv.gz")

BATCH_SIZE = 256
# BATCH_SIZE = 1024
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0
MOMENTUM = 0

EPOCH = 5

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_id, (cells, features, targets) in enumerate(train_loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward
        outputs = model(features)
        loss = F.mse_loss(targets,outputs)
        loss.backward()

        # optimize
        optimizer.step()

        # LOGGING
        if not batch_id % 50:
               print("Epoch: {}  |  Batch: {}/{}  |  Cost: {}".format(epoch+1, batch_id, len(train_loader), loss))




def main():
    print("start reading weightPCA--")
    weightPCA = pd.read_csv(FP_MULTIOME_TRAIN_weightPCA, compression = 'gzip').values
    weight = torch.from_numpy(weightPCA).to(torch.float32).T
    print("finish reading weightPCA--")
    print("weightPCA shape: ", weightPCA.shape)
    test_multi = SingleCellDataset(FP_MULTIOME_TEST_INPUTS)
    # testloader_multi = DataLoader(test_multi, batch_size=BATCH_SIZE)
    train_multi = SingleCellDataset(FP_MULTIOME_TRAIN_INPUTS,None, FP_MULTIOME_TRAIN_TARGETS_k_centers)
    trainloader_multi = DataLoader(train_multi, batch_size=BATCH_SIZE)

    # weight = torch.ones(228942, 1000)

    model = LinearRegression(weight)
    # print(model.LR)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)


    # cells, features, targets = next(iter(trainloader_multi))

    # output = model(features)

    # print(output.shape)

    # training loop
    train_loss_total = []
    for epoch in range(EPOCH):
        train_loss = train(model, trainloader_multi, optimizer, epoch)
        train_loss_total.append(train_loss)
        # LOGGING
        if epoch % 1 == 0:
            print("Epoch: {}/{}  === Train Cost: {}".format(epoch + 1, 20000, train_loss))
    
    torch.save(model.state_dict(),os.path.join("model_epoch{}.pt".format(EPOCH)))

if __name__ == "__main__":
    main()
