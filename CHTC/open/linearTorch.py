import os
import sys

import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl

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

FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")

FP_MULTIOME_TRAIN_TARGETS_k_centers = os.path.join(DATA_DIR,"k_centers_targets.csv.gz")
FP_MULTIOME_TRAIN_weightPCA = os.path.join(DATA_DIR,"weightPCA.csv.gz")
FP_MULTIOME_INPUTS_PROJECTION = os.path.join(DATA_DIR,"test/inputs_projection.pickle")

BATCH_SIZE = 256
# BATCH_SIZE = 1024
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 0
MOMENTUM = 0

START_EPOCH = 0
EPOCH = 1

##====================== train and test function
def compute_epoch_loss(model, data_loader):
    model.eval()
    curr_loss, num_examples = 0.,0
    with torch.no_grad():
        for cells, features, targets in data_loader:
            features = features.to(DEVICE)

            targets = targets.to(DEVICE)

            outputs = model(features)

            loss = F.mse_loss(outputs,targets, reduction = "sum")
            num_examples += targets.size(0)
            curr_loss += loss
        curr_loss /= num_examples
        return curr_loss

def train(model, train_loader, optimizer, epoch, start_time):
    model.train()
    model = model.to(DEVICE)
    for batch_id, (_, _, targets) in enumerate(train_loader):
        targets = targets.to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward
        outputs = model(batch_id)
        loss = F.mse_loss(targets,outputs)
        loss.backward()

        # optimize
        optimizer.step()

        # LOGGING
        if not batch_id % 1:
               print("Epoch: {}  |  Batch: {}/{}  |  Cost: {}".format(epoch+1+START_EPOCH, batch_id, len(train_loader), loss))
               print(f"-- Time elapsed: {(time.time() - start_time)/60} min --")
        
        if batch_id > 3:
            break

    train_loss = compute_epoch_loss(model, train_loader)

    return train_loss



def main():
    start_time = time.time()
    print("start reading weightPCA--")
    print(f"-- Time elapsed: {(time.time() - start_time)/60} min --")
    #weightPCA = pd.read_csv(FP_MULTIOME_TRAIN_weightPCA, compression = 'gzip').values
    #weight = torch.from_numpy(weightPCA).to(torch.float32).T
    weight = torch.ones(228942, 1000)

    print("finish reading weightPCA--")
    print(f"-- Time elapsed: {(time.time() - start_time)/60} min --")
    print("weightPCA shape: ", weight.shape)
    # test_multi = SingleCellDataset(FP_MULTIOME_TEST_INPUTS)
    # testloader_multi = DataLoader(test_multi, batch_size=BATCH_SIZE)
    train_multi = SingleCellDataset(FP_MULTIOME_TRAIN_INPUTS,None, FP_MULTIOME_TRAIN_TARGETS_k_centers)
    trainloader_multi = DataLoader(train_multi, batch_size=BATCH_SIZE)


    #if (NAME OF CONCAT FILE IS PRESENT):
    #    read that file
    #else:
    #pkl.dumps(INPUT_PROJECTIONS)
    # save the file
    if os.path.exists(FP_MULTIOME_INPUTS_PROJECTION):
        print("Projection Exists")
        with open(FP_MULTIOME_INPUTS_PROJECTION, 'rb') as handle:
            projections = pkl.load(handle)
    else:
        print("Projection Doesn't Exists")
        projections = {}
        weight = weight.to(DEVICE)
        for batch_id, (_, features, _) in enumerate(trainloader_multi):
            features = features.to(DEVICE)
            projections[batch_id] = features.matmul(weight)
        with open(FP_MULTIOME_INPUTS_PROJECTION, 'wb') as handle:
            pkl.dump(projections, handle, protocol=pkl.HIGHEST_PROTOCOL)


    model = LinearRegression(projections, DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)

    # training loop
    train_loss_total = []
    if START_EPOCH > 0:
        model.load_state_dict(torch.load("model_epoch{}.pt".format(START_EPOCH)))
    print("start training--")
    print(f"-- Time elapsed: {(time.time() - start_time)/60} min --")
    for epoch in range(EPOCH):
        train_loss = train(model, trainloader_multi, optimizer, epoch, start_time)
        train_loss_total.append(train_loss)
        # LOGGING
        if epoch % 1 == 0:
            print("Epoch: {}/{}  === Train Cost: {}".format(epoch + 1 + START_EPOCH, START_EPOCH + EPOCH, train_loss))
            print(f"-- Time elapsed: {(time.time() - start_time)/60} min --")

    torch.save(model.state_dict(),os.path.join("model_epoch{}.pt".format(START_EPOCH + EPOCH)))
    ##====================== record cost
    with open(f"cost_epoch{START_EPOCH + EPOCH}.txt", "w") as f:
        for s in train_loss_total:
            f.write(str(s) +"\n")

if __name__ == "__main__":
    main()
