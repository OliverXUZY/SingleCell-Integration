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
from utils import time_str
import utils

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
FP_MULTIOME_INPUTS_PROJECTION = os.path.join(DATA_DIR,"inputs_projection.pickle")

BATCH_SIZE = 256
# BATCH_SIZE = 1024
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 0
MOMENTUM = 0

START_EPOCH = 0
EPOCH = 10

##====================== train and test function
def compute_epoch_loss(model, data_loader):
    model.eval()
    curr_loss, num_examples = 0.,0
    with torch.no_grad():
        for batch_id, (cells, features, targets) in enumerate(data_loader):

            targets = targets.to(DEVICE)

            outputs = model(batch_id)

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
        if not batch_id % 50:
               utils.log("Epoch: {}  |  Batch: {}/{}  |  Cost: {}".format(epoch+1+START_EPOCH, batch_id, len(train_loader), loss))
               utils.log(f"-- Time elapsed: {time_str(time.time() - start_time)} --")
        

    train_loss = compute_epoch_loss(model, train_loader)

    return train_loss



def main():
    ckpt_name = 'linearTorch_PCA_LR_Cluster'
    ckpt_path = os.path.join('./save', ckpt_name)

    utils.ensure_path(ckpt_path)
    utils.set_log_path(ckpt_path)

    start_time = time.time()
    
    train_multi = SingleCellDataset(FP_MULTIOME_TRAIN_INPUTS,None, FP_MULTIOME_TRAIN_TARGETS_k_centers)
    trainloader_multi = DataLoader(train_multi, batch_size=BATCH_SIZE)

    #if (NAME OF CONCAT FILE IS PRESENT):
    #    read that file
    #else:
    #pkl.dumps(INPUT_PROJECTIONS)
    # save the file
    if os.path.exists(FP_MULTIOME_INPUTS_PROJECTION):
        utils.log("Projection Exists")
        with open(FP_MULTIOME_INPUTS_PROJECTION, 'rb') as handle:
            projections = pkl.load(handle)
    else:
        utils.log("Projection Doesn't Exists, start constructing:")
        utils.log(f"-- Time elapsed: {time_str(time.time() - start_time)} --")
        projections = {}
        weight = weight.to(DEVICE)
        for batch_id, (_, features, _) in enumerate(trainloader_multi):
            features = features.to(DEVICE)
            projections[batch_id] = features.matmul(weight)
        
        utils.log("Construction done, start saveing:")
        utils.log(f"-- Time elapsed: {time_str(time.time() - start_time)} --")
        with open(FP_MULTIOME_INPUTS_PROJECTION, 'wb') as handle:
            pkl.dump(projections, handle, protocol=pkl.HIGHEST_PROTOCOL)
        utils.log(f"-- Time elapsed: {time_str(time.time() - start_time)} --")



    model = LinearRegression(projections, DEVICE)
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)

    # training loop
    train_loss_total = []
    if START_EPOCH > 0:
        model.load_state_dict(torch.load("model_epoch{}.pt".format(START_EPOCH)))
    utils.log("start training--")
    utils.log(f"-- Time elapsed: {time_str(time.time() - start_time)} --")

    for epoch in range(EPOCH):
        train_loss = train(model, trainloader_multi, optimizer, epoch, start_time)
        train_loss_total.append(train_loss)
        # LOGGING
        if epoch % 1 == 0:
            utils.log("Epoch: {}/{}  === Train Cost: {}".format(epoch + 1 + START_EPOCH, START_EPOCH + EPOCH, train_loss))
            utils.log(f"-- Time elapsed: {time_str(time.time() - start_time)} --")

        ckpt = {
        'file': __file__,
        'start_epoch': START_EPOCH,
        'epoch': epoch,
        'model_state_dict': model.state_dict()
        }

        torch.save(ckpt, os.path.join(ckpt_path, 'epoch-last.pth'))
        if epoch % 5 == 0:
            torch.save(ckpt, os.path.join(ckpt_path, 'epoch-{}.pth'.format(START_EPOCH + epoch + 1)))

        ##====================== record cost
        with open(os.path.join(ckpt_path,f"cost_epoch{START_EPOCH + EPOCH}.txt"), "w") as f:
            for s in train_loss_total:
                f.write(str(s) +"\n")

if __name__ == "__main__":
    main()
