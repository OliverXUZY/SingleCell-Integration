import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

def test10(inputs):
    return torch.matmul(inputs, torch.ones(228942, 10))  # for SingleCellDataset_test

def test20(inputs):
    return torch.matmul(inputs, torch.ones(228942, 20))  # for SingleCellDataset_test

def testfull(inputs):
    return torch.matmul(inputs, torch.ones(228942, 23418)) # SingleCellDataset


class LinearRegression(nn.Module):
    def __init__(self, projections, DEVICE) -> None:
        super(LinearRegression, self).__init__()

        self.LR = nn.Linear(1000,100, bias=True)
        self.projections = projections
        self.DEVICE = DEVICE

    def forward(self, batch_id):
        x = self.projections[batch_id]
        output = self.LR(x.to(self.DEVICE))
        return output

def recover_from_cluster(raw_pred, cluster_model_labels, nz_index, n_targets):
    # raw_pred: prediction for those clusters, size: batch_size*raw_pred
    # cluster_model: the model used for clustering
    # nz_index: the index for all non-zero values (columns), size: number of non-zero columns
    # n_targets: how many targets in total
    # batch_size: number of observations (rows)
    batch_size = raw_pred.shape[0]
    pred = np.zeros([batch_size, n_targets])
    n_y = 100 # how many clusters do we have
    for i_y in range(n_y):          # i_y is the index for clusters
        i_nz = np.where(cluster_model_labels==i_y)    #i_nz is the index (in the nonzero items) where the item belongs to cluster i_y
        pred[:,nz_index[i_nz]] = np.tile(raw_pred[:,i_y],[1,len(i_nz)]).T    # nz_index[i_nz] is the index (in the full prediction) where the item belongs to cluster i_y
    return pred
