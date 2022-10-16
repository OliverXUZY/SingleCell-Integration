import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from func.prepro import SingleCellDataset, SingleCellDataset_test

def test10(inputs):
    return torch.matmul(inputs, torch.ones(228942, 10))  # for SingleCellDataset_test

def testfull(inputs):
    return torch.matmul(inputs, torch.ones(228942, 23418)) # SingleCellDataset