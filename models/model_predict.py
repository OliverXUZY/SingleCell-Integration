import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from func.prepro import SingleCellDataset, SingleCellDataset_test

def test10(inputs):
    return torch.matmul(inputs, torch.ones(228942, 10))  # for SingleCellDataset_test

def test20(inputs):
    return torch.matmul(inputs, torch.ones(228942, 20))  # for SingleCellDataset_test

def testfull(inputs):
    return torch.matmul(inputs, torch.ones(228942, 23418)) # SingleCellDataset


class LinearRegression(nn.Module):
    def __init__(self, weightPCA) -> None:
        super(LinearRegression, self).__init__()

        self.LR = nn.Linear(1000,100, bias=True)
        self.weightPCA = weightPCA

    def forward(self, input):
        input = input.matmul(self.weightPCA)
        output = self.LR(input)
        return output