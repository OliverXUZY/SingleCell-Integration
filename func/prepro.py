import h5py
import hdf5plugin

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

def readH5pyFile(filename):
    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        # print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]
        # print(a_group_key)

        # get the object type for a_group_key: usually group or dataset
        # print(type(f[a_group_key])) 

        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])
        
        group = f[a_group_key]      # returns as a h5py dataset object

        # print(list(group.keys()))

        d = {}
        for i in list(group.keys()):
            d[i] = group[i][()]
        
    return d


class SingleCellDataset(Dataset):
    def __init__(self, path_to_input_file, path_to_target_file) -> None:
        self.input_file = path_to_input_file
        self.target_file = path_to_target_file
        with h5py.File(self.input_file, "r") as f:
            a_group_key = list(f.keys())[0]            
            group = f[a_group_key]      # returns as a h5py dataset object
            
            self.cells = group['axis1'][:].astype(str)
            self.features = group['axis0'][:].astype(str)
        
        with h5py.File(self.target_file, "r") as f:
            a_group_key = list(f.keys())[0]            
            group = f[a_group_key]      # returns as a h5py dataset object
            
            self.targets = group['axis0'][:].astype(str)

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, index):
        with h5py.File(self.input_file, "r") as f:
            a_group_key = list(f.keys())[0]
            group = f[a_group_key]      # returns as a h5py dataset object
            
            cells, inputs = self.cells[index], group['block0_values'][index]
        
        with h5py.File(self.target_file, "r") as f:
            a_group_key = list(f.keys())[0]            
            group = f[a_group_key]      # returns as a h5py dataset object
            
            targets = group['block0_values'][index]
        
        return cells, inputs, targets
            