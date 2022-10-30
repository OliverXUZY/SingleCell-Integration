import h5py
import hdf5plugin
import numpy as np
import pandas as pd

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

def readH5pyFile_cols(filename, startcol = 0, endcol = 2500):
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
        d["cell_ids"] = group['axis1'][:].astype(str)
        d["features"] = group['axis0'][startcol:endcol].astype(str)
        d["matrix"] = group['block0_values'][:,startcol:endcol]
        
    return d

class SingleCellDataset(Dataset):
    def __init__(self,  path_to_input_file = None, path_to_target_file = None, path_to_kmeans_target = None) -> None:
        self.input_file = path_to_input_file
        self.target_file = path_to_target_file
        self.kmeans_target_file = path_to_kmeans_target
        with h5py.File(self.input_file, "r") as f:
            a_group_key = list(f.keys())[0]            
            group = f[a_group_key]      # returns as a h5py dataset object
            
            self.cells = group['axis1'][:].astype(str)
            self.features = group['axis0'][:].astype(str)
        
        ## only load target file while we train the model
        if self.target_file:
            with h5py.File(self.target_file, "r") as f:
                a_group_key = list(f.keys())[0]            
                group = f[a_group_key]      # returns as a h5py dataset object
                
                self.targets = group['axis0'][:].astype(str)
        
        if self.kmeans_target_file:
            self.k_centers = np.float32(pd.read_csv(self.kmeans_target_file,compression='gzip').values)

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, index):
        with h5py.File(self.input_file, "r") as f:
            a_group_key = list(f.keys())[0]
            group = f[a_group_key]      # returns as a h5py dataset object
            
            cells, inputs = self.cells[index], group['block0_values'][index,:] # add only for testing
        
        if not self.target_file and not self.kmeans_target_file:
            return cells, inputs
        
        if self.target_file:
            with h5py.File(self.target_file, "r") as f:
                a_group_key = list(f.keys())[0]            
                group = f[a_group_key]      # returns as a h5py dataset object
                
                targets = group['block0_values'][index,:]  # add only for testing

        elif self.kmeans_target_file:
            targets = self.k_centers[index,:]

        return cells, inputs, targets


class SingleCellDataset_test(Dataset):
    def __init__(self,  path_to_input_file = None, path_to_target_file = None, testresponse = 10) -> None:
        self.input_file = path_to_input_file
        self.target_file = path_to_target_file
        self.testresponse = testresponse
        with h5py.File(self.input_file, "r") as f:
            a_group_key = list(f.keys())[0]            
            group = f[a_group_key]      # returns as a h5py dataset object
            
            self.cells = group['axis1'][:].astype(str)
            self.features = group['axis0'][:].astype(str)
        
        ## only load target file while we train the model
        if self.target_file:
            with h5py.File(self.target_file, "r") as f:
                a_group_key = list(f.keys())[0]            
                group = f[a_group_key]      # returns as a h5py dataset object
                
                self.targets = group['axis0'][:self.testresponse].astype(str)

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, index):
        with h5py.File(self.input_file, "r") as f:
            a_group_key = list(f.keys())[0]
            group = f[a_group_key]      # returns as a h5py dataset object
            
            cells, inputs = self.cells[index], group['block0_values'][index,:] # add only for testing
        
        if not self.target_file:
            return cells, inputs
        with h5py.File(self.target_file, "r") as f:
            a_group_key = list(f.keys())[0]            
            group = f[a_group_key]      # returns as a h5py dataset object
            
            targets = group['block0_values'][index,:self.testresponse]  # add only for testing
        
        return cells, inputs, targets
            