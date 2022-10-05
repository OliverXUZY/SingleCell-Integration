import os
import argparse
import pwd
import sys
sys.path.append("/Users/zyxu/Documents/py/kris")

import numpy as np
import pandas as pd
import h5py
import hdf5plugin
import func.prepro as pp
from tqdm import tqdm
import gc

import seaborn as sns
import matplotlib.pyplot as plt

custom_colors = ["#a8e6cf","#dcedc1","#ffd3b6","#ffaaa5","#ff8b94"]
palette = sns.set_palette(sns.color_palette(custom_colors))

DATA_DIR = "open-problems-multimodal"
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


preMat_protein = str(sys.argv[1])  # 
preMat_RNA = str(sys.argv[2])  # 


def prot_submission(preMat, eva_protein):
    numCells = preMat.shape[0]
    assert numCells == 48663
    numProt = preMat.shape[1]
    assert numProt == 140
    
    sub = pd.DataFrame({"row_id": range(48663 * 140), "target": preMat.reshape(-1)})
    return sub


def RNA_submission(preMat, eva_RNA, RNAs, cellId_RNA):
    """
    preMat: the big prediction matrix, rows encoded by cellId_RNA, cols encoded by RNAs
    eva_RNA: evaluation_ids matrix
    """
    cell_dict = {}
    for i in range(cellId_RNA.shape[0]):
        cell_dict[cellId_RNA[i]] = i

    gene_dict = {}
    for i,value in enumerate(RNAs):
        gene_dict[value] = i
    
    retval = []
    for i in range(eva_RNA.shape[0]):
        rowId = cell_dict[eva_RNA.iloc[i,1]]
        colId = gene_dict[eva_RNA.iloc[i,2]]
        
        retval.append({"cell_id": eva_RNA.iloc[i,1], "gene_id": eva_RNA.iloc[i,2], "target": preMat[rowId, colId]})
    return retval

def main(preMat_protein, preMat_RNA):
    ## read evaluation csv
    evaluation = pd.read_csv(f"{FP_EVALUATION_IDS}")

    ## protein prediction
    eva_protein = evaluation.iloc[:48663*140]
    sub_protein = prot_submission(preMat_protein,eva_protein)

    ## RNA prediction
    eva_RNA = evaluation.iloc[48663*140:]

    train_multi_targets = pp.readH5pyFile(FP_MULTIOME_TRAIN_TARGETS)
    test_multi_inputs = pp.readH5pyFile(FP_MULTIOME_TEST_INPUTS)

    RNAs = train_multi_targets["axis0"].astype(str)
    cellId_RNA = test_multi_inputs["axis1"].astype(str)

    sub_RNA = RNA_submission(preMat_RNA, eva_RNA, RNAs, cellId_RNA)

    df = pd.concat((sub_protein,sub_RNA)).reset_index(drop=True)

    df.to_csv("submission.csv.gz",compression='gzip')

if __name__ == "__main__":
    # print(preMat_protein, preMat_RNA)
    main(preMat_protein, preMat_RNA)


    