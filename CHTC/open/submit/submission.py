import os
from os.path import exists
# import argparse
# import pwd
import sys

import time
import numpy as np
import pandas as pd
import h5py
import hdf5plugin
from tqdm import tqdm
# import gc

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

from prepro import SingleCellDataset, SingleCellDataset_test
import model_predict
from model_predict import LinearRegression, recover_from_cluster



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
FP_MULTIOME_TRAIN_weightPCA = os.path.join(DATA_DIR,"weightPCA.csv.gz")

FP_cluster_labels = os.path.join(DATA_DIR,"cluster_labels.npy")

# preMat_DIR = "preMat"
# preMat_protein = os.path.join(preMat_DIR,str(sys.argv[1]))  # 
# preMat_RNA = os.path.join(preMat_DIR, str(sys.argv[2]))  # 

FP_submission_protein = os.path.join(DATA_DIR,"submission_protein.csv.gz")

### parameters
BATCH_SIZE = 2048

START_EPOCH = 15


def prot_submission(preMat_path, eva_protein):
    preMat = pd.read_csv(preMat_path, compression="gzip").values
    numCells = preMat.shape[0]
    assert numCells == 48663
    print("num cells right")
    numProt = preMat.shape[1]
    print("num proteins right")
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

## This function is optimized by below v2
# def eva_dataframe(dataloader, responses, model_pre, eva_RNA = None):
#     """
#     only for RNA prediction, since multiome is too large, we exchange with this minibatch method, each time, we predict a minibatch
#     then we check whether this batch contains any (cell_id, RNA) pair contained in evaluation matrix, if it does, extract those

#     dataloader: test loader
#     responses: the response columns
#     model_pre: any model support forward prediction
#     eva_RNA: the evaluation matrix contains only multi_ome

#     eva_protein = evaluation.iloc[:48663*140]
#     eva_RNA = evaluation.iloc[48663*140:]

#     usage:
#     y_df = eva_dataframe(testloader_multi, trainloader_multi.dataset.targets, model_pre, eva_RNA)
#     """

#     y_df_total = pd.DataFrame()
#     eval_cells = eva_RNA.cell_id.unique()
#     i=0
#     for cell_ids, inputs in tqdm(dataloader):
#         y_hat_batch = model_pre(inputs).cpu().detach().numpy()
#         y_df = pd.DataFrame(y_hat_batch).T
#         y_df.columns = cell_ids
#         y_df['responses'] = responses
#         y_df = pd.melt(y_df,id_vars = 'responses', value_vars=y_df.columns[:y_df.shape[1]-1]) 
#         y_df.columns = ["gene_id", "cell_id", "target"]

#         # check cell_ids first, whether any cells in evaluations
#         y_df = y_df[y_df['cell_id'].isin(eval_cells)]
#         if y_df.empty:
#             continue
        
#         temp = eva_RNA[eva_RNA['cell_id'].isin(y_df.cell_id)]
#         eva_RNA_cell_gene_selected = temp[temp['gene_id'].isin(y_df.gene_id)]
#         y_df = pd.merge(eva_RNA_cell_gene_selected,y_df, how = "inner")

#         y_df_total  = pd.concat([y_df_total,y_df])
#         i += 1
#         # if i >= 5:
#         #     break
#     return y_df_total


## v2
def eva_dataframe(dataloader, responses, model_pre, eva_RNA, cluster_model_labels = None, nz_index = None):
    """
    only for RNA prediction, since multiome is too large, we exchange with this minibatch method, each time, we predict a minibatch
    then we check whether this batch contains any (cell_id, RNA) pair contained in evaluation matrix, if it does, extract those

    dataloader: test loader
    responses: the response columns
    model_pre: any model support forward prediction: input to 100 responses
    eva_RNA: the evaluation matrix contains only multi_ome

    eva_protein = evaluation.iloc[:48663*140]
    eva_RNA = evaluation.iloc[48663*140:]

    usage:
    y_df = eva_dataframe(testloader_multi, trainloader_multi.dataset.targets, model_pre, eva_RNA)
    """
    start_time = time.time()
    y_list = []
    eval_cells = eva_RNA.cell_id.unique()
    i=1
    for cell_ids, inputs in dataloader:
        ## first filter the cells to be estimated
        cell_ids = pd.Series(cell_ids)
        inputs = inputs[cell_ids.isin(eval_cells)]
        if inputs.shape[0] == 0:
            continue

        y_hat_batch = model_pre(inputs).cpu().detach().numpy()

        ### 100 responses to 23418 responses
        y_hat_batch = recover_from_cluster(y_hat_batch, cluster_model_labels, nz_index, 23418)

        y_df = pd.DataFrame(y_hat_batch).T
        y_df.columns = cell_ids[cell_ids.isin(eval_cells)]
        y_df['responses'] = responses
        y_df = pd.melt(y_df,id_vars = 'responses', value_vars=y_df.columns[:y_df.shape[1]-1]) 
        y_df.columns = ["gene_id", "cell_id", "target"]

        
        temp = eva_RNA[eva_RNA['cell_id'].isin(y_df.cell_id)]
        eva_RNA_cell_gene_selected = temp[temp['gene_id'].isin(y_df.gene_id)]
        y_df = pd.merge(eva_RNA_cell_gene_selected,y_df, how = "inner")

        y_list.append(y_df)

        print(f"Batch {i} | {len(dataloader)}. Time elapsed: {(time.time() - start_time)/60} min")
        i+=1

    return pd.concat(y_list)

def main():
    # read evaluation csv
    evaluation = pd.read_csv(f"{FP_EVALUATION_IDS}")

    if not exists(FP_submission_protein):
        ## protein prediction
        eva_protein = evaluation.iloc[:48663*140]
        sub_protein = prot_submission(preMat_protein,eva_protein)
        sub_protein.to_csv(FP_submission_protein, index = False,compression='gzip')
        print("---protein done!---")
    else:
        print("---protein already exists!---")

    ## RNA prediction
    eva_RNA = evaluation.iloc[48663*140:]

    test_multi = SingleCellDataset(FP_MULTIOME_TEST_INPUTS)
    testloader_multi = DataLoader(test_multi, batch_size=BATCH_SIZE)
    train_multi = SingleCellDataset(FP_MULTIOME_TRAIN_INPUTS,FP_MULTIOME_TRAIN_TARGETS)
    trainloader_multi = DataLoader(train_multi, batch_size=BATCH_SIZE)

    with open(FP_cluster_labels, 'rb') as f:
        a = np.load(f)
    cluster_model_labels = a[0,:]
    nz_index = a[1,:]

    print("start reading weightPCA--")
    weightPCA = pd.read_csv(FP_MULTIOME_TRAIN_weightPCA, compression = 'gzip').values
    weight = torch.from_numpy(weightPCA).to(torch.float32).T
    # weight = torch.ones(228942, 1000)
    print("finish reading weightPCA--")
    print("weightPCA shape: ", weight.shape)
    model = LinearRegression(weight)
    model.load_state_dict(torch.load("model_epoch{}.pt".format(START_EPOCH)))
    # model_pre = model_predict.testfull
    model_pre = model
    sub_RNA = eva_dataframe(testloader_multi, trainloader_multi.dataset.targets, model_pre, eva_RNA, cluster_model_labels = cluster_model_labels, nz_index = nz_index)
    sub_RNA = sub_RNA.sort_values(by=['row_id'])

    sub_RNA.to_csv(f"submission_RNA_full_epoch{START_EPOCH}.csv.gz", index = False,compression='gzip')

    # df = pd.concat((sub_protein,sub_RNA)).reset_index(drop=True)

    
if __name__ == "__main__":
    main()


    