#!/bin/bash

# First, copy the compressed tar file from /staging into the working directory,
#  and un-tar it to reveal your large input file(s) or directories:
# cp /staging/zxu444/train_multi_inputs.h5 ./
# cp /staging/zxu444/train_multi_targets.h5 ./
# cp /staging/zxu444/test_multi_inputs.h5 ./
# echo bash---
# echo $(ls)
# echo bash---
# rm train_multi_inputs.h5 train_multi_targets.h5 test_multi_inputs.h5

# python testStaging.py $1
mkdir linearTorch_PCA_LR_Cluster
python linearTorch.py $1
tar -cvf linearTorch_PCA_LR_Cluster.tar.gz linearTorch_PCA_LR_Cluster