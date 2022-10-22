#!/bin/bash

# First, copy the compressed tar file from /staging into the working directory,
#  and un-tar it to reveal your large input file(s) or directories:
cp /staging/syan58/train_multi_inputs.h5 ./
cp /staging/syan58/train_multi_targets.h5 ./

python testStaging.py $1