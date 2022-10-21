#!/bin/bash

# First, copy the compressed tar file from /staging into the working directory,
#  and un-tar it to reveal your large input file(s) or directories:
cp /staging/username/large_input.tar.gz ./
cp /staging/username/large_input.tar.gz ./

python testStaging.py $1 $2