import sys
import os
import pandas as pd
import prepro as pp
print('sys.argv is', sys.argv)
print("current working directory is", os.getcwd())

# file1 = sys.argv[1] ## test_cite_inputs
# file2 = sys.argv[2] ## test_multi_inputs

multi_input = pp.readH5pyFile("train_multi_inputs.h5") ## file1 is train_multi_inputs
multi_target = pp.readH5pyFile("train_multi_targets.h5") ## file2 is train_multi_targets
res1 = 0  ## your first prediction matrix based on test_cite_inputs

res2 = 0  ## your second prediction matrix based on test_multi_inputs

# res1.to_csv("preMat_protein.csv.gz",compression='gzip')
# res2.to_csv("preMat_RNA.csv.gz",compression='gzip')

print(len(multi_input))
print(len(multi_target))