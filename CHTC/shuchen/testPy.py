import sys
import pandas as pd
print('sys.argv is', sys.argv)


cite_input = pd.read_csv(file1)  ## file1 is test_cite_inputs
multi_input = pd.read_csv(file2) ## file2 is test_multi_inputs

res1 = 0  ## your first prediction matrix based on test_cite_inputs

res2 = 0  ## your second prediction matrix based on test_multi_inputs

res1.to_csv("preMat_protein.csv.gz",compression='gzip')
res2.to_csv("preMat_RNA.csv.gz",compression='gzip')
