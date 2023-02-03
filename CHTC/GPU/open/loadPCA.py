import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

from sklearn.decomposition import IncrementalPCA

DATA_DIR = "/staging/zxu444"

BATCH_SIZE = 1024
IPCA_N_COMPONENTS = 1000
filename = f"{DATA_DIR}/test_ipca_ncom{IPCA_N_COMPONENTS}_batch{BATCH_SIZE}.sav"

ipca2 = joblib.load(f"{filename}")
print(ipca2.components_.shape)

weight = np.float32(ipca2.components_)

pd.DataFrame(weight).to_csv("weightPCA.csv.gz", index = False,compression='gzip')


