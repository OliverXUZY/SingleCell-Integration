import h5py
import pandas as pd

def readH5pyFile(filename):
    with h5py.File(filename, "r") as f:
        a_group_key = list(f.keys())[0]
        data = list(f[a_group_key])
        group = f[a_group_key]      # returns as a h5py dataset object
        d = {}
        for i in list(group.keys()):
            d[i] = group[i][()]
    return d

def create_internal_split(overall, metadata, holdout_day=4):
    metadata.cell_id = metadata.cell_id.astype("str")
    internal_holdout = metadata[metadata["day"] == holdout_day].cell_id
    
    # create a DataFrame from the h5py objects
    overall_df = pd.DataFrame(overall['block0_values'])
    overall_df.columns = overall["axis0"].astype("str")
    overall_df["cell_id"] = overall["axis1"].astype("str")
    overall_df = overall_df.set_index("cell_id")
    
    # create the split and return
    holdout_df = overall_df[overall_df.index.isin(internal_holdout)]
    train_df = overall_df[~overall_df.index.isin(internal_holdout)]
    return train_df, holdout_df