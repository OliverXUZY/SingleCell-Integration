import h5py
import hdf5plugin
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