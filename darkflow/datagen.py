from utils.data_utils import hf5_to_npy

channel = 'chan3_32Dim_graph_objtype'
file = "/home/pjawahar/Projects/DarkFlow/Data/DMData/h5py/background/data_chan3.hf5"

data_f, met = hf5_to_npy(file,channel)