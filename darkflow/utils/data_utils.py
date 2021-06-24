import torch
import torch.nn as nn
import pandas as pd
import random
import h5py
import numpy as np
import scipy.sparse as sp
import os


def read_h5py(filename, dataset_name='data'):
    print("Reading data from file ", filename)
    h5f = h5py.File(filename, 'r')
    data = np.array(h5f[dataset_name])
    h5f.close()

    return data


def save_h5py(data, filename, dataset_name='data'):
    print('Saving data in ', filename)
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset(dataset_name, data=data)
    h5f.close()
    print('** Done **')


def save_npy(data, filename):
    print('Saving data in ', filename)
    np.save(filename, data)
    print('** Done **')


def read_npy(filename):
    print("Reading data from file ", filename)
    return np.load(filename, allow_pickle=True)


def save_csv(data, filename):
    print('Saving CSV file in ', filename)
    df = pd.DataFrame(data)
    df.to_csv(filename, header=None , index=None)
    print('** Done **')

def csv_to_hf5(directory, channel):
    data = []    
    for fname in os.scandir(directory):
        with open(fname, 'r') as file:
            for line in file.readlines():
                line = line.replace(';', ',')
                line = line.rstrip(',\n')
                line = line.split(',')
                data.append(line)
    
    #Find the longest line in the data 
    longest_line = max(data, key = len)
        
    #Set the maximum number of columns
    max_col_num = len(longest_line)

    #Set the columns names
    col_names = ['event_ID', 'process_ID', 'event_weight', 'MET', 'MET_Phi']

    for i in range(1, (int((max_col_num-5)/5))+1):
        col_names.append('obj'+str(i))
        col_names.append('E'+str(i))
        col_names.append('pt'+str(i))
        col_names.append('eta'+str(i))
        col_names.append('phi'+str(i))

    #Create a dataframe from the list, using the column names from before
    df = pd.DataFrame(data, columns=col_names)
    df.fillna(value= -999, inplace=True)


    #hdf5 code that will sort it into:
    #event_ID, process_ID, event_weight, MET_values (MET and MET_Phi) and Objects with variable values
    indices = [index for index, val in enumerate(col_names) if val.startswith('obj')] 
    list_of_lists = [col_names[i: j] for i, j in zip([0] + indices, indices + ([len(col_names)] if indices[-1] != len(col_names) else []))]

    #Change the path below for a specific name for the hf5 file
    out_file = h5py.File('/home/pjawahar/Projects/DarkFlow/Data/data_%s.hf5' %channel, 'w')

    dt = h5py.special_dtype(vlen = str)

    out_file.create_dataset('event_ID', data = df['event_ID'], compression = 'gzip', dtype = dt)
    out_file.create_dataset('process_ID', data = df['process_ID'], compression = 'gzip', dtype = dt)
    out_file.create_dataset('event_weight', data = df['event_weight'].astype('float16'), compression = 'gzip')
    out_file.create_dataset('MET_values', data = df[['MET', 'MET_Phi']].astype('float64'), compression = 'gzip')

    for entry in list_of_lists[1:]:
        out_file.create_dataset(entry[0], data = df[entry], compression = 'gzip', dtype = dt)

    out_file.close()

def hf5_to_npy(file, channel):
    print('Starting to process h5py file . . .')

    data_save_path = "/home/pjawahar/Projects/DarkFlow/"
    f = h5py.File(file, "r")
    keys = list(f.keys())
    data_all = []

    for k in keys:
        data_all.append(f.get(k))

    met_values = np.array(data_all[0])
    met = met_values[:,0]
    event_ID = np.array(data_all[1])
    event_weight = np.array(data_all[2])
    process_ID = np.array(data_all[-1])

    # Extracting objects data
    data = data_all[3:-1]
    # Preparing to convert dataset to array
    d = np.array(data[0])

    for p in data[1:]:
        d = np.concatenate((d,p), axis=1)

    jets_per_evt = 13 #13
    obj_per_evt = 3 #3

    jets = np.full((d.shape[0],jets_per_evt,5), 0, dtype=float) #pad with -999 might be the error with the results
    bjets = np.full((d.shape[0],obj_per_evt,5), 0, dtype=float)
    MPlus = np.full((d.shape[0],obj_per_evt,5), 0, dtype=float)
    MMinus = np.full((d.shape[0],obj_per_evt,5), 0, dtype=float)
    EPlus = np.full((d.shape[0],obj_per_evt,5), 0, dtype=float)
    EMinus = np.full((d.shape[0],obj_per_evt,5), 0, dtype=float)
    Gamma = np.full((d.shape[0],obj_per_evt,5), 0, dtype=float)
    mult = np.full((d.shape[0], 7), 0, dtype=float) # count the object multiplicities per event
    HT = [] # HT
    mEff = [] #Effective mass
        
    for i in range(d.shape[0]):
        ct_j = 0
        ct_bj = 0
        ct_mp = 0
        ct_mm = 0
        ct_ep = 0
        ct_em = 0
        ct_g = 0
        
        c_j = 0
        c_bj = 0
        c_mp = 0
        c_mm = 0
        c_ep = 0
        c_em = 0
        c_g = 0

        for j in range(d.shape[1]):
            if d[i][j] == 'j':
                c_j += 1
                if ct_j < jets_per_evt:
                    x = d[i][j+1:j+5]
                    x = np.append(x,[1])
                    jets[i][ct_j] = x #d[i][j+1:j+5]
                    ct_j += 1
                    HT[i] = HT[i] + x[1]
                    mEff[i] = mEff[i] + x[1]
            elif d[i][j] == 'b':
                c_bj += 1
                if ct_bj < obj_per_evt:
                    x = d[i][j+1:j+5]
                    x = np.append(x,[2])
                    bjets[i][ct_bj] = x #d[i][j+1:j+5]
                    ct_bj += 1
                    mEff[i] = mEff[i] + x[1]
            elif d[i][j] == 'm+':
                c_mp += 1
                if ct_mp < obj_per_evt:
                    x = d[i][j+1:j+5]
                    x = np.append(x,[3])
                    MPlus[i][ct_mp] = x #d[i][j+1:j+5]
                    ct_mp += 1
                    mEff[i] = mEff[i] + x[1]
            elif d[i][j] == 'm-':
                c_mm += 1
                if ct_mm < obj_per_evt:
                    x = d[i][j+1:j+5]
                    x = np.append(x,[4])
                    MMinus[i][ct_mm] = x #d[i][j+1:j+5]
                    ct_mm += 1
                    mEff[i] = mEff[i] + x[1]
            elif d[i][j] == 'e+':
                c_ep += 1
                if ct_ep < obj_per_evt:
                    x = d[i][j+1:j+5]
                    x = np.append(x,[5])
                    EPlus[i][ct_ep] = x #d[i][j+1:j+5]
                    ct_ep += 1
                    mEff[i] = mEff[i] + x[1]
            elif d[i][j] == 'e-':
                c_em += 1
                if ct_em < obj_per_evt:
                    x = d[i][j+1:j+5]
                    x = np.append(x,[6])
                    EMinus[i][ct_em] = x #d[i][j+1:j+5]
                    ct_em += 1
                    mEff[i] = mEff[i] + x[1]
            elif d[i][j] == 'g':
                c_g += 1
                if ct_g < obj_per_evt:
                    x = d[i][j+1:j+5]
                    x = np.append(x,[7])
                    Gamma[i][ct_g] = x #d[i][j+1:j+5]
                    ct_g += 1
                    mEff[i] = mEff[i] + x[1]
            else:
                flag = 1

        mult[i] = np.array([c_j, c_bj, c_mp, c_mm, c_ep, c_em, c_g])
        mEff[i] = mEff[i] + met[i]

    data_f = np.concatenate((jets,bjets,MPlus,MMinus,EPlus,EMinus,Gamma), axis=1)
    data_f = np.reshape(data_f, (data_f.shape[0], 1, data_f.shape[1], data_f.shape[2]))
    met = np.reshape(met, (met.shape[0], 1))
    met = np.concatenate((met, HT), axis=1) # Concat HT
    met = np.concatenate((met, mEff), axis=1) # Concat mEff
    met = np.concatenate((met, event_weight), axis=1) # Concat weight
    met_mult = np.concatenate((met, mult), axis=1) # Concat the multiplicitites
    
    # print('Data shape: ', data_f.shape)
    save_npy(data_f, data_save_path + 'Data/DMData/npy/d_%s.npy' %channel)
    save_npy(met, data_save_path + 'Data/DMData/npy/met_%s.npy' %channel)
    print('**Done**')

    return data_f, met

def save_run_history(best_model, model, model_save_path, model_name, x_graph, train_y_rec, train_y_kl, train_y_loss, hist_name):
    # Save the model
    print('Saving run history ...')
    torch.save(best_model.state_dict(), model_save_path + 'BEST_%s.pt' %model_name)
    torch.save(model.state_dict(), model_save_path + '%s.pt' %model_name)

    # store training history
    # outFile = h5py.File(model_save_path + hist_name + '_%s_.h5' %model_name, "w")
    # outFile.create_dataset('epoch', data=x_graph, compression='gzip')

    # outFile.create_dataset('train_loss_reco', data = np.array(train_y_rec), compression='gzip')
    # outFile.create_dataset('train_loss_kl', data = train_y_kl, compression='gzip')
    # outFile.create_dataset('train_loss', data = train_y_loss, compression='gzip')
    # outFile.close()
    print('** Done **')

def build_graph(x):
    # print('BUilding graph ...')

    # define node features 
    features = sp.csr_matrix(x[0,0], dtype=np.float32)
    # define adjacency for full connected undirected graph
    adj = np.ones((x.shape[2], x.shape[2]))
    adj = sp.coo_matrix(adj)    # convert to sparse matrix

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return features, adj

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    with np.errstate(divide='ignore'):
        r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_inv[np.isnan(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
