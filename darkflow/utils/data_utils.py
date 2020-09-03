import torch
import torch.nn as nn
import pandas as pd
import random
import h5py
import numpy as np
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


def save_csv(data, columns, filename):
    print('Saving CSV file in ', filename)
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
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

    out_file.create_dataset('event_ID', data = df['event_ID'].astype('float16'), compression = 'gzip')
    out_file.create_dataset('process_ID', data = df['process_ID'], compression = 'gzip', dtype = dt)
    out_file.create_dataset('event_weight', data = df['event_weight'].astype('int'), compression = 'gzip')
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

    jets = np.full((d.shape[0],13,4), -999, dtype=float)
    bjets = np.full((d.shape[0],6,4), -999, dtype=float)
    MPlus = np.full((d.shape[0],6,4), -999, dtype=float)
    MMinus = np.full((d.shape[0],6,4), -999, dtype=float)
    EPlus = np.full((d.shape[0],6,4), -999, dtype=float)
    EMinus = np.full((d.shape[0],6,4), -999, dtype=float)
    Gamma = np.full((d.shape[0],6,4), -999, dtype=float)
        
    for i in range(d.shape[0]):
        ct_j = 0
        ct_bj = 0
        ct_mm = 0
        ct_ep = 0
        ct_em = 0
        ct_g = 0
        ct_mp = 0

        for j in range(d.shape[1]):
            if d[i][j] == 'j':
                jets[i][ct_j] = d[i][j+1:j+5]
                ct_j += 1
            elif d[i][j] == 'b':
                bjets[i][ct_bj] = d[i][j+1:j+5]
                ct_bj += 1
            elif d[i][j] == 'm+':
                MPlus[i][ct_mp] = d[i][j+1:j+5]
                ct_mp += 1
            elif d[i][j] == 'm-':
                MMinus[i][ct_mm] = d[i][j+1:j+5]
                ct_mm += 1
            elif d[i][j] == 'e+':
                EPlus[i][ct_ep] = d[i][j+1:j+5]
                ct_ep += 1
            elif d[i][j] == 'e-':
                EMinus[i][ct_em] = d[i][j+1:j+5]
                ct_em += 1
            elif d[i][j] == 'g':
                Gamma[i][ct_g] = d[i][j+1:j+5]
                ct_g += 1
            else:
                flag = 1

    data_f = np.concatenate((jets,bjets,MPlus,MMinus,EPlus,EMinus,Gamma), axis=1)
    data_f = np.reshape(data_f, (data_f.shape[0], 1, data_f.shape[1], data_f.shape[2]))
    met = np.reshape(met, (met.shape[0], 1))
    met = np.c_[met, event_weight]
    
    save_npy(data_f, data_save_path + 'Data/d_sm_%s.npy' %channel)
    save_npy(met, data_save_path + 'Data/met_%s.npy' %channel)
    print('**Done**')

    return data_f, met_values, event_weight

def save_run_history(best_model, model, model_save_path, model_name, x_graph, train_y_rec, train_y_kl, train_y_loss, hist_name):
    # Save the model
    print('Saving run history ...')
    torch.save(best_model.state_dict(), model_save_path + 'BEST_%s.pt' %model_name)
    torch.save(model.state_dict(), model_save_path + '%s.pt' %model_name)

    # store training history
    outFile = h5py.File(model_save_path + hist_name + '_%s_.h5' %model_name, "w")
    outFile.create_dataset('epoch', data=x_graph, compression='gzip')

    outFile.create_dataset('train_loss_reco', data = np.array(train_y_rec), compression='gzip')
    outFile.create_dataset('train_loss_kl', data = train_y_kl, compression='gzip')
    outFile.create_dataset('train_loss', data = train_y_loss, compression='gzip')
    outFile.close()
    print('** Done **')
