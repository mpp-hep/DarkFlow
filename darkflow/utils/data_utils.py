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

def csv_to_hdf5(directory):
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
    out_file = h5py.File('data_N.hf5', 'w')

    dt = h5py.special_dtype(vlen = str)

    out_file.create_dataset('event_ID', data = df['event_ID'].astype('float16'), compression = 'gzip')
    out_file.create_dataset('process_ID', data = df['process_ID'], compression = 'gzip', dtype = dt)
    out_file.create_dataset('event_weight', data = df['event_weight'].astype('int'), compression = 'gzip')
    out_file.create_dataset('MET_values', data = df[['MET', 'MET_Phi']].astype('float16'), compression = 'gzip')

    for entry in list_of_lists[1:]:
        out_file.create_dataset(entry[0], data = df[entry], compression = 'gzip', dtype = dt)

    out_file.close()

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
