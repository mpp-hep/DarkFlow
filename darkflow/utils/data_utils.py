import torch
import torch.nn as nn
import pandas as pd
import random
import h5py
import numpy as np


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
