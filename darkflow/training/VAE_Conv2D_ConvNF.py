##################################################################################
# VAE on DarkMachine dataset with 3D Sparse Loss                                 # 
# Author: B. Orzani (Universidade Estadual Paulista, Brazil), M. Pierini (CERN)  #
##################################################################################

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from pickle import dump
import numpy as np
import h5py
from tqdm import tqdm

from darkflow.utils.data_utils import save_npy, read_npy
from darkflow.utils.network_utils import compute_loss
from darkflow.utils.train_utils import train_net, test_net
import darkflow.networks.VAE_NF_Conv2D as VAE

# Hyperparameters
data_save_path = "/home/pjawahar/Projects/DarkFlow/"
Data_filename = data_save_path + 'Data/d_sm_10fb.npy'
Data_bsm_filename = data_save_path + 'Data/d_bsm_10fb_test.npy'
Met_filename = data_save_path + 'Data/met_sm_10fb.npy'
Met_bsm_filename = data_save_path + 'Data/met_bsm_10fb_test.npy'
model_name = "VAE_ConvNF_Conv2D_runBestModel_2EP"
num_epochs = 2
num_classes = 1
training_fraction = 0.8
batch_size = 16
test_batchsize = 1
learning_rate = 0.001
latent_dim = 10
beta = 1


########################## TRAINING ##########################

# model = ConvNet()
# model = VAE.PlanarVAE()
model = VAE.HouseholderSylvesterVAE()
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Read data
# d = read_npy(Data_filename)
# met = read_npy(Met_filename)
d = read_npy(Data_filename)
d_bsm = read_npy(Data_bsm_filename)
met = read_npy(Met_filename)
met_bsm = read_npy(Met_bsm_filename)

print('Starting to process data ...')
weight = met[:,1]
evtId =  met[:,0]
weight_bsm = met_bsm[:,1]
evtId_bsm =  met_bsm[:,0]
met = np.array(met[:,-2:], dtype='f')

# suffle data
d, met, weight, evtId = shuffle(d, met, weight, evtId, random_state=0)
d_bsm, weight_bsm, evtId_bsm = shuffle(d_bsm, weight_bsm, evtId_bsm, random_state=0)

# Taking samples where pT>20GeV
idx = []
for i in range(d.shape[0]):
    if((d[i,:,3,0]*d[i,:,3,0]+d[i,:,3,1]*d[i,:,3,1])>400):
        idx.append(i)
d = d[idx,:,:,:]
met = met[idx,:]
weight = weight[idx]
evtId = evtId[idx]

# standardize particle inputs
scaler_p = StandardScaler()
d_shape = d.shape
d = np.reshape(d, (d_shape[0], d_shape[2]*d_shape[3]))
scaler_p.fit(d)
d = scaler_p.transform(d)
d = np.reshape(d, d_shape)

scaler_p = StandardScaler()
d_bsm_shape = d_bsm.shape
d_bsm = np.reshape(d_bsm, (d_bsm_shape[0], d_bsm_shape[2]*d_bsm_shape[3]))
scaler_p.fit(d_bsm)
d_bsm = scaler_p.transform(d_bsm)
d_bsm = np.reshape(d_bsm, d_bsm_shape)

#Build test set
d_test = d[:2013,:,:,:]
weight_test = weight[:2013]
x_test = np.append(d_test, d_bsm, axis=0)

d = d[2014:,:,:,:]
weight = weight[2014:]

# # standardize met inputs
# scaler_met = StandardScaler()
# scaler_met.fit(met)
# met = scaler_met.transform(met)
# save the scalers
# dump(scaler_p, open(data_save_path + 'darkflow/models/run4/%s_particleScaler.pkl' %model_name, 'wb'))
# dump(scaler_met, open(data_save_path + 'darkflow/models/run4/%s_metScaler.pkl' %model_name, 'wb'))

i_train = int(d.shape[0]*training_fraction)
# training data
x_train = d[:i_train,:,:,:]
# met_train = met[:i_train,:]
weight_train = weight[:i_train]
# evtId_train = evtId[:i_train]
# Val data
x_val = d[i_train:,:,:,:]
# met_val = met[i_train:,:]
weight_val = weight[i_train:]
# evtId_val = evtId[i_train:]
print('Done; x_train shape: ', x_train.shape, 'x_val shape: ', x_val.shape, 'x_test shape: ', x_test.shape)
# for now, only train with particles. 
# met to be concatenated to the first dense layer in the encoder

train_loader = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(dataset=x_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=x_test, batch_size=test_batchsize, shuffle=False)
weight_train_loader = DataLoader(dataset=weight_train, batch_size=batch_size, shuffle=False)
weight_val_loader = DataLoader(dataset=weight_val, batch_size=batch_size, shuffle=False)
weight_test_loader = DataLoader(dataset=weight_test, batch_size=test_batchsize, shuffle=False)

# to store training history
x_graph = []
train_y_rec = []
train_y_kl = []
train_y_loss = []
val_y_rec = []
val_y_kl = []
val_y_loss = []
test_y_rec = []
test_y_kl = []
test_y_loss = []

# print('Model Parameter: ', model)

print('Initiating training, validation processes ...')
for epoch in range(num_epochs):
    x_graph.append(epoch)
    print('Starting to train ...')

    # training
    tr_loss_aux = 0.0
    tr_kl_aux = 0.0
    tr_rec_aux = 0.0
    for y, (x_train, wt_train) in tqdm(enumerate(zip(train_loader, weight_train_loader))):
        if y == (len(train_loader) - 1): break

        tr_loss, tr_kl, tr_eucl = train_net(model, x_train, wt_train, optimizer, batch_size=batch_size)
        
        tr_loss_aux += tr_loss
        tr_kl_aux += tr_kl
        tr_rec_aux += tr_eucl

    print('Moving to validation stage ...')
    # validation
    val_loss_aux = 0.0
    val_kl_aux = 0.0
    val_rec_aux = 0.0

    for y, (x_val, wt_val) in tqdm(enumerate(zip(val_loader, weight_val_loader))):
        if y == (len(val_loader) - 1): break
        
        #Test
        val_loss, val_kl, val_eucl = test_net(model, x_val, wt_val, batch_size=batch_size)

        val_loss_aux += val_loss
        val_kl_aux += val_kl
        val_rec_aux += val_eucl

    train_y_loss.append(tr_loss_aux.cpu().detach().numpy()/(len(train_loader)))
    train_y_kl.append(tr_kl_aux.cpu().detach().numpy()/(len(train_loader)))
    train_y_rec.append(tr_rec_aux.cpu().detach().numpy()/(len(train_loader)))
        
    val_y_loss.append(val_loss_aux/(len(val_loader)))
    val_y_kl.append(val_kl_aux/(len(val_loader)))
    val_y_rec.append(val_rec_aux/(len(val_loader)))
        
    print('Epoch: {} -- Train loss: {}  -- Val loss: {}'.format(epoch, 
                                                                 tr_loss_aux/(len(train_loader)), 
                                                                 val_loss_aux/(len(val_loader))))
    if (epoch == 0):
        best_val_loss = val_loss_aux/(len(val_loader))
        best_model = model
    if (val_loss_aux/(len(val_loader))<best_val_loss):
        best_model = model
        best_val_loss = val_loss_aux/(len(val_loader))
        print('Best Model Yet')
    

########################## TESTING ##########################
# print('Starting the Testing Process ...')
# test_ev_rec = []
# test_ev_kl = []
# test_ev_loss = []
# for y, (x_test, wt_test) in tqdm(enumerate(zip(test_loader, weight_test_loader))):
#     if y == (len(test_loader) - 1): break
    
#     #Test
#     # print('\rbatch: ',y, end='')
#     te_loss, te_kl, te_eucl = test_net(model, x_test, wt_test, batch_size=test_batchsize)

#     # print('loss: ', te_loss, 'kl: ', te_kl, )
    
#     test_ev_loss.append(te_loss.cpu().detach().numpy())
#     test_ev_kl.append(te_kl.cpu().detach().numpy())
#     test_ev_rec.append(te_eucl.cpu().detach().numpy())
# # print('loss: ', test_ev_loss)
# save_npy(np.array(test_ev_loss), data_save_path + 'Data/TestResults/%s_loss.npy' %model_name)
# save_npy(np.array(test_ev_kl), data_save_path + 'Data/TestResults/%s_kl.npy' %model_name)
# save_npy(np.array(test_ev_rec), data_save_path + 'Data/TestResults/%s_rec.npy' %model_name)

# print('Testing Complete')

# Save the model
print('Saving run history ...')
torch.save(best_model.state_dict(), data_save_path + 'darkflow/models/run4/BEST_%s.pt' %model_name)
torch.save(model.state_dict(), data_save_path + 'darkflow/models/run4/%s.pt' %model_name)

# store training history
outFile = h5py.File(data_save_path + 'darkflow/models/run4/%s_TrainingHistory.h5' %model_name, "w")
outFile.create_dataset('epoch', data=x_graph, compression='gzip')

outFile.create_dataset('train_loss_reco', data = np.array(train_y_rec), compression='gzip')
outFile.create_dataset('train_loss_kl', data = train_y_kl, compression='gzip')
outFile.create_dataset('train_loss', data = train_y_loss, compression='gzip')
outFile.close()

# store val history
outFile = h5py.File(data_save_path + 'darkflow/models/run4/%s_ValHistory.h5' %model_name, "w")
outFile.create_dataset('epoch', data=x_graph, compression='gzip')

outFile.create_dataset('val_loss_reco', data = val_y_rec, compression='gzip')
outFile.create_dataset('val_loss_kl', data = val_y_kl, compression='gzip')
outFile.create_dataset('val_loss', data = val_y_loss, compression='gzip')

outFile.close()
print('Network Run Complete')

