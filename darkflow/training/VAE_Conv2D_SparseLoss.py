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
import darkflow.networks.VAE_Conv2D as VAE

# Hyperparameters
data_save_path = "/home/pjawahar/Projects/DarkFlow/"
Data_filename = data_save_path + 'Data/d_sm_10fb.npy'
Met_filename = data_save_path + 'Data/met_sm_10fb.npy'
model_name = "VAE_Conv2D_SparseLoss_run3_5EP"
num_epochs = 5
num_classes = 1
training_fraction = 0.7
batch_size = 16
learning_rate = 0.001
latent_dim = 10
beta = 1.

# Train
def train_net(model, x_train, wt_train, optimizer):
    input_train = x_train[:, :, :, :].cuda()
    wt_train = wt_train[:].cuda()
    model.train()   

    output_train = model(input_train)
    tr_loss, tr_kl, tr_eucl = compute_loss(model, input_train, wt_train)
    # add this batch loss to total loss
    # tr_loss_aux += tr_loss
    # tr_kl_aux += tr_kl
    # tr_rec_aux += tr_eucl

    # Backprop and perform Adam optimisation
    optimizer.zero_grad()
    tr_loss.backward()
    optimizer.step()

    return tr_loss, tr_kl, tr_eucl

# Test/Validate
def test_net(model, x_test, wt_test):
    model.eval()
    with torch.no_grad():
        input_test = x_test[:, :, :, :].cuda()
        wt_test = wt_test[:].cuda()
        te_loss, te_kl, te_eucl = compute_loss(model, input_test, wt_test)

    return te_loss, te_kl, te_eucl

#Sparse loss function    
def compute_loss(model, x, weight):
    # print('computing loss ...')
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_decoded = model.decode(z)
    
    # Euclidean distance 
    pdist = nn.PairwiseDistance(p=2) 
    x_pos = torch.zeros(batch_size,3,25).cuda()
    # Removes the channel dimension to make the following calculations easier
    x_pos = x[:,0,:,:] 
    # Changes the dimension of the tensor so that dist is the distance between every 
    # pair of input and output pixels
    x_pos = x_pos.view(batch_size, 4, 1, 26) 
    
    x_decoded_pos = torch.zeros(batch_size,4,26).cuda()
    # Removes the channel dimension to make the following calculations easier
    x_decoded_pos = x_decoded[:,0,:,:] 
    
    # Changes the dimension of the tensor so that dist is the distance between 
    # every pair of input and output pixels
    x_decoded_pos = x_decoded_pos.view(batch_size, 4, 26, 1) 
    x_decoded_pos = torch.repeat_interleave(x_decoded_pos, 26, -1) 
    
    dist = torch.pow(pdist(x_pos, x_decoded_pos),2)
    
    # Gets the value of the distance between the closest output pixels to all the 
    # input pixels of the images in a batch (all features of the pixels)
    ieo = torch.min(dist, dim = 1) 
    
    # Gets the value of the distance between the closest input pixels to all the 
    # output pixels of the images in a batch (all features of the pixels)
    oei = torch.min(dist, dim = 2) 
    
    # Symmetrical euclidean distances
    eucl = ieo.values + oei.values 

    # Managing weight tensor shape to prepare for incorporation into loss terms
    weight = weight.unsqueeze(1)

    # Average symmetrical euclidean distance per image with weight per event incorporated
    eucl = torch.sum(weight * eucl) / batch_size 
    
    reconstruction_loss = - eucl            
    # Compares mu = mean, sigma = exp(0.5 * logvar) gaussians with standard gaussians, with weight per event incorporated
    KL_divergence = 0.5 * torch.sum(weight * (torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1.0)).sum() / batch_size 
    ELBO = reconstruction_loss - (beta * KL_divergence) 
    loss = - ELBO
    return loss, (beta * KL_divergence), eucl

########################## TRAINING ##########################

# model = ConvNet()
model = VAE.ConvNet()
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Read data
d = read_npy(Data_filename)
met = read_npy(Met_filename)

print('Starting to process data ...')
weight = met[:,1]
evtId =  met[:,0]
met = np.array(met[:,-2:], dtype='f')

# suffle data
d, met, weight, evtId = shuffle(d, met, weight, evtId, random_state=0)

# Taking samples where pT>20TeV
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
# standardize met inputs
scaler_met = StandardScaler()
scaler_met.fit(met)
met = scaler_met.transform(met)
# save the scalers
dump(scaler_p, open(data_save_path + 'darkflow/models/run4/%s_particleScaler.pkl' %model_name, 'wb'))
dump(scaler_met, open(data_save_path + 'darkflow/models/run4/%s_metScaler.pkl' %model_name, 'wb'))

i_train = int(d.shape[0]*training_fraction)
# training data
x_train = d[:i_train,:,:,:]
met_train = met[:i_train,:]
weight_train = weight[:i_train]
evtId_train = evtId[:i_train]
# test data
x_test = d[i_train:,:,:,:]
met_test = met[i_train:,:]
weight_test = weight[i_train:]
evtId_test = evtId[i_train:]
print('Done; x_train shape: ', x_train.shape, 'x_test shape: ', x_test.shape)
# for now, only train with particles. 
# met to be concatenated to the first dense layer in the encoder

train_loader = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=x_test, batch_size=batch_size, shuffle=False)
weight_train_loader = DataLoader(dataset=weight_train, batch_size=batch_size, shuffle=False)
weight_test_loader = DataLoader(dataset=weight_test, batch_size=batch_size, shuffle=False)

# to store training history
x_graph = []
train_y_rec = []
train_y_kl = []
train_y_loss = []
test_y_rec = []
test_y_kl = []
test_y_loss = []
test_ev_rec = []
test_ev_kl = []
test_ev_loss = []


print('Initiating training, testing processes ...')
for epoch in range(num_epochs):
    x_graph.append(epoch)
    print('Starting to train ...')

    # training
    tr_loss_aux = 0.0
    tr_kl_aux = 0.0
    tr_rec_aux = 0.0
    for y, (x_train, wt_train) in tqdm(enumerate(zip(train_loader, weight_train_loader))):
        if y == (len(train_loader) - 1): break

        # input_train = x_train[:, :, :, :].cuda()
        # wt_train = wt_train[:].cuda()

        # Train
        # print('\rbatch: ',y, end='')
        tr_loss, tr_kl, tr_eucl = train_net(model, x_train, wt_train, optimizer)
        # output_train = model(input_train)

        # tr_loss, tr_kl, tr_eucl = compute_loss(model, input_train, wt_train)
        # add this batch loss to total loss
        tr_loss_aux += tr_loss
        tr_kl_aux += tr_kl
        tr_rec_aux += tr_eucl

        # Backprop and perform Adam optimisation
        # optimizer.zero_grad()
        # tr_loss.backward()
        # optimizer.step()

    print('Moving to validation stage ...')
    # validation
    te_loss_aux = 0.0
    te_kl_aux = 0.0
    te_rec_aux = 0.0
    for y, (x_test, wt_test) in tqdm(enumerate(zip(test_loader, weight_test_loader))):
        if y == (len(test_loader) - 1): break
        
        #Test
        # print('\rbatch: ',y, end='')
        te_loss, te_kl, te_eucl = test_net(model, x_test, wt_test)

        # input_test = x_test[:, :, :, :].cuda()
        # wt_test = wt_test[:].cuda()
        # te_loss, te_kl, te_eucl = compute_loss(model, input_test, wt_test)

        # add this batch loss to total loss
        te_loss_aux += te_loss
        te_kl_aux += te_kl
        te_rec_aux += te_eucl

    train_y_loss.append(tr_loss_aux.cpu().detach().numpy()/(len(train_loader)))
    train_y_kl.append(tr_kl_aux.cpu().detach().numpy()/(len(train_loader)))
    train_y_rec.append(tr_rec_aux.cpu().detach().numpy()/(len(train_loader)))
        
    test_y_loss.append(te_loss_aux/(len(test_loader)))
    test_y_kl.append(te_kl_aux/(len(test_loader)))
    test_y_rec.append(te_rec_aux/(len(test_loader)))
        
    print('Epoch: {} -- Train loss: {}  -- Test loss: {}'.format(epoch, 
                                                                 tr_loss_aux/(len(train_loader)), 
                                                                 te_loss_aux/(len(test_loader))))

# Save the model
torch.save(model.state_dict(), data_save_path + 'darkflow/models/run4/%s.pt' %model_name)

# store training history
outFile = h5py.File(data_save_path + 'darkflow/models/run4/%s_Traininghistory.h5' %model_name, "w")
dt = h5py.special_dtype(vlen=np.dtype('float64'))
outFile.create_dataset('epoch', data=x_graph, compression='gzip', dtype = dt)

outFile.create_dataset('train_loss_reco', data = np.array(train_y_rec), compression='gzip', dtype = dt)
outFile.create_dataset('train_loss_kl', data = train_y_kl, compression='gzip', dtype = dt)
outFile.create_dataset('train_loss', data = train_y_loss, compression='gzip', dtype = dt)

# outFile.create_dataset('test_loss_reco', data = test_y_rec, compression='gzip', dtype = dt)
# outFile.create_dataset('test_loss_kl', data = test_y_kl, compression='gzip', dtype = dt)
# outFile.create_dataset('test_loss', data = test_y_loss, compression='gzip', dtype = dt)

outFile.close()
print('Network Run Complete')

