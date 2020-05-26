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

# Hyperparameters
model_name = "VAE_Conv2D_SparseLoss"
num_epochs = 500
num_classes = 1
training_fraction = 0.7
batch_size = 30
learning_rate = 0.001
latent_dim = 10
beta = 1.

##################  DEFINE MODEL  #####################

class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
    
            self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,4), stride=(1), padding=(0))
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 16, kernel_size=(5,1), stride=(1), padding=(0))
            self.bn2 = nn.BatchNorm2d(16)
            self.conv3 = nn.Conv2d(16, 8, kernel_size=(7,1), stride=(1), padding=(0))
            self.bn3 = nn.BatchNorm2d(8)
            #
            self.dense1 = nn.Linear(112,20)
            self.dnn_bn1 = nn.BatchNorm1d(20)
            self.dense2_mean = nn.Linear(20, latent_dim)
            self.dense2_logvar = nn.Linear(20, latent_dim)
            #
            self.dense3 = nn.Linear(latent_dim, 20)
            self.dnn_bn3 = nn.BatchNorm1d(20)
            self.dense4 = nn.Linear(20, 112)
            self.dnn_bn4 = nn.BatchNorm1d(112)
            self.conv4 = nn.ConvTranspose2d(8, 16, kernel_size=(7,1), stride=(1), padding=(0))
            self.bn4 = nn.BatchNorm2d(16)
            self.conv5 = nn.ConvTranspose2d(16, 32, kernel_size=(5,1), stride=(1), padding=(0))
            self.bn5 = nn.BatchNorm2d(32)
            self.conv6 = nn.ConvTranspose2d(32, 1, kernel_size=(3,4), stride=(1), padding=(0))
    
        def encode(self, x):
            # Conv Layer 1
            print(x.size())
            out = self.conv1(x)
            out = self.bn1(out)
            out = torch.relu(out)
            print(out.size())
            # Conv Layer 2
            out = self.conv2(out)
            out = self.bn2(out)
            out = torch.relu(out)
            # Conv Layer 3            
            out = self.conv3(out)
            out = self.bn3(out)
            out = torch.relu(out)
            # flatten
            out = out.view(out.size(0), -1)
            # dense Layer 1
            out = self.dense1(out)
            out = self.dnn_bn1(out)
            out = torch.relu(out)
            # dense Layer 2
            mean  = self.dense2_mean(out)
            logvar = self.dense2_logvar(out)
            return mean, logvar
    
        def decode(self, z):
            # dense Layer 3
            out = self.dense3(z)
            out = self.dnn_bn3(out)
            out = torch.relu(out)
            # dense Layer 4
            out = self.dense4(out)
            out = self.dnn_bn4(out)
            out = torch.relu(out)
            # reshape
            out = out.view(batch_size, 8, 14, 1)
            # DeConv Layer 1
            out = self.conv4(out)
            out = self.bn4(out)
            out = torch.relu(out)
            # DeConv Layer 2
            out = self.conv5(out)
            out = self.bn5(out)
            out = torch.relu(out)
            # DeConv Layer 6
            out = self.conv6(out)
            return out
    
        def reparameterize(self, mean, logvar):
            z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
            return z
    
        def forward(self, x):
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            out = self.decode(z)
            return out
    
def compute_loss(model, x):
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
    
    # Average symmetrical euclidean distance per image
    eucl = torch.sum(eucl) / batch_size 
    
    reconstruction_loss = - eucl            
    # Compares mu = mean, sigma = exp(0.5 * logvar) gaussians with standard gaussians
    KL_divergence = 0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1.0).sum() / batch_size 
    ELBO = reconstruction_loss - (beta * KL_divergence) 
    loss = - ELBO
    return loss, (beta * KL_divergence), eucl

########################## TRAINING ##########################

model = ConvNet()
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load data
#f = h5py.File("/eos/project/d/dshep/DARKMACHINES/sm/sm.h5", "r")
f = h5py.File("/eos/project/d/dshep/DARKMACHINES/sm/4top_10fb.h5", "r")
particles = ['Bjets', 'MuPlus', 'MuMinus', 'ElePlus', 'EleMinus', 'Gamma']
d = np.array(f.get("Jets"), dtype='f')
for p in particles:
    d = np.concatenate((d, np.array(f.get(p), dtype='f')), axis=1) 
# add channel
d = np.reshape(d, (d.shape[0], 1, d.shape[1], d.shape[2]))
met = np.array(f.get("EventFeatures"))
weight = met[:,1]
evtId =  met[:,0]
met = np.array(met[:,-2:], dtype='f')

# suffle data
d, met, weight, evtId = shuffle(d, met, weight, evtId, random_state=0)

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
dump(scaler_p, open('../models/%s_particleScaler.pkl' %model_name, 'wb'))
dump(scaler_met, open('../models/%s_metScaler.pkl' %model_name, 'wb'))

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

# for now, only train with particles. 
# met to be concatenated to the first dense layer in the encoder

train_loader = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=x_test, batch_size=batch_size, shuffle=False)

# to store training history
x_graph = []
train_y_rec = []
train_y_kl = []
train_y_loss = []
test_y_rec = []
test_y_kl = []
test_y_loss = []

for epoch in range(num_epochs):
    x_graph.append(epoch)

    # training
    tr_loss_aux = 0.0
    tr_kl_aux = 0.0
    tr_rec_aux = 0.0
    for y, (x_train) in enumerate(train_loader):
        if y == (len(train_loader) - 1): break

        input_train = x_train[:, :, :, :].cuda()
        # Train
        output_train = model(input_train)
        tr_loss, tr_kl, tr_eucl = compute_loss(model, input_train)
        # add this batch loss to total loss
        tr_loss_aux += tr_loss
        tr_kl_aux += tr_kl
        tr_rec_aux += tr_eucl

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()
        
    # validation
    te_loss_aux = 0.0
    te_kl_aux = 0.0
    te_rec_aux = 0.0
    for y, (x_test) in enumerate(test_loader):
        if y == (len(test_loader) - 1): break
        input_test = x_test[:, :, :, :].cuda()
        te_loss, te_kl, te_eucl = compute_loss(model, input_test)
        # add this batch loss to total loss
        te_loss_aux += te_loss
        te_kl_aux += te_kl
        te_rec_aux += te_eucl

    train_y_loss.append(tr_loss_aux.cpu().detach().numpy()/(len(train_loader)))
    train_y_kl.append(tr_kl_aux.cpu().detach().numpy()/(len(train_loader)))
    train_y_rec.append(tr_rec_aux.cpu().detach().numpy()/(len(train_loader)))
        
    test_y_loss.append(te_loss_aux/(len(train_loader)))
    test_y_kl.append(te_kl_aux/(len(train_loader)))
    test_y_rec.append(te_rec_aux/(len(train_loader)))
        
    print('Epoch: {} -- Train loss: {}  -- Test loss: {}'.format(epoch, 
                                                                 tr_loss_aux/(len(train_loader)), 
                                                                 tr_loss_aux/(len(test_loader))))

# Save the model
torch.save(model.state_dict(), '../models/%s.pt' %model_name)

# store training history
outFile = h5py.File('../models/%s_Traininghistory.h5' %model_name, "w")
outFile.create_dataset('epoch', data=x_graph, compression='gzip')

outFile.create_dataset('train_loss_reco', data = np.array(train_y_rec), compression='gzip')
outFile.create_dataset('train_loss_kl', data = train_y_kl, compression='gzip')
outFile.create_dataset('train_loss', data = train_y_loss, compression='gzip')

outFile.create_dataset('test_loss_reco', data = test_y_rec, compression='gzip')
outFile.create_dataset('test_loss_kl', data = test_y_kl, compression='gzip')
outFile.create_dataset('test_loss', data = test_y_loss, compression='gzip')

outFile.close()

