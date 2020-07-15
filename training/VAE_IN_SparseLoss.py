###################################################################################
# VAE on DarkMachine dataset with 3D Sparse Loss and Interaction Networks         # 
# Authors: B. Orzani (Universidade Estadual Paulista, Brazil), M. Pierini (CERN)  #
#          P. Jawahar (Worcester Polytechnic Institute)
###################################################################################

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import itertools
from torch.autograd.variable import *
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from pickle import dump
import numpy as np
import h5py
import math

# Hyperparameters
model_name = "VAE_IN_SparseLoss"
num_epochs = 500
num_classes = 1
training_fraction = 0.7
batch_size = 10
learning_rate = 0.001
latent_dim = 10
beta = 1.

################################

def train_net(model, x_train, optimizer):
    input_train = x_train[:, :, :].cuda()
    # wt_train = wt_train[:].cuda()   

    output_train = model(input_train)
    tr_loss, tr_kl, tr_eucl = compute_loss(model, input_train)
    # add this batch loss to total loss
    # tr_loss_aux += tr_loss
    # tr_kl_aux += tr_kl
    # tr_rec_aux += tr_eucl

    # Backprop and perform Adam optimisation
    optimizer.zero_grad()
    tr_loss.backward()
    optimizer.step()

    return tr_loss, tr_kl, tr_eucl

def test_net(model, x_test):
    model.eval()
    with torch.no_grad():
        input_test = x_test[:, :, :].cuda()
        # wt_test = wt_test[:].cuda()
        te_loss, te_kl, te_eucl = compute_loss(model, input_test)
    return te_loss, te_kl, te_eucl


def assign_matrices(N, Nr):
        Rr = torch.zeros(N, Nr)
        Rs = torch.zeros(N, Nr)
        receiver_sender_list = [i for i in itertools.product(range(N), range(N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
                Rr[r, i] = 1
                Rs[s, i] = 1
        Rr = Variable(Rr).cuda()
        Rs = Variable(Rs).cuda()
        return Rr, Rs

def tmul(x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])

##################  DEFINE MODEL  #####################

# processes a PxDo input with aggregation function (sum) + MLP to return a n_targets array
class INaggregator(nn.Module):
        def __init__(self, name, Do, n_targets, hidden):
                super(INaggregator, self).__init__()
                self.name = name
                self.Do = Do
                self.n_targets = n_targets
                self.hidden = hidden

                self.fc1 = nn.Linear(self.Do, self.hidden).cuda()
                self.fc2 = nn.Linear(self.hidden, self.hidden).cuda()
                self.fc3 = nn.Linear(self.hidden, self.n_targets).cuda()

        def forward(self, x):
                #sum over N
                O = torch.sum(x, dim=2)
                # apply MLP
                N = nn.functional.relu(self.fc1(O.view(-1, self.Do)))
                N = nn.functional.relu(self.fc2(N))
                N = self.fc3(N)
                return N 

class InLayer(nn.Module):
        def __init__(self, name, N, P, Do, De, hidden):
                super(InLayer, self).__init__()

                self.name = name
                self.N = N
                self.P = P
                self.De = De
                self.Do = Do
                self.hidden = hidden
                self.Nr = N*(N-1)
                self.Rr, self.Rs = assign_matrices(self.N, self.Nr)    

                # initialize the MLPs 
                self.fr1 = nn.Linear(2 * P, hidden).cuda()
                self.fr2 = nn.Linear(hidden, hidden).cuda()
                self.fr3 = nn.Linear(hidden, De).cuda()

                self.fo1 = nn.Linear(P + De, hidden).cuda()
                self.fo2 = nn.Linear(hidden, hidden).cuda()
                self.fo3 = nn.Linear(hidden, Do).cuda()

        # turn a PxN input into a PxDo output
        def forward(self, x):

                Orr = tmul(x, self.Rr)
                Ors = tmul(x, self.Rs)
                B = torch.cat([Orr, Ors], 1)

                ### First MLP ###
                B = torch.transpose(B, 1, 2).contiguous()
                B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P)))
                B = nn.functional.relu(self.fr2(B))
                E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
                del B
                E = torch.transpose(E, 1, 2).contiguous()
                Ebar = tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
                del E
                C = torch.cat([x, Ebar], 1)
                del Ebar
                C = torch.transpose(C, 1, 2).contiguous()
                C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.De)))
                C = nn.functional.relu(self.fo2(C))
                O = nn.functional.relu(self.fo3(C).view(-1, self.Do, self.N))
                del C
                return O
            
class InVAE(nn.Module):
        def __init__(self, N, P, Nz):
                super(InVAE, self).__init__()

                self.Nae = N
                self.Pae = P
                self.Nz = Nz

                # Encoder
                # (NxP -> NxDo_1)
                self.Do_1 = 10
                De_1 = 20
                self.enc_InLayer1 = InLayer("INenc1", self.Nae, self.Pae, self.Do_1, De_1, 64)
                # (NxDo_1 -> NxDo_2)   
                self.Do_2 = 20
                De_2 = 30
                self.enc_InLayer2 = InLayer("INenc2", self.Nae, self.Do_1, self.Do_2, De_2, 64)
                # aggregate De_2 quantities to Nz latent mu
                self.mean = INaggregator("mean", self.Do_2, self.Nz, 64)
                self.logvar = INaggregator("logvar", self.Do_2, self.Nz, 64)

                #Decoder
                dec_hidden = int(math.sqrt(self.Nae*self.Do_2*self.Nz))

                self.dec_dnn1 = nn.Linear(self.Nz, dec_hidden).cuda()
                self.dec_dnn2 = nn.Linear(dec_hidden, dec_hidden).cuda()
                self.dec_dnn3 = nn.Linear(dec_hidden, self.Nae*self.Do_2).cuda()

                # NxDo_2 -> NxDo_1
                self.dec_InLayer1 = InLayer("IN_dec1", self.Nae, self.Do_2, self.Do_1, De_2, 64)
                # NxDo_1  -> NxP
                self.dec_InLayer2 = InLayer("IN_dec2", self.Nae, self.Do_1, self.Pae, De_1, 64)


        def reparameterize(self, mean, logvar):
            z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
            return z
    
        def encode(self, x):
                Enc = self.enc_InLayer1(x)
                Enc = self.enc_InLayer2(Enc)
                mean  = self.mean(Enc)
                logvar = self.logvar(Enc)
                return mean, logvar

        def decode(self, z):
                Dec = nn.functional.relu(self.dec_dnn1(z))
                Dec = nn.functional.relu(self.dec_dnn2(Dec))
                Dec = nn.functional.relu(self.dec_dnn3(Dec))
                Dec = Dec.view(-1, self.Do_2, self.Nae)
                Dec = self.dec_InLayer1(Dec)
                Dec = self.dec_InLayer2(Dec)
                return Dec

        def forward(self, x):
                mean, logvar = self.encode(x)
                z = self.reparameterize(mean, logvar)
                return self.decode(z)


def compute_loss(model, x_pos):
    mean, logvar = model.encode(x_pos)
    z = model.reparameterize(mean, logvar)
    x_decoded_pos = model.decode(z)
    
    # Euclidean distance 
    pdist = nn.PairwiseDistance(p=2) 

    # Changes the dimension of the tensor so that dist is the distance between 
    # every pair of input and output pixels
    x_pos = x_pos.view(batch_size, 4, 1, 26) 
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

# Load data
f = h5py.File("/eos/project/d/dshep/DARKMACHINES/bsm_10fb/gluino_01_10fb.h5", "r")
particles = ['Bjets', 'MuPlus', 'MuMinus', 'ElePlus', 'EleMinus', 'Gamma']
d = np.array(f.get("Jets"), dtype='f')
for p in particles:
    d = np.concatenate((d, np.array(f.get(p), dtype='f')), axis=1) 
# transpose constituents index and feature index (to match what IN expects)
d = np.swapaxes(d, 1, 2)
met = np.array(f.get("EventFeatures"))
weight = met[:,1]
evtId =  met[:,0]
met = np.array(met[:,-2:], dtype='f')

# suffle data
d, met, weight, evtId = shuffle(d, met, weight, evtId, random_state=0)
# standardize particle inputs
scaler_p = StandardScaler()
d_shape = d.shape

d = np.reshape(d, (d_shape[0], d_shape[2]*d_shape[1]))
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

model = InVAE(d.shape[2], d.shape[1],10)
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

i_train = int(d.shape[0]*training_fraction)
# training data
x_train = d[:i_train,:,:]
met_train = met[:i_train,:]
weight_train = weight[:i_train]
evtId_train = evtId[:i_train]
# test data
x_test = d[i_train:,:,:]
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

        # input_train = x_train[:, :, :].cuda()
        # # Train
        # output_train = model(input_train)
        # tr_loss, tr_kl, tr_eucl = compute_loss(model, input_train)
        tr_loss, tr_kl, tr_eucl = train_net(model, x_train, optimizer)
        # add this batch loss to total loss
        tr_loss_aux += tr_loss
        tr_kl_aux += tr_kl
        tr_rec_aux += tr_eucl
        # Backprop and perform Adam optimisation
        # optimizer.zero_grad()
        # tr_loss.backward()
        # optimizer.step()

    # validation
    te_loss_aux = 0.0
    te_kl_aux = 0.0
    te_rec_aux = 0.0
    for y, (x_test) in enumerate(test_loader):
        if y == (len(test_loader) - 1): break
        # input_test = x_test[:, :, :].cuda()
        # te_loss, te_kl, te_eucl = compute_loss(model, input_test)
        te_loss, te_kl, te_eucl = test_net(model, x_test)
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

