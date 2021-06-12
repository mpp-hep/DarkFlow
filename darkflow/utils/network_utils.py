import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np


#Sparse loss function    
def compute_loss(x, weight, x_decoded, mean, logvar, batch_size=1, beta=1):
    # print('computing loss ...')
    # mean, logvar = model.encode(x)
    # z = model.reparameterize(mean, logvar)
    # x_decoded = model.decode(z)

    
    # Euclidean distance 
    pdist = nn.PairwiseDistance(p=2) 
    x_pos = torch.zeros(batch_size,3,25)#.cuda()
    # Removes the channel dimension to make the following calculations easier
    x_pos = x[:,0,:,:] 
    # Changes the dimension of the tensor so that dist is the distance between every 
    # pair of input and output pixels
    x_pos = x_pos.view(batch_size, 4, 1, 38) #32 full(13+3), 17-4LJ
    
    x_decoded_pos = torch.zeros(batch_size,4,38)#.cuda()
    # Removes the channel dimension to make the following calculations easier
    x_decoded_pos = x_decoded[:,0,:,:] 
    
    # Changes the dimension of the tensor so that dist is the distance between 
    # every pair of input and output pixels
    x_decoded_pos = x_decoded_pos.view(batch_size, 4, 38, 1) 
    x_decoded_pos = torch.repeat_interleave(x_decoded_pos, 38, -1) 
    
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

#Sparse loss function    
def compute_gcn_loss(x, weight, x_decoded, mean, logvar, batch_size=1, beta=1):
    
    # Euclidean distance 
    pdist = nn.PairwiseDistance(p=2) 
    x_pos = torch.zeros(batch_size,5,31)#.cuda()
    # Removes the channel dimension to make the following calculations easier
    x_pos = x
    # Changes the dimension of the tensor so that dist is the distance between every 
    # pair of input and output pixels
    x_pos = x_pos.view(batch_size, 5, 1, 31) #32 full(13+3), 17-4LJ
    
    x_decoded_pos = torch.zeros(batch_size,5,31)#.cuda()
    # Removes the channel dimension to make the following calculations easier
    x_decoded_pos = x_decoded
    
    # Changes the dimension of the tensor so that dist is the distance between 
    # every pair of input and output pixels
    x_decoded_pos = x_decoded_pos.view(batch_size, 5, 31, 1) 
    x_decoded_pos = torch.repeat_interleave(x_decoded_pos, 31, -1) 
    
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

# Train
def train_net(model, x_train, adj_train, wt_train, optimizer, batch_size):
    input_train = x_train.cuda()
    adj_train = adj_train.cuda()
    wt_train = wt_train[:].cuda()
    model.train()   

    x_decoded, z_mu, z_var, log_det_j, z0, zk = model(input_train, adj_train)

    tr_loss, tr_kl, tr_eucl = compute_gcn_loss(input_train, wt_train, x_decoded, z_mu, z_var, batch_size=batch_size)
    
    # Backprop and perform Adam optimisation
    optimizer.zero_grad()
    tr_loss.backward()
    optimizer.step()

    return tr_loss, tr_kl, tr_eucl, model, input_train, x_decoded

# Test/Validate
def test_net(model, x_test, adj_test, wt_test, batch_size):
    model.eval()
    with torch.no_grad():
        input_test = x_test.cuda()
        adj_test = adj_test.cuda()
        wt_test = wt_test[:].cuda()

        x_decoded, z_mu, z_var, log_det_j, z0, zk = model(input_test, adj_test)
        
        te_loss, te_kl, te_eucl = compute_gcn_loss(input_test, wt_test, x_decoded, z_mu, z_var, batch_size=batch_size)

    return te_loss, te_kl, te_eucl
