import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from pickle import dump
import numpy as np
import h5py


##################  DEFINE MODEL  #####################
class ConvNet(nn.Module):
        def __init__(self, args):
            super(ConvNet, self).__init__()

            self.latent_dim = args.latent_dim
            self.test_mode = args.test_net

            if args.train_net:
                self.batch_size = args.batch_size
            if args.test_net:
                self.batch_size = args.test_batch_size
                
            self.q_z_output_dim = args.q_z_output_dim
            self.z_size = self.latent_dim
    
            self.q_z_nn = nn.Sequential(
                  nn.Conv2d(1, 32, kernel_size=(3,4), stride=(1), padding=(0)),
                  nn.BatchNorm2d(32),
                  nn.ReLU(),
                  nn.Conv2d(32, 16, kernel_size=(5,1), stride=(1), padding=(0)),
                  nn.BatchNorm2d(16),
                  nn.ReLU(),
                  nn.Conv2d(16, 8, kernel_size=(7,1), stride=(1), padding=(0)),
                  nn.BatchNorm2d(8),
                  nn.ReLU()
                  ) 
            #
            self.dense1 = nn.Linear(161,self.q_z_output_dim) #161 - full+mult, 41-4LJ
            self.dnn_bn1 = nn.BatchNorm1d(self.q_z_output_dim)
            self.q_z_mean = nn.Linear(self.q_z_output_dim, self.latent_dim)
            self.q_z_logvar = nn.Linear(self.q_z_output_dim, self.latent_dim)
            #
            self.dense3 = nn.Linear(self.latent_dim, self.q_z_output_dim)
            self.dnn_bn3 = nn.BatchNorm1d(self.q_z_output_dim)
            self.dense4 = nn.Linear(self.q_z_output_dim, 161)
            self.dnn_bn4 = nn.BatchNorm1d(161)
            #
            self.p_x_nn = nn.Sequential(
                  nn.ConvTranspose2d(8, 16, kernel_size=(7,1), stride=(1), padding=(0)),
                  nn.BatchNorm2d(16),
                  nn.ReLU(),
                  nn.ConvTranspose2d(16, 32, kernel_size=(5,1), stride=(1), padding=(0)),
                  nn.BatchNorm2d(32),
                  nn.ReLU(),
                  nn.ConvTranspose2d(32, 1, kernel_size=(3,4), stride=(1), padding=(0))
                  ) 
            # log-det-jacobian = 0 without flows
            self.ldj = 0
    
        def encode(self, x, y):
          
            out = self.q_z_nn(x)
            # flatten
            out = out.view(out.size(0), -1)
            # concat met
            out = torch.cat((out, y),axis=1)
            # dense Layer 1
            out = self.dense1(out)
            out = self.dnn_bn1(out)
            out = torch.relu(out)
            # dense Layer 2
            mean  = self.q_z_mean(out)
            logvar = self.q_z_logvar(out)
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
            # remove met
            out = out[:,:-1]
            # reshape
            out = out.view(out.size(0), 8, 20, 1) #20-full, 5-4LJ
            # DeConv
            out = self.p_x_nn(out)
            
            return out
    
        def reparameterize(self, mean, logvar):
            z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
            return z
    
        def forward(self, x, y):
            self.met = y
            mean, logvar = self.encode(x, self.met)
            z = self.reparameterize(mean, logvar)
            out = self.decode(z)
            return out, mean, logvar, self.ldj, z, z

