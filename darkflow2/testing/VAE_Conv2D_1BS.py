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


batch_size = 1
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
            # print(x.size())
            out = self.conv1(x)
            out = self.bn1(out)
            out = torch.relu(out)
            # print(out.size())
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