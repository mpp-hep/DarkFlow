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
import darkflow.testing.VAE_NF_Conv2D_1BS as VAE 
from darkflow.utils.network_utils import compute_loss


# Test/Validate
def test_net(model, x_test, wt_test, batch_size):
    with torch.no_grad():
        input_test = x_test[:, :, :, :].cuda()
        wt_test = wt_test[:].cuda()
        x_decoded, z_mu, z_var, log_det_j, z0, zk = model(input_test)

        te_loss, te_kl, te_eucl = compute_loss(input_test, wt_test, x_decoded, z_mu, z_var, batch_size=batch_size)

    return te_loss, te_kl, te_eucl

# Hyperparameters
data_save_path = "/home/pjawahar/Projects/DarkFlow/"
Data_filename = data_save_path + 'Data/d_sm_10fb.npy'
Data_bsm_filename = data_save_path + 'Data/d_bsm_10fb_test.npy'
Met_filename = data_save_path + 'Data/met_sm_10fb.npy'
Met_bsm_filename = data_save_path + 'Data/met_bsm_10fb_test.npy'
model_name = "VAE_Conv2D_IAF_run_2EP"
num_classes = 1
batch_size = 1
learning_rate = 0.001
latent_dim = 10
beta = 1.

# model = VAE.PlanarVAE()
model = VAE.IAFVAE()
model = model.cuda()
model.load_state_dict(torch.load(data_save_path + 'darkflow/models/run4/IAF/BEST_VAE_IAF_Conv2D_runBestModel_2EP.pt', map_location=torch.device('cpu')))
model.eval()

#Read data
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

# # standardize met inputs
# scaler_met = StandardScaler()
# scaler_met.fit(met)
# met = scaler_met.transform(met)

d_test = d[:2013,:,:,:]
weight = weight[:2013]
x_test = np.append(d_test, d_bsm, axis=0)
weight_test = np.append(weight, weight_bsm, axis=0)
print('Done; x_test shape: ', x_test.shape, 'wt_test shape: ', weight_test.shape, 'd shape: ', d_test.shape, 'd_bsm shape: ', d_bsm.shape)

test_loader = DataLoader(dataset=x_test, batch_size=batch_size, shuffle=False)
weight_test_loader = DataLoader(dataset=weight_test, batch_size=batch_size, shuffle=False)

print('Starting the Testing Process ...')
test_ev_rec = []
test_ev_kl = []
test_ev_loss = []
for y, (x_test, wt_test) in tqdm(enumerate(zip(test_loader, weight_test_loader))):
    if y == (len(test_loader) - 1): break
    
    #Test
    # print('\rbatch: ',y, end='')
    te_loss, te_kl, te_eucl = test_net(model, x_test, wt_test, batch_size=batch_size)

    # print('loss: ', te_loss, 'kl: ', te_kl, )
    
    test_ev_loss.append(te_loss.cpu().detach().numpy())
    test_ev_kl.append(te_kl.cpu().detach().numpy())
    test_ev_rec.append(te_eucl.cpu().detach().numpy())
# print('loss: ', test_ev_loss)
save_npy(np.array(test_ev_loss), data_save_path + 'Data/TestResults/IAF_loss_2EP_BEST.npy')
save_npy(np.array(test_ev_kl), data_save_path + 'Data/TestResults/IAF_kl_2EP_BEST.npy')
save_npy(np.array(test_ev_rec), data_save_path + 'Data/TestResults/IAF_rec_2EP_BEST.npy')

print('Testing Complete')





