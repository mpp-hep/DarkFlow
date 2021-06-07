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
import scipy.sparse as sp
from pickle import dump
import numpy as np
import h5py
from tqdm import tqdm

from darkflow.utils.data_utils import save_npy, save_csv, read_npy, save_run_history, build_graph
from darkflow.utils.network_utils import compute_loss, train_net, test_net
import darkflow.networks.VAE_NF_GCN as VAE


class GCNNetRunner:
    def __init__(self, args):

        # Hyperparameters
        self.data_save_path = args.data_save_path
        self.model_save_path = args.model_save_path
        self.Data_filename = args.Data_filename
        self.Data_bsm_filename = args.Data_bsm_filename
        self.Met_filename = args.Met_filename
        self.Met_bsm_filename = args.Met_bsm_filename
        self.model_name = args.model_name
        self.num_epochs = args.num_epochs
        self.num_classes = args.num_classes
        self.training_fraction = args.training_fraction
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.learning_rate = args.learning_rate
        self.latent_dim = args.latent_dim
        self.beta = args.beta
        # self.test_model_path = args.test_model_path
        self.test_data_save_path = args.test_data_save_path

        self.network = args.network
        self.flow = args.flow 
        # print(args.flow, self.flow)
        self.channel = args.channel
        if self.channel == 'chan1':
            self.num_test_ev_sm = 10000
        elif self.channel == 'chan2a':
            self.num_test_ev_sm = 5868
        elif self.channel == 'chan2b':
            self.num_test_ev_sm = 89000
        elif self.channel == 'chan3':
            self.num_test_ev_sm = 1025333

        if self.flow == 'noflow':
            self.model = VAE.GCNNet(args)
            self.flow_ID = 'NoF'
        elif self.flow == 'planar':
            self.model = VAE.PlanarVAE(args)
            self.flow_ID = 'Planar'
        elif self.flow == 'orthosnf':
            self.model = VAE.OrthogonalSylvesterVAE(args)
            self.flow_ID = 'Ortho'
        elif self.flow == 'householdersnf':
            self.model = VAE.HouseholderSylvesterVAE(args)
            self.flow_ID = 'House'
        elif self.flow == 'triangularsnf':
            self.model = VAE.TriangularSylvesterVAE(args)
            self.flow_ID = 'Tri'
        elif self.flow == 'iaf':
            self.model = VAE.IAFVAE(args)
            self.flow_ID = 'IAF'
        elif self.flow == 'convflow':
            self.model = VAE.ConvFlowVAE(args)
            self.flow_ID = 'ConvF'
        else:
            raise ValueError('Invalid flow choice')
        
        self.model_name = self.model_name%self.flow_ID
        self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.preprocess_data()

    def preprocess_data(self):
        #Read data
        d = read_npy(self.Data_filename)
        d_bsm = read_npy(self.Data_bsm_filename)
        met = read_npy(self.Met_filename)
        met_bsm = read_npy(self.Met_bsm_filename)

        print('Starting to process data ...')
        weight = met[:,1]
        Met =  met[:,0]
        weight_bsm = np.ones(d_bsm.shape[0])#met_bsm[:,1]
        Met_bsm =  met_bsm[:,0]

        # suffle data
        d, weight, Met = shuffle(d, weight, Met, random_state=0)
        d_bsm, weight_bsm, Met_bsm = shuffle(d_bsm, weight_bsm, Met_bsm, random_state=0)

        # standardize particle inputs
        scaler_p = StandardScaler()
        d_shape = d.shape
        d = np.reshape(d, (d_shape[0], d_shape[2]*d_shape[3]))
        scaler_p.fit(d)
        d = scaler_p.transform(d)
        d = np.reshape(d, d_shape)

        scaler_b = StandardScaler()
        d_bsm_shape = d_bsm.shape
        d_bsm = np.reshape(d_bsm, (d_bsm_shape[0], d_bsm_shape[2]*d_bsm_shape[3]))
        scaler_b.fit(d_bsm)
        d_bsm = scaler_b.transform(d_bsm)
        d_bsm = np.reshape(d_bsm, d_bsm_shape)

        # # standardize met inputs
        # Met = np.reshape(Met, (Met.shape[0],1))
        # scaler_met = StandardScaler()
        # scaler_met.fit(Met)
        # Met = scaler_met.transform(Met)

        # Met_bsm = np.reshape(Met_bsm, (Met_bsm.shape[0],1))
        # scaler_mb = StandardScaler()
        # scaler_mb.fit(Met_bsm)
        # Met_bsm = scaler_mb.transform(Met_bsm)

        # # manage Met shapes to concatenate with d
        # met_pad = np.full((Met.shape[0],3), 0, dtype=float) 
        # met_bsm_pad = np.full((Met_bsm.shape[0],3), 0, dtype=float)
        # paddedMet = np.concatenate((Met,met_pad), axis=1)
        # paddedMet_bsm = np.concatenate((Met_bsm,met_bsm_pad), axis=1)

        # # concatenate d and Met
        # paddedMet = np.reshape(paddedMet, (paddedMet.shape[0],1,1,paddedMet.shape[1]))
        # paddedMet_bsm = np.reshape(paddedMet_bsm, (paddedMet_bsm.shape[0],1,1,paddedMet_bsm.shape[1]))
        # d = np.concatenate((d,paddedMet), axis=2)
        # d_bsm = np.concatenate((d_bsm,paddedMet_bsm), axis=2)

        # Set aside bkg samples to form test set
        # num_test_ev_sm = 1025333     #1025333 for chan3 | 10000 for chan1 | 89000 for chan2b | 5868 for chan2a
        self.d_test = d[:num_test_ev_sm,:,:,:]
        self.Met_sm = Met[:num_test_ev_sm,:] 
        self.weight_sm = weight[:num_test_ev_sm]

        # Build test set
        self.x_test = np.append(self.d_test, d_bsm, axis=0)
        self.met_test = np.append(self.Met_sm, Met_bsm, axis=0)
        self.weight_test = np.append(self.weight_sm, weight_bsm, axis=0)
        
        # Remaining data for train and val
        self.d = d[(num_test_ev_sm+1):,:,:,:]
        self.Met_d = Met[(num_test_ev_sm+1):,:] 
        self.weight = weight[(num_test_ev_sm+1):]

        # save the scalers
        # dump(scaler_p, open(data_save_path + 'darkflow/models/run4/%s_particleScaler.pkl' %model_name, 'wb'))
        # dump(scaler_met, open(data_save_path + 'darkflow/models/run4/%s_metScaler.pkl' %model_name, 'wb'))

        # build train and val sets
        i_train = int(self.d.shape[0]*self.training_fraction)
        # training data
        self.x_train = self.d[:i_train,:,:,:]
        self.met_train = self.Met_d[:i_train,:] 
        self.weight_train = self.weight[:i_train]
        
        # Val data
        self.x_val = self.d[i_train:,:,:,:]
        self.met_val = self.Met_d[i_train:,:]
        self.weight_val = self.weight[i_train:]
        
        print('Done; x_train shape: ', self.x_train.shape, 'x_val shape: ', self.x_val.shape, 'x_test shape: ', self.x_test.shape, 'met_train shape: ', self.met_train.shape, 'met_val shape: ', self.met_val.shape)

        # build the graphs
        self.features_train, self.adj_train = build_graph(x_train)
        self.features_val, self.adj_val = build_graph(x_val)
        self.features_test, self.adj_test = build_graph(x_test)

    # def preprocess_data_withMult(self):
    #     #Read data
    #     d = read_npy(self.Data_filename)
    #     d_bsm = read_npy(self.Data_bsm_filename)
    #     met = read_npy(self.Met_filename)
    #     met_bsm = read_npy(self.Met_bsm_filename)

    #     print('Starting to process data ...')
    #     weight = met[:,-1]
    #     Met =  met[:,0]
    #     mult = met[:,1:-1] # event object pultiplicities
    #     weight_bsm = np.ones(d_bsm.shape[0])#met_bsm[:,1]
    #     Met_bsm =  met_bsm[:,0]
    #     mult_bsm = met_bsm[:,1:-1]

    #     # suffle data
    #     d, weight, Met, mult = shuffle(d, weight, Met, mult, random_state=0)
    #     d_bsm, weight_bsm, Met_bsm, mult_bsm = shuffle(d_bsm, weight_bsm, Met_bsm, mult_bsm, random_state=0)

    #     # standardize particle inputs
    #     scaler_p = StandardScaler()
    #     d_shape = d.shape
    #     d = np.reshape(d, (d_shape[0], d_shape[2]*d_shape[3]))
    #     scaler_p.fit(d)
    #     d = scaler_p.transform(d)
    #     d = np.reshape(d, d_shape)

    #     scaler_b = StandardScaler()
    #     d_bsm_shape = d_bsm.shape
    #     d_bsm = np.reshape(d_bsm, (d_bsm_shape[0], d_bsm_shape[2]*d_bsm_shape[3]))
    #     scaler_b.fit(d_bsm)
    #     d_bsm = scaler_b.transform(d_bsm)
    #     d_bsm = np.reshape(d_bsm, d_bsm_shape)

    #     # standardize met inputs
    #     Met = np.reshape(Met, (Met.shape[0],1))
    #     scaler_met = StandardScaler()
    #     scaler_met.fit(Met)
    #     Met = scaler_met.transform(Met)

    #     Met_bsm = np.reshape(Met_bsm, (Met_bsm.shape[0],1))
    #     scaler_mb = StandardScaler()
    #     scaler_mb.fit(Met_bsm)
    #     Met_bsm = scaler_mb.transform(Met_bsm)

    #     # manage Met shapes to concatenate with d
    #     met_pad = np.full((Met.shape[0],3), 0, dtype=float) 
    #     met_bsm_pad = np.full((Met_bsm.shape[0],3), 0, dtype=float)
    #     paddedMet = np.concatenate((Met,met_pad), axis=1)
    #     paddedMet_bsm = np.concatenate((Met_bsm,met_bsm_pad), axis=1)

    #     # concatenate d and Met
    #     paddedMet = np.reshape(paddedMet, (paddedMet.shape[0],1,1,paddedMet.shape[1]))
    #     paddedMet_bsm = np.reshape(paddedMet_bsm, (paddedMet_bsm.shape[0],1,1,paddedMet_bsm.shape[1]))
    #     d = np.concatenate((d,paddedMet), axis=2)
    #     d_bsm = np.concatenate((d_bsm,paddedMet_bsm), axis=2)

    #     # concatenate multiplicities back to Met
    #     Met = np.concatenate((Met, mult), axis=1)
    #     Met_bsm = np.concatenate((Met_bsm, mult_bsm), axis=1)

    #     # Set aside bkg samples to form test set
    #     num_test_ev_sm = self.num_test_ev_sm     
    #     self.d_test = d[:num_test_ev_sm,:,:,:]
    #     self.Met_sm = Met[:num_test_ev_sm,:] 
    #     self.weight_sm = weight[:num_test_ev_sm]

    #     # Build test set
    #     self.x_test = np.append(self.d_test, d_bsm, axis=0)
    #     self.met_test = np.append(self.Met_sm, Met_bsm, axis=0)
    #     self.weight_test = np.append(self.weight_sm, weight_bsm, axis=0)
        
    #     # Remaining data for train and val
    #     self.d = d[(num_test_ev_sm+1):,:,:,:]
    #     self.Met_d = Met[(num_test_ev_sm+1):,:] 
    #     self.weight = weight[(num_test_ev_sm+1):]

    #     # save the scalers
    #     # dump(scaler_p, open(data_save_path + 'darkflow/models/run4/%s_particleScaler.pkl' %model_name, 'wb'))
    #     # dump(scaler_met, open(data_save_path + 'darkflow/models/run4/%s_metScaler.pkl' %model_name, 'wb'))

    #     # build train and val sets
    #     i_train = int(self.d.shape[0]*self.training_fraction)
    #     # training data
    #     self.x_train = self.d[:i_train,:,:,:]
    #     self.met_train = self.Met_d[:i_train,:] 
    #     self.weight_train = self.weight[:i_train]
        
    #     # Val data
    #     self.x_val = self.d[i_train:,:,:,:]
    #     self.met_val = self.Met_d[i_train:,:]
    #     self.weight_val = self.weight[i_train:]
        
    #     print('Done; x_train shape: ', self.x_train.shape, 'x_val shape: ', self.x_val.shape, 'x_test shape: ', self.x_test.shape, 'met_train shape: ', self.met_train.shape, 'met_val shape: ', self.met_val.shape)
        

    def trainer(self):
        self.train_loader = DataLoader(dataset = self.features_train, batch_size = self.test_batch_size, shuffle=False, drop_last=True)
        self.adjTr_loader = DataLoader(dataset = self.adj_train, batch_size = self.test_batch_size, shuffle=False, drop_last=True)
        self.weight_train_loader = DataLoader(dataset = self.weight_train, batch_size = self.test_batch_size, shuffle=False, drop_last=True)

        self.val_loader = DataLoader(dataset = self.features_val, batch_size = self.test_batch_size, shuffle=False, drop_last=True)
        self.adjVa_loader = DataLoader(dataset = self.adj_val, batch_size = self.test_batch_size, shuffle=False, drop_last=True)
        self.weight_val_loader = DataLoader(dataset = self.weight_val, batch_size = self.test_batch_size, shuffle=False, drop_last=True)

        # to store training history
        self.x_graph = []
        self.train_y_rec = []
        self.train_y_kl = []
        self.train_y_loss = []
        self.val_y_rec = []
        self.val_y_kl = []
        self.val_y_loss = []

        # print('Model Parameter: ', self.model)
        print('Model Type: %s'%self.flow_ID)
        print('Initiating training, validation processes ...')
        for epoch in range(self.num_epochs):
            self.x_graph.append(epoch)
            print('Starting to train ...')

            # training
            tr_loss_aux = 0.0
            tr_kl_aux = 0.0
            tr_rec_aux = 0.0
            for y, (x_train, adj_tr, wt_train) in tqdm(enumerate(zip(self.train_loader, self.adjTr_loader, self.weight_train_loader))):
                if y == (len(self.train_loader)): break

                tr_loss, tr_kl, tr_eucl, self.model = train_net(self.model, x_train, adj_tr, wt_train, self.optimizer, batch_size=self.test_batch_size)
                
                tr_loss_aux += tr_loss
                tr_kl_aux += tr_kl
                tr_rec_aux += tr_eucl

            print('Moving to validation stage ...')
            # validation
            val_loss_aux = 0.0
            val_kl_aux = 0.0
            val_rec_aux = 0.0

            for y, (x_val, adj_va, wt_val) in tqdm(enumerate(zip(self.val_loader, self.adjVa_loader, self.weight_val_loader))):
                if y == (len(self.val_loader)): break
                
                #Test
                val_loss, val_kl, val_eucl = test_net(self.model, x_val, adj_va, wt_val, batch_size=self.test_batch_size)

                val_loss_aux += val_loss
                val_kl_aux += val_kl
                val_rec_aux += val_eucl

            self.train_y_loss.append(tr_loss_aux.cpu().detach().numpy()/(len(self.train_loader)))
            self.train_y_kl.append(tr_kl_aux.cpu().detach().numpy()/(len(self.train_loader)))
            self.train_y_rec.append(tr_rec_aux.cpu().detach().numpy()/(len(self.train_loader)))
                
            self.val_y_loss.append(val_loss_aux/(len(self.val_loader)))
            self.val_y_kl.append(val_kl_aux/(len(self.val_loader)))
            self.val_y_rec.append(val_rec_aux/(len(self.val_loader)))
                
            print('Epoch: {} -- Train loss: {}  -- Val loss: {}'.format(epoch, 
                                                                         tr_loss_aux/(len(self.train_loader)), 
                                                                         val_loss_aux/(len(self.val_loader))))
            if (epoch == 0):
                self.best_val_loss = val_loss_aux/(len(self.val_loader))
                self.best_model = self.model
            if (val_loss_aux/(len(self.val_loader))<self.best_val_loss):
                self.best_model = self.model
                self.best_val_loss = val_loss_aux/(len(self.val_loader))
                print('Best Model Yet')


        # Save the model
        save_run_history(self.best_model, self.model, self.model_save_path, self.model_name, 
                            self.x_graph, self.train_y_rec, self.train_y_kl, self.train_y_loss, hist_name='TrainHistory')
        # save_run_history(self.best_model, self.model, self.model_save_path, self.model_name, 
                            # self.x_graph, self.val_y_rec, self.val_y_kl, self.val_y_loss, hist_name='ValHistory')

        print('Network Run Complete')

    def tester(self):

        print('Model Type: %s'%self.flow_ID)
        
        # load model
        self.model.load_state_dict(torch.load(self.model_save_path + '%s.pt' %self.model_name, map_location=torch.device('cpu')))

        # load data
        self.test_loader = DataLoader(dataset=self.features_test, batch_size=self.test_batch_size, shuffle=False, drop_last=True)
        self.adjTe_loader = DataLoader(dataset=self.adj_test, batch_size=self.test_batch_size, shuffle=False, drop_last=True)
        self.weight_test_loader = DataLoader(dataset=self.weight_test, batch_size=self.test_batch_size, shuffle=False, drop_last=True)

        print('Starting the Testing Process ...')
        self.test_ev_rec = []
        self.test_ev_kl = []
        self.test_ev_loss = []
        for y, (x_test, adj_te, wt_test) in tqdm(enumerate(zip(self.test_loader, self.adjTe_loader, self.weight_test_loader))):
            if y == (len(self.test_loader)): break
            
            #Test
            te_loss, te_kl, te_eucl = test_net(self.model, x_test, adj_te, wt_test, batch_size=self.test_batch_size)
            
            self.test_ev_loss.append(te_loss.cpu().detach().numpy())
            self.test_ev_kl.append(te_kl.cpu().detach().numpy())
            self.test_ev_rec.append(te_eucl.cpu().detach().numpy())
        # print('loss: ', test_ev_loss)
        save_npy(np.array(self.test_ev_loss), self.test_data_save_path + '%s_loss.npy' %self.model_name)
        save_npy(np.array(self.test_ev_kl), self.test_data_save_path + '%s_kl.npy' %self.model_name)
        save_npy(np.array(self.test_ev_rec), self.test_data_save_path + '%s_rec.npy' %self.model_name)
        # save_csv(data= np.array(self.test_ev_kl), filename= self.test_data_save_path + 'rec_%s.csv' %self.model_name)
        # save_csv(data= np.array(self.test_ev_rec), filename= self.test_data_save_path + 'rec1_%s.csv' %self.model_name)

        print('Testing Complete')

    # def infer(self):






        

