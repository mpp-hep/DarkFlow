# ##################################################################################
# # VAE on DarkMachine dataset with 3D Sparse Loss                                 # 
# # Author: B. Orzani (Universidade Estadual Paulista, Brazil), M. Pierini (CERN)  #
# ##################################################################################

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# import torchvision.datasets
# from sklearn.utils import shuffle
# from sklearn.preprocessing import StandardScaler
# from pickle import dump
# import numpy as np
# import h5py
# from tqdm import tqdm

# from darkflow.utils.data_utils import save_npy, read_npy
# from darkflow.utils.network_utils import compute_loss, train_net, test_net
# import darkflow.networks.VAE_NF_Conv2D as VAE


# class ConvNetRunner:
#     def __init__(self, args):

#         # Hyperparameters
#         self.data_save_path = args.data_save_path
#         self.model_save_path = args.model_save_path
#         self.Data_filename = args.Data_filename
#         self.Data_bsm_filename = args.Data_bsm_filename
#         self.Met_filename = args.Met_filename
#         self.Met_bsm_filename = args.Met_bsm_filename
#         self.model_name = args.model_name
#         self.num_epochs = args.num_epochs
#         self.num_classes = args.num_classes
#         self.training_fraction = args.training_fraction
#         self.batch_size = args.batch_size
#         self.test_batch_size = args.test_batch_size
#         self.learning_rate = args.learning_rate
#         self.latent_dim = args.latent_dim
#         self.beta = args.beta
#         self.test_model_path = args.test_model_path
#         self.test_data_save_path = args.test_data_save_path

#         self.network = args.network
#         self.flow = args.flow 

#         if self.flow == 'none':
#             self.model = VAE.ConvNet(args)
#         if self.flow == 'planar':
#             self.model = VAE.PlanarVAE(args)
#         if self.flow == 'orthosnf':
#             self.model = VAE.OrthogonalSylvesterVAE(args)
#         if self.flow == 'householdersnf':
#             self.model = VAE.HouseholderSylvesterVAE(args)
#         if self.flow == 'triangularsnf':
#             self.model = VAE.TriangularSylvesterVAE(args)
#         if self.flow == 'iaf':
#             self.model = VAE.IAFVAE(args)
#         # if self.flow == 'convflow':
#         #     self.model = VAE.ConvFlow(args)
#         else:
#             raise ValueError('Invalid flow choice')
        
#         self.model = self.model.cuda()
#         self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#         self.preprocess_data()

#     def preprocess_data(self):
#         #Read data
#         d = read_npy(Data_filename)
#         d_bsm = read_npy(Data_bsm_filename)
#         met = read_npy(Met_filename)
#         met_bsm = read_npy(Met_bsm_filename)

#         print('Starting to process data ...')
#         weight = met[:,1]
#         evtId =  met[:,0]
#         weight_bsm = met_bsm[:,1]
#         evtId_bsm =  met_bsm[:,0]
#         met = np.array(met[:,-2:], dtype='f')

#         # suffle data
#         d, met, weight, evtId = shuffle(d, met, weight, evtId, random_state=0)
#         d_bsm, weight_bsm, evtId_bsm = shuffle(d_bsm, weight_bsm, evtId_bsm, random_state=0)

#         # Taking samples where pT>20GeV
#         idx = []
#         for i in range(d.shape[0]):
#             if((d[i,:,3,0]*d[i,:,3,0]+d[i,:,3,1]*d[i,:,3,1])>400):
#                 idx.append(i)
#         d = d[idx,:,:,:]
#         met = met[idx,:]
#         weight = weight[idx]
#         evtId = evtId[idx]

#         # standardize particle inputs
#         scaler_p = StandardScaler()
#         d_shape = d.shape
#         d = np.reshape(d, (d_shape[0], d_shape[2]*d_shape[3]))
#         scaler_p.fit(d)
#         d = scaler_p.transform(d)
#         d = np.reshape(d, d_shape)

#         scaler_p = StandardScaler()
#         d_bsm_shape = d_bsm.shape
#         d_bsm = np.reshape(d_bsm, (d_bsm_shape[0], d_bsm_shape[2]*d_bsm_shape[3]))
#         scaler_p.fit(d_bsm)
#         d_bsm = scaler_p.transform(d_bsm)
#         d_bsm = np.reshape(d_bsm, d_bsm_shape)

#         #Build test set
#         self.d_test = d[:2013,:,:,:]
#         weight_test = weight[:2013]
#         self.x_test = np.append(d_test, d_bsm, axis=0)
#         self.weight_test = np.append(weight_test, weight_bsm, axis=0)

#         self.d = d[2014:,:,:,:]
#         self.weight = weight[2014:]

#         # # standardize met inputs
#         # scaler_met = StandardScaler()
#         # scaler_met.fit(met)
#         # met = scaler_met.transform(met)
#         # save the scalers
#         # dump(scaler_p, open(data_save_path + 'darkflow/models/run4/%s_particleScaler.pkl' %model_name, 'wb'))
#         # dump(scaler_met, open(data_save_path + 'darkflow/models/run4/%s_metScaler.pkl' %model_name, 'wb'))

#         # build train and val sets
#         i_train = int(self.d.shape[0]*self.training_fraction)
#         # training data
#         self.x_train = self.d[:i_train,:,:,:]
#         # met_train = met[:i_train,:]
#         self.weight_train = self.weight[:i_train]
#         # evtId_train = evtId[:i_train]
        
#         # Val data
#         self.x_val = d[i_train:,:,:,:]
#         # met_val = met[i_train:,:]
#         self.weight_val = weight[i_train:]
#         # evtId_val = evtId[i_train:]
#         print('Done; x_train shape: ', self.x_train.shape, 'x_val shape: ', self.x_val.shape, 'x_test shape: ', self.x_test.shape)
#         # for now, only train with particles. 
#         # met to be concatenated to the first dense layer in the encoder

#     def trainer(self):
#         self.train_loader = DataLoader(dataset = self.x_train, batch_size = self.batch_size, shuffle=False)
#         self.val_loader = DataLoader(dataset = self.x_val, batch_size = self.batch_size, shuffle=False)
#         # self.test_loader = DataLoader(dataset = self.x_test, batch_size = self.test_batchsize, shuffle=False)
#         self.weight_train_loader = DataLoader(dataset = self.weight_train, batch_size = self.batch_size, shuffle=False)
#         self.weight_val_loader = DataLoader(dataset = self.weight_val, batch_size = self.batch_size, shuffle=False)
#         # self.weight_test_loader = DataLoader(dataset = self.weight_test, batch_size = self.test_batchsize, shuffle=False)

#         # to store training history
#         self.x_graph = []
#         self.train_y_rec = []
#         self.train_y_kl = []
#         self.train_y_loss = []
#         self.val_y_rec = []
#         self.val_y_kl = []
#         self.val_y_loss = []

#         # print('Model Parameter: ', self.model)

#         print('Initiating training, validation processes ...')
#         for epoch in range(self.num_epochs):
#             self.x_graph.append(epoch)
#             print('Starting to train ...')

#             # training
#             tr_loss_aux = 0.0
#             tr_kl_aux = 0.0
#             tr_rec_aux = 0.0
#             for y, (x_train, wt_train) in tqdm(enumerate(zip(self.train_loader, self.weight_train_loader))):
#                 if y == (len(self.train_loader) - 1): break

#                 tr_loss, tr_kl, tr_eucl, self.model = train_net(self.model, x_train, wt_train, self.optimizer, batch_size=self.batch_size)
                
#                 tr_loss_aux += tr_loss
#                 tr_kl_aux += tr_kl
#                 tr_rec_aux += tr_eucl

#             print('Moving to validation stage ...')
#             # validation
#             val_loss_aux = 0.0
#             val_kl_aux = 0.0
#             val_rec_aux = 0.0

#             for y, (x_val, wt_val) in tqdm(enumerate(zip(self.val_loader, self.weight_val_loader))):
#                 if y == (len(self.val_loader) - 1): break
                
#                 #Test
#                 val_loss, val_kl, val_eucl = test_net(self.model, x_val, wt_val, batch_size=self.batch_size)

#                 val_loss_aux += val_loss
#                 val_kl_aux += val_kl
#                 val_rec_aux += val_eucl

#             self.train_y_loss.append(tr_loss_aux.cpu().detach().numpy()/(len(train_loader)))
#             self.train_y_kl.append(tr_kl_aux.cpu().detach().numpy()/(len(train_loader)))
#             self.train_y_rec.append(tr_rec_aux.cpu().detach().numpy()/(len(train_loader)))
                
#             self.val_y_loss.append(val_loss_aux/(len(val_loader)))
#             self.val_y_kl.append(val_kl_aux/(len(val_loader)))
#             self.val_y_rec.append(val_rec_aux/(len(val_loader)))
                
#             print('Epoch: {} -- Train loss: {}  -- Val loss: {}'.format(epoch, 
#                                                                          tr_loss_aux/(len(self.train_loader)), 
#                                                                          val_loss_aux/(len(self.val_loader))))
#             if (epoch == 0):
#                 self.best_val_loss = val_loss_aux/(len(self.val_loader))
#                 self.best_model = self.model
#             if (val_loss_aux/(len(self.val_loader))<self.best_val_loss):
#                 self.best_model = self.model
#                 self.best_val_loss = val_loss_aux/(len(self.val_loader))
#                 print('Best Model Yet')


#         # Save the model
#         save_run_history(self.best_model, self.model, self.model_save_path, self.model_name, 
#                             self.x_graph, self.train_y_rec, self.train_y_kl, self.train_y_loss, hist_name='TrainHistory')
#         # save_run_history(self.best_model, self.model, self.model_save_path, self.model_name, 
#                             # self.x_graph, self.val_y_rec, self.val_y_kl, self.val_y_loss, hist_name='ValHistory')

#         print('Network Run Complete')

#     def tester(self):
        
#         # load model
#         self.model.load_state_dict(torch.load(self.test_model_path, map_location=torch.device('cpu')))

#         # load data
#         self.test_loader = DataLoader(dataset=self.x_test, batch_size=self.test_batch_size, shuffle=False)
#         self.weight_test_loader = DataLoader(dataset=self.weight_test, batch_size=self.test_batch_size, shuffle=False)

#         print('Starting the Testing Process ...')
#         self.test_ev_rec = []
#         self.test_ev_kl = []
#         self.test_ev_loss = []
#         for y, (x_test, wt_test) in tqdm(enumerate(zip(self.test_loader, self.weight_test_loader))):
#             if y == (len(self.test_loader) - 1): break
            
#             #Test
#             te_loss, te_kl, te_eucl = test_net(self.model, x_test, wt_test, batch_size=self.test_batch_size)
            
#             self.test_ev_loss.append(te_loss.cpu().detach().numpy())
#             self.test_ev_kl.append(te_kl.cpu().detach().numpy())
#             self.test_ev_rec.append(te_eucl.cpu().detach().numpy())
#         # print('loss: ', test_ev_loss)
#         save_npy(np.array(self.test_ev_loss), self.test_data_save_path + 'IAF_loss_2EP_BEST.npy')
#         save_npy(np.array(self.test_ev_kl), self.test_data_save_path + 'IAF_kl_2EP_BEST.npy')
#         save_npy(np.array(self.test_ev_rec), self.test_data_save_path + 'IAF_rec_2EP_BEST.npy')

#         print('Testing Complete')

#     # def infer(self):






        

