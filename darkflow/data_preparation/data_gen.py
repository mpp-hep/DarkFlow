import numpy as np
import h5py

from darkflow.utils.data_utils import save_npy
from progress.spinner import Spinner

data_save_path = "/home/pjawahar/Projects/DarkFlow/"

# Load data file
print('Starting to read .h5py file ...')
state = 'N'
spinner = Spinner('Reading .h5py file: ')
while state != 'Fin':
    f = h5py.File("/home/pjawahar/Projects/DarkFlow/Data/sm_10fb.h5", "r")
    spinner.next()
    state = 'Fin'
spinner.finish()

#Read and concatenate particles
print('Starting to process each particle data and concatenating with Jets ...')
state = 'N'
spinner = Spinner('Concatenating particles: ')
while state != 'Fin':
    particles = ['Bjets', 'MuPlus', 'MuMinus', 'ElePlus', 'EleMinus', 'Gamma']
    d = np.array(f.get("Jets"), dtype='f')
    # d = d[(d[:,3,0]*d[:,3,0]+d[:,3,1]*d[:,3,1])>400]  #Used to generate training dataset
    for p in particles:
        d = np.concatenate((d, np.array(f.get(p), dtype='f')), axis=1) 
    spinner.next()
    state = 'Fin'
spinner.finish()

# add channel
d = np.reshape(d, (d.shape[0], 1, d.shape[1], d.shape[2]))
save_npy(d, data_save_path + 'Data/d_sm_10fb_jets.npy')
print('Done')

Read and save Event Features
print('Starting to process Event Features ...')
state = 'N'
spinner = Spinner('Reading .h5py file: ')
while state != 'Fin':
    met = np.array(f.get("EventFeatures"))
    spinner.next()
    state = 'Fin'
spinner.finish()
save_npy(met, data_save_path + 'Data/met_sm_10fb.npy')
print('Done')

