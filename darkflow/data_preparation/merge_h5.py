import glob
import h5py
import sys
import numpy as np

dirIN = sys.argv[1]
fileOUT = sys.argv[2]

filesIN = glob.glob("%s/*h5" %dirIN)

file0 = h5py.File(filesIN[0])
EventContent = file0.get("EventContent")
ParticleContent = file0.get("ParticleContent")

ProcessID = np.array(file0.get('ProcessID'))
EventFeatures = np.array(file0.get('EventFeatures'))
Jets = np.array(file0.get('Jets'))
Bjets = np.array(file0.get('Bjets'))
MuPlus = np.array(file0.get('MuPlus'))
MuMinus = np.array(file0.get('MuMinus'))
ElePlus = np.array(file0.get('ElePlus'))
EleMinus = np.array(file0.get('EleMinus'))
Gamma = np.array(file0.get('Gamma'))

for fileIN in filesIN[1:]:
    f = h5py.File(fileIN)
    ProcessID = np.concatenate((ProcessID, f.get("ProcessID")), axis = 0)
    EventFeatures  =np.concatenate((EventFeatures, f.get('EventFeatures')), axis = 0)
    Jets = np.concatenate((Jets, f.get('Jets')), axis = 0)
    Bjets = np.concatenate((Bjets, f.get('Bjets')), axis = 0)
    MuPlus = np.concatenate((MuPlus, f.get('MuPlus')), axis = 0)
    MuMinus = np.concatenate((MuMinus, f.get('MuMinus')), axis = 0)
    ElePlus = np.concatenate((ElePlus, f.get('ElePlus')), axis = 0)
    EleMinus = np.concatenate((EleMinus, f.get('EleMinus')), axis = 0)
    Gamma = np.concatenate((Gamma, f.get('Gamma')), axis = 0)

outFile = h5py.File(fileOUT, "w")
outFile.create_dataset('ProcessID', data = ProcessID, compression='gzip')
outFile.create_dataset('EventContent', data = EventContent, compression='gzip')
outFile.create_dataset('EventFeatures', data=EventFeatures, compression='gzip')
outFile.create_dataset('ParticleContent', data = ParticleContent, compression='gzip')
outFile.create_dataset('Jets', data=Jets, compression='gzip')
outFile.create_dataset('Bjets', data=Bjets, compression='gzip')
outFile.create_dataset('MuPlus', data = MuPlus, compression='gzip')
outFile.create_dataset('MuMinus', data = MuMinus, compression='gzip')
outFile.create_dataset('ElePlus', data = ElePlus, compression='gzip')
outFile.create_dataset('EleMinus', data = EleMinus, compression='gzip')
outFile.create_dataset('Gamma', data = Gamma, compression='gzip')
outFile.close()



