#########################################################################
# Convert DarkMachine csv file to fixed-size numpy arrays in h5 format  #
# Author: M. Pierini (CERN)                                             #
#########################################################################

import numpy as np
import sys
import argparse
import csv
import ROOT as rt
import h5py

def readBlock(data, pid, maxP):
    myPs = np.array([])
    myZero = np.zeros((1,4))
    for p in data:
        if p.find("%s," %pid) == -1: continue
        my_P_data = p.split(",")
        myPmom = rt.TLorentzVector()
        myPmom.SetPtEtaPhiE(float(my_P_data[2])/1000., float(my_P_data[3]), float(my_P_data[4]), float(my_P_data[1])/1000.)
        myP = np.array([myPmom.Px(), myPmom.Py(), myPmom.Eta(), myPmom.M()])
        myP = myP.reshape((1,4))
        myPs = np.concatenate((myPs, myP), axis=0) if myPs.size else myP
    for i in range(myPs.shape[0], maxP):
        myPs = np.concatenate((myPs, myZero), axis=0) if myPs.size else myZero
    if myPs.shape[0] > maxP:
        myPs =  myPs[:maxP, :]
    myPs = np.reshape(myPs, (1,myPs.shape[0], myPs.shape[1]))
    return myPs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help="Input csv file", required=True)
    parser.add_argument('-o', '--output', type=str, help="Output h5 file", required=True)
    parser.add_argument('--jet', type=int, help="Maximum number of jets to store", default=10)
    parser.add_argument('--bjet', type=int, help="Maximum number of b-jets to store", default=4)
    parser.add_argument('--muplus', type=int, help="Maximum number of Mu+ to store", default=2)
    parser.add_argument('--muminus', type=int, help="Maximum number of Mu- to store", default=2)
    parser.add_argument('--eleplus', type=int, help="Maximum number of ele+ to store", default=2)
    parser.add_argument('--eleminus', type=int, help="Maximum number of ele- to store", default=2)
    parser.add_argument('--gamma', type=int, help="Maximum number of gamma to store", default=4)
    args = parser.parse_args()


    dataIn = args.input
    maxJet = args.jet
    maxBjet = args.bjet
    maxMuPlus = args.muplus
    maxMuMinus = args.muminus
    maxElePlus = args.eleplus
    maxEleMinus = args.eleminus
    maxGamma = args.gamma

    with open(dataIn) as csvfile:

        Jets = np.array([])
        Bjets = np.array([])
        MuPlus = np.array([])
        MuMinus = np.array([])
        ElePlus = np.array([])
        EleMinus = np.array([])
        Gamma = np.array([])
        ProcessID = np.array([])
        Event = np.array([])

        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            #a new event
            data = row[0].split(";")
            # 
            myProcessID = np.array([data[1]])
            myProcessID.reshape((1,1))
            #
            MET = rt.TLorentzVector()
            MET.SetPtEtaPhiM(float(data[3])/1000., 0., float(data[4]), 0.)
            myEvent = np.array([int(data[0]), float(data[2]), MET.Px(), MET.Py()])
            myEvent = np.reshape(myEvent, (1, myEvent.shape[0]))
            #
            myJets = readBlock(data, "j", maxJet)
            myBjets = readBlock(data, "b", maxBjet)
            myElePlus = readBlock(data, "e+", maxElePlus)
            myEleMinus = readBlock(data, "e-", maxEleMinus)
            myMuPlus = readBlock(data, "m+", maxMuPlus)
            myMuMinus = readBlock(data, "m-", maxMuMinus)
            myGamma = readBlock(data,"g", maxGamma)
        
            Jets = np.concatenate((Jets, myJets), axis=0) if Jets.size else myJets
            Bjets = np.concatenate((Bjets, myBjets), axis=0) if Bjets.size else myBjets
            MuPlus = np.concatenate((MuPlus, myMuPlus), axis=0) if MuPlus.size else myMuPlus
            MuMinus = np.concatenate((MuMinus, myMuMinus), axis=0) if MuMinus.size else myMuMinus
            ElePlus = np.concatenate((ElePlus, myElePlus), axis=0) if ElePlus.size else myElePlus
            EleMinus = np.concatenate((EleMinus, myEleMinus), axis=0) if EleMinus.size else myEleMinus
            Gamma = np.concatenate((Gamma, myGamma), axis=0) if Gamma.size else myGamma
            Event = np.concatenate((Event, myEvent), axis=0) if Event.size else myEvent
            ProcessID = np.concatenate((ProcessID, myProcessID), axis=0) if ProcessID.size else myProcessID

        # outputFile
        outFile = h5py.File(args.output, "w")
        outFile.create_dataset('ProcessID', data = ProcessID, compression='gzip')
        outFile.create_dataset('EventContent', data = ["evtID", "weight", "METx", "METy"], compression='gzip')
        outFile.create_dataset('EventFeatures', data=Event, compression='gzip')
        outFile.create_dataset('ParticleContent', data = ['pX', 'pY', 'Eta', 'M'], compression='gzip')
        outFile.create_dataset('Jets', data=Jets, compression='gzip')
        outFile.create_dataset('Bjets', data=Bjets, compression='gzip')
        outFile.create_dataset('MuPlus', data = MuPlus, compression='gzip')
        outFile.create_dataset('MuMinus', data = MuMinus, compression='gzip')
        outFile.create_dataset('ElePlus', data = ElePlus, compression='gzip')
        outFile.create_dataset('EleMinus', data = EleMinus, compression='gzip')
        outFile.create_dataset('Gamma', data = Gamma, compression='gzip')
                               
        outFile.close()
