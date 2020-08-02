############################################################################# 
# Convert all DarkMachine csv file to fixed-size numpy arrays in h5 format  #
# Author: M. Pierini (CERN)                                                 #
#############################################################################          

import glob
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help="Input directory with csv files", required=True)
    parser.add_argument('-o', '--output', type=str, help="Output directory with h5 files", required=True)
    parser.add_argument('-t', '--time', type=str, default = "500000", help="Runtime in sec")

    args = parser.parse_args()

    outDir = args.output.split("/")[-1]
    os.system("mkdir %s" %outDir)
    i = 0
    for fileIN in glob.glob("%s/*csv" %args.input):
        print(fileIN)
        outFile = fileIN.split("/")[-1].replace(".csv", "")
        if os.path.exists("%s/%s.h5" %(args.output, outFile)): continue
        script = open("%s/%s.src" %(outDir, outFile), "w")
        script.write("#!/bin/bash\n")
        script.write("python %s/csv_to_numpy.py -i %s -o %s/%s.h5" %(os.getcwd(), fileIN, args.output, outFile))
        script.close()

        # condor    
        script = open("%s/%s.condor" %(outDir, outFile), "w")
        script.write("executable            = %s/%s/%s.src\n" %(os.getcwd(), outDir, outFile))
        script.write("universe       = vanilla\n")
        script.write("output                = %s/%s/%s.out\n" %(os.getcwd(), outDir, outFile))
        script.write("error                 = %s/%s/%s.err\n"  %(os.getcwd(), outDir, outFile))
        script.write("log                   = %s/%s/%s.log\n" %(os.getcwd(), outDir, outFile))
        script.write("+MaxRuntime           = %s\n" %args.time)
        script.write("queue\n")
        script.close()
        os.system("chmod a+x %s/%s/%s.src" %(os.getcwd(), outDir, outFile))
        os.system("condor_submit %s/%s/%s.condor" %(os.getcwd(), outDir, outFile))
        i = i+1
        print("submitting job n. %i: %s/%s/%s.condor" %(i, os.getcwd(), outDir, outFile))
        
