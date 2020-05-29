############################################################################# 
# Convert all DarkMachine csv file to fixed-size numpy arrays in h5 format  #
# Author: M. Pierini (CERN)                                                 #
#############################################################################          

import glob
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help="Input csv file", required=True)
    parser.add_argument('-o', '--output', type=str, help="Output directory with h5 files", required=True)
    parser.add_argument('-n', '--nsplit', type=int, default=1, help="Number of output h5 files", required=True)
    parser.add_argument('-t', '--time', type=str, default = "500000", help="Runtime in sec")

    args = parser.parse_args()
    fileIN = args.input
    
    outDir = args.output.split("/")[-1]
    os.system("mkdir %s" %outDir)
    # how many events?
    os.system("wc %s > count.txt" %fileIN)
    count_file = open("count.txt", "r")
    line = count_file.readline()
    count = int(" ".join(line.split()).split(" ")[0])
    count_file.close()
    os.system("rm count.txt")
    for i in range(args.nsplit):
        outFile = fileIN.split("/")[-1].replace(".csv", "_%i" %i)
        if os.path.exists("%s/%s_%i.h5" %(args.output, outFile, i)): continue
        script = open("%s/%s_%i.src" %(outDir, outFile, i), "w")
        script.write("#!/bin/bash\n")
        i_first = int(float(count)/args.nsplit)*i
        i_last = int(float(count)/args.nsplit)*(i+1)
        script.write("python %s/csv_to_numpy.py -i %s -o %s/%s.h5 -f %i -l %i" %(os.getcwd(), fileIN, args.output, outFile, i_first, min(i_last,count)))
        script.close()

        # condor    
        script = open("%s/%s_%i.condor" %(outDir, outFile, i), "w")
        script.write("executable            = %s/%s/%s_%i.src\n" %(os.getcwd(), outDir, outFile, i))
        script.write("universe       = vanilla\n")
        script.write("output                = %s/%s/%s_%i.out\n" %(os.getcwd(), outDir, outFile, i))
        script.write("error                 = %s/%s/%s_%i.err\n"  %(os.getcwd(), outDir, outFile, i))
        script.write("log                   = %s/%s/%s_%i.log\n" %(os.getcwd(), outDir, outFile, i))
        script.write("+MaxRuntime           = %s\n" %args.time)
        script.write("queue\n")
        script.close()
        os.system("chmod a+x %s/%s/%s_%i.src" %(os.getcwd(), outDir, outFile, i))
        os.system("condor_submit %s/%s/%s_%i.condor" %(os.getcwd(), outDir, outFile, i))
        i = i+1
        print("submitting job n. %i: %s/%s/%s_%i.condor" %(i, os.getcwd(), outDir, outFile, i))
        
