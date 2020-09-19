#!/bin/bash
#SBATCH -n 8                # Number of cores
#SBATCH -N 1
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00         # Runtime in HH:MM:SS
#SBATCH -o test_4_output.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e test_4_output.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pjawahar@wpi.edu
#SBATCH --requeue
#SBATCH --mem 96G

# Train
python3 -u ~/Projects/DarkFlow/darkflow/testing/VAE_Conv2D_SparseLoss_Test.py
