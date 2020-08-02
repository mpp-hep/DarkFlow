#!/bin/bash
#SBATCH -n 8                # Number of cores
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C P100
#SBATCH -t 24:00:00         # Runtime in HH:MM:SS
#SBATCH -o 5_train_output.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e 5_train_output.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pjawahar@wpi.edu
#SBATCH --requeue
#SBATCH --mem 196G

# Train
python3 -u ~/Projects/DarkFlow/darkflow/training/VAE_Conv2D_SparseLoss_K80.py #--train_net True --configs_file ~/dr_alcoc$





