#!/bin/bash
#SBATCH -n 8                # Number of cores
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C K80
#SBATCH -p short
#SBATCH -t 24:00:00         # Runtime in HH:MM:SS
#SBATCH -o 6_train_output.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e 6_train_output.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pjawahar@wpi.edu
#SBATCH --requeue
#SBATCH --mem 226G

# Train
python3 -u ~/Projects/DarkFlow/darkflow/training/VAE_Conv2D_HouseSNF.py #--train_net True --configs_file ~/dr_alcoc$





