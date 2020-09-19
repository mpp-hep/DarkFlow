#!/bin/bash
#SBATCH -n 8                # Number of cores
#SBATCH -N 1
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00         # Runtime in HH:MM:SS
#SBATCH -o 4train_output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e 4train_output_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pjawahar@wpi.edu
#SBATCH --requeue
#SBATCH --mem 256G

# Train
python3 -u ~/Projects/DarkFlow/darkflow/training/VAE_Conv2D_IAF.py #--train_net True --configs_file ~/dr_alcoc$





