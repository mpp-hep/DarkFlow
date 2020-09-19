#!/bin/bash
#SBATCH -n 8                # Number of cores
#SBATCH -N 1
#SBATCH --partition=gpu:4
#SBATCH --gres=gpu:1
#SBATCH -C V100
#SBATCH -p emmanuel
#SBATCH -t 20:00:00         # Runtime in HH:MM:SS
#SBATCH -o train_output.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e train_output.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pjawahar@wpi.edu
#SBATCH --requeue
#SBATCH --mem 226G

# Train
python3 -u ~/Projects/DarkFlow/darkflow/training/VAE_Conv2D_HouseSNF.py #--train_net True --configs_file ~/dr_alcocudio/AlcoAudio_master/alcoaudio/configs/turing_configs.json
