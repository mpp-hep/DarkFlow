#!/bin/bash
#SBATCH -n 8                # Number of cores
#SBATCH -N 1
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00         # Runtime in HH:MM:SS
#SBATCH -o main_output_%j.out  # File to which STDOUT will be written, %j insert$
#SBATCH -e main_output_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pjawahar@wpi.edu
#SBATCH --requeue
#SBATCH --mem 200G

# Train
python3 -u ~/Projects/DarkFlow/darkflow/main.py --flow iaf --test_net True --configs_file ~/Projects/DarkFlow/darkflow/configs/configs.json



