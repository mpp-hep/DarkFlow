#!/bin/bash
#SBATCH -n 4                # Number of cores
#SBATCH -N 1
#SBATCH -C K80
#SBATCH -t 05:00:00         # Runtime in HH:MM:SS
#SBATCH -o data_gen.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e data_gen.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pjawahar@wpi.edu
#SBATCH --requeue
#SBATCH --mem 96G

# Train
python3 -u ~/Projects/DarkFlow/darkflow/data_preparation/data_gen.py #--configs_file ../AlcoAudio_master/alcoaudio/configs/turing_configs.json
