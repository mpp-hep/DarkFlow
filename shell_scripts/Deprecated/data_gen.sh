#!/bin/bash
#SBATCH -n 4                # Number of cores
#SBATCH -N 1
#SBATCH -p emmanuel
#SBATCH -t 10:00:00         # Runtime in HH:MM:SS
#SBATCH -o data_gen.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e data_gen.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pjawahar@wpi.edu
#SBATCH --requeue
#SBATCH --mem 32G

# Train
python3 -u ~/dr_alcocudio/AlcoAudio_master/alcoaudio/datagen/data_processor.py --configs_file ../AlcoAudio_master/alcoaudio/configs/turing_configs.json
